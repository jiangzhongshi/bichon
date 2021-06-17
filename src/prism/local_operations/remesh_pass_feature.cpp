#include <igl/doublearea.h>
#include <igl/facet_components.h>
#include <igl/vertex_triangle_adjacency.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>
#include <igl/boundary_facets.h>
#include <igl/cotmatrix.h>
#include <igl/harmonic.h>
#include <igl/remove_unreferenced.h>

#include "prism/phong/query_correspondence.hpp"
#include "prism/phong/projection.hpp"
#include <algorithm>
#include <highfive/H5Easy.hpp>
#include <prism/cage_check.hpp>
#include <queue>

#include "local_mesh_edit.hpp"
#include "prism/PrismCage.hpp"
#include "prism/cage_utils.hpp"
#include "prism/energy/prism_quality.hpp"
#include "prism/feature_utils.hpp"
#include "prism/geogram/AABB.hpp"
#include "prism/spatial-hash/AABB_hash.hpp"
#include "remesh_with_feature.hpp"
#include "retain_triangle_adjacency.hpp"
#include "validity_checks.hpp"

namespace collapse {
bool satisfy_link_condition(const std::vector<Vec3i> &,
                            const std::vector<Vec3i> &,
                            const std::vector<Vec3i> &, int, int,
                            std::vector<std::pair<int, int>> &,
                            std::vector<std::pair<int, int>> &);
}

namespace prism::local_validity {
// This is a quite generic attempt, with feature trackee surgery.
// input:
//     old_id
//     new_tris
//     chain_id: for trackee surgery
// Output:
//     distrubuted sub trackee
//     local control points to be assigned later
// Return:
//    flag == 0 is success.
int attempt_feature_remesh(const PrismCage &pc,
                           const std::vector<std::set<int>> &map_track,
                           const prism::local::RemeshOptions &option,
                           // specified infos below
                           double old_quality,
                           const std::vector<int> &old_fid,
                           const std::vector<Vec3i> &moved_tris,
                           std::vector<std::set<int>> &sub_trackee,
                           std::vector<RowMatd> &local_cp) {
  auto &base = pc.base, &top = pc.top, &mid = pc.mid;
  auto &F = pc.F;
  auto num_freeze = pc.ref.aabb->num_freeze;

  std::vector<Vec3i> old_tris;
  for (auto f : old_fid) old_tris.push_back(F[f]);

  spdlog::trace("old_tris {}", old_tris);
   auto quality_before = (old_quality >= 0)
                            ? old_quality
                            : max_quality_on_tris(base, mid, top, old_tris);
  // auto quality_before = max_quality_on_tris(base, mid, top, old_tris);
  auto quality_after = max_quality_on_tris(base, mid, top, moved_tris);
  spdlog::trace("Quality compare {} -> {}", quality_before, quality_after);
  if (std::isnan(quality_after) || !std::isfinite(quality_after)) return 4;
  if ((quality_after > option.relax_quality_threshold) &&
      quality_after > quality_before)  // if the quality is too large, not allow
                                       // it to increase.
    return 4;
  //  volume check
  if (!volume_check(base, mid, top, moved_tris, num_freeze)) {
    return 1;
  }

  auto ic =
      dynamic_intersect_check(pc.base, pc.F, old_fid, moved_tris,
                              *pc.base_grid) &&
      dynamic_intersect_check(pc.top, pc.F, old_fid, moved_tris, *pc.top_grid);
  if (!ic) return 2;

  spdlog::trace("old tris {}", old_tris);
  spdlog::trace("move tris {}", moved_tris);
  sub_trackee.clear();
  if (!feature_handled_distort_check(pc, option, moved_tris, old_fid,
                                     sub_trackee))
    return 3;
  if (option.curve_checker.first.has_value() &&
      !(std::any_cast<std::function<bool(
            const PrismCage &, const std::vector<int> &,
            const std::vector<Vec3i> &, std::vector<RowMatd> &)>>(
          option.curve_checker.first))(pc, old_fid, moved_tris, local_cp)) {
    return 5;
  }

  return 0;
}

}  // namespace prism::local_validity

constexpr auto project_vertex_to_segment =
    [](const RowMatd &refV, const std::vector<int> segment_id, const Vec3d &e0,
       const Vec3d &e1, std::vector<Vec3d> &proj) -> bool {
  // Note that this is inclusive (begin and end, for no particular reason)
  proj.resize(segment_id.size());

  auto cumlength = std::vector<double>(1, 0.);
  for (auto i = 1; i < segment_id.size(); i++) {
    cumlength.push_back(
        cumlength[i - 1] +
        (refV.row(segment_id[i]) - refV.row(segment_id[i - 1])).norm());
  }
  auto poly_length = cumlength.back();
  proj.front() = e0;
  for (int i = 1, n = segment_id.size(); i <= n - 1; i++) {
    double alpha = cumlength[i] / poly_length;
    proj[i] = e0 * (1 - alpha) + e1 * alpha;
  }
  proj.back() = e1;

  return true;  // success flag for snap.
};

auto populate_tracker_shells(const std::vector<Vec3i> &FF,
                             const std::vector<std::set<int>> &map_track,
                             const std::set<int> &ref_faces_affected,
                             const std::vector<int> &old_fids) {
  std::set<int> shells;  //(old_fids.begin(), old_fids.end());
  std::queue<int> Q;
  for (auto f : old_fids) Q.push(f);
  while (!Q.empty()) {
    auto fi = Q.front();
    Q.pop();
    auto it = shells.lower_bound(fi);
    if (it == shells.end() || *it != fi) {
      if (!non_empty_intersect(map_track[fi], ref_faces_affected)) continue;
      shells.insert(fi);
      for (int j = 0; j < 3; j++) Q.push(FF[fi][j]);
    }
  }
  for (auto &f : old_fids) shells.insert(f);
  spdlog::trace("shells {}", shells);
  return std::move(shells);
}

auto expand_affected_shells(const PrismCage &pc, const std::vector<Vec3i> &FF,
       std::vector<std::vector<int>> &refVF, const std::vector<int> &newseg,
       std::vector<int> &old_fids) -> bool {
  // figure out the reference faces affected, and further, the affected
  // shells.
  std::set<int> ref_faces_affected;
  for (int i = 0; i < newseg.size(); i++) {
    if (i == 0 || i == newseg.size() - 1) continue;
    set_add_to(refVF[newseg[i]], ref_faces_affected);
  }
  spdlog::trace("affected faces {}", ref_faces_affected);
  // populate affected shells.
  auto affected_shells =
      populate_tracker_shells(FF, pc.track_ref, ref_faces_affected, old_fids);
  old_fids = std::vector<int>(affected_shells.begin(), affected_shells.end());

  RowMati F_affected(ref_faces_affected.size(), 3);
  for (auto [i, it] = std::tuple(0, ref_faces_affected.begin());
       it != ref_faces_affected.end(); it++, i++) {
    F_affected.row(i) = pc.ref.F.row(*it);
  }
  Eigen::VectorXd areas_snapped;
  igl::doublearea(pc.ref.V, F_affected, areas_snapped);
  if (areas_snapped.minCoeff() < 1e-10) {
    spdlog::debug("Snapping squeeze area={}", areas_snapped.minCoeff());
    return false;
  }
  return true;
};

constexpr auto select_middle = [](auto &refV, auto &newseg) -> int {
  std::vector<double> current_length;
  for (int i = 0; i < newseg.size() - 1; i++)
    current_length.push_back(
        (refV.row(newseg[i]) - refV.row(newseg[i + 1])).norm());
  double half_length =
      std::accumulate(current_length.begin(), current_length.end(), 0.) / 2.;
  double cum_len = current_length[0];
  double min_error = half_length * 2;
  int min_id = -1;
  for (int i = 1; i < current_length.size(); i++) {
    double round_error = std::abs(cum_len - half_length);
    if (min_error > round_error) {
      min_error = round_error;
      min_id = i;
    }
    cum_len += current_length[i];
  }
  spdlog::trace("slice {}", min_id);
  return min_id;
};

namespace prism::local {
int feature_collapse_pass(PrismCage &pc, RemeshOptions &option) {
  auto &meta_edges = pc.meta_edges;

  // meta_edges maps a single edge on the middle surface to a chain of edges on
  // reference.
  auto &F = pc.F;
  auto &V = pc.mid;
  using queue_entry = std::tuple<double, int /*f*/,
                                 int /*e*/, int /*u0*/, int /*u1*/, int /*ts*/>;
  // the smaller ones will be on top.
  std::priority_queue<queue_entry, std::vector<queue_entry>, std::greater<queue_entry>> queue;
  int orig_fnum = F.size();
  int inf_node = pc.mid.size();

  // chains in mv
  {
    auto meta_chains = prism::glue_meta_together(pc.meta_edges);
    RowMati mF;
    vec2eigen(pc.F, mF);
    std::vector<std::vector<int>> VF, VFi;
    igl::vertex_triangle_adjacency(pc.mid.size(), mF, VF, VFi);

    std::vector<Eigen::Vector2i> v0v1;
    for (auto ch : meta_chains) {
      for (auto v0 = std::next(ch.begin()); std::next(v0) != ch.end(); v0++) {
        auto v1 = std::next(v0);
        auto v2 = std::prev(v0);
        auto [f, e] = prism::vv2fe(*v0, *v1, mF, VF);
        queue.push({(V[*v0] - V[*v1]).norm() + (V[*v0] - V[*v2]).norm(), f, e, *v0, *v1, 0});
        v0v1.emplace_back(*v0, *v1);
      }
      if (ch.front() == ch.back() && ch.size() > 2) { // circular
        auto v0 = ch.front(), v2 = ch.back();
        auto v1 = *std::next(ch.begin());
        auto [f, e] = prism::vv2fe(v0, v1, mF, VF);
        queue.push({(V[v0] - V[v1]).norm() + (V[v0] - V[v2]).norm(), f, e, v0, v1, 0});
        v0v1.emplace_back(v0, v1);
      }
    }
  }
  spdlog::debug("size of queue {}", queue.size());
  std::vector<int> feature_occurence(pc.mid.size(), 0);
  for (auto [k, ignore] : meta_edges) {
    feature_occurence[k.first] ++;
    feature_occurence[k.second] ++;
  }

  auto [FF, FFi] = prism::local::triangle_triangle_adjacency(F);
  std::vector<std::vector<int>> refVF, refVFi;
  igl::vertex_triangle_adjacency(pc.ref.V.rows(), pc.ref.F, refVF, refVFi);
   auto find_u2 = [&F, &feature_occurence, &meta_edges](auto n0, auto u0, auto u1){
     if (feature_occurence[u0] > 2) return -1;
      auto u2 = -1;
    for (auto [f, e] : n0) {  
           // find the adjacent (in-chain) vertex of u0 except u1.
      auto v = F[f][(e + 1) % 3];
      if (v != u1 && feature_occurence[v] > 0 &&
          meta_edges.find({v, u0}) != meta_edges.end()) {
            u2 = v;
      }
    }
    return u2;
    };
  std::vector<int> rejections_steps(8, 0);
  int global_tick = 0;
  while (!queue.empty()) {
    auto [l, f, e, v0, v1, ignore] = queue.top();
    queue.pop();
    if (f == -1 || FF[f][e] == -1) continue;  // skip collapsed

    auto u0 = F[f][e], u1 = F[f][(e + 1) % 3];
    if (u0 == u1 || u0 != v0 ||
        u1 != v1)  // vid changed, means the edge is outdated.
      continue;

    l = (V[v1] - V[v0]).norm();
    if (l * 2.5 > option.sizing_field(V[u0]) * option.target_adjustment[u0] +
                      option.sizing_field(V[u1]) * option.target_adjustment[u1])
      continue;  // skip if l > 4/5*(s1+s2)/2

    spdlog::trace(">>>>>>LinkCondition check {} {}", f, e);
    // collapse and misc checks.
    std::vector<std::pair<int, int>> n0, n1;
    if (!collapse::satisfy_link_condition(F, FF, FFi, f, e, n0, n1)) {
      rejections_steps[0]++;
      continue;
    }

    spdlog::trace("LinkCondition pass, attempt {} {}", f, e);

    auto f1 = FF[f][e], e1 = FFi[f][e];  // fake face and edge

    // snapping location, proposing remove u0.
    auto seg1 = meta_edges.find({u0, u1});
    assert(seg1 != meta_edges.end() && (seg1->second.second[0]) != -1);
    int chain_id = seg1->second.first;
    spdlog::trace("chain {} ", chain_id);
   
    auto u2 = find_u2(n0, u0, u1);
    if (u2 == -1) {
      spdlog::trace(">>>> Hit Head ");
      rejections_steps[0]++;
      continue;
    }
    spdlog::trace("u1 {} <- u0 {} <- u2 {}", u1, u0, u2);
    auto seg0 = meta_edges.find({u2, u0});
    assert(seg0 != meta_edges.end());

    if (chain_id != seg0->second.first) {
      spdlog::trace("different chain");
      rejections_steps[0]++;
      continue;
    }

    // concatentate seg(u2,u0) and seg(u1,u0) and snap/project.
    std::vector<int> newseg = seg0->second.second;
    if (newseg.back()!= seg1->second.second.front()) {
      spdlog::trace("newseg {}", newseg);
      spdlog::trace("seg1 {}", seg1->second.second);
      spdlog::dump_backtrace();
      exit(1);
    }
    assert(newseg.back() == seg1->second.second.front());
    newseg.insert(newseg.end(), seg1->second.second.begin() + 1,
                  seg1->second.second.end());
    std::vector<Vec3d> seg_snaps;
    project_vertex_to_segment(pc.ref.V, newseg, V[u2], V[u1], seg_snaps);

    std::vector<Vec3d> seg_restore(newseg.size());
    for (int i = 0; i < newseg.size(); i++) {
      seg_restore[i] = pc.ref.V.row(newseg[i]);        // whether refV or inpV?
      if (i == 0 || i == newseg.size() - 1) continue;  // bitwise exact
      pc.ref.V.row(newseg[i]) = seg_snaps[i];
    }
    std::vector<int> old_fids;
    for (auto [f, e] : n0) old_fids.push_back(f);
    for (auto [f, e] : n1) old_fids.push_back(f);
    std::sort( old_fids.begin(), old_fids.end() );
    old_fids.erase( std::unique( old_fids.begin(), old_fids.end() ), old_fids.end() );

    auto rollback = [&](){
      for (auto i = 0; i < newseg.size(); i++) {
        pc.ref.V.row(newseg[i]) = seg_restore[i];
      }
    };
    if (!expand_affected_shells(pc, FF, refVF, newseg, old_fids)) {
      rejections_steps[0]++;
      rollback();
      continue;
    }

    std::tuple<Vec3d, Vec3d> recover_coordinates{pc.base[u1], pc.top[u1]};
    std::vector<Vec3i> moved_tris;  //(old_fids.size() - 2);
    std::vector<int> new_fid;
    for (auto f : old_fids) {
      auto new_f = F[f];
      for (auto j = 0; j < 3; j++) {
        if (new_f[j] == u0) new_f[j] = u1;
      }
      if (new_f[0] == new_f[1] || new_f[1] == new_f[2] || new_f[0] == new_f[2])
        continue;
      new_fid.push_back(f);
      moved_tris.emplace_back(new_f);
    }
    auto new_shifts = prism::local_validity::triangle_shifts(moved_tris);
    spdlog::trace("moved_tris {}", moved_tris);

    auto [new_it, it_flag] =
        meta_edges.emplace(std::pair(u2, u1), std::pair(chain_id, newseg));
    if (!it_flag) {
      spdlog::trace("already exist.");
      rejections_steps[0]++;
      rollback();
      continue;
    }

    std::vector<std::set<int>> new_tracks;
    std::vector<RowMatd> local_cp;
    // auto flag = prism::local_validity::attempt_feature_remesh(
        // pc, pc.track_ref, option, -1, old_fids, moved_tris, new_tracks, local_cp);

    auto flag = [&pc, &option, f0 = f, &f1, &u0, &u1,
                 &new_tracks, &local_cp,
                 &old_fids , &moved_tris = moved_tris](int repeat_num) -> int {
      for (auto rp = 0; rp < repeat_num; rp++) {
        local_cp.clear();
        new_tracks.clear();
        auto flag = prism::local_validity::attempt_feature_remesh(
            pc, pc.track_ref, option, -1, old_fids, moved_tris, new_tracks, local_cp);

        if (flag != 1)  // if fail not due to volume, accept the conclusion
          return flag;
        pc.base[u1] = (pc.base[u1] + pc.mid[u1]) / 2;
        pc.top[u1] = (pc.top[u1] + pc.mid[u1]) / 2;
      }
      return 1;
    }(3);

    spdlog::trace("Attempt Feature Collapse, {} {} pass: {}{}", f, e,
                  (flag == 0), flag);

    if (flag != 0) {
      rollback();
      std::tie(pc.base[u1], pc.top[u1]) = recover_coordinates;
      meta_edges.erase(new_it);
      spdlog::trace("Rollback V");
      rejections_steps[flag]++;
      continue;
    }

    assert(new_fid.size() == new_shifts.size());

    prism::edge_collapse(F, FF, FFi, f, e);
    spdlog::trace("EdgeCollapse done {} {}", f, e);
    assert(new_fid.size() == new_tracks.size());
    prism::local_validity::post_operation(pc, option, old_fids, new_fid, new_tracks, local_cp);

    // shifts
    shift_left(new_fid, new_shifts, F, FF, FFi);
    spdlog::trace("insert {} {}", u2, u1);
    meta_edges.erase(seg0);
    meta_edges.erase(seg1);
    auto [newf, newe] = [&](){
      for (int i = 0; i < new_fid.size(); i++) {
      auto f = new_fid[i];
      for (auto e = 0; e < 3; e++) {
        auto v0 = F[f][e], v1 = F[f][(e + 1) % 3];
        if (v0 == u2 && v1 == u1) {
          return std::pair(f,e);
        }
      }
    }
     return std::pair(-1,-1);
    }();
    if (newf == -1) throw std::runtime_error("not pushed in feature collapse");
    std::vector<std::pair<int, int>> newn0;
    prism::get_star_edges(F, FF, FFi, newf, newe, newn0);
    auto newu2 = find_u2(newn0, u2, u1);
    if (newu2 == -1) continue;
    queue.push({(V[u2] - V[u1]).norm() + (V[u2] - V[newu2]).norm(), f, e, v0, v1, global_tick});

    global_tick++;
    spdlog::trace("Feature Edge Collapsed {} {}", f, e);
  }
  spdlog::info("Pass Snapper Feature Collapse total {}. lk{}, v{} i{} n{} q{} c{}",
               global_tick, rejections_steps[0], rejections_steps[1],
               rejections_steps[2], rejections_steps[3], rejections_steps[4],
               rejections_steps[5]);
  F.resize(orig_fnum);
  Eigen::VectorXi vid_map, vid_ind;  // new to old
  pc.cleanup_empty_faces(vid_map, vid_ind);
  for (int i = 0; i < vid_ind.size(); i++) {
    option.target_adjustment[i] = option.target_adjustment[vid_ind[i]];
  }
  option.target_adjustment.resize(vid_ind.size());

  return global_tick;
}
}  // namespace prism::local

namespace prism::local {
int feature_slide_pass(PrismCage &pc, RemeshOptions &option) {
  auto &meta_edges = pc.meta_edges;
  auto &F = pc.F;
  auto &V = pc.mid;
  auto [FF, FFi] = prism::local::triangle_triangle_adjacency(F);
  auto &inpV = pc.ref.inpV;

  int global_ticks = 0;
  std::vector<int> verts_on_feat;
  for (auto [m, ign] : meta_edges) {
    verts_on_feat.push_back(m.first);
    verts_on_feat.push_back(m.second);
  }
  std::sort(verts_on_feat.begin(), verts_on_feat.end());
  verts_on_feat.erase(std::unique(verts_on_feat.begin(), verts_on_feat.end()),
                      verts_on_feat.end());

  std::vector<std::vector<int>> refVF, refVFi;
  igl::vertex_triangle_adjacency(pc.ref.V.rows(), pc.ref.F, refVF, refVFi);
  std::vector<std::vector<int>> VF, VFi;
  igl::vertex_triangle_adjacency(
      V.size(), Eigen::Map<RowMati>(F[0].data(), F.size(), 3), VF, VFi);
  std::vector<int> rejections_steps(8, 0);
  for (auto vid : verts_on_feat) {
    auto nb = VF[vid], nbi = VFi[vid];
    // only do sth when two feature
    std::vector<decltype(meta_edges.begin())> nb_feat;
    for (auto i = 0; i < nb.size(); i++) {
      auto v0 = vid, v1 = F[nb[i]][(nbi[i] + 1) % 3],
           v2 = F[nb[i]][(nbi[i] + 2) % 3];
      auto it = meta_edges.find({v0, v1});
      if (it != meta_edges.end()) nb_feat.emplace_back(it);
      it = meta_edges.find({v2, v0});
      if (it != meta_edges.end()) nb_feat.emplace_back(it);
    }
    if (nb_feat.size() != 2) continue;
    auto it0 = nb_feat[0], it1 = nb_feat[1];
    if (it0->second.first != it1->second.first) continue;  // different chain
    int chain_id = it0->second.first;
    if (it0->first.first == vid) std::swap(it0, it1);
    // (v2) -->--it0-->--(vid)-->--it1-->--v1
    assert(it0->first.second == vid && it1->first.first == vid);
    auto v1 = it1->first.second, v2 = it0->first.first;
    spdlog::trace("u1 {} <- u0 {} <- u2 {}", v1, vid, v2);

    std::vector<int> newseg = it0->second.second;
    newseg.insert(newseg.end(), it1->second.second.begin() + 1,
                  it1->second.second.end());
    if (newseg.size() < 4) continue;  // no need to slide

    auto slice_id = select_middle(inpV, newseg);  // invoked
#ifndef NDEBUG
    if (slice_id == -1) {
      spdlog::error("Cannot find a position to snap, this should not happen.");
      spdlog::dump_backtrace();
    }
#endif
    if (slice_id + 1 == it0->second.second.size()) continue;  // no shift
    std::vector<int> seg0(newseg.begin(), newseg.begin() + slice_id + 1);
    std::vector<int> seg1(newseg.begin() + slice_id, newseg.end());
    auto center_refid = seg1.front();
    std::vector<Vec3d> seg_snaps, seg1_snaps;
    spdlog::trace("{} -> {} -> {}", seg0, center_refid, seg1);

    project_vertex_to_segment(inpV, seg0, V[v2], inpV.row(center_refid),
                              seg_snaps);
    project_vertex_to_segment(inpV, seg1, inpV.row(center_refid), V[v1],
                              seg1_snaps);

    seg_snaps.insert(seg_snaps.end(), seg1_snaps.begin() + 1, seg1_snaps.end());

    std::vector<Vec3d> seg_restore(newseg.size());
    for (int i = 0; i < newseg.size(); i++) {
      seg_restore[i] = pc.ref.V.row(newseg[i]);
      if (i == 0 || i == newseg.size() - 1) continue;  // bitwise exact
      pc.ref.V.row(newseg[i]) = seg_snaps[i];
    }
    std::tuple<Vec3d, Vec3d, Vec3d> recover_coordinates{
        pc.base[vid], pc.mid[vid], pc.top[vid]};

    auto rollback = [&, seg0_r = it0->second.second, seg1_r = it1->second.second]() {
      for (auto i = 0; i < newseg.size(); i++) {
        pc.ref.V.row(newseg[i]) = seg_restore[i];
      }
      std::tie(pc.base[vid], pc.mid[vid], pc.top[vid]) = recover_coordinates;
      it0->second.second = std::move(seg0_r);
      it1->second.second = std::move(seg1_r);
    };
    std::vector<int> old_fids = nb;
    if (!expand_affected_shells(pc, FF, refVF, newseg, old_fids)) {
      rejections_steps[0]++;
      rollback();
      continue;
    }
    std::vector<Vec3i> moved_tris;
    for (auto f : old_fids) {
      moved_tris.push_back(pc.F[f]);
    }
    double old_quality = prism::local_validity::max_quality_on_tris(
      pc.base, pc.mid, pc.top, moved_tris);

    // parallel offset
    pc.base[vid] += inpV.row(center_refid) - pc.mid[vid];
    pc.top[vid] += inpV.row(center_refid) - pc.mid[vid];
    pc.mid[vid] = inpV.row(center_refid);

    it0->second.second = std::move(seg0);
    it1->second.second = std::move(seg1);
    std::vector<std::set<int>> new_tracks;
    std::vector<RowMatd> local_cp;
    auto flag = prism::local_validity::attempt_feature_remesh(
        pc, pc.track_ref, option, old_quality, old_fids, moved_tris, new_tracks, local_cp);
    spdlog::trace("Attempt Feature Slide, {} pass: {}{}", vid, (flag == 0),
                  flag);
    if (flag != 0) {
      rollback();
      spdlog::trace("Rollback V");
      rejections_steps[flag]++;
      continue;
    }
    spdlog::trace("Slide happening {}", vid);
    global_ticks++;
    auto &new_fids = old_fids;
    assert(moved_tris.size() == new_tracks.size());
    prism::local_validity::post_operation(pc, option, old_fids, new_fids, new_tracks, local_cp);
  }
  spdlog::info("Snapper Feature Slide. {}", global_ticks);
  return global_ticks;
}

int feature_split_pass(PrismCage &pc, prism::local::RemeshOptions &option) {
  auto &meta_edges = pc.meta_edges;
  auto &F = pc.F;
  auto &V = pc.mid;
  auto &inpV = pc.ref.inpV;
  auto input_vnum = V.size();
  auto [FF, FFi] = prism::local::triangle_triangle_adjacency(F);

  using queue_entry =
      std::tuple<double, int /*f*/, int /*e*/, int /*u0*/, int /*u1*/>;
  std::multiset<queue_entry> mutlqueue;
  std::priority_queue<queue_entry> queue;
  for (int f = 0; f < F.size(); f++) {
    for (auto e : {0, 1, 2}) {
      auto v0 = F[f][e], v1 = F[f][(e + 1) % 3];
      if (meta_edges.find({v0, v1}) == meta_edges.end())
        continue;  // this can be done with an easier way if know vv2fe.
      queue.push({(V[v0] - V[v1]).norm(), f, e, v0, v1});
    }
  }

  std::vector<std::vector<int>> refVF, refVFi;
  igl::vertex_triangle_adjacency(pc.ref.V.rows(), pc.ref.F, refVF, refVFi);
  std::vector<std::vector<int>> VF, VFi;
  igl::vertex_triangle_adjacency(
      V.size(), Eigen::Map<RowMati>(F[0].data(), F.size(), 3), VF, VFi);
  std::vector<int> rejections_steps(8, 0);
  int global_tick = 0;
  while (!queue.empty()) {
    auto [l, f0, e0, u0, u1] = queue.top();
    queue.pop();
    if (f0 == -1 || FF[f0][e0] == -1) continue;  // skip boundary
    if (auto u0_ = F[f0][e0], u1_ = F[f0][(e0 + 1) % 3];
        u0_ == u1_ || u0_ != u0 ||
        u1_ != u1)  // vid changed, means the edge is outdated.
      continue;

    if (std::abs(l) * 1.5 <
        (option.sizing_field(V[u0]) * option.target_adjustment[u0] +
         option.sizing_field(V[u1]) * option.target_adjustment[u1])) {
      //  spdlog::debug("skip");
      continue;  // skip if l < 4/3*(s1+s2)/2
    }

    auto f1 = FF[f0][e0], e1 = FFi[f0][e0];
    if (f1 == -1) continue;  // boundary check
    auto v0 = F[f0][(e0 + 2) % 3];
    auto v1 = F[f1][(e1 + 2) % 3];
    spdlog::trace(">>>>>> Entering: {}-{} {}-{} {}->{} {}-{}", f0, e0, f1, e1,
                  u0, u1, v0, v1);

    auto current_meta_it = meta_edges.find({u0, u1});
    assert(current_meta_it != meta_edges.end());

    auto [cid, segs] = current_meta_it->second;
    if (segs.size() <= 2) continue;
    auto slice_id = select_middle(inpV, segs);
    spdlog::trace("sliceid {} seglen {}", slice_id, segs.size());

    if (slice_id >= segs.size()) {
      spdlog::warn("violation");
      spdlog::dump_backtrace();
      continue;
    };
    auto seg0 = std::vector<int>(segs.begin(), segs.begin() + slice_id + 1);
    auto seg1 = std::vector<int>(segs.begin() + slice_id, segs.end());
    auto center_refid = seg1.front();
    std::vector<Vec3d> seg_snaps, seg1_snaps;
    spdlog::trace("{} -> {} -> {}", seg0, center_refid, seg1);

    project_vertex_to_segment(inpV, seg0, V[u0], inpV.row(center_refid),
                              seg_snaps);
    project_vertex_to_segment(inpV, seg1, inpV.row(center_refid), V[u1],
                              seg1_snaps);

    seg_snaps.insert(seg_snaps.end(), seg1_snaps.begin() + 1, seg1_snaps.end());
    assert(segs.size() == seg_snaps.size());
    std::vector<Vec3d> seg_restore(segs.size());
    for (int i = 0; i < segs.size(); i++) {
      seg_restore[i] = pc.ref.V.row(segs[i]);
      if (i == 0 || i == segs.size() - 1) continue;  // bitwise exact
      pc.ref.V.row(segs[i]) = seg_snaps[i];
    }
    auto ux = pc.mid.size();

    auto [new_it0, it_flag0] =
        meta_edges.emplace(std::pair(u0, ux), std::pair(cid, seg0));
    auto [new_it1, it_flag1] =
        meta_edges.emplace(std::pair(ux, u1), std::pair(cid, seg1));
    if (it_flag0==false  || it_flag1==false) {
      spdlog::error("should be able to emplace");
      throw std::runtime_error("Check feature split pass.");
    }

    auto rollback = [&, &newseg = segs, &it0 = new_it0, &it1 = new_it1]() {
      for (auto i = 0; i < newseg.size(); i++) {
        pc.ref.V.row(newseg[i]) = seg_restore[i];
      }
      meta_edges.erase(it0);
      meta_edges.erase(it1);
      if (pc.mid.size() > ux) {
        pc.mid.pop_back();
        pc.base.pop_back();
        pc.top.pop_back();
      }
    };
    std::vector<int> old_fids = {f0, f1};
    if (!expand_affected_shells(pc, FF, refVF, segs, old_fids)) {
      rejections_steps[0]++;
      rollback();
      continue;
    }

    auto mid_pos = inpV.row(center_refid);
    pc.base.push_back((pc.base[u0] + pc.base[u1]) / 2);
    pc.mid.push_back(mid_pos);
    pc.top.push_back((pc.top[u0] + pc.top[u1]) / 2);
    pc.base.back() += mid_pos - (pc.mid[u0] + pc.mid[u1]) / 2;
    pc.top.back() += mid_pos - (pc.mid[u0] + pc.mid[u1]) / 2;

    std::vector<int> new_fid = {f0, f1, int(F.size()), int(F.size() + 1)};
    auto moved_tris = std::vector<Vec3i>{F[f0], F[f1], F[f0], F[f1]};
    moved_tris[0][e0] = ux;
    moved_tris[1][e1] = ux;
    moved_tris[2][(e0 + 1) % 3] = ux;
    moved_tris[3][(e1 + 1) % 3] = ux;
    for (auto f : old_fids) {
      if (f != f0 && f != f1) {
        moved_tris.emplace_back(F[f]);
        new_fid.emplace_back(f);
      }
    }
    auto new_shifts = prism::local_validity::triangle_shifts(moved_tris);

    std::vector<std::set<int>> new_tracks;
    std::vector<RowMatd> local_cp;
    auto flag = prism::local_validity::attempt_feature_remesh(
        pc, pc.track_ref, option, -1, old_fids, moved_tris, new_tracks, local_cp);
    if (flag != 0) {
      rollback();
      spdlog::trace("Rollback V");
      rejections_steps[flag]++;
      continue;
    }
    spdlog::trace("Feature split happening {} {}", u0, u1);

    prism::edge_split(pc.mid.size() - 1, F, FF, FFi, f0, e0);

    prism::local_validity::post_operation(pc, option, old_fids, new_fid, new_tracks, local_cp);

    meta_edges.erase(current_meta_it);
    // shifts
    global_tick++;
    shift_left(new_fid, new_shifts, F, FF, FFi);
    option.target_adjustment.push_back(
        (option.target_adjustment[u0] + option.target_adjustment[u1]) / 2);

    // no pushing for now.
  }
  spdlog::info("Complete a pass of feature split. {}", global_tick);
  return global_tick;
}

auto inverse_project_discrete =
    [](const auto &base, const auto &mid, const auto &top, const auto &f,
       const auto &pp) -> std::optional<std::array<double, 3>> {
  auto [v0, v1, v2] = f;
  bool split = v1 > v2;
  auto tup = std::array<double, 3>();
  if (!prism::phong::phong_projection(
          {base[v0], base[v1], base[v2], mid[v0], mid[v1], mid[v2]}, pp, split,
          tup) &&
      (!prism::phong::phong_projection(
          {mid[v0], mid[v1], mid[v2], top[v0], top[v1], top[v2]}, pp, split,
          tup))) {
    // neither in top nor bottom.
    return {};
  }
  return tup;
};

auto smooth_local_trackee = [](PrismCage &pc, auto f) {
  auto &inpV = pc.ref.inpV;
  auto &refV = pc.ref.V;
  auto &refF = pc.ref.F;
  auto &track = pc.track_ref[f];
  RowMati trackF(track.size(), 3);
  for (auto [i, it] = std::tuple(0, track.begin()); i < track.size();
       i++, it++) {
    trackF.row(i) = refF.row(*it);
  }
  Eigen::VectorXi I, J;
  igl::remove_unreferenced(refV.rows(), trackF, I, J);
  std::for_each(trackF.data(), trackF.data() + trackF.size(),
                [&I](auto &a) { a = I(a); });
  auto localV = RowMatd(J.size(), 3);
  for (auto v = 0; v < J.size(); v++) {
    localV.row(v) = refV.row(J[v]);
  }
  RowMati E;
  igl::boundary_facets(trackF, E);
  auto local_fixed = std::vector<bool>(J.size());
  for (auto i = 0; i < E.rows(); i++) {
    local_fixed[E(i, 0)] = true;
    local_fixed[E(i, 1)] = true;
  }
  for (auto v = 0; v < J.size(); v++) {  // exclude outsiders.
    if (local_fixed[v]) continue;
    Vec3d pp = refV.row(J[v]);
    auto tup = inverse_project_discrete(pc.base, pc.mid, pc.top, pc.F[f], pp);
    if (!tup) local_fixed[v] = true;
  }

  // harmonic stuff.
  Eigen::VectorXi b(localV.rows());
  RowMatd bc(localV.rows(), 3);
  auto num_fixed = 0;
  for (auto v = 0; v < local_fixed.size(); v++) {
    if (local_fixed[v]) {
      b[num_fixed] = v;
      bc.row(num_fixed) = refV.row(J[v]);
      num_fixed++;
    }
  }
  b.conservativeResize(num_fixed);
  bc.conservativeResize(num_fixed, 3);
  RowMatd local_W;
  igl::harmonic(trackF, b, bc, 1, local_W);
  return std::tuple(std::move(J), std::move(localV), std::move(local_W));
  // spdlog::info("\n{}", W);
};

int smooth_tracked_reference(PrismCage &pc,
                             prism::local::RemeshOptions &option) {
  auto single = [&](auto &FF, auto sh_id) {
    auto [J, localV, localW] = smooth_local_trackee(pc, sh_id);

    auto rollback = [&pc, &J = J, &localV = localV]() {
      for (auto v = 0; v < J.size(); v++) {
        pc.ref.V.row(J[v]) = localV.row(v);
      }
    };
    for (auto v = 0; v < J.size(); v++) {
      pc.ref.V.row(J[v]) = localW.row(v);
    }

    auto old_fidset =
        populate_tracker_shells(FF, pc.track_ref, pc.track_ref[sh_id], {sh_id});
    auto old_fids = std::vector<int>(old_fidset.begin(), old_fidset.end());
    auto moved_tris = std::vector<Vec3i>();
    for (auto f : old_fids) moved_tris.emplace_back(pc.F[f]);

    std::vector<std::set<int>> new_tracks;
    std::vector<RowMatd> local_cp;
    auto flag = prism::local_validity::attempt_feature_remesh(
        pc, pc.track_ref, option, -1, old_fids, moved_tris, new_tracks,
        local_cp);
    if (flag != 0) {
      spdlog::debug("Reject {} Rollback", flag);
      rollback();
      return flag;
    }
    spdlog::debug("happening {} ", sh_id);
    prism::local_validity::post_operation(pc, option, old_fids, old_fids,
                                          new_tracks, local_cp);
    return 0;
  };
  auto [FF, FFi] = prism::local::triangle_triangle_adjacency(pc.F);
  auto cnt = 0;
  for (auto sh_id = 0; sh_id < pc.F.size(); sh_id++) {
    auto flag= single(FF, sh_id);
    if (flag == 0) cnt ++;
  }
  spdlog::info("Reference Mesh Smoother {}", cnt);
  return cnt;
}

}  // namespace prism::local
