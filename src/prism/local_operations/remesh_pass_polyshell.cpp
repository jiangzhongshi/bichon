#include <igl/vertex_triangle_adjacency.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <highfive/H5Easy.hpp>
#include <prism/PrismCage.hpp>
#include <prism/cage_check.hpp>
#include <prism/geogram/AABB.hpp>
#include <prism/polyshell_utils.hpp>
#include <queue>

#include "prism/cage_utils.hpp"
#include "prism/energy/prism_quality.hpp"
#include "prism/energy/smoother_pillar.hpp"
#include "prism/feature_utils.hpp"
#include "prism/local_operations/local_mesh_edit.hpp"
#include "prism/local_operations/remesh_pass.hpp"
#include "prism/local_operations/retain_triangle_adjacency.hpp"
#include "prism/local_operations/validity_checks.hpp"
#include "prism/spatial-hash/AABB_hash.hpp"
#include "remesh_with_feature.hpp"

namespace collapse {
bool satisfy_link_condition(const std::vector<Vec3i> &,
                            const std::vector<Vec3i> &,
                            const std::vector<Vec3i> &, int, int,
                            std::vector<std::pair<int, int>> &,
                            std::vector<std::pair<int, int>> &);
}

constexpr auto select_middle = [](auto &segs) {
  assert(segs.size() >= 2);
  // change to fractional based.
  auto cumlength = std::vector<double>(segs.size(), 0.);
  for (auto i = 1; i < segs.size(); i++) {
    cumlength[i] = (cumlength[i - 1] + (segs[i] - segs[i - 1]).norm());
  }
  auto half_len = cumlength.back() / 2;
  auto it = std::find_if(cumlength.begin(), cumlength.end(),
                         [half_len](auto &x) { return x > half_len; });
  assert(it != cumlength.begin() && "should not be zero cumlength");
  auto it0 = std::prev(it);
  auto id = std::distance(cumlength.begin(), it0);
  auto offset = (half_len - *it0) / (*it - *it0);
  if (offset < 1e-3) offset = 0.;
  if (offset > 1 - 1e-3) {
    id++;
    offset = 0.;
  }
  return std::tuple(id, offset);
};

constexpr auto split_segs_in_the_middle = [](const Vec3d &p0, const Vec3d &p1,
                                             const RowMatd &inpV,
                                             const std::vector<int> &segs) {
  // perform zig append. minor duplication wrt zig_constructor
  assert(segs.size() >= 2 && "Switching to the end-inclusive segs now");
  std::vector<Vec3d> seg_pos(segs.size());
  seg_pos.front() = p0;
  seg_pos.back() = p1;
  for (auto i = 1; i < segs.size() - 1; i++)
    seg_pos[i] = inpV.row(segs[i]);


  auto [slice_id, offset] = select_middle(seg_pos);
  spdlog::trace("sliceid {} seglen {}", slice_id, seg_pos.size());

  assert(slice_id < seg_pos.size());
  auto mid_pos = Vec3d();
  if (offset == 0.) { // exactly at one of the vertices.
    mid_pos = seg_pos[slice_id]; // bitwise
  } else {
    mid_pos = seg_pos[slice_id + 1] * offset + seg_pos[slice_id] * (1 - offset);
  }

  auto seg0 = std::vector<int>();
  auto seg1 = std::vector<int>();

  seg0 = std::vector<int>(segs.begin(), segs.begin() + slice_id + 2);
  seg1 = std::vector<int>(segs.begin() + slice_id, segs.end());
  if (offset == 0.) {
    seg0.pop_back();
    assert(seg0.back() == seg1.front());
  }
  spdlog::trace("segs {} -> {} + {}", segs, seg0, seg1);
  return std::tuple(std::move(seg0), std::move(seg1), mid_pos);
};

int prism::local::zig_collapse_pass(PrismCage &pc, RemeshOptions &option) {
  auto &meta_edges = pc.meta_edges;

  // meta_edges maps a single edge on the middle surface to a chain of edges on
  // reference.
  auto &F = pc.F;
  auto &V = pc.mid;
  using queue_entry = std::tuple<double /*should negative*/, int /*f*/,
                                 int /*e*/, int /*u0*/, int /*u1*/, int /*ts*/>;
  std::priority_queue<queue_entry> queue;
  int orig_fnum = F.size();
  int inf_node = pc.mid.size();
  for (int f = 0; f < orig_fnum; f++) {
    for (auto e : {0, 1, 2}) {
      auto v0 = F[f][e], v1 = F[f][(e + 1) % 3];
      if (meta_edges.find({v0, v1}) == meta_edges.end())
        continue; // this can be done with an easier way if know vv2fe.
      queue.push({-(V[v0] - V[v1]).norm(), f, e, v0, v1, 0});
    }
  }
  std::set<int> feature_verts;
  for (auto [k, ignore] : meta_edges) {
    feature_verts.insert(k.first);
    feature_verts.insert(k.second);
  }

  auto [FF, FFi] = prism::local::triangle_triangle_adjacency(F);
  std::vector<std::vector<int>> refVF, refVFi;
  igl::vertex_triangle_adjacency(pc.ref.V.rows(), pc.ref.F, refVF, refVFi);
  std::vector<int> rejections_steps(10, 0);
  // pop
  int global_tick = 0;
  while (!queue.empty()) {
    auto [l, f, e, v0, v1, ignore] = queue.top();
    l = std::abs(l);
    queue.pop();
    if (f == -1 || FF[f][e] == -1)
      continue; // skip collapsed

    auto u0 = F[f][e], u1 = F[f][(e + 1) % 3];
    if (u0 == u1 || u0 != v0 ||
        u1 != v1) // vid changed, means the edge is outdated.
      continue;
    assert((V[u1] - V[u0]).norm() == l &&
           "Outdated entries will be ignored, this condition can actually "
           "replace the previous");

    if (l * 2.5 > option.sizing_field(V[u0]) * option.target_adjustment[u0] +
                      option.sizing_field(V[u1]) * option.target_adjustment[u1])
      continue; // skip if l > 4/5*(s1+s2)/2

    spdlog::trace(">>>>>>LinkCondition check {} {}", f, e);
    // collapse and misc checks.
    std::vector<std::pair<int, int>> n0, n1;
    if (!collapse::satisfy_link_condition(F, FF, FFi, f, e, n0, n1)) {
      rejections_steps[0]++;
      continue;
    }

    spdlog::trace("LinkCondition pass, attempt {} {}", f, e);

    auto f1 = FF[f][e], e1 = FFi[f][e]; // fake face and edge

    // snapping location, proposing remove u0.
    auto seg1 = meta_edges.find({u0, u1});
    assert(seg1 != meta_edges.end());
    auto u2 = -1;
    for (auto [f, e] :
         n0) { // find the adjacent (in-chain) vertex of u0 except u1.
      auto v = F[f][(e + 1) % 3];
      if (v != u1 && feature_verts.find(v) != feature_verts.end() &&
          meta_edges.find({v, u0}) != meta_edges.end())
        u2 = v;
    }
    if (u2 == -1) {
      spdlog::trace(">>>> Hit Head ");
      rejections_steps[0]++;
      continue;
    }
    spdlog::trace("u1 {} <- u0 {} <- u2 {}", u1, u0, u2);
    auto seg0 = meta_edges.find({u2, u0});
    assert(seg0 != meta_edges.end());

    int chain_id = seg1->second.first;
    spdlog::trace("chain {} ", chain_id);
    if (chain_id != seg0->second.first) {
      spdlog::trace("different chain");
      rejections_steps[0]++;
      continue;
    }

    // concatentate seg(u2,u0) and seg(u1,u0)
    auto newseg = seg0->second.second;
    if (newseg.back() != seg1->second.second.front()) { // exact vertex
      assert(seg1->second.second[1] == newseg.back());
      newseg.pop_back();
    }
    newseg.pop_back();
    newseg.insert(newseg.end(), seg1->second.second.begin(),
                  seg1->second.second.end());

    std::vector<int> old_fids;
    for (auto [f, e] : n0)
      old_fids.push_back(f);

    std::tuple<std::vector<int> /*newfid*/, std::vector<int> /*shifts*/,
               std::vector<std::set<int>> /*track*/>
        checker;

    std::tuple<Vec3d, Vec3d> recover_coordinates{pc.base[u1], pc.top[u1]};
    std::vector<Vec3i> moved_tris; //(old_fids.size() - 2);
    std::vector<int> new_fid;
    for (auto f : old_fids) {
      auto new_f = F[f];
      for (auto j = 0; j < 3; j++) {
        if (new_f[j] == u0)
          new_f[j] = u1;
      }
      if (new_f[0] == new_f[1] || new_f[1] == new_f[2] || new_f[0] == new_f[2])
        continue;
      new_fid.push_back(f);
      moved_tris.emplace_back(new_f);
    }
    auto new_shifts = prism::local_validity::triangle_shifts(moved_tris);
    std::vector<std::set<int>> new_tracks;
    spdlog::trace("moved_tris {}", moved_tris);

    auto [new_it, it_flag] =
        meta_edges.emplace(std::pair(u2, u1), std::pair(chain_id, newseg));

    std::vector<RowMatd> local_cp;
    auto flag = prism::local_validity::attempt_zig_remesh(
        pc, pc.track_ref, option, -1, old_fids, moved_tris, new_tracks,
        local_cp);

    spdlog::trace("Attempt Feature Collapse, {} {} pass: {}{}", f, e,
                  (flag == 0), flag);

    if (flag != 0) {
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
    prism::local_validity::post_operation(pc, option, old_fids, new_fid,
                                          new_tracks, local_cp);

    // shifts
    shift_left(new_fid, new_shifts, F, FF, FFi);
    spdlog::trace("insert {} {}", u2, u1);
    meta_edges.erase(seg0);
    meta_edges.erase(seg1);
    for (int i = 0; i < new_fid.size(); i++) {
      auto f = new_fid[i];
      for (auto e = 0; e < 3; e++) {
        auto v0 = F[f][e], v1 = F[f][(e + 1) % 3];
        if (v0 == u2 && v1 == u1)
          queue.push({-(V[v0] - V[v1]).norm(), f, e, v0, v1, global_tick});
      }
    }

    global_tick++;
    spdlog::trace("Feature Edge Collapsed {} {}", f, e);
  }
  spdlog::info("Pass Zig Feature Collapse total {}. lk{}, v{} i{} n{} q{} c{}, zv "
               "{} zi {} zn {}",
               global_tick, rejections_steps[0], rejections_steps[1],
               rejections_steps[2], rejections_steps[3], rejections_steps[4],
               rejections_steps[5], rejections_steps[6], rejections_steps[7],
               rejections_steps[8]);
  F.resize(orig_fnum);
  Eigen::VectorXi vid_map, vid_ind; // new to old
  pc.cleanup_empty_faces(vid_map, vid_ind);
  for (int i = 0; i < vid_ind.size(); i++) {
    option.target_adjustment[i] = option.target_adjustment[vid_ind[i]];
  }
  option.target_adjustment.resize(vid_ind.size());

  return global_tick;
}

int prism::local::zig_slide_pass(PrismCage &pc, RemeshOptions &option) {
  auto &meta_edges = pc.meta_edges;
  auto &F = pc.F;
  auto &V = pc.mid;
  auto [FF, FFi] = prism::local::triangle_triangle_adjacency(F);
  auto &inpV = pc.ref.V;

  int global_ticks = 0;
  std::vector<int> verts_on_feat;
  for (auto [m, ign] : meta_edges) {
    verts_on_feat.push_back(m.first);
    verts_on_feat.push_back(m.second);
  }
  std::sort(verts_on_feat.begin(), verts_on_feat.end());
  verts_on_feat.erase(std::unique(verts_on_feat.begin(), verts_on_feat.end()),
                      verts_on_feat.end());

  std::vector<std::vector<int>> VF, VFi;
  igl::vertex_triangle_adjacency(
      V.size(), Eigen::Map<RowMati>(F[0].data(), F.size(), 3), VF, VFi);
  std::vector<int> rejections_steps(8, 0);
  for (auto vid : verts_on_feat) {
    auto nb = VF[vid], nbi = VFi[vid];
    // only do sth when two feature
    std::vector<PrismCage::meta_type_t::iterator> nb_feat;
    for (auto i = 0; i < nb.size(); i++) {
      auto v0 = vid, v1 = F[nb[i]][(nbi[i] + 1) % 3],
           v2 = F[nb[i]][(nbi[i] + 2) % 3];
      auto it = meta_edges.find({v0, v1});
      if (it != meta_edges.end())
        nb_feat.emplace_back(it);
      it = meta_edges.find({v2, v0});
      if (it != meta_edges.end())
        nb_feat.emplace_back(it);
    }
    if (nb_feat.size() != 2)
      continue;
    auto it0 = nb_feat[0], it1 = nb_feat[1];
    if (it0->second.first != it1->second.first)
      continue; // different chain
    int chain_id = it0->second.first;
    if (it0->first.first == vid)
      std::swap(it0, it1);
    // (v2) -->--it0-->--(vid)-->--it1-->--v1
    assert(it0->first.second == vid && it1->first.first == vid);
    auto v1 = it1->first.second, v2 = it0->first.first;
    spdlog::trace("u1 {} <- u0 {} <- u2 {}", v1, vid, v2);

    auto newseg = it0->second.second; // seg combines.
    if (newseg.back() != it1->second.second.front()) {
      assert(newseg.back() == it1->second.second[1]);
      newseg.pop_back();
    }
    newseg.pop_back();
    newseg.insert(newseg.end(), it1->second.second.begin(),
                  it1->second.second.end());
    // if (newseg.size() <= 3)
      // continue; // no need to slide

    auto [seg0, seg1, mid_pos] =
        split_segs_in_the_middle(V[v2], V[v1], inpV, newseg);
    // if (offset == 0. && slice_id + 1 == it0->second.second.size())
    // continue;  // no shift
    spdlog::trace("seg size {} + {} from {}", seg0.size(), seg1.size(), newseg.size());
    assert(seg0.size() >= 2 && seg1.size() >= 2);

    auto recover_coordinates =
        std::tuple(pc.base[vid], pc.mid[vid], pc.top[vid], it0->second.second,
                   it1->second.second);

    auto rollback = [&]() {
      std::tie(pc.base[vid], pc.mid[vid], pc.top[vid], it0->second.second,
               it1->second.second) = recover_coordinates;
    };

    auto &old_fids = nb;
    std::vector<Vec3i> moved_tris;
    for (auto f : old_fids)
      moved_tris.push_back(pc.F[f]);

    auto old_quality = prism::local_validity::max_quality_on_tris(
        pc.base, pc.mid, pc.top, moved_tris);
    { // modifications
      // parallel offset
      pc.base[vid] += mid_pos - pc.mid[vid];
      pc.top[vid] += mid_pos - pc.mid[vid];
      pc.mid[vid] = mid_pos;
      it0->second.second = std::move(seg0);
      it1->second.second = std::move(seg1);
    }
    std::vector<std::set<int>> new_tracks;
    std::vector<RowMatd> local_cp;
    auto flag = prism::local_validity::attempt_zig_remesh(
        pc, pc.track_ref, option, old_quality, old_fids, moved_tris, new_tracks,
        local_cp);
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
    prism::local_validity::post_operation(pc, option, old_fids, new_fids,
                                          new_tracks, local_cp);
  }
  spdlog::info("Zig Feature slide complete: {}/{}", global_ticks,
               verts_on_feat.size());
  return 0;
}

int prism::local::zig_split_pass(PrismCage &pc,
                                 prism::local::RemeshOptions &option) {
  auto &meta_edges = pc.meta_edges;
  auto &F = pc.F;
  auto &V = pc.mid;
  auto &inpV = pc.ref.V;
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
        continue; // this can be done with an easier way if know vv2fe.
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
    if (f0 == -1 || FF[f0][e0] == -1)
      continue; // skip boundary
    if (auto u0_ = F[f0][e0], u1_ = F[f0][(e0 + 1) % 3];
        u0_ == u1_ || u0_ != u0 ||
        u1_ != u1) // vid changed, means the edge is outdated.
      continue;

    if (std::abs(l) * 1.5 <
        (option.sizing_field(V[u0]) * option.target_adjustment[u0] +
         option.sizing_field(V[u1]) * option.target_adjustment[u1])) {
      //  spdlog::debug("skip");
      continue; // skip if l < 4/3*(s1+s2)/2
    }

    auto f1 = FF[f0][e0], e1 = FFi[f0][e0];
    if (f1 == -1)
      continue; // boundary check
    auto v0 = F[f0][(e0 + 2) % 3];
    auto v1 = F[f1][(e1 + 2) % 3];
    spdlog::trace(">>>>>> Entering: {}-{} {}-{} {}->{} {}-{}", f0, e0, f1, e1,
                  u0, u1, v0, v1);

    auto current_meta_it = meta_edges.find({u0, u1});
    assert(current_meta_it != meta_edges.end());

    auto [cid, segs] = current_meta_it->second;

    auto [seg0, seg1, mid_pos] =
        split_segs_in_the_middle(V[u0], V[u1], inpV, segs);
    auto ux = pc.mid.size();

    auto [new_it0, it_flag0] =
        meta_edges.emplace(std::pair(u0, ux), std::pair(cid, seg0));
    auto [new_it1, it_flag1] =
        meta_edges.emplace(std::pair(ux, u1), std::pair(cid, seg1));
    assert(it_flag0 && it_flag1);

    auto rollback = [&, &newseg = segs, &it0 = new_it0, &it1 = new_it1]() {
      meta_edges.erase(it0);
      meta_edges.erase(it1);
      if (pc.mid.size() > ux) {
        pc.mid.pop_back();
        pc.base.pop_back();
        pc.top.pop_back();
      }
    };

    std::vector<int> old_fids = {f0, f1};
    std::vector<int> new_fid = {f0, f1, int(F.size()), int(F.size() + 1)};
    auto moved_tris = std::vector<Vec3i>{F[f0], F[f1], F[f0], F[f1]};
    moved_tris[0][e0] = ux;
    moved_tris[1][e1] = ux;
    moved_tris[2][(e0 + 1) % 3] = ux;
    moved_tris[3][(e1 + 1) % 3] = ux;
    auto new_shifts = prism::local_validity::triangle_shifts(moved_tris);

    pc.mid.push_back(mid_pos);
    pc.base.push_back((pc.base[u0] + pc.base[u1]) / 2);
    pc.top.push_back((pc.top[u0] + pc.top[u1]) / 2);
    pc.base.back() += mid_pos - (pc.mid[u0] + pc.mid[u1]) / 2;
    pc.top.back() += mid_pos - (pc.mid[u0] + pc.mid[u1]) / 2;

    std::array<Vec3d, 3> newlocation{pc.base.back(), pc.mid.back(),
                                     pc.top.back()};
    std::vector<std::set<int>> new_tracks;
    std::vector<RowMatd> local_cp;
    auto flag = 1;
    auto alpha = 1.;
    while (flag == 1) { // vc problem
      pc.base.back() = newlocation[0] * (alpha) + (1 - alpha) * newlocation[1];
      pc.top.back() = newlocation[2] * (alpha) + (1 - alpha) * newlocation[1];
      flag = prism::local_validity::attempt_zig_remesh(pc, pc.track_ref, option,
                                                       -1, old_fids, moved_tris,
                                                       new_tracks, local_cp);
      alpha *= 0.8;
      if (alpha < 1e-2)
        break;
    }
    if (flag != 0) {
      rollback();
      spdlog::trace("Rollback V");
      rejections_steps[flag]++;
      continue;
    }
    spdlog::trace("Feature split happening {} {}", u0, u1);

    prism::edge_split(pc.mid.size() - 1, F, FF, FFi, f0, e0);

    prism::local_validity::post_operation(pc, option, old_fids, new_fid,
                                          new_tracks, local_cp);

    meta_edges.erase(current_meta_it);
    // shifts
    global_tick++;
    shift_left(new_fid, new_shifts, F, FF, FFi);
    option.target_adjustment.push_back(
        (option.target_adjustment[u0] + option.target_adjustment[u1]) / 2);
  }
  spdlog::info("Complete a pass of feature split. {}: rejections {}",
               global_tick, rejections_steps);
  return global_tick;
}

int prism::local::zig_comb_pass(PrismCage &pc,
                                RemeshOptions &option) { // only zoom.
  auto &meta_edges = pc.meta_edges;
  auto &F = pc.F;
  auto &V = pc.mid;
  auto [FF, FFi] = prism::local::triangle_triangle_adjacency(F);

  auto &inpV = pc.ref.V;

  int global_ticks = 0;
  if (pc.zig_base.rows() == 0)
    throw std::runtime_error("comb empty");

  std::vector<std::vector<int>> VF, VFi;
  igl::vertex_triangle_adjacency(
      V.size(), Eigen::Map<RowMati>(F[0].data(), F.size(), 3), VF, VFi);

  auto vv2fe = [](const std::vector<Vec3i> &F) {
    std::map<std::pair<int, int>, std::pair<int, int>> v2fe;
    for (int i = 0; i < F.size(); i++) {
      for (int j = 0; j < 3; j++) {
        v2fe.emplace(std::pair<int, int>{F[i][j], F[i][(j + 1) % 3]},
                     std::pair<int, int>{i, j});
      }
    }
    return std::move(v2fe);
  }(F);

  auto rejections_steps = std::vector<int>(10, 0);
  for (auto [m, c_segs] : meta_edges) { // iterate over edges
    auto [v0, v1] = m;
    auto [f, e] = vv2fe.at({v0, v1});
    auto f1 = FF[f][e], e1 = FFi[f][e];
    auto &[cid, segs] = c_segs;

    auto old_fids = VF[v0];
    old_fids.insert(old_fids.end(), VF[v1].begin(), VF[v1].end());
    std::sort(old_fids.begin(), old_fids.end());
    old_fids.erase(std::unique(old_fids.begin(), old_fids.end()),
                   old_fids.end());
    auto moved_tris = std::vector<Vec3i>();
    for (auto i : old_fids)
      moved_tris.push_back(F[i]);

    std::map<int, std::pair<Vec3d, Vec3d>> restorer;
    {
      auto &refV = pc.ref.V;
      auto start = pc.mid[v0];
      auto end = pc.mid[v1];
      auto padded_segs = segs;
      if (segs.empty()) {
        continue;
      }
      std::vector<Vec3d> local_mid(0);
      local_mid.reserve(segs.size() + 2);
      if (refV.row(segs.front()) != start) { // add to front.
        local_mid.push_back(start);
        padded_segs.insert(padded_segs.begin(), -1);
      }
      for (auto s : segs)
        local_mid.push_back(refV.row(s)); // 1,2,...,n
      if (refV.row(segs.back()) != end) { // add to back
        local_mid.push_back(end);
        padded_segs.push_back(-1);
      }
      int n = local_mid.size();
      auto avg_pillar_len = 0.;
      for (auto i = 0; i < n; i++) {
        auto sid = padded_segs[i];
        if (sid == -1)
          continue;
        Vec3d pillar = (pc.zig_top.row(sid) - pc.zig_base.row(sid)) / 2;
        avg_pillar_len += pillar.norm();
      }
      avg_pillar_len /= segs.size();
      for (auto i = 0; i < n; i++) {
        auto sid = padded_segs[i];
        if (sid == -1)
          continue;
        restorer.emplace(sid, std::pair<Vec3d, Vec3d>{pc.zig_base.row(sid),
                                                      pc.zig_top.row(sid)});
        Vec3d pillar = (pc.zig_top.row(sid) - pc.zig_base.row(sid)) / 2;
        double leng = pillar.norm();
        auto targ_leng = std::min(option.zig_thick, 2 * avg_pillar_len);
        pc.zig_base.row(sid) = pc.ref.V.row(sid) - targ_leng * pillar / leng;
        pc.zig_top.row(sid) = pc.ref.V.row(sid) + targ_leng * pillar / leng;
      }
    }
    auto rollback = [&]() {
      for (auto [sid, bt] : restorer) {
        pc.zig_base.row(sid) = bt.first;
        pc.zig_top.row(sid) = bt.second;
      }
    };
    std::vector<std::set<int>> new_tracks;
    std::vector<RowMatd> local_cp;
    auto flag = prism::local_validity::attempt_zig_remesh(
        pc, pc.track_ref, option, -1., old_fids, moved_tris, new_tracks,
        local_cp);
    spdlog::trace("Attempt Comb, {} {} pass: {}{}", v0, v1, (flag == 0), flag);
    if (flag != 0) {
      spdlog::trace("Rollback pc.zig_base/top");
      rollback();
      rejections_steps[flag]++;
      continue;
    }
    spdlog::trace("Comb happening {} {}", v0, v1);
    global_ticks++;
    auto &new_fids = old_fids;
    assert(moved_tris.size() == new_tracks.size());
    prism::local_validity::post_operation(pc, option, old_fids, new_fids,
                                          new_tracks, local_cp);
  }
  spdlog::info("zig comb complete: {}/{}", global_ticks, meta_edges.size());
  return global_ticks;
}
