#include <igl/boundary_facets.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <highfive/H5Easy.hpp>
#include <queue>

#include "local_mesh_edit.hpp"
#include "prism/PrismCage.hpp"
#include "prism/cage_utils.hpp"
#include "prism/energy/prism_quality.hpp"
#include "prism/feature_utils.hpp"
#include "prism/geogram/AABB.hpp"
#include "prism/spatial-hash/AABB_hash.hpp"
#include "remesh_pass.hpp"
#include "retain_triangle_adjacency.hpp"
#include "validity_checks.hpp"

namespace prism::local_validity {
int attempt_collapse(
    const PrismCage &pc, const std::vector<std::set<int>> &map_track,
    const prism::local::RemeshOptions &option,
    // specified infos below
    const std::vector<std::pair<int, int>> &neighbor0,
    const std::vector<std::pair<int, int>> &neighbor1, int f0, int f1, int u0,
    int u1, bool feature_enable,
    std::tuple<std::vector<int> /*oldfid*/, std::vector<int> /*newfid*/,
               std::vector<int> /*shifts*/,
               std::vector<std::set<int>> /*track*/> &checker) {
  auto &base = pc.base, &top = pc.top, &mid = pc.mid;
  auto &F = pc.F;
  auto &refV = pc.ref.V;
  auto &refF = pc.ref.F;
  auto &tree = *pc.ref.aabb;

  std::vector<Vec3i> moved_tris, old_tris;
  moved_tris.reserve(neighbor0.size() + neighbor1.size() - 4);

  std::vector<int> new_fid, old_fid;

  auto inter_safe = [&pc, &option, &old_fid, &moved_tris]() {
    if (!option.dynamic_hashgrid)
      return intersect_check(pc.base, pc.top, moved_tris, *pc.ref.aabb);
    else
      return dynamic_intersect_check(pc.base, pc.F, old_fid, moved_tris,
                                     *pc.base_grid) &&
             dynamic_intersect_check(pc.top, pc.F, old_fid, moved_tris,
                                     *pc.top_grid);
  };
  for (auto [f, e] : neighbor0) {
    assert(F[f][e] == u0);
    auto new_tris = F[f];
    old_fid.push_back(f);
    if (new_tris[0] == u1 || new_tris[1] == u1 || new_tris[2] == u1)
      continue;  // collapsed faces
    new_tris[e] = u1;
    moved_tris.emplace_back(new_tris);
    new_fid.push_back(f);
  }
  for (auto [f, e] : neighbor1) {
    auto new_tris = pc.F[f];
    if (new_tris[0] == u0 || new_tris[1] == u0 || new_tris[2] == u0)
      continue;  // collapsed faces
    old_fid.push_back(f);
    moved_tris.emplace_back(new_tris);
    new_fid.push_back(f);
  }

  // note that new_fid is geometric cyclic.
  assert(moved_tris.size() == neighbor0.size() + neighbor1.size() - 4);

  spdlog::trace("Quality check");
  for (auto f : old_fid)
    old_tris.emplace_back(Vec3i{F[f][0], F[f][1], F[f][2]});
  auto quality_before = max_quality_on_tris(base, mid, top, old_tris);
  auto quality_after = max_quality_on_tris(base, mid, top, moved_tris);
  spdlog::trace("Quality compare {} -> {}", quality_before, quality_after);
  if (std::isnan(quality_after)) return 4;
  if ((quality_after > option.collapse_quality_threshold) &&
      quality_after > quality_before)  // if the quality is too large, not allow
                                       // it to increase.
    return 4;
  //  volume check
  auto vc = volume_check(base, mid, top, moved_tris, tree.num_freeze);
  if (!vc) return 1;

  if (!inter_safe()) return 2;
  // get new subdivide types
  auto new_shifts = prism::local_validity::triangle_shifts(moved_tris);
  std::vector<std::set<int>> distributed_trackee;
  if (!feature_handled_distort_check(pc, option, moved_tris, old_fid,
                                     distributed_trackee))
    return 3;
  checker = std::tuple(std::move(old_fid), std::move(new_fid),
                       std::move(new_shifts), std::move(distributed_trackee));
  return 0;
}
}  // namespace prism::local_validity

namespace collapse {
bool satisfy_link_condition(const std::vector<Vec3i> &F,
                            const std::vector<Vec3i> &FF,
                            const std::vector<Vec3i> &FFi, int f, int e,
                            std::vector<std::pair<int, int>> &neighbor0,
                            std::vector<std::pair<int, int>> &neighbor1) {
  auto f0 = f, e0 = e;
  auto f1 = FF[f0][e0], e1 = FFi[f0][e0];
  auto u0 = F[f0][e0], u1 = F[f1][e1];
  if (f1 == -1) return false;  // no boundary here
  assert(f1 != -1);
  assert(e1 != -1);
  assert(F[f1][(e1 + 1) % 3] == u0);
  // clockwise
  auto flag0 = prism::get_star_edges(F, FF, FFi, f0, e0, neighbor0);
  auto flag1 = prism::get_star_edges(F, FF, FFi, f1, e1, neighbor1);
  if (!flag0) return false;
  std::vector<int> nv0(neighbor0.size()), nv1(neighbor1.size());
  for (int i = 0; i < neighbor0.size(); i++) {
    auto [f, e] = neighbor0[i];
    nv0[i] = F[f][(e + 1) % 3];
  }
  for (int i = 0; i < neighbor1.size(); i++) {
    auto [f, e] = neighbor1[i];
    nv1[i] = F[f][(e + 1) % 3];
  }

  std::sort(nv0.begin(), nv0.end());
  std::sort(nv1.begin(), nv1.end());
  decltype(nv0) inter;
  std::set_intersection(nv0.begin(), nv0.end(), nv1.begin(), nv1.end(),
                        std::back_inserter(inter));
  if (inter.size() == 2)
    return true;
  else
    return false;
}

}  // namespace collapse

int prism::local::wildcollapse_pass(PrismCage &pc, RemeshOptions &option) {
  auto attempt_operation = option.use_polyshell? local_validity::attempt_zig_remesh: local_validity::attempt_feature_remesh;
  auto vv2fe = [](auto &F) {
    std::map<std::pair<int, int>, std::pair<int, int>> v2fe;
    for (int i = 0; i < F.rows(); i++) {
      for (int j = 0; j < 3; j++) {
        v2fe.emplace({F(i, j), F(i, (j + 1) % 3)}, {i, j});
      }
    }
    return v2fe;
  };
  auto &F = pc.F;
  auto &V = pc.mid;
  using queue_entry = std::tuple<double /*should negative*/, int /*f*/,
                                 int /*e*/, int /*u0*/, int /*u1*/, int /*ts*/>;
  std::priority_queue<queue_entry> queue;

  // build connectivity
  auto [FF, FFi] = prism::local::triangle_triangle_adjacency(F);

  // enqueue
  for (auto f = 0; f < F.size(); f++) {
    for (auto e : {0, 1, 2}) {
      auto v0 = F[f][e], v1 = F[f][(e + 1) % 3];
      if (FF[f][e] == -1) continue;
      queue.push({-(V[v0] - V[v1]).norm(), f, e, v0, v1,
                  0});  // both half edges are pushed.
    }
  }
  std::vector<bool> skip_flag(pc.mid.size(), false);
  for (auto [m, ignore] : pc.meta_edges) {
    auto [v0, v1] = m;
    skip_flag[v0] = true;
    skip_flag[v1] = true;
  }
  for (int i = 0; i < pc.ref.aabb->num_freeze; i++) skip_flag[i] = true;

  std::vector<int> rejections_steps(8, 0);
  // pop
  int global_tick = 0;
  RowMati timestamp = RowMati::Zero(F.size(), 3);
  while (!queue.empty()) {
    auto [l, f, e, v0, v1, tick] = queue.top();
    l = std::abs(l);
    queue.pop();
    if (f == -1 || FF[f][e] == -1) continue;

    auto u0 = F[f][e], u1 = F[f][(e + 1) % 3];
    if (tick != timestamp(f, e)) continue;
    assert((V[u1] - V[u0]).norm() == l &&
           "Outdated entries will be ignored, this condition can actually "
           "replace the previous");

    if (l * 2.5 > option.sizing_field(V[u0]) * option.target_adjustment[u0] +
                      option.sizing_field(V[u1]) * option.target_adjustment[u1])
      continue;  // skip if l > 4/5*(s1+s2)/2
    spdlog::trace(">>>>>>>> LinkCondition check {} {}", f, e);
    // collapse and misc checks.
    std::vector<std::pair<int, int>> n0, n1;
    if (!collapse::satisfy_link_condition(F, FF, FFi, f, e, n0, n1)) {
      rejections_steps[0]++;
      continue;
    }
    spdlog::trace("LinkCondition pass, attempt {} {}", f, e);
    if (option.collapse_valence_threshold > 0 &&  // enable
        n0.size() + n1.size() - 4 > option.collapse_valence_threshold) {
      spdlog::trace("avoiding too high valence", f, e);
      continue;
    }
    auto f1 = FF[f][e], e1 = FFi[f][e];
    std::tuple<std::vector<int>, std::vector<int> /*newfid*/,
               std::vector<int> /*shifts*/,
               std::vector<std::set<int>> /*track*/>
        checker;
    std::tuple<Vec3d, Vec3d> recover_coordinates{pc.base[u1], pc.top[u1]};

    auto [old_fid, new_fid, moved_tris] = [&neighbor0 = n0, &neighbor1 = n1,
                                           &F = pc.F, &u1, &u0]() {
      std::vector<Vec3i> moved_tris, old_tris;
      moved_tris.reserve(neighbor0.size() + neighbor1.size() - 4);

      std::vector<int> new_fid, old_fid;
      for (auto [f, e] : neighbor0) {
        assert(F[f][e] == u0);
        auto new_tris = F[f];
        old_fid.push_back(f);
        if (new_tris[0] == u1 || new_tris[1] == u1 || new_tris[2] == u1)
          continue;  // collapsed faces
        new_tris[e] = u1;
        moved_tris.emplace_back(new_tris);
        new_fid.push_back(f);
      }
      for (auto [f, e] : neighbor1) {
        auto &new_tris = F[f];
        if (new_tris[0] == u0 || new_tris[1] == u0 || new_tris[2] == u0)
          continue;  // collapsed faces
        old_fid.push_back(f);
        moved_tris.emplace_back(new_tris);
        new_fid.push_back(f);
      }
      return std::tuple(old_fid, new_fid, moved_tris);
    }();
    auto new_shifts = prism::local_validity::triangle_shifts(moved_tris);

    auto new_tracks = std::vector<std::set<int>>();
    std::vector<RowMatd> local_cp;
    auto flag = [&pc, &option, &skip_flag, f0 = f, &f1, &u0, &u1, &attempt_operation,
                 &new_tracks, &local_cp,
                 &old_fids = old_fid, &moved_tris = moved_tris](int repeat_num) -> int {
      if (skip_flag[u0])  // skip if singularity or feature
        return 4;
      // if (skip_flag[u1])
        // repeat_num = 1;  // no shrinking on feature for now.
      for (auto rp = 0; rp < repeat_num; rp++) {
        auto flag = attempt_operation(
            pc, pc.track_ref, option, -1, old_fids, moved_tris, new_tracks, local_cp);

        if (flag != 1)  // if fail not due to volume, accept the conclusion
          return flag;
        pc.base[u1] = (pc.base[u1] + pc.mid[u1]) / 2;
        pc.top[u1] = (pc.top[u1] + pc.mid[u1]) / 2;
      }
      return 1;
    }(5);
    spdlog::trace("Attempt Collapse, {} {} pass: {}", f, e, flag);

    if (flag != 0) {
      rejections_steps[flag]++;
      std::tie(pc.base[u1], pc.top[u1]) = recover_coordinates;
      continue;  // still failing
    }

    assert(new_fid.size() == new_shifts.size());

    prism::edge_collapse(F, FF, FFi, f, e);
    spdlog::trace("EdgeCollapse done {} {}", f, e);
    if (option.curve_checker.second.has_value())
      (std::any_cast<
          std::function<void(const std::vector<int> &, const std::vector<int> &,
                             const std::vector<RowMatd> &)>>(
          option.curve_checker.second))(old_fid, new_fid, local_cp);
    assert(old_fid.size() == new_fid.size() + 2);
    if (pc.top_grid != nullptr) {
      spdlog::trace("HashGrid Update");
      for (auto f : old_fid) {
        pc.top_grid->remove_element(f);
        pc.base_grid->remove_element(f);
      }
      pc.top_grid->insert_triangles(pc.top, F, new_fid);
      pc.base_grid->insert_triangles(pc.base, F, new_fid);
    }

    assert(new_fid.size() == new_tracks.size());
    for (int i = 0; i < new_tracks.size(); i++) {
      pc.track_ref[new_fid[i]] = new_tracks[i];
    }

    // shifts
    shift_left(new_fid, new_shifts, F, FF, FFi);

    // Push the modified edges back in the queue
    global_tick++;
    for (int i = 0; i < new_fid.size(); i++) {
      auto f = new_fid[i];
      for (auto e = 0; e < 3; e++) {
        auto u0 = F[f][e], u1 = F[f][(e + 1) % 3];
        if (FF[f][e] == -1) {
          timestamp(f, e) = -1;
          continue;
        }
        auto f1 = FF[f][e], e1 = FFi[f][e];
        queue.push({-(V[u1] - V[u0]).norm(), f, e, u0, u1, global_tick});
        queue.push({-(V[u1] - V[u0]).norm(), f1, e1, u1, u0, global_tick});
        timestamp(f, e) = global_tick;
        timestamp(f1, e1) = global_tick;
        spdlog::trace("pq {} {} {} {} {}", f, e, u0, u1, global_tick);
      }
    }
    spdlog::trace("Edge Collapsed {} {}", f, e);
    if (global_tick % 1000 == 0) {
      int sum = 0;
      for (auto v : F) sum += v[0] * v[1] * v[2];
      spdlog::info("{} F size {} checksum {}", global_tick,
                   F.size() - 2 * global_tick, sum);
    }
  }
  spdlog::info("Pass Collapse total {}. lk{}, v{} i{} d{} q{} c{}", global_tick,
               rejections_steps[0], rejections_steps[1], rejections_steps[2],
               rejections_steps[3], rejections_steps[4], rejections_steps[5]);
  Eigen::VectorXi vid_ind, vid_map;
  pc.cleanup_empty_faces(vid_map, vid_ind);
  for (int i = 0; i < vid_ind.size(); i++) {
    option.target_adjustment[i] = option.target_adjustment[vid_ind[i]];
  }
  option.target_adjustment.resize(vid_ind.size());
  return global_tick;
}