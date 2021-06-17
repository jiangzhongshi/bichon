#include "remesh_pass.hpp"

#include <igl/boundary_facets.h>
#include <igl/doublearea.h>
#include <igl/volume.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <prism/energy/smoother_pillar.hpp>
#include <queue>

#include "local_mesh_edit.hpp"
#include "prism/PrismCage.hpp"
#include "prism/cage_utils.hpp"
#include "prism/energy/prism_quality.hpp"
#include "prism/geogram/AABB.hpp"
#include "prism/spatial-hash/AABB_hash.hpp"
#include "retain_triangle_adjacency.hpp"
#include "validity_checks.hpp"

namespace prism::local_validity {

bool unified_intersection_check(const PrismCage &pc,
                                const std::vector<int> &old_fid,
                                const std::vector<Vec3i> &moved_tris,
                                bool dynamic) {
  bool ic = false;
  if (!dynamic)
    ic = intersect_check(pc.base, pc.top, moved_tris, *pc.ref.aabb);
  else {
    ic = dynamic_intersect_check(pc.base, pc.F, old_fid, moved_tris,
                                 *pc.base_grid) &&
         dynamic_intersect_check(pc.top, pc.F, old_fid, moved_tris,
                                 *pc.top_grid);
  }
  return ic;
}

int attempt_split(
    const PrismCage &pc, const std::vector<std::set<int>> &map_track,
    const prism::local::RemeshOptions &option,
    // double distortion_bound, bool improve_quality,
    // specified infos below
    int f0, int f1, int e0, int e1,
    std::tuple<std::vector<int> /*fid*/, std::vector<int>, /*shift*/
               std::vector<std::set<int>>,                 /*track*/
               std::vector<RowMatd>                        /*local_cp*/
               > &checker) {
  auto &base = pc.base, &top = pc.top, &mid = pc.mid;
  auto &F = pc.F;
  auto &refV = pc.ref.V;
  auto &refF = pc.ref.F;
  auto &tree = *pc.ref.aabb;

  int ux = mid.size() - 1;
  std::vector<Vec3i> new_tris = {F[f0], F[f1], F[f0], F[f1]};
  // conform to the convention inside `bool edge_split(int,int,int,int)`
  new_tris[0][e0] = ux;
  new_tris[1][e1] = ux;
  new_tris[2][(e0 + 1) % 3] = ux;
  new_tris[3][(e1 + 1) % 3] = ux;
  auto quality_before = max_quality_on_tris(base, mid, top, {F[f0], F[1]});
  auto quality_after = max_quality_on_tris(base, mid, top, new_tris);
  spdlog::trace("Quality compare {} -> {}", quality_before, quality_after);
  if (std::isnan(quality_after) ||
      option.split_improve_quality && quality_after > quality_before) {
    return 4;
  }

  std::vector<int> new_fid = {f0, f1, static_cast<int>(F.size()),
                              static_cast<int>(F.size() + 1)};
  std::vector<int> old_fids = {f0, f1};

  //  volume check
  auto vc = volume_check(base, mid, top, new_tris, tree.num_freeze);
  if (!vc) {
    spdlog::trace("Split Vol fail");
    return 1;
  }
  if (!unified_intersection_check(pc, old_fids, new_tris,
                                  option.dynamic_hashgrid)) {
    spdlog::trace("Split: Failed inter check");
    return 2;
  }

  auto new_shifts = prism::local_validity::triangle_shifts(new_tris);
  std::vector<std::set<int>> sub_refs;

  std::set<int> combined_tracks;
  for (auto f : old_fids) set_add_to(map_track[f], combined_tracks);

  auto sub_refs_optional = distort_check(
      base, mid, top, new_tris, combined_tracks, refV, refF,
      option.distortion_bound, tree.num_freeze, option.dynamic_hashgrid);
  if (sub_refs_optional) sub_refs = sub_refs_optional.value();
  if (sub_refs.size() == 0) {
    spdlog::trace("failed map check f{}e{}, restore", f0, e0);
    return 3;
  }

  std::vector<RowMatd> local_cp;
  if (option.curve_checker.first.has_value() &&
      !(std::any_cast<std::function<bool(
            const PrismCage &, const std::vector<int> &,
            const std::vector<Vec3i> &, std::vector<RowMatd> &)>>(
          option.curve_checker.first))(pc, old_fids, new_tris, local_cp)) {
    return 5;
  }

  checker = std::tuple(std::move(new_fid), std::move(new_shifts),
                       std::move(sub_refs), std::move(local_cp));

  return 0;
}

}  // namespace prism::local_validity
namespace prism::local {
void wildflip_pass(PrismCage &pc, const RemeshOptions &option) {
  auto attempt_operation = option.use_polyshell? local_validity::attempt_zig_remesh: local_validity::attempt_feature_remesh;
  auto &F = pc.F;
  auto &V = pc.mid;
  using queue_entry =
      std::tuple<double, int /*f*/, int /*e*/, int /*u0*/, int /*u1*/, int>;
  std::priority_queue<queue_entry> queue;

  // build connectivity
  auto [FF, FFi] = prism::local::triangle_triangle_adjacency(F);

  std::set<std::pair<int, int>> skip_edges;
  for (auto [m, ignore] : pc.meta_edges) {
    auto [v0, v1] = m;
    skip_edges.insert({v0, v1});
    skip_edges.insert({v1, v0});
    // skip_edges.insert({std::min(v0, v1), std::max(v0, v1)});
  }

  // enqueue
  for (auto [f, flags] = std::pair(0, RowMati(RowMati::Zero(F.size(), 3)));
       f < F.size(); f++) {
    for (auto e : {0, 1, 2}) {
      auto v0 = F[f][e], v1 = F[f][(e + 1) % 3];
      if (v0 > v1 || flags(f, e) == 1) continue;
      if (FF[f][e] == -1) continue;
      if (skip_edges.find({v0, v1}) != skip_edges.end()) continue;
      queue.push({(V[v0] - V[v1]).norm(), f, e, v0, v1, 0});
      flags(f, e) = 1;
      flags(FF[f][e], FFi[f][e]) = 1;
    }
  }

  std::vector<int> valence(V.size(), 0);
  for (auto f : F)
    for (int j = 0; j < 3; j++) valence[f[j]]++;

  int global_tick = 0;
  RowMati timestamp = RowMati::Zero(F.size(), 3);
  std::vector<int> rejection_steps(8, 0);
  // pop
  while (!queue.empty()) {
    auto [l, f, e, u0, u1, tick] = queue.top();
    queue.pop();
    if (f == -1 || FF[f][e] == -1) continue;  // skip boundary
    if (skip_edges.find({u0, u1}) != skip_edges.end()) continue;

    if (auto u0_ = F[f][e], u1_ = F[f][(e + 1) % 3];
        u0_ == u1_ || u0_ != u0 ||
        u1_ != u1)  // vid changed, means the edge is outdated.
      continue;
    else
      assert((V[u1_] - V[u0_]).norm() == l &&
             "Outdated entries will be ignored, this condition can actually "
             "replace the previous");

    auto f1 = FF[f][e], e1 = FFi[f][e];
    assert(f1 != -1);
    auto f0 = f, e0 = e;
    auto v0 = F[f0][(e0 + 2) % 3];
    auto v1 = F[f1][(e1 + 2) % 3];

    // check valency energy.
    constexpr auto valence_energy = [](int i0, int i1, int i2, int i3) {
      double e = 0;
      for (auto i : {i0, i1, i2, i3}) e += (i - 6) * (i - 6);
      return e;
    };
    if (valence_energy(valence[u0], valence[v0], valence[u1], valence[v1]) <
        valence_energy(valence[u0] - 1, valence[v0] + 1, valence[u1] - 1,
                       valence[v1] + 1))
      continue;
    std::tuple<std::vector<int>, std::vector<std::set<int>>> checker;

    auto [old_fid, new_fid, moved_tris] = [f0, f1, e0, e1, v0, v1,
                                           &F = pc.F]() {
      std::vector<Vec3i> moved_tris{F[f0], F[f1]};
      std::vector<int> old_fid{f0, f1};
      auto e01 = (e0 + 1) % 3;
      auto e11 = (e1 + 1) % 3;
      moved_tris[0][e01] = v1;
      moved_tris[1][e11] = v0;
      return std::tuple(old_fid, old_fid, moved_tris);
    }();
    auto new_shifts = prism::local_validity::triangle_shifts(moved_tris);

    std::vector<RowMatd> local_cp;
    std::vector<std::set<int>> new_tracks;
    int flag = attempt_operation(
        pc, pc.track_ref, option, -1., old_fid, moved_tris, new_tracks,
        local_cp);

    if (flag > 0) {
      rejection_steps[flag]++;
      continue;
    }

    if (!prism::edge_flip(F, FF, FFi, f, e)) {
      rejection_steps[0]++;
      continue;
    }
    // shifts
    shift_left(new_fid, new_shifts, F, FF, FFi);
    prism::local_validity::post_operation(pc, option, old_fid, new_fid,
                                          new_tracks, local_cp);

    // Push the modified edges back in the queue
    // Not removing replaced ones since additional checks are in place: v0,v1
    // and norm==l.
    global_tick++;
    auto push_to_queue = [&queue, &F, &V, &timestamp, &global_tick](auto f,
                                                                    auto v) {
      auto e = -1;
      auto face = F[f];
      for (int i = 0; i < 3; i++)
        if (face[i] == v) e = i;
      if (e == -1) spdlog::error("push queue wrong");
      auto u0 = F[f][e], u1 = F[f][(e + 1) % 3];
      if (u0 > u1) {
        return;
      }
      queue.push({(V[u1] - V[u0]).norm(), f, e, u0, u1, global_tick});
    };
    if (v0 > v1)
      push_to_queue(f1, v1);
    else
      push_to_queue(f1, v0);
    if (v0 < u0) push_to_queue(f0, v0);
    if (u0 < v1) push_to_queue(f0, u0);
    if (v1 < u1) push_to_queue(f1, v1);
    if (u1 < v0) push_to_queue(f1, u1);

    valence[v0]++;
    valence[v1]++;
    valence[u0]--;
    valence[u1]--;
  }
  spdlog::info("Flip {} Done, Rej t{} v{} i{} d{} q{}", global_tick,
               rejection_steps[0], rejection_steps[1], rejection_steps[2],
               rejection_steps[3], rejection_steps[4]);
}

int wildsplit_pass(PrismCage &pc, RemeshOptions &option) {
  auto attempt_operation = option.use_polyshell? local_validity::attempt_zig_remesh: local_validity::attempt_feature_remesh;
  auto &F = pc.F;
  auto &V = pc.mid;
  auto input_vnum = V.size();
  using queue_entry =
      std::tuple<double, int /*f*/, int /*e*/, int /*u0*/, int /*u1*/>;
  std::priority_queue<queue_entry> queue;

  // build connectivity
  auto [FF, FFi] = prism::local::triangle_triangle_adjacency(F);

  std::set<std::pair<int, int>> skip_edges;
  for (auto [m, ignore] : pc.meta_edges) {
    auto [v0, v1] = m;
    skip_edges.insert({v0, v1});
    skip_edges.insert({v1, v0});
  }

  // enqueue
  for (auto [f, flags] = std::pair(0, RowMati(RowMati::Zero(F.size(), 3)));
       f < F.size(); f++) {
    for (auto e : {0, 1, 2}) {
      auto v0 = F[f][e], v1 = F[f][(e + 1) % 3];
      if (v0 > v1 || flags(f, e) == 1) continue;
      if (FF[f][e] == -1) continue;  // skip boundary
      if (skip_edges.find({v0, v1}) != skip_edges.end()) continue;
      queue.push({(V[v0] - V[v1]).norm(), f, e, v0, v1});
      flags(f, e) = 1;
      flags(FF[f][e], FFi[f][e]) = 1;
    }
  }

  std::vector<int> rejections_steps(8, 0);
  int global_tick = 0;
  // pop
  while (!queue.empty()) {
    auto [l, f0, e0, u0, u1] = queue.top();
    queue.pop();
    if (skip_edges.find({u0, u1}) != skip_edges.end()) continue;

    if (f0 == -1 || FF[f0][e0] == -1) continue;  // skip boundary
    if (auto u0_ = F[f0][e0], u1_ = F[f0][(e0 + 1) % 3];
        u0_ == u1_ || u0_ != u0 ||
        u1_ != u1)  // vid changed, means the edge is outdated.
      continue;
    // spdlog::debug("l {} with {} {}", l, option.sizing_field(V[u0]) *
    // option.target_adjustment[u0], option.sizing_field(V[u1]) *
    // option.target_adjustment[u1]);
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
    std::array<Vec3d, 3> newlocation{(pc.base[u0] + pc.base[u1]) / 2,
                                     (pc.mid[u0] + pc.mid[u1]) / 2,
                                     (pc.top[u0] + pc.top[u1]) / 2};

    spdlog::trace("Attempting: {}-{} {}-{} {}->{} {}-{}", f0, e0, f1, e1, u0,
                  u1, v0, v1);
    auto ux = pc.mid.size();
    std::vector<int> old_fids = {f0, f1};
    std::vector<int> new_fid = {f0, f1, int(F.size()), int(F.size() + 1)};
    auto moved_tris = std::vector<Vec3i>{F[f0], F[f1], F[f0], F[f1]};
    moved_tris[0][e0] = ux;
    moved_tris[1][e1] = ux;
    moved_tris[2][(e0 + 1) % 3] = ux;
    moved_tris[3][(e1 + 1) % 3] = ux;
    auto new_shifts = prism::local_validity::triangle_shifts(moved_tris);

    auto alpha = 1.;
    auto flag = 1;
    pc.base.push_back(newlocation[0]);
    pc.mid.push_back(newlocation[1]);
    pc.top.push_back(newlocation[2]);
    std::vector<std::set<int>> new_tracks;
    std::vector<RowMatd> local_cp;
    while (flag == 1) {  // vc problem
      pc.base.back() = newlocation[0] * (alpha) + (1 - alpha) * newlocation[1];
      pc.top.back() = newlocation[2] * (alpha) + (1 - alpha) * newlocation[1];

      flag = attempt_operation(
          pc, pc.track_ref, option, -1, old_fids, moved_tris,
          new_tracks, local_cp);
      alpha *= 0.8;
      if (alpha < 1e-2) break;
    }

    if (flag != 0) {
      spdlog::trace("Split Attempt Failed {}-{} {}-{}", f0, e0, f1, e1);
      rejections_steps[flag]++;
      pc.base.pop_back();
      pc.mid.pop_back();
      pc.top.pop_back();
      continue;
    }
    assert(new_fid.size() == new_tracks.size());
    prism::edge_split(V.size() - 1, F, FF, FFi, f0, e0);

    prism::local_validity::post_operation(pc, option, old_fids, new_fid,
                                          new_tracks, local_cp);
    // shifts
    global_tick++;
    shift_left(new_fid, new_shifts, F, FF, FFi);
    option.target_adjustment.push_back(
        (option.target_adjustment[u0] + option.target_adjustment[u1]) / 2);

    auto push_to_queue = [&queue, &F, &V, input_vnum](auto f, auto v) {
      auto e = -1;
      auto face = F[f];
      for (int i = 0; i < 3; i++)
        if (face[i] == v) e = i;
      if (e == -1) spdlog::error("push queue wrong");
      auto u0 = F[f][e], u1 = F[f][(e + 1) % 3];
      if (u0 > u1 || u1 >= input_vnum) {
        return;
      }
      queue.push({(V[u1] - V[u0]).norm(), f, e, u0, u1});
      spdlog::trace("pushed {} {} {} {}", f, e, u0, u1);
    };

    auto fx0 = F.size() - 2;
    auto fx1 = F.size() - 1;
    spdlog::trace("Consider {} {} {} {}", u0, u1, v0, v1, fx0, fx1);
    if (v0 < u0) push_to_queue(fx0, v0);
    if (v1 < u1) push_to_queue(fx1, v1);
    if (u1 < v0 && new_shifts[0] != 0) push_to_queue(f0, u1);
    if (u0 < v1 && new_shifts[1] != 0) push_to_queue(f1, u0);
  }
  spdlog::info("Split Done {}, Rejections v{} i{} d{} q{} c{}", global_tick,
               rejections_steps[1], rejections_steps[2], rejections_steps[3],
               rejections_steps[4], rejections_steps[5]);

  std::set<int> low_quality_vertices;
  for (auto [v0, v1, v2] : F) {
    if (prism::energy::triangle_quality({V[v0], V[v1], V[v2]}) > 10) {
      low_quality_vertices.insert(v0);
      low_quality_vertices.insert(v1);
      low_quality_vertices.insert(v2);
    }
  }
  for (auto v : low_quality_vertices) option.target_adjustment[v] /= (2 * 1.5);
  for (auto &u : option.target_adjustment)
    u = std::max(std::min(1.5 * u, 1.), 1e-2);

  spdlog::info("Post Split Adjustments: low_quality {}/{}",
               low_quality_vertices.size(), V.size());
  return global_tick;
}
}  // namespace prism::local
