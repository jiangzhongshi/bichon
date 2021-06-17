#include "section_remesh.hpp"

#include <igl/Timer.h>
#include <igl/boundary_facets.h>
#include <igl/doublearea.h>
#include <igl/is_edge_manifold.h>
#include <igl/parallel_for.h>
#include <igl/remove_unreferenced.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/volume.h>
#include <igl/writeDMAT.h>
#include <igl/write_triangle_mesh.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <prism/energy/smoother_pillar.hpp>
#include <queue>
#include <random>

#include "../energy/map_distortion.hpp"
#include "../phong/projection.hpp"
#include "../phong/query_correspondence.hpp"
#include "../predicates/inside_octahedron.hpp"
#include "local_mesh_edit.hpp"
#include "mesh_coloring.hpp"
#include "prism/PrismCage.hpp"
#include "prism/energy/prism_quality.hpp"
#include "prism/geogram/AABB.hpp"
#include "prism/predicates/positive_prism_volume_12.hpp"
#include "retain_triangle_adjacency.hpp"
#include "validity_checks.hpp"

namespace collapse {
bool satisfy_link_condition(const std::vector<Vec3i>& F,
                            const std::vector<Vec3i>& FF,
                            const std::vector<Vec3i>& FFi, int f, int e,
                            std::vector<std::pair<int, int>>& neighbor0,
                            std::vector<std::pair<int, int>>& neighbor1);
}
namespace prism::section_validity {

void cleanup_empty_faces(std::vector<Vec3d>& V, std::vector<Vec3i>& F,
                         std::vector<std::set<int>>& track_ref,
                         std::vector<double>& target_adjustment) {
  constexpr auto remove_zero_rows = [](const auto& vecF, RowMati& mat) {
    std::vector<Vec3i> newF;
    newF.reserve(vecF.size());
    for (int i = 0; i < vecF.size(); i++) {
      auto& f = vecF[i];
      if (f[0] != f[1])
        newF.push_back(f);
      else
        newF.push_back({-1, -1, -1});
    }
    mat = Eigen::Map<RowMati>(newF[0].data(), newF.size(), 3);
  };

  RowMati mF;
  remove_zero_rows(F, mF);
  Eigen::VectorXi NI, NJ;
  igl::remove_unreferenced(V.size(), mF, NI, NJ);

  // assuming NJ is sorted ascending
  for (int i = 0; i < NJ.size(); i++) {
    V[i] = V[NJ[i]];
    target_adjustment[i] = target_adjustment[NJ[i]];
  }
  V.resize(NJ.size());
  target_adjustment.resize(NJ.size());

  int cur = 0;
  for (int i = 0; i < F.size(); i++) {
    if (F[i][0] == F[i][1]) continue;
    if (track_ref[i].size() == 0) spdlog::error("Zero Tracer");
    if (i != cur) track_ref[cur] = std::move(track_ref[i]);
    for (int j = 0; j < 3; j++) F[cur][j] = NI[F[i][j]];
    cur++;
  }
  track_ref.resize(cur);
  F.resize(cur);
}

bool intersect_check(const std::vector<Vec3d>& V,
                     const std::vector<Vec3i>& tris,
                     const prism::geogram::AABB& tree_base,
                     const prism::geogram::AABB& tree_top) {
  spdlog::trace("In IC 2x{}", tris.size());
  igl::Timer timer;
  timer.start();
  for (auto [v0, v1, v2] : tris) {
    spdlog::trace("ic v {} {} {}", v0, v1, v2);
    if (tree_base.intersects_triangle({V[v0], V[v1], V[v2]},
                                      v0 < tree_base.num_freeze))
      return false;
    if (tree_top.intersects_triangle({V[v0], V[v1], V[v2]},
                                     v0 < tree_top.num_freeze))
      return false;
  }
  auto elapsed = timer.getElapsedTimeInMicroSec();
  spdlog::trace("IC true {}", elapsed);
  return true;
}

bool distort_check(
    const std::vector<Vec3d>& V, const std::vector<Vec3i>& tris,
    const std::set<int>& combined_trackee,  // indices to prism pcF tracked
    const std::vector<Vec3d>& base, const std::vector<Vec3d>& mid,
    const std::vector<Vec3d>& top, const std::vector<Vec3i>& pcF,
    double distortion_bound, int num_freeze,
    std::vector<std::set<int>>& distributed_refs) {
  spdlog::trace("In DC ct#{}, tris{}", combined_trackee.size(), tris.size());
  igl::Timer timer;
  timer.start();
  distributed_refs.resize(tris.size());
  for (int i = 0; i < tris.size(); i++) {
    auto cur_tri =
        std::array<Vec3d, 3>{V[tris[i][0]], V[tris[i][1]], V[tris[i][2]]};

    for (auto t : combined_trackee) {  // for every traked original triangle.
      spdlog::trace("test sh {}, tri {}", t, tris[i]);
      auto [v0, v1, v2] = pcF[t];
      std::array<Vec3d, 3> base_vert{base[v0], base[v1], base[v2]};
      std::array<Vec3d, 3> mid_vert{mid[v0], mid[v1], mid[v2]};
      std::array<Vec3d, 3> top_vert{top[v0], top[v1], top[v2]};
      std::array<bool, 3> oct_type;
      prism::determine_convex_octahedron(base_vert, top_vert, oct_type,
                                         num_freeze > v0);
      bool intersected_prism = false;
      if (num_freeze <= v0 || tris[i][0] != v0) {
        intersected_prism =
            prism::triangle_intersect_octahedron(base_vert, mid_vert, oct_type,
                                                 cur_tri, num_freeze > v0) ||
            prism::triangle_intersect_octahedron(mid_vert, top_vert, oct_type,
                                                 cur_tri, num_freeze > v0);
      } else {
        intersected_prism = prism::singularless_triangle_intersect_octahedron(
                                base_vert, mid_vert, oct_type, cur_tri) ||
                            prism::singularless_triangle_intersect_octahedron(
                                mid_vert, top_vert, oct_type, cur_tri);
      }
      if (!intersected_prism) continue;
      for (int tc = (v0 < num_freeze) ? 1 : 0; tc < 3; tc++) {
        auto pillar = top_vert[tc] - base_vert[tc];
        auto distortion = prism::energy::map_max_cos_angle(pillar, cur_tri);
        if (distortion < distortion_bound) {
          spdlog::trace("DotProduct {}", distortion);
          spdlog::trace("sh{} with {}-{}-{}, tc{}, distortion: {}", t, v0, v1,
                        v2, tc, distortion);
          spdlog::trace("pillar {} tc {}", pillar, tc);
          spdlog::trace("cur{} {} {}\n{}\n{}\n{}", tris[i][0], tris[i][1],
                        tris[i][2], cur_tri[0], cur_tri[1], cur_tri[2]);
          spdlog::trace("mid{} {} {}\n{}\n{}\n{}", v0, v1, v2, mid_vert[0],
                        mid_vert[1], mid_vert[2]);
          return false;
        }
      }
      distributed_refs[i].insert(t);
    }
  }

  auto elapsed = timer.getElapsedTimeInMicroSec();
  spdlog::trace("DC true {}", elapsed);

  //
  return true;
}
}  // namespace prism::section_validity

namespace prism::section {

constexpr auto max_quality_on_tris = [](const auto& mid,
                                        const std::vector<Vec3i>& moved_tris) {
  double quality = 0;

  for (auto [v0, v1, v2] : moved_tris) {
    auto q = prism::energy::triangle_quality({mid[v0], mid[v1], mid[v2]});
    if (std::isnan(q)) return std::numeric_limits<double>::infinity();
    quality = std::max(quality, q);
  }
  return quality;
};

int attempt_relocate(const PrismCage& pc, const prism::geogram::AABB& base_tree,
                     const prism::geogram::AABB& top_tree,
                     std::vector<Vec3d>& V, const std::vector<Vec3i>& F,
                     const std::vector<std::set<int>>& map_track,
                     double distortion_bound,
                     // specified infos below
                     const std::vector<int>& nb, int vid, Vec3d relocation,
                     std::vector<std::set<int>>& sub_trackee) {
  std::vector<Vec3i> nb_tris(nb.size());
  for (int i = 0; i < nb.size(); i++) nb_tris[i] = F[nb[i]];
  auto quality_before = max_quality_on_tris(V, nb_tris);
  V[vid] = relocation;

  auto quality_after = max_quality_on_tris(V, nb_tris);
  spdlog::trace("Quality compare {} -> {}", quality_before, quality_after);
  if (std::isnan(quality_after) || quality_after > quality_before) {
    return 4;
  };
  //  volume check
  if (!prism::section_validity::intersect_check(V, nb_tris, base_tree,
                                                top_tree)) {
    spdlog::trace("failed inter check {},  restore", vid);
    return 2;
  }

  std::set<int> combined_tracks;
  for (auto f : nb) {
    std::merge(map_track[f].begin(), map_track[f].end(),
               combined_tracks.begin(), combined_tracks.end(),
               std::inserter(combined_tracks, combined_tracks.begin()));
  }

  auto dc = prism::section_validity::distort_check(
      V, nb_tris, combined_tracks, pc.base, pc.mid, pc.top, pc.F,
      distortion_bound, pc.ref.aabb->num_freeze, sub_trackee);
  if (!dc) {
    spdlog::trace("failed map check {}, restore", vid);
    return 3;
  }

  return 0;
}

int attempt_collapse(
    const PrismCage& pc, const prism::geogram::AABB& base_tree,
    const prism::geogram::AABB& top_tree, const std::vector<Vec3d>& V,
    const std::vector<Vec3i>& F, const std::vector<std::set<int>>& map_track,
    double distortion_bound, double improve_quality_threshold,
    // specified infos below
    std::vector<std::pair<int, int>>& neighbor0,
    std::vector<std::pair<int, int>>& neighbor1, int f0, int f1, int u0, int u1,
    std::tuple<std::vector<int> /*newfid*/,
               std::vector<std::set<int>> /*track*/>& checker) {
  std::vector<Vec3i> moved_tris, old_tris;
  std::vector<int> new_fid;
  moved_tris.reserve(neighbor0.size() - 2);
  for (auto [f, e] : neighbor0) {
    assert(F[f][e] == u0);
    auto new_tris = F[f];
    if (new_tris[0] == u1 || new_tris[1] == u1 || new_tris[2] == u1) continue;
    old_tris.push_back(new_tris);
    new_tris[e] = u1;
    moved_tris.emplace_back(new_tris);
    new_fid.push_back(f);
  }
  // note that new_fid is geometric cyclic.
  assert(moved_tris.size() == neighbor0.size() - 2);

  spdlog::trace("Quality check");
  auto quality_before = max_quality_on_tris(V, old_tris);
  auto quality_after = max_quality_on_tris(V, moved_tris);
  spdlog::trace("Quality compare {} -> {}", quality_before, quality_after);
  if (std::isnan(quality_after)) return 4;
  if ((quality_after > improve_quality_threshold) &&
      quality_after > quality_before)  // if the quality is too large, not allow
                                       // it to increase.
    return 4;
  if (quality_after > quality_before && quality_after > 30)
    return 4;  // if increase
  //  volume check

  auto ic = prism::section_validity::intersect_check(V, moved_tris, base_tree,
                                                     top_tree);
  if (!ic) return 2;
  // get new subdivide types

  std::set<int> combined_tracks;
  for (auto fe : neighbor0) {
    auto f = fe.first;
    std::merge(map_track[f].begin(), map_track[f].end(),
               combined_tracks.begin(), combined_tracks.end(),
               std::inserter(combined_tracks, combined_tracks.begin()));
  }
  std::vector<std::set<int>> sub_refs;
  auto dc = prism::section_validity::distort_check(
      V, moved_tris, combined_tracks, pc.base, pc.mid, pc.top, pc.F,
      distortion_bound, pc.ref.aabb->num_freeze, sub_refs);
  if (!dc) return 3;

  checker = std::tuple(std::move(new_fid), std::move(sub_refs));
  return 0;
}

int attempt_flip(const PrismCage& pc, const prism::geogram::AABB& base_tree,
                 const prism::geogram::AABB& top_tree,
                 const std::vector<Vec3d>& V, const std::vector<Vec3i>& F,
                 const std::vector<std::set<int>>& map_track,
                 double distortion_bound,
                 // specified infos below
                 int f0, int f1, int e0, int e1, int v0, int v1,
                 std::vector<std::set<int>> /*track*/
                     & checker) {
  std::vector<Vec3i> moved_tris{F[f0], F[f1]};

  auto quality_before = max_quality_on_tris(V, moved_tris);
  auto e01 = (e0 + 1) % 3;
  auto e11 = (e1 + 1) % 3;
  moved_tris[0][e01] = v1;
  moved_tris[1][e11] = v0;
  auto quality_after = max_quality_on_tris(V, moved_tris);
  if (quality_after > quality_before) return 4;

  auto ic = prism::section_validity::intersect_check(V, moved_tris, base_tree,
                                                     top_tree);
  if (!ic) {
    return 2;
  }

  std::set<int> combined_tracks;
  std::merge(map_track[f0].begin(), map_track[f0].end(), map_track[f1].begin(),
             map_track[f1].end(),
             std::inserter(combined_tracks, combined_tracks.begin()));
  auto dc = prism::section_validity::distort_check(
      V, moved_tris, combined_tracks, pc.base, pc.mid, pc.top, pc.F,
      distortion_bound, pc.ref.aabb->num_freeze, checker);
  if (!dc) {
    return 3;
  }
  return 0;
}

int attempt_split(
    const PrismCage& pc, const prism::geogram::AABB& base_tree,
    const prism::geogram::AABB& top_tree, const std::vector<Vec3d>& V,
    const std::vector<Vec3i>& F, std::vector<std::set<int>>& map_track,
    double distortion_bound, double improve_quality, int f0, int f1, int e0,
    int e1,
    std::tuple<std::vector<int> /*fid*/, std::vector<std::set<int>> /*track*/
               >& checker) {
  int ux = V.size() - 1;
  std::vector<Vec3i> new_tris{F[f0], F[f1], F[f0], F[f1]};
  // conform to bool edge_split
  new_tris[0][e0] = ux;
  new_tris[1][e1] = ux;
  new_tris[2][(e0 + 1) % 3] = ux;
  new_tris[3][(e1 + 1) % 3] = ux;
  auto quality_before = max_quality_on_tris(V, {F[f0], F[1]});
  auto quality_after = max_quality_on_tris(V, new_tris);
  spdlog::trace("Quality compare {} -> {}", quality_before, quality_after);
  if (std::isnan(quality_after) ||
      (improve_quality && quality_after > quality_before)) {
    return 4;
  };

  std::vector<int> new_fid{f0, f1, static_cast<int>(F.size()),
                           static_cast<int>(F.size() + 1)};

  auto ic = prism::section_validity::intersect_check(V, new_tris, base_tree,
                                                     top_tree);
  if (!ic) {
    return 2;
  }
  std::vector<std::set<int>> sub_refs;
  std::set<int> combined_tracks;
  std::merge(map_track[f0].begin(), map_track[f0].end(), map_track[f1].begin(),
             map_track[f1].end(),
             std::inserter(combined_tracks, combined_tracks.begin()));
  auto dc = prism::section_validity::distort_check(
      V, new_tris, combined_tracks, pc.base, pc.mid, pc.top, pc.F,
      distortion_bound, pc.ref.aabb->num_freeze, sub_refs);

  if (!dc) {
    return 3;
  }

  checker = std::tuple(std::move(new_fid), std::move(sub_refs));
  return 0;
}

int wildcollapse_pass(const PrismCage& pc,
                      const prism::geogram::AABB& base_tree,
                      const prism::geogram::AABB& top_tree,
                      RemeshOptions& option, std::vector<Vec3d>& V,
                      std::vector<Vec3i>& F,
                      std::vector<std::set<int>>& track_to_prism,
                      std::vector<double>& target_adjustment) {
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
      queue.push({-(V[v0] - V[v1]).norm(), f, e, v0, v1, 0});
    }
  }

  std::vector<int> rejections_steps(6, 0);
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
    // if (u0 == u1 || u0 != v0 ||
    // u1 != v1)  // vid changed, means the edge is outdated.
    // continue;
    assert((V[u1] - V[u0]).norm() == l &&
           "Outdated entries will be ignored, this condition can actually "
           "replace the previous");

    if (std::abs(l) > (option.sizing_field(V[u0]) * target_adjustment[u0] +
                       option.sizing_field(V[u1]) * target_adjustment[u1]) /
                          2. * (4 / 5.))
      continue;  // skip if l too long

    spdlog::trace("LinkCondition check {} {}", f, e);
    // collapse and misc checks.
    std::vector<std::pair<int, int>> n0, n1;
    if (!collapse::satisfy_link_condition(F, FF, FFi, f, e, n0, n1)) {
      rejections_steps[0]++;
      continue;
    }
    spdlog::trace("LinkCondition pass, attempt {} {}", f, e);
    auto f1 = FF[f][e], e1 = FFi[f][e];
    std::tuple<std::vector<int> /*newfid*/,
               std::vector<std::set<int>> /*track*/>
        checker;
    int flag = prism::section::attempt_collapse(
        pc, base_tree, top_tree, V, F, track_to_prism, option.distortion_bound,
        option.collapse_quality_threshold, n0, n1, f, f1, u0, u1, checker);
    spdlog::trace("Attempt Collapse, {} {} pass: {}", f, e,
                  flag == 0 ? true : false);
    if (flag != 0) {
      rejections_steps[flag]++;
      // test the reverse
      std::swap(n0, n1);
      std::swap(f, f1);
      std::swap(u0, u1);
      std::swap(v0, v1);
      std::swap(e, e1);
      // if (!collapse::satisfy_link_condition(F, FF, FFi, f, e, n0, n1))
      // continue;
      flag = prism::section::attempt_collapse(
          pc, base_tree, top_tree, V, F, track_to_prism,
          option.distortion_bound, option.collapse_improve_quality, n0, n1, f,
          f1, u0, u1, checker);

      if (flag != 0) continue;  // still failing
    }
    auto& [new_fid, new_tracks] = checker;

    prism::edge_collapse(F, FF, FFi, f, e);
    spdlog::trace("EdgeCollapse done {} {}", f, e);

    assert(new_fid.size() == new_tracks.size());
    for (int i = 0; i < new_tracks.size(); i++) {
      track_to_prism[new_fid[i]] = new_tracks[i];
    }

    // shifts

    // Push the modified edges back in the queue
    // Not removing replaced ones since additional checks are in place: v0,v1
    // and norm==l.

    // Push the modified edges back in the queue
    global_tick++;
    for (int i = 0; i < new_fid.size(); i++) {
      auto f = new_fid[i];
      for (auto e = 0; e < 3; e++) {
        auto v0 = F[f][e], v1 = F[f][(e + 1) % 3];
        if (FF[f][e] == -1) {
          timestamp(f, e) = -1;
          continue;
        }
        auto f_1 = FF[f][e], e_1 = FFi[f][e];
        queue.push({-(V[v1] - V[v0]).norm(), f, e, v0, v1, global_tick});
        queue.push({-(V[v1] - V[v0]).norm(), f_1, e_1, v1, v0, global_tick});
        timestamp(f, e) = global_tick;
        timestamp(f_1, e_1) = global_tick;
        spdlog::trace("pq {} {} {} {} {}", f, e, v0, v1, global_tick);
      }
    }
    spdlog::trace("Edge Collapsed {} {}", f, e);
  }
  spdlog::info("Pass Collapse total {}. lk{}, v{} i{} d{} q{}", global_tick,
               rejections_steps[0], rejections_steps[1], rejections_steps[2],
               rejections_steps[3], rejections_steps[4]);
  prism::section_validity::cleanup_empty_faces(V, F, track_to_prism,
                                               target_adjustment);
  return global_tick;
}

void wildsplit_pass(const PrismCage& pc, const prism::geogram::AABB& base_tree,
                    const prism::geogram::AABB& top_tree, RemeshOptions& option,
                    std::vector<Vec3d>& V, std::vector<Vec3i>& F,
                    std::vector<std::set<int>>& track_ref,
                    std::vector<double>& target_adjustment) {
  auto overrefine_limit = 1e-3;
  auto input_vnum = V.size();
  using queue_entry =
      std::tuple<double, int /*f*/, int /*e*/, int /*u0*/, int /*u1*/>;
  std::priority_queue<queue_entry> queue;

  // build connectivity
  auto [FF, FFi] = prism::local::triangle_triangle_adjacency(F);

  // enqueue
  for (auto [f, flags] = std::pair(0, RowMati(RowMati::Zero(F.size(), 3)));
       f < F.size(); f++) {
    for (auto e : {0, 1, 2}) {
      auto v0 = F[f][e], v1 = F[f][(e + 1) % 3];
      if (v0 > v1 || flags(f, e) == 1) continue;
      if (FF[f][e] == -1) continue;  // skip boundary
      double len = (V[v0] - V[v1]).norm();
      if (len > overrefine_limit) queue.push({len, f, e, v0, v1});
      flags(f, e) = 1;
      flags(FF[f][e], FFi[f][e]) = 1;
    }
  }

  std::array<int, 6> rejections_steps{0, 0, 0, 0, 0, 0};
  // pop
  while (!queue.empty()) {
    auto [l, f0, e0, u0, u1] = queue.top();
    queue.pop();

    if (f0 == -1 || FF[f0][e0] == -1) continue;  // skip boundary
    if (auto u0_ = F[f0][e0], u1_ = F[f0][(e0 + 1) % 3];
        u0_ == u1_ || u0_ != u0 ||
        u1_ != u1)  // vid changed, means the edge is outdated.
      continue;
    if (std::abs(l) < (option.sizing_field(V[u0]) * target_adjustment[u0] +
                       option.sizing_field(V[u1]) * target_adjustment[u1]) /
                          2. * (4 / 3.)) {
      continue;  // skip if l too short
    }

    auto f1 = FF[f0][e0], e1 = FFi[f0][e0];
    if (f1 == -1) continue;  // boundary check
    auto v0 = F[f0][(e0 + 2) % 3];
    auto v1 = F[f1][(e1 + 2) % 3];

    Vec3d mid_value = (V[u0] + V[u1]) / 2;

    spdlog::trace("Attempting: {}-{} {}-{} {}->{} {}-{}", f0, e0, f1, e1, u0,
                  u1, v0, v1);
    std::tuple<std::vector<int> /*fid*/, std::vector<std::set<int>>> checker;

    int flag = 1;
    V.push_back(mid_value);
    flag = prism::section::attempt_split(
        pc, base_tree, top_tree, V, F, track_ref, option.distortion_bound,
        option.split_improve_quality, f0, f1, e0, e1, checker);
    if (flag != 0) {
      V.pop_back();
      spdlog::trace("Split Attempt Failed {}-{} {}-{}", f0, e0, f1, e1);
      rejections_steps[flag]++;
      continue;
    }
    prism::edge_split(V.size() - 1, F, FF, FFi, f0, e0);

    auto& [new_fid, new_tracks] = checker;
    assert(new_fid.size() == new_tracks.size());
    track_ref.resize(F.size());
    for (int i = 0; i < new_tracks.size(); i++) {
      track_ref[new_fid[i]] = new_tracks[i];
    }

    auto push_to_queue = [&queue, &F, &V, input_vnum, &overrefine_limit](
                             auto f, auto v) {
      auto e = -1;
      auto face = F[f];
      for (int i = 0; i < 3; i++)
        if (face[i] == v) e = i;
      if (e == -1) spdlog::error("push queue wrong");
      auto u0 = F[f][e], u1 = F[f][(e + 1) % 3];
      if (u0 > u1 || u1 >= input_vnum) {
        return;
      }
      auto len = (V[u1] - V[u0]).norm();
      if (len > overrefine_limit)  // avoid over-refinement
        queue.push({len, f, e, u0, u1});
      spdlog::trace("pushed {} {} {} {}", f, e, u0, u1);
    };

    auto fx0 = F.size() - 2;
    auto fx1 = F.size() - 1;
    spdlog::trace("Consider {} {} {} {} {} {}", u0, u1, v0, v1, fx0, fx1);
    if (v0 < u0) push_to_queue(fx0, v0);
    if (v1 < u1) push_to_queue(fx1, v1);
    rejections_steps[0]++;
    target_adjustment.push_back(
        (target_adjustment[u0] + target_adjustment[u1]) / 2);
  }
  spdlog::info("Split Done {}, Rejections v{} i{} d{} q{}", rejections_steps[0],
               rejections_steps[1], rejections_steps[2], rejections_steps[3],
               rejections_steps[4]);

  std::set<int> low_quality_vertices;
  std::vector<double> all_qualities;
  for (auto [v0, v1, v2] : F) {
    if (prism::energy::triangle_quality({V[v0], V[v1], V[v2]}) > 30) {
      low_quality_vertices.insert(v0);
      low_quality_vertices.insert(v1);
      low_quality_vertices.insert(v2);
    }
  }
  for (auto v : low_quality_vertices) target_adjustment[v] /= (2 * 1.5);
  for (auto& u : target_adjustment) {
    u *= 1.5;
    u = (u > 1.) ? 1. : u;
    u = (u < 1e-2) ? 1e-2 : u;
  }

  spdlog::info("Post Split Adjustments: low_quality {}/{}",
               low_quality_vertices.size(), V.size());
}

void smooth_single(const PrismCage& pc, const prism::geogram::AABB& base_tree,
                   const prism::geogram::AABB& top_tree,
                   double distortion_bound, int vid,
                   const std::vector<std::vector<int>>& VF,
                   const std::vector<std::vector<int>>& VFi,
                   const std::vector<bool>& skip, std::vector<Vec3d>& V,
                   std::vector<Vec3i>& F,
                   std::vector<std::set<int>>& track_to_prism) {
  if (skip[vid]) return;
  spdlog::trace("smooth single: {}", vid);
  Vec3d mid_value;
  {
    Vec3d new_pos(0, 0, 0);
    std::set<int> neighborverts;
    for (auto f : VF[vid]) {
      for (int j = 0; j < 3; j++) neighborverts.insert(F[f][j]);
    }
    // neighborverts.erase(vid);
    for (auto v : neighborverts) new_pos += V[v];
    new_pos /= neighborverts.size();
    // new_direction -= V[vid];

    if (!::prism::project_to_ref_mesh(pc, track_to_prism, VF[vid], new_pos,
                                      mid_value))
      return;
  }

  spdlog::trace("attempt smooth: {}", vid);
  std::vector<std::set<int>> new_tracks;
  Vec3d old_loc = V[vid];
  int flag = prism::section::attempt_relocate(
      pc, base_tree, top_tree, V, F, track_to_prism, distortion_bound, VF[vid],
      vid, mid_value, new_tracks);

  if (flag > 0) {
    V[vid] = old_loc;
    spdlog::trace("Smoother checker failed.");
    return;
  } else {
    spdlog::trace("Smoother Sucess. Distributing tracks");
    for (int i = 0; i < VF[vid].size(); i++) {
      track_to_prism[VF[vid][i]] = std::move(new_tracks[i]);
    }
  }
};

void localsmooth_pass(const PrismCage& pc,
                      const prism::geogram::AABB& base_tree,
                      const prism::geogram::AABB& top_tree,
                      RemeshOptions& option, std::vector<Vec3d>& V,
                      std::vector<Vec3i>& F,
                      std::vector<std::set<int>>& track_ref) {
  std::vector<std::vector<int>> VF, VFi, groups;
  std::vector<bool> skip_flag(V.size(), false);
  {
    RowMati mF, mE;
    vec2eigen(F, mF);
    igl::vertex_triangle_adjacency(V.size(), mF, VF, VFi);
    prism::local::vertex_coloring(mF, groups);
    igl::boundary_facets(mF, mE);
    for (int i = 0; i < mE.rows(); i++)
      for (auto j : {0, 1}) {
        skip_flag[mE(i, j)] = true;
      }
    for (int i = 0; i < pc.ref.aabb->num_freeze; i++) skip_flag[i] = true;
    spdlog::info("Smoothing with {} Groups {} bnd", groups.size(), mE.rows());
  }

  std::srand(0);
  std::random_device rd;
  std::mt19937 mtg(rd());
  for (auto& gr : groups) {
    std::shuffle(gr.begin(), gr.end(), mtg);
    igl::parallel_for(
        gr.size(),
        [&gr, &pc = std::as_const(pc), &VF, &VFi,
         distortion_bound = option.distortion_bound, &skip_flag, &track_ref,
         &base_tree = std::as_const(base_tree),
         &top_tree = std::as_const(top_tree), &V, &F](auto ii) {
          smooth_single(pc, base_tree, top_tree, distortion_bound, gr[ii], VF,
                        VFi, skip_flag, V, F, track_ref);
        },
        size_t(option.parallel ? 1 : gr.size()));
  }
}

void check_manifold(const std::vector<Vec3i>& F, const std::vector<Vec3i>& FF,
                    const std::vector<Vec3i>& FFi) {
  auto [nFF, nFFi] = prism::local::triangle_triangle_adjacency(F);
  for (int i = 0; i < F.size(); i++) {
    for (int j = 0; j < 3; j++) {
      if (nFF[i][j] != FF[i][j] || nFFi[i][j] != FFi[i][j]) {
        spdlog::error("Not Match {} {}", i, j);
        exit(1);
      }
    }
  }
}
void wildflip_pass(const PrismCage& pc, const prism::geogram::AABB& base_tree,
                   const prism::geogram::AABB& top_tree, RemeshOptions& option,
                   std::vector<Vec3d>& V, std::vector<Vec3i>& F,
                   std::vector<std::set<int>>& track_ref) {
  using queue_entry =
      std::tuple<double, int /*f*/, int /*e*/, int /*u0*/, int /*u1*/>;
  std::priority_queue<queue_entry> queue;

  // build connectivity
  auto [FF, FFi] = prism::local::triangle_triangle_adjacency(F);

  // enqueue
  for (auto [f, flags] = std::pair(0, RowMati(RowMati::Zero(F.size(), 3)));
       f < F.size(); f++) {
    for (auto e : {0, 1, 2}) {
      auto v0 = F[f][e], v1 = F[f][(e + 1) % 3];
      if (v0 > v1 || flags(f, e) == 1) continue;
      if (FF[f][e] == -1) continue;
      queue.push({(V[v0] - V[v1]).norm(), f, e, v0, v1});
      flags(f, e) = 1;
      flags(FF[f][e], FFi[f][e]) = 1;
    }
  }

  std::vector<int> valence(V.size(), 0);
  for (auto f : F)
    for (int j = 0; j < 3; j++) valence[f[j]]++;

  std::array<int, 5> rejection_steps{0, 0, 0, 0, 0};
  int global_tick = 0;
  // pop
  while (!queue.empty()) {
    auto [l, f, e, u0, u1] = queue.top();
    queue.pop();
    if (f == -1 || FF[f][e] == -1) continue;  // skip boundary

    if (auto u0_ = F[f][e], u1_ = F[f][(e + 1) % 3];
        u0_ == u1_ || u0_ != u0 ||
        u1_ != u1)  // vid changed, means the edge is outdated.
      continue;
    else
      assert((V[u1_] - V[u0_]).norm() == l &&
             "Outdated entries will be ignored, this condition can actually "
             "replace the previous");

    auto f1 = FF[f][e], e1 = FFi[f][e];
    if (f1 == -1) continue;  // boundary check
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
    std::vector<std::set<int>> new_tracks;
    int flag = prism::section::attempt_flip(pc, base_tree, top_tree, V, F,
                                            track_ref, option.distortion_bound,
                                            f0, f1, e0, e1, v0, v1, new_tracks);
    if (flag > 0) {
      rejection_steps[flag]++;
      continue;
    }
    std::vector<int> new_fid{f, f1};

    if (!prism::edge_flip(F, FF, FFi, f, e)) {
      rejection_steps[0]++;
      continue;
    }

    assert(new_fid.size() == new_tracks.size());
    for (int i = 0; i < new_tracks.size(); i++) {
      track_ref[new_fid[i]] = new_tracks[i];
    }

    // Push the modified edges back in the queue
    // Not removing replaced ones since dditional checks are in place: v0,v1 and
    // norm==l.
    global_tick++;
    auto push_to_queue = [&queue, &F, &V](auto f, auto v) {
      auto e = -1;
      auto face = F[f];
      for (int i = 0; i < 3; i++)
        if (face[i] == v) e = i;
      if (e == -1) spdlog::error("push queue wrong");
      auto u0 = F[f][e], u1 = F[f][(e + 1) % 3];
      if (u0 > u1) {
        return;
      }

      queue.push({(V[u1] - V[u0]).norm(), f, e, u0, u1});
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
  spdlog::info("Flip {} Done, Rej t{}, v{} i{} d{} q{}", global_tick,
               rejection_steps[0], rejection_steps[1], rejection_steps[2],
               rejection_steps[3], rejection_steps[4]);
}

}  // namespace prism::section