#include <igl/boundary_facets.h>
#include <igl/parallel_for.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/volume.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>
#include <prism/energy/prism_quality.hpp>
#include <algorithm>
#include <prism/energy/smoother_pillar.hpp>
#include <prism/predicates/triangle_triangle_intersection.hpp>

#include "mesh_coloring.hpp"
#include "prism/PrismCage.hpp"
#include "prism/cage_utils.hpp"
#include "prism/cgal/triangle_triangle_intersection.hpp"
#include "prism/geogram/AABB.hpp"
#include "prism/intersections.hpp"
#include "prism/spatial-hash/AABB_hash.hpp"
#include "remesh_pass.hpp"
#include "validity_checks.hpp"

namespace prism::local {
int smooth_prism(PrismCage &pc, int vid,
                 const std::vector<std::vector<int>> &VF,
                 const std::vector<std::vector<int>> &VFi,
                 const RemeshOptions &option, bool true_zoom_false_rotate,
                 const std::vector<bool> &skip) {
  auto attempt_operation = option.use_polyshell? local_validity::attempt_zig_remesh: local_validity::attempt_feature_remesh;
  if (vid < pc.ref.aabb->num_freeze)
    return 1;  // only skip singularity, not boundary or feature
  std::optional<std::pair<Vec3d, Vec3d>> great_prism;
  spdlog::trace("zoom or rotate for vid {}", vid);
  if (true_zoom_false_rotate) {
    great_prism = zoom(pc.base, pc.mid, pc.top, pc.F, VF, VFi, vid,
                       option.target_thickness);
  } else
    great_prism = rotate(pc.base, pc.mid, pc.top, pc.F, VF, VFi, vid,
                         option.target_thickness);
  // great_prism =
  // zoom_and_rotate(pc.base, pc.mid, pc.top, pc.F, pc.ref.aabb->num_freeze, VF,
  // VFi, vid, target_thick);
  if (!great_prism) {
    spdlog::debug("Zoom and Rotate failed.");
    return 1;
  }

  std::array<Vec3d, 3> relocations;
  std::tie(relocations[0], relocations[2]) = great_prism.value();

  std::vector<RowMatd> local_cp;
  std::vector<Vec3i> moved_tris;
  std::vector<std::set<int>> new_tracks;
  for (auto &f : VF[vid]) {
    moved_tris.push_back(pc.F[f]);
  }
  auto &new_fid = VF[vid];
  auto &old_fid = VF[vid];

  auto flag = 1;
  double alpha = 1.;
  std::tuple<Vec3d, Vec3d, Vec3d> old_locations{pc.base[vid], pc.mid[vid],
                                                pc.top[vid]};

  double old_quality = prism::local_validity::max_quality_on_tris(
      pc.base, pc.mid, pc.top, moved_tris);
  bool enable_feature_separation =
      skip[vid];  // TODO, maybe seperate the two cases of feature vs. real
                  // skip.
  while (flag == 1 || flag == 2) {  // shrink if there is a volume failure.
    pc.base[vid] = relocations[0] * (alpha) + (1 - alpha) * pc.mid[vid];
    pc.top[vid] = relocations[2] * (alpha) + (1 - alpha) * pc.mid[vid];
    flag = attempt_operation(
        pc, pc.track_ref, option, old_quality, old_fid, moved_tris, new_tracks,
        local_cp);
    spdlog::trace("newb {}", pc.base[vid]);
    spdlog::trace("newt {}", pc.top[vid]);
    alpha *= 0.8;
    if (alpha < 1e-2) break;
  }
  if (flag > 0) {
    spdlog::trace("ZoomRotate checker failed.");
    std::tie(pc.base[vid], pc.mid[vid], pc.top[vid]) = old_locations;
    return flag;
  }
  prism::local_validity::post_operation(pc, option, old_fid, new_fid,
                                        new_tracks, local_cp);
  spdlog::trace("ZoomRotate SUCCEED, move to next.");
  return 0;
}

int smooth_single(PrismCage &pc, int vid,
                  const std::vector<std::vector<int>> &VF,
                  const std::vector<std::vector<int>> &VFi,
                  const RemeshOptions &option, const std::vector<bool> &skip) {
  auto attempt_operation = option.use_polyshell? local_validity::attempt_zig_remesh: local_validity::attempt_feature_remesh;
  if (skip[vid]) return 1;

  spdlog::trace("smooth attempt: {}", vid);
  auto new_direction = prism::smoother_direction(
      pc.base, pc.mid, pc.top, pc.F, pc.ref.aabb->num_freeze, VF, VFi, vid);

  if (!new_direction) {
    spdlog::trace("No better location.");
    return 1;
  }

  std::tuple<Vec3d, Vec3d, Vec3d> old_locations{pc.base[vid], pc.mid[vid],
                                                pc.top[vid]};
  std::array<Vec3d, 3> relocations{pc.base[vid] + new_direction.value(),
                                   pc.mid[vid] + new_direction.value(),
                                   pc.top[vid] + new_direction.value()};
  auto query = [&ref = pc.ref](
                   const Vec3d &s, const Vec3d &t,
                   const std::set<int> &total_trackee) -> std::optional<Vec3d> {
    if (ref.aabb->enabled)  // this can be discarded if no performance benefit
                            // is found.
      return ref.aabb->segment_query(s, t);
    std::array<Vec3d, 2> seg_query{s, t};
    for (auto f : total_trackee) {
      auto v0 = ref.F(f, 0), v1 = ref.F(f, 1), v2 = ref.F(f, 2);
      auto mid_intersect = prism::cgal::segment_triangle_intersection(
          seg_query, {ref.V.row(v0), ref.V.row(v1), ref.V.row(v2)});
      if (mid_intersect) return mid_intersect;
    }
    return {};
  };

  if (true) {  // project onto reference
    std::set<int> total_trackee;
    for (auto f : VF[vid])
      total_trackee.insert(pc.track_ref[f].begin(), pc.track_ref[f].end());
    std::optional<Vec3d> mid_intersect;
    for (int i = 0; i < 20; i++) {
      mid_intersect = query(relocations[0], relocations[2], total_trackee);
      if (mid_intersect) break;
      relocations[0] = (pc.base[vid] + relocations[0]) / 2;
      relocations[2] = (pc.top[vid] + relocations[2]) / 2;
    }
    if (!mid_intersect) {
      spdlog::trace("Pan mid failed.");
      return 1;
    }
    relocations[1] = mid_intersect.value();
    spdlog::trace("found new mid {}", relocations[1]);
  }

  std::vector<RowMatd> local_cp;
  std::vector<Vec3i> moved_tris;
  std::vector<std::set<int>> new_tracks;
  for (auto &f : VF[vid]) {
    moved_tris.push_back(pc.F[f]);
  }
  auto &new_fid = VF[vid];
  auto &old_fid = VF[vid];

  double old_quality = prism::local_validity::max_quality_on_tris(
      pc.base, pc.mid, pc.top, moved_tris);
  pc.base[vid] = relocations[0];
  pc.mid[vid] = relocations[1];
  pc.top[vid] = relocations[2];
  int flag = attempt_operation(
      pc, pc.track_ref, option, old_quality, old_fid, moved_tris, new_tracks,
      local_cp);

  if (flag > 0) {
    spdlog::trace("Pan checker failed.");
    std::tie(pc.base[vid], pc.mid[vid], pc.top[vid]) = old_locations;
    return flag;
  }

  prism::local_validity::post_operation(pc, option, old_fid, new_fid,
                                        new_tracks, local_cp);
  spdlog::trace("SUCCEED, move to next.");
  return 0;
};
}  // namespace prism::local

double total_energy(const std::vector<Vec3d> &V, const std::vector<Vec3i> &F) {
  std::set<int> low_quality_vertices;
  double total_quality = 0;
  double max_quality = 0;
  for (auto [v0, v1, v2] : F) {
    auto q = prism::energy::triangle_quality({V[v0], V[v1], V[v2]});
    total_quality += q;
    max_quality = std::max(max_quality, q);
  }

  spdlog::info("Total Q fnum {}, avg {}, max {}", F.size(),
               total_quality / F.size(), max_quality);
  return max_quality;
};

void prism::local::localsmooth_pass(PrismCage &pc,
                                    const RemeshOptions &option) {
#ifndef NDEBUG
  {
    std::vector<Vec3d> tetV;
    std::vector<Vec4i> tetT;
    prism::cage_utils::tetmesh_from_prismcage(
        pc.base, pc.mid, pc.top, pc.F, pc.ref.aabb->num_freeze, tetV, tetT);
    RowMatd tV;
    RowMati tT;
    vec2eigen(tetV, tV);
    vec2eigen(tetT, tT);
    Eigen::VectorXd vols;
    igl::volume(tV, tT, vols);
    spdlog::warn("Volumes {} {} {}", vols.minCoeff(), vols.maxCoeff(),
                 vols.mean());
  }
#endif
  std::vector<std::vector<int>> VF, VFi, groups;
  std::vector<bool> skip_flag(pc.mid.size(), false);
  for (auto [m, ignore] : pc.meta_edges) {
    auto [v0, v1] = m;
    skip_flag[v0] = true;
    skip_flag[v1] = true;
  }
  for (int i = 0; i < pc.ref.aabb->num_freeze; i++) skip_flag[i] = true;
  {
    RowMati mF, mE;
    vec2eigen(pc.F, mF);
    igl::vertex_triangle_adjacency(pc.mid.size(), mF, VF, VFi);
    prism::local::vertex_coloring(mF, groups);
    igl::boundary_facets(mF, mE);
    for (int i = 0; i < mE.rows(); i++)
      for (auto j : {0, 1}) skip_flag[mE(i, j)] = true;
  }
  spdlog::info("Smoothing: Group Count {}", groups.size());

  std::srand(0);
  if (option.parallel && pc.top_grid != nullptr) {
    spdlog::error("Multithread hashmap is not safe. Todo: move to TBB.");
  }
  std::vector<int> stats(pc.mid.size(), 0);
  for (auto &gr : groups)
    igl::parallel_for(
        gr.size(),
        [&gr, &pc, &VF, &VFi, &option, &skip_flag, &stats](auto ii) {
          stats[gr[ii]] = smooth_single(pc, gr[ii], VF, VFi, option, skip_flag);
        },
        size_t(option.parallel ? 1 : pc.mid.size()));
  for (auto &gr : groups)
    igl::parallel_for(
        gr.size(),
        [&gr, &pc, &VF, &VFi, &option, &skip_flag, &stats](auto ii) {
          stats[gr[ii]] =
              smooth_prism(pc, gr[ii], VF, VFi, option, false, skip_flag);
          stats[gr[ii]] =
              smooth_prism(pc, gr[ii], VF, VFi, option, true, skip_flag);
          
        },
        size_t(option.parallel ? 1 : pc.mid.size()));

  spdlog::info("Finished Smoothing");
  total_energy(pc.mid, pc.F);
  return;
}

namespace prism::local {

void legacy_smooth_prism(PrismCage &pc, int vid,
                         const std::vector<std::vector<int>> &VF,
                         const std::vector<std::vector<int>> &VFi,
                         const prism::local::RemeshOptions &option,
                         const std::vector<bool> &skip, bool on_base) {
  if (skip[vid]) return;

  auto great_prism = prism::smoother_location_legacy(
      pc.base, pc.mid, pc.top, pc.F, pc.ref.aabb->num_freeze, VF, VFi, vid,
      on_base);
  if (!great_prism) {
    spdlog::trace("Legacy Smooth failed.");
    return;
  }

  std::array<Vec3d, 3> relocations{pc.base[vid], pc.mid[vid], pc.top[vid]};
  relocations[on_base ? 0 : 2] = great_prism.value();
  relocations[1] = (relocations[0] + relocations[2]) / 2;

  std::vector<std::set<int>> checker;
  auto flag = 1;
  double alpha = 1.;
  throw std::runtime_error("legacy code.");
  // flag = prism::local_validity::attempt_relocate(
      // pc, pc.track_ref, option, VF[vid], vid, relocations, {}, checker);
  if (flag > 0) {
    spdlog::trace("ZoomRotate checker failed.");
  } else {
    for (int i = 0; i < VF[vid].size(); i++) {
      pc.track_ref[VF[vid][i]] = std::move(checker[i]);
    }
    spdlog::trace("ZoomRotate SUCCEED, move to next.");
  }
}

void shellsmooth_pass(PrismCage &pc, const RemeshOptions &option) {
  std::vector<std::vector<int>> VF, VFi, groups;
  std::vector<bool> skip_flag(pc.mid.size(), false);
  {
    RowMati mF, mE;
    vec2eigen(pc.F, mF);
    igl::vertex_triangle_adjacency(pc.mid.size(), mF, VF, VFi);
    prism::local::vertex_coloring(mF, groups);
    igl::boundary_facets(mF, mE);
    spdlog::info("boundary, mE {}", mE.rows());
    for (int i = 0; i < mE.rows(); i++)
      for (auto j : {0, 1}) {
        skip_flag[mE(i, j)] = true;
      }
    for (int i = 0; i < pc.ref.aabb->num_freeze; i++) skip_flag[i] = true;
  }
  spdlog::info("Group Count {}", groups.size());
  std::srand(0);

  for (auto &gr : groups)
    igl::parallel_for(
        gr.size(),
        [&gr, &pc, &VF, &VFi, &option, &skip_flag](auto ii) {
          legacy_smooth_prism(pc, gr[ii], VF, VFi, option, skip_flag, true);
          legacy_smooth_prism(pc, gr[ii], VF, VFi, option, skip_flag, false);
        },
        size_t(option.parallel ? 1 : pc.mid.size()));

  return;
}

}  // namespace prism::local
