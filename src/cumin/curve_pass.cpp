#include <igl/boundary_facets.h>
#include <igl/parallel_for.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/volume.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <algorithm>

#include "curve_validity.hpp"
#include "prism/PrismCage.hpp"
#include "prism/cage_utils.hpp"
#include "prism/geogram/AABB.hpp"
#include "prism/intersections.hpp"
#include "prism/local_operations/mesh_coloring.hpp"
#include "prism/local_operations/remesh_pass.hpp"
#include "prism/spatial-hash/AABB_hash.hpp"
namespace prism::curve {
auto smooth_prism(const PrismCage &pc, int vid,
                  const std::vector<std::vector<int>> &VF,
                  const std::vector<std::vector<int>> &VFi,
                  const prism::local::RemeshOptions &option,
                  const std::vector<bool> &skip) {
  if (vid < pc.ref.aabb->num_freeze)
    return false;  // only skip singularity, not boundary or feature

  std::vector<std::set<int>> checker;
  auto flag = 1;
  double alpha = 1.;
  std::vector<RowMatd> local_cp;
  std::vector<Vec3i> moved_tris;
  for (auto f : VF[vid]) moved_tris.push_back(pc.F[f]);
  if (option.curve_checker.first.has_value() &&
      !(std::any_cast<std::function<bool(
            const PrismCage &, const std::vector<int> &,
            const decltype(moved_tris) &, decltype(local_cp) &)>>(
          option.curve_checker.first))(pc, VF[vid], moved_tris, local_cp)) {
    spdlog::debug("Curving checker failed.");
    return false;
  }

  if (option.curve_checker.second.has_value())
    (std::any_cast<
        std::function<void(const std::vector<int> &, const std::vector<int> &,
                           const std::vector<RowMatd> &)>>(
        option.curve_checker.second))(VF[vid], VF[vid], local_cp);

  spdlog::trace("Curving SUCCEED, move to next.");
  return true;
}
}  // namespace prism::curve

void prism::curve::localcurve_pass(const PrismCage &pc,
                                   const prism::local::RemeshOptions &option) {
  std::vector<std::vector<int>> VF, VFi, groups;
  std::vector<bool> skip_flag(pc.mid.size(), false);

  for (int i = 0; i < pc.ref.aabb->num_freeze; i++) skip_flag[i] = true;
  {
    RowMati mF, mE;
    vec2eigen(pc.F, mF);
    igl::vertex_triangle_adjacency(pc.mid.size(), mF, VF, VFi);
  }
  auto succ = 0;
  for (auto vid = 0; vid < pc.mid.size(); vid++) {
    auto flag = smooth_prism(pc, vid, VF, VFi, option, skip_flag);
    if (flag) succ ++;
  }

  spdlog::info("Finished Curve Smoothing {}/{}", succ, pc.mid.size());
  return;
}
