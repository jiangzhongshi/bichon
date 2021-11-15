#ifndef PRISM_LOCAL_OPERATIONS_VALIDITY_CHECKS_HPP
#define PRISM_LOCAL_OPERATIONS_VALIDITY_CHECKS_HPP

#include <any>

#include "../common.hpp"
#include "../geogram/AABB.hpp"
#include "local_mesh_edit.hpp"
#include "prism/energy/prism_quality.hpp"
namespace prism {
struct HashGrid;
}
struct PrismCage;
namespace prism::local {
struct RemeshOptions;
}

namespace prism::local_validity {
enum PolyOpError {
  kSuccess = 0,
  link,
  quality,
  kVolume,
  kIntersect,
  twofeature,
  sub_volume,
  sub_intersect,
  distort,
  curve,
  feature,
  others,
  kMax
};
// old quality is used within smooth, to record before entering the attempter.
PolyOpError attempt_zig_remesh(const PrismCage &pc,
                               const std::vector<std::set<int>> &map_track,
                               const prism::local::RemeshOptions &option,
                               // specified infos below
                               double old_quality,
                               const std::vector<int> &old_fid,
                               const std::vector<Vec3i> &moved_tris,
                               std::vector<std::set<int>> &sub_trackee,
                               std::vector<RowMatd> &local_cp);

PolyOpError attempt_feature_remesh(const PrismCage &pc,
                           const std::vector<std::set<int>> &map_track,
                           const prism::local::RemeshOptions &option,
                           // specified infos below
                           double old_quality, const std::vector<int> &old_fid,
                           const std::vector<Vec3i> &moved_tris,
                           std::vector<std::set<int>> &sub_trackee,
                           std::vector<RowMatd> &local_cp);

double max_quality_on_tris(const std::vector<Vec3d> &base,
                           const std::vector<Vec3d> &mid,
                           const std::vector<Vec3d> &top,
                           const std::vector<Vec3i> &moved_tris);

constexpr auto triangle_shifts = [](auto &moved_tris) {
  std::vector<int> new_shifts(moved_tris.size());
  for (int i = 0; i < moved_tris.size(); i++) {
    auto [s, mt, shift] = tetra_split_AorB(moved_tris[i]);
    moved_tris[i] = mt;
    new_shifts[i] = shift;
  }
  return std::move(new_shifts);
};

// this augments igl::volume numerical check with predicate check.
bool prism_positivity_with_numerical(const std::array<Vec3d, 6> &verts,
                                     const std::array<bool, 3> &constrained = {
                                         false, false, false});

bool dynamic_intersect_check(
    const std::vector<Vec3d> &base, const std::vector<Vec3i> &F,
    const std::vector<int>
        &vec_removed, // proposed removal face_id to be ignored in the test.
    const std::vector<Vec3i> &tris, // proposed addition triangles
    const prism::HashGrid &grid);

int attempt_local_edit(
    const PrismCage &pc, const std::vector<std::set<int>> &map_track,
    const prism::local::RemeshOptions &option, const std::vector<int> &old_fid,
    const std::vector<Vec3i> &old_tris, const std::vector<Vec3i> &moved_tris,
    std::tuple<std::vector<int> /*newfid*/, std::vector<int> /*shifts*/,
               std::vector<std::set<int>> /*track*/> &checker);

int attempt_relocate(PrismCage &pc, const std::vector<std::set<int>> &map_track,
                     const prism::local::RemeshOptions &,
                     // specified infos below
                     const std::vector<int> &nb, int vid,
                     const std::array<Vec3d, 3> /*b-m-t*/ &relocations,
                     bool feature_enable, std::vector<std::set<int>> &trackee);

int attempt_flip(const PrismCage &pc,
                 const std::vector<std::set<int>> &map_track,
                 const prism::local::RemeshOptions &,
                 // specified infos below
                 int f0, int f1, int e0, int e1, int v0, int v1,
                 std::tuple<std::vector<int>,          /*shift*/
                            std::vector<std::set<int>> /*track*/
                            > &);

int attempt_split(
    const PrismCage &pc, const std::vector<std::set<int>> &map_track,
    const prism::local::RemeshOptions &,
    // specified infos below
    int f0, int f1, int e0, int e1,
    std::tuple<std::vector<int> /*fid*/, std::vector<int>, /*shift*/
               std::vector<std::set<int>>                  /*track*/
               > &checker);

int attempt_collapse(
    const PrismCage &pc, const std::vector<std::set<int>> &map_track,
    const prism::local::RemeshOptions &,
    // specified infos below
    const std::vector<std::pair<int, int>> &neighbor0,
    const std::vector<std::pair<int, int>> &neighbor1, int f0, int f1, int u0,
    int u1, bool feature_enable,
    std::tuple<std::vector<int> /*newfid*/, std::vector<int> /*shifts*/,
               std::vector<std::set<int>> /*track*/> &);

// (1) if any new volume is negative
// requires to find vector<new Tri>, and their (modified) base-top V

bool volume_check(const std::vector<Vec3d> &base, const std::vector<Vec3d> &mid,
                  const std::vector<Vec3d> &top, const std::vector<Vec3i> &tris,
                  int num_cons = 0);
bool volume_check(const std::vector<Vec3d> &base, const std::vector<Vec3d> &top,
                  const std::vector<Vec3i> &tris, int num_cons = 0);
// (2) if new prism intersect with ref-sheet
// same as (1)

bool intersect_check(const std::vector<Vec3d> &base,
                     const std::vector<Vec3d> &top,
                     const std::vector<Vec3i> &tris,
                     const prism::geogram::AABB &tree);

// (3) if distortion exceeds the bound, or flip
// from vector<new Tri>, need to call placement(),
// and update position, compute distortion for each Tri

std::optional<std::vector<std::set<int>>>
distort_check(const std::vector<Vec3d> &base,
              const std::vector<Vec3d> &mid, // placed new verts
              const std::vector<Vec3d> &top, const std::vector<Vec3i> &tris,
              const std::set<int> &combined_trackee, // indices to ref.F tracked
              const RowMatd &refV, const RowMati &refF, double distortion_bound,
              int num_freeze, bool bundled_intersection = false);

bool feature_handled_distort_check(const PrismCage &pc,
                                   const prism::local::RemeshOptions &option,
                                   const std::vector<Vec3i> &moved_tris,
                                   const std::vector<int> &old_fid,
                                   std::vector<std::set<int>> &sub_trackee);
void post_operation(PrismCage &pc, const prism::local::RemeshOptions &option,
                    const std::vector<int> &old_fids,
                    const std::vector<int> &new_fids,
                    const std::vector<std::set<int>> &new_tracks,
                    std::vector<RowMatd> &local_cp);
} // namespace prism::local_validity

#endif