#ifndef PRISM_ENERGY_SMOOTHER_PILLAR_HPP
#define PRISM_ENERGY_SMOOTHER_PILLAR_HPP
#include <prism/common.hpp>

namespace prism {
RowMatd one_ring_volumes(const std::vector<Vec3d>& base,
                         const std::vector<Vec3d>& mid,
                         const std::vector<Vec3d>& top,
                         const std::vector<Vec3i>& F,
                         const std::vector<int>& nb,
                         const std::vector<int>& nbi,
                         const std::array<Vec3d, 3>& modify = {
                             Vec3d(0, 0, 0), Vec3d(0, 0, 0), Vec3d(0, 0, 0)});

double get_min_step_to_singularity(const std::vector<Vec3d>& base,
                                   const std::vector<Vec3d>& mid,
                                   const std::vector<Vec3d>& top,
                                   const std::vector<Vec3i>& F,
                                   const std::vector<int>& nb,
                                   const std::vector<int>& nbi,
                                   std::array<bool, 3> /*base,mid,top*/ change,
                                   const Vec3d& direction, int num_freeze = 0);

// Pan: parallel move pillar for a better triangle quality, considering deprecate it.
std::optional<Vec3d> smoother_direction(
    const std::vector<Vec3d>& base, const std::vector<Vec3d>& mid,
    const std::vector<Vec3d>& top, const std::vector<Vec3i>& F, int num_freeze,
    const std::vector<std::vector<int>>& VF,
    const std::vector<std::vector<int>>& VFi, int vid);

// Legacy version of zoom and rotate, for prism full quality
std::optional<std::pair<Vec3d, Vec3d>> zoom_and_rotate(
    const std::vector<Vec3d>& base, const std::vector<Vec3d>& mid,
    const std::vector<Vec3d>& top, const std::vector<Vec3i>& F, int num_freeze,
    const std::vector<std::vector<int>>& VF,
    const std::vector<std::vector<int>>& VFi, int vid, double target_height);

std::optional<Vec3d> smoother_location_legacy(
    const std::vector<Vec3d>& base, const std::vector<Vec3d>& mid,
    const std::vector<Vec3d>& top, const std::vector<Vec3i>& F, int freeze,
    const std::vector<std::vector<int>>& VF,
    const std::vector<std::vector<int>> VFi, int vid, bool on_base);

std::optional<std::pair<Vec3d, Vec3d>> zoom(
    const std::vector<Vec3d>& base, const std::vector<Vec3d>& mid,
    const std::vector<Vec3d>& top, const std::vector<Vec3i>& F,
    const std::vector<std::vector<int>>& VF,
    const std::vector<std::vector<int>>& VFi, int vid,
    double target_thickness);


std::optional<std::pair<Vec3d, Vec3d>> rotate(
    const std::vector<Vec3d>& base, const std::vector<Vec3d>& mid,
    const std::vector<Vec3d>& top, const std::vector<Vec3i>& F,
    const std::vector<std::vector<int>>& VF,
    const std::vector<std::vector<int>>& VFi, int vid, double);
}  // namespace prism

#endif