#ifndef PRISM_SPATIAL_HASH_SELF_INTERSECTION_HPP
#define PRISM_SPATIAL_HASH_SELF_INTERSECTION_HPP

#include "../common.hpp"
namespace prism::spatial_hash {
std::vector<std::pair<int, int>> self_intersections(
    const std::vector<Vec3d> &V, const std::vector<Vec3i> &F);

// raw routine for single-layer tetrashell candidates test. To be split into
// multiple stages for integration.
std::set<std::pair<int, int>> tetrashell_self_intersections(
    const std::vector<Vec3d> &base, const std::vector<Vec3d> &top,
    const std::vector<Vec3i> &F);

std::function<void(const std::pair<int, int> &)> find_offending_pairs(
    const std::vector<Vec3i> &F, const std::vector<Vec3d> &tetV,
    const std::vector<Vec4i> &tetT,
    std::set<std::pair<int, int>> &offend_pairs);
}  // namespace prism::spatial_hash

#endif