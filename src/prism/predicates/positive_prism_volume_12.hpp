#ifndef PRISM_POSITIVE_PRISM_VOLUME_12_HPP
#define PRISM_POSITIVE_PRISM_VOLUME_12_HPP

#include "../common.hpp"
namespace prism::predicates {
bool positive_prism_volume(const std::array<Vec3d, 6>& verts,
                           const std::array<bool, 3>& constrained,
                           bool numerical = false);
bool positive_prism_volume(const std::array<Vec3d, 6>& verts);

// this is not yet an exact predicate.
bool positive_nonlinear_prism(const std::array<Vec3d, 6>& verts,
                              const std::array<bool, 3>& constrained);
}  // namespace prism::predicates
#endif