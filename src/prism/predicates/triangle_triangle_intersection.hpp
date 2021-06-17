#ifndef PRISM_PREDICATES_TRIANGLE_TRIANGLE_INTERSECTION_HPP
#define PRISM_PREDICATES_TRIANGLE_TRIANGLE_INTERSECTION_HPP

#include <array>
#include "../common.hpp"

namespace prism::predicates {
bool triangle_triangle_overlap(const std::array<Vec3d, 3>& tri0,
                                    const std::array<Vec3d, 3>& tri1);
bool segment_triangle_overlap(const std::array<Vec3d, 2>& seg,
                                    const std::array<Vec3d, 3>& tri1);
}  // namespace prism

#endif