#ifndef PRISM_CGAL_TRIANGLE_TRIANGLE_INTERSECTION_HPP
#define PRISM_CGAL_TRIANGLE_TRIANGLE_INTERSECTION_HPP

#include <array>
#include <optional>
#include "../common.hpp"

namespace prism{struct Hit;}
namespace prism::cgal {
[[deprecated]] bool triangle_triangle_overlap(const std::array<Vec3d, 3>& tri0,
                                    const std::array<Vec3d, 3>& tri1);
[[deprecated]] bool segment_triangle_overlap(const std::array<Vec3d, 2>& seg,
                                    const std::array<Vec3d, 3>& tri1);

std::optional<Vec3d> segment_triangle_intersection(const std::array<Vec3d, 2>& seg,
                                    const std::array<Vec3d, 3>& tri1);
}  // namespace prism

#endif