#ifndef PRISM_CGAL_TRIANGLE_TETRAHEDRON_INTERSECTION_HPP
#define PRISM_CGAL_TRIANGLE_TETRAHEDRON_INTERSECTION_HPP

#include <array>
#include "../common.hpp"

namespace prism::cgal {
[[deprecated]]
bool triangle_tetrahedron_intersection(const std::array<Vec3d, 3>&,
                                    const std::array<Vec3d, 4>&,
                                    bool wireless=false);
bool triangle_tetrahedron_intersection_wireless(
    const std::array<Vec3d, 3>& triangle,
    const std::array<Vec3d, 4>& tetrahedron);
}  // namespace prism

#endif