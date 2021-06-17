#ifndef PRISM_CGAL_TETRAHEDRON_TETRAHEDRON_INTERSECTION_HPP
#define PRISM_CGAL_TETRAHEDRON_TETRAHEDRON_INTERSECTION_HPP

#include <array>
#include "../common.hpp"

namespace prism::cgal {
[[deprecated]]
bool tetrahedron_tetrahedron_intersection(const std::array<Vec3d, 4>&,
                                    const std::array<Vec3d, 4>&);
}  // namespace prism

#endif