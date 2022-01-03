#pragma once

#include <array>
#include "../common.hpp"

namespace prism::predicates {
bool tetrahedron_tetrahedron_overlap(const std::array<Vec3d, 4>&, const std::array<Vec3d, 4>&);
bool triangle_tetrahedron_overlap(const std::array<Vec3d, 3>& tri, const std::array<Vec3d, 4>& tet);

} // namespace prism::predicates