#pragma once

#include "prism/common.hpp"
#include "tetra_utils.hpp"
namespace prism::tet {
void insert_all_points(
    prism::tet::tetmesh_t& tetmesh,
    const std::vector<int>& hint_tid,
    const std::vector<Vec3d>& points);
void insert_triangles(prism::tet::tetmesh_t& tetmesh, const std::vector<std::array<size_t,3>>& tris);
} // namespace prism::tet