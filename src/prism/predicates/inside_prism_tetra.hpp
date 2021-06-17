// Inside tests for triangulated prism

#ifndef PRISM_PREDICATES_INSIDE_PRISM_TETRA_HPP
#define PRISM_PREDICATES_INSIDE_PRISM_TETRA_HPP

#include "../common.hpp"
namespace prism::predicates {
bool point_in_tetrahedron(const Vec3d& point, const Vec3d& T0, const Vec3d& T1,
                          const Vec3d& T2, const Vec3d& T3);

bool point_in_prism(const Vec3d& point, bool tetra_split_AB,
                    const std::array<Vec3d, 6>& verts);


// [[deprecated("Not used now, tri-tet intersect is more informative")]]
bool triangle_intersects_prism(const std::array<Vec3d, 3>& tri_v, bool tetra_split_AB,
                    const std::array<Vec3d, 6>& pri_v);
                
}  // namespace prism::predicates

#endif