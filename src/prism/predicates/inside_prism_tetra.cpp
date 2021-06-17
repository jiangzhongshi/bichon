#include "inside_prism_tetra.hpp"

#include <geogram/numerics/predicates.h>

#include "prism/predicates/triangle_triangle_intersection.hpp"
namespace prism::predicates {

bool point_in_tetrahedron(const Vec3d& point, const Vec3d& T0, const Vec3d& T1,
                          const Vec3d& T2, const Vec3d& T3) {
  auto orient3D = [](const auto& a, const auto& b, const auto& c,
                     const auto& d) { return GEO::PCK::orient_3d(a, b, c, d); };
  return orient3D(T0.data(), T3.data(), T1.data(), point.data()) >= 0 &&
         orient3D(T1.data(), T3.data(), T2.data(), point.data()) >= 0 &&
         orient3D(T0.data(), T1.data(), T2.data(), point.data()) >= 0 &&
         orient3D(T0.data(), T2.data(), T3.data(), point.data()) >= 0;
}

bool point_in_prism(const Vec3d& point, bool tetra_split_AB,
                    const std::array<Vec3d, 6>& verts) {
  auto tets = tetra_split_AB ? TETRA_SPLIT_A : TETRA_SPLIT_B;
  auto sing = verts[0] == verts[3];
  for (int i = sing?1:0; i < 3; i++)
    if (point_in_tetrahedron(point, verts[tets[i][0]], verts[tets[i][1]],
                             verts[tets[i][2]], verts[tets[i][3]]))
      return true;
  return false;
}

bool triangle_intersects_prism(const std::array<Vec3d, 3>& tri_pts,
                               bool tetra_split_AB,
                               const std::array<Vec3d, 6>& verts) {
  auto bnd = tetra_split_AB ? PRISM_BOUNDARY_A : PRISM_BOUNDARY_B;

  for (auto& t : tri_pts)
    if (point_in_prism(t, tetra_split_AB, verts)) return true;
  auto sing = verts[0] == verts[3];
  if (sing) { // the following is very specific to the pre-defined PRISM_BOUNDARY, careful when reset.
    bnd[1] = {0, 4, 5};
    bnd[2][0] = -1;
    bnd[5][0] = -1;
  }
  for (auto& t : bnd) {
    if (t[0] == -1)continue;
    std::array<Vec3d, 3> boundary_triangle{verts[t[0]], verts[t[1]],
                                           verts[t[2]]};
    if (prism::predicates::triangle_triangle_overlap(boundary_triangle,
                                                     tri_pts))
      return true;
  }
  return false;
}

}  // namespace prism::predicates