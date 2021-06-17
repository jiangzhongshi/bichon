#include "tangent_orientation.hpp"

#include "../predicates/inside_prism_tetra.hpp"

namespace prism::phong {
bool tangent_orientation(const std::array<Vec3d, 6>& stacked_V,
                         const std::array<Vec3d, 3>& triangle) {
  double EPS = 1e-8;
  Vec3d tri_normal = (triangle[1] - triangle[0])
                         .cross(triangle[2] - triangle[0])
                         .stableNormalized();
  for (int i = 0; i < 3; i++) {
    Vec3d fiber = stacked_V[i + 3] - stacked_V[i];
    if (fiber.dot(tri_normal) < EPS) return false;
  }
  return true;
}

}  // namespace prism::phong
