#include "positive_prism_volume_12.hpp"

#include <geogram/numerics/predicates.h>
#include <spdlog/spdlog.h>

#include <Eigen/Dense>

bool prism::predicates::positive_prism_volume(
    const std::array<Vec3d, 6> &verts) {
  using GEO::PCK::orient_3d;
  for (int i = 0; i < 12; i++) {
    if (orient_3d(verts[TWELVE_TETRAS[i][0]].data(),
                  verts[TWELVE_TETRAS[i][1]].data(),
                  verts[TWELVE_TETRAS[i][2]].data(),
                  verts[TWELVE_TETRAS[i][3]].data()) <= 0)
      return false;
  }
  return true;
}

bool prism::predicates::positive_prism_volume(
    const std::array<Vec3d, 6> &verts, const std::array<bool, 3> &constrained,
    bool numerical) {
  using GEO::PCK::orient_3d;
  for (int i = 0; i < 3; i++) {
    if (constrained[i]) continue;
    for (int j = 0; j < 4; j++) {
      auto &tet = TWELVE_TETRAS[i * 4 + j];
      if (numerical) {  // also check numerical validity
        RowMat3d local_verts;
        for (int k = 1; k < 4; k++)
          local_verts.row(k - 1) = verts[tet[k]] - verts[tet[0]];
        if (local_verts.determinant() <= 0) {
          spdlog::trace("vol: failing numerical");
          return false;
        }
      }

      if (orient_3d(verts[tet[0]].data(), verts[tet[1]].data(),
                    verts[tet[2]].data(), verts[tet[3]].data()) <= 0) {
        spdlog::trace("vol: failing predicate");
        return false;
      }
    }
  }
  return true;
}
#include <CGAL/MP_Float.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/ostr.h>
bool prism::predicates::positive_nonlinear_prism(
    const std::array<Vec3d, 6> &verts, const std::array<bool, 3> &constrained) {
  using Scalar = CGAL::MP_Float;
  constexpr std::array<std::array<int, 4>, 3> reorder = {
      {{1, 3, 0, 2}, {5, 7, 4, 6}, {10, 11, 8, 9}}};  // a,c, b1,b2
  for (int i = 0; i < 3; i++) {
    if (constrained[i]) continue;
    std::array<Scalar, 4> vols;
    for (int j = 0; j < 4; j++) {
      auto ti = reorder[i][j];
      auto &tet = TWELVE_TETRAS[ti];
      Eigen::Matrix<Scalar, 3, 3> local_verts;
      for (int k = 1; k < 4; k++)
        for (int l = 0; l < 3; l++)
          local_verts(k - 1, l) =
              Scalar(verts[tet[k]][l]) - Scalar(verts[tet[0]][l]);
      vols[j] = local_verts.determinant();
    }
    // spdlog::trace("vols {}", vols);
    auto [a, c, b1, b2] = vols;
    auto b = b1 + b2;
    if (a <= 0 || c <= 0) return false;
    // spdlog::trace("delta {}", b * b - 4 * a * c);
    if (b <= 0 && b * b > 4 * a * c) return false;
  }
  return true;
}