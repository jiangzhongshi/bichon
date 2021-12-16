
#include "test_common.hpp"

#include <prism/geogram/AABB_tet.hpp>

#include <highfive/H5Easy.hpp>

/**
 * Tetrahedral mesh as size, and the related queries.
 * 
 */
TEST_CASE("AABB-tet-self-query")
{
  H5Easy::File file("../tests/data/cube_tetra_10.h5", H5Easy::File::ReadOnly);
  auto bgV = H5Easy::load<RowMatd>(file, "V");
  auto bgT = H5Easy::load<RowMati>(file, "T");

  auto bgTree = prism::geogram::AABB_tet(bgV,bgT);

  for (auto i=0; i<bgT.rows(); i++) {
    auto P = std::array<Vec3d, 4>();
    for (auto j=0; j<4; j++) {
      P[j] = bgV.row(bgT(i,j));
    }
    auto result = bgTree.overlap_tetra(P);
    // the current tet should be found, ensures that the order is not screwed up.
    REQUIRE_NE(std::find(result.begin(), result.end(), i), result.end());
  }
}
