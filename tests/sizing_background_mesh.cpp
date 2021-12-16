
#include "test_common.hpp"

#include <prism/geogram/AABB_tet.hpp>

#include <highfive/H5Easy.hpp>

/**
 * Tetrahedral mesh as size, and the related queries.
 * 
 */
TEST_CASE("size-mesh")
{
  H5Easy::File file("../tests/data/cube_tetra_10.h5", H5Easy::File::ReadOnly);
  auto bgV = H5Easy::load<RowMatd>(file, "V");
  auto bgT = H5Easy::load<RowMati>(file, "T");

  auto bgTree = prism::geogram::AABB_tet(bgV,bgT);
}