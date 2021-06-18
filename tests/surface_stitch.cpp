#include <doctest.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <cumin/curve_common.hpp>
#include <cumin/curve_utils.hpp>
#include <cumin/stitch_surface_to_volume.hpp>
#include <highfive/H5Easy.hpp>
#include <map>
#include <prism/common.hpp>

TEST_CASE("stitch-surface") {
  RowMatd V;
  RowMati T(1, 4);
  T << 0, 1, 2, 3;
  V = RowMatd::Identity(4, 4);
  V = V.rightCols(3).eval();
  spdlog::info(V);

  std::vector<RowMatd> cp(1);  // surface cp
  RowMatd mB = V.topRows(3);
  RowMatd mT = mB.array() + 1;
  RowMati mF(1, 3);
  mF << 0, 1, 2;
  int order = 3;
  cp[0].setRandom((order + 1) * (order + 2) / 2, 3);  // cubic

  auto helper = prism::curve::magic_matrices(3,3);
  const auto elevlag_from_bern =helper.elev_lag_from_bern; 
  const auto vec_dxyz =helper.volume_data.vec_dxyz;

  RowMatd nodes;
  RowMati p4T;
  prism::curve::stitch_surface_to_volume(mB, mT, mF, cp, V, T, nodes, p4T);
}