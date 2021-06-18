#include <doctest.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

#include <cumin/curve_common.hpp>
#include <tuple>

#include "cumin/bernstein_eval.hpp"
#include "cumin/curve_common.hpp"
#include "cumin/curve_utils.hpp"

using RowMatX3d = Eigen::Matrix<double, -1, 3, Eigen::RowMajor>;
std::tuple<double, RowMatX3d> mips_energy(const RowMatX3d& nodes,
                                          const std::vector<RowMatd>& dxyz,
                                          bool with_grad = false);

TEST_CASE("cute-mips") {
  auto& helper = prism::curve::magic_matrices(3,3);
  auto &vec_dxyz = helper.volume_data.vec_dxyz;
  auto &elevlag_from_bern = helper.elev_lag_from_bern;
  auto tri4_cod = codecs_gen_id(helper.tri_order + 1, 2);
  auto tet4_cod = codecs_gen_id(helper.tri_order + 1, 3);

  RowMati codecs_o4(35, 4), codecs_o9(220, 4);
  vec2eigen(codecs_gen(4, 3), codecs_o4);
  vec2eigen(codecs_gen(9, 3), codecs_o9);
  RowMatX3d nodes35 = codecs_o4.rightCols(3).cast<double>() / 3;
  {
    auto [v, _] = mips_energy(nodes35, vec_dxyz, false);
    CHECK_EQ(v, doctest::Approx(504));
  }
  nodes35(0, 0) += 0.1;
  auto [val, grad] = mips_energy(nodes35, vec_dxyz, true);
  auto [val1, _] = mips_energy(nodes35 - 1e-6 * grad, vec_dxyz, false);
  CHECK_LT(val1, val);
}

auto get_l2b = [](std::string filename) {
  H5Easy::File file1("../python/curve/data/" + filename);
  return H5Easy::load<RowMatd>(file1, "l2b");
};

#include "cumin/inversion_check.hpp"
TEST_CASE("recursive-check") {
    auto helper = prism::curve::magic_matrices(3, 3);
  auto file = H5Easy::File("../buildr/after.h5", H5Easy::File::ReadOnly);
  auto lagr = H5Easy::load<RowMatd>(file, "lagr");
  auto p4T = H5Easy::load<RowMati>(file, "cells");
  for (auto i : {1902}) {
    auto t = i;
    RowMatX3d nodes35(35, 3);
    for (auto j = 0; j < 35; j++) {
      nodes35.row(j) = lagr.row(p4T(t, j));
    }
    spdlog::enable_backtrace(100);
    // spdlog::set_level(spdlog::level::trace);
    if (!prism::curve::tetrahedron_inversion_check(
                nodes35, helper.volume_data.vol_codec, helper.volume_data.vol_jac_codec, helper.volume_data.vol_bern_from_lagr,
                helper.volume_data.vol_jac_bern_from_lagr))  {
      spdlog::warn("negative");
    }
  }
}

auto save_cute = [](std::string name, auto& V, auto& T) {
  RowMatd mV;
  RowMati mT;
  vec2eigen(V, mV);
  vec2eigen(T, mT);
  auto file = H5Easy::File(name, H5Easy::File::Overwrite);
  H5Easy::dump(file, "V", mV);
  H5Easy::dump(file, "T", mT);
};

#include <cumin/high_order_optimization.hpp>
namespace prism::curve{

}
TEST_CASE("cute-collapse") { 
  spdlog::set_pattern("[%l] %v");
  std::string in_file = "../buildr/before_opt.h5";
  auto file = H5Easy::File(in_file, H5Easy::File::ReadOnly);
  auto lagr = H5Easy::load<RowMatd>(file, "lagr");  //<double>*3
  auto p4T = H5Easy::load<RowMati>(file, "cells");  //<int>*35

  auto helper = prism::curve::magic_matrices(3, 3);
  REQUIRE_EQ(helper.volume_data.vol_codec.rows(), p4T.cols());
  spdlog::info("Codec {}x{}", helper.volume_data.vol_codec.rows(),
   helper.volume_data.vol_codec.cols());
   spdlog::info("codec row {}", helper.volume_data.vol_codec.row(0));
  // prism::curve::edge_collapsing( lagr, p4T, 1e2);
  // prism::curve::edge_swapping( lagr, p4T, 1e2);
    for (int pass = 1; pass <= 1; pass++) {
    spdlog::info("======== Optimization Pass {}/{} ========", pass, 6);
    auto col = prism::curve::edge_collapsing(lagr, p4T,  100);
    CHECK_EQ(col, 281);

    spdlog::set_level(spdlog::level::debug);
    auto swa = prism::curve::edge_swapping(lagr, p4T, 100);
    CHECK_EQ(swa, 46);

    prism::curve::vertex_star_smooth(lagr, p4T,  0, 1);
  }
  // auto ofile = H5Easy::File(in_file + "_out.h5", H5Easy::File::Overwrite);
  // H5Easy::dump(ofile, "lagr", lagr);
  // H5Easy::dump(ofile, "cells", p4T);
}

TEST_CASE("cute-collapse-edit") { 
  spdlog::set_pattern("[%l] %v");
  std::string in_file = "../buildr/before_opt.h5";
  auto file = H5Easy::File(in_file, H5Easy::File::ReadOnly);
  auto lagr = H5Easy::load<RowMatd>(file, "lagr");  //<double>*3
  auto p4T = H5Easy::load<RowMati>(file, "cells");  //<int>*35

  auto helper = prism::curve::magic_matrices(2, 3);
  REQUIRE_EQ(helper.volume_data.vol_codec.rows(), p4T.cols());
  spdlog::info("Codec {}x{}", helper.volume_data.vol_codec.rows(),
   helper.volume_data.vol_codec.cols());
  // prism::curve::edge_collapsing( lagr, p4T, 1e2);
  // prism::curve::edge_swapping( lagr, p4T, 1e2);
    for (int pass = 1; pass <= 6; pass++) {
    spdlog::info("======== Optimization Pass {}/{} ========", pass, 6);
    auto col = prism::curve::cutet_collapse(lagr, p4T, 100);
    // CHECK_EQ(col, 281);

    spdlog::set_level(spdlog::level::info);
    auto swa = prism::curve::cutet_swap(lagr, p4T, 100);
    // CHECK_EQ(swa, 46);

    prism::curve::vertex_star_smooth(lagr, p4T, 3, 1);
  }
}