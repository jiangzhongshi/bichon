#include <doctest.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/per_vertex_normals.h>
#include <igl/ray_mesh_intersect.h>
#include <igl/upsample.h>
#include <igl/vertex_triangle_adjacency.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <highfive/H5Easy.hpp>
#include <map>
#include <prism/common.hpp>
#include <prism/energy/map_distortion.hpp>
#include <prism/local_operations/section_remesh.hpp>
#include <prism/phong/projection.hpp>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>
#include <utility>


#include "cumin/bernstein_eval.hpp"
#include "cumin/curve_common.hpp"
#include "cumin/curve_utils.hpp"
#include "prism/PrismCage.hpp"
#include "prism/geogram/AABB.hpp"
#include "prism/local_operations/remesh_pass.hpp"

TEST_CASE("Bernstein Evaluators") {
  RowMati short_codecs(35, 4);
  short_codecs << 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 3, 1, 0, 0, 1,
      3, 0, 0, 3, 0, 1, 0, 0, 3, 1, 0, 1, 0, 3, 0, 0, 1, 3, 0, 3, 0, 0, 1, 0, 3,
      0, 1, 0, 0, 3, 1, 1, 0, 0, 3, 0, 1, 0, 3, 0, 0, 1, 3, 2, 2, 0, 0, 2, 0, 2,
      0, 0, 2, 2, 0, 2, 0, 0, 2, 0, 2, 0, 2, 0, 0, 2, 2, 2, 1, 1, 0, 1, 2, 1, 0,
      1, 1, 2, 0, 2, 1, 0, 1, 1, 2, 0, 1, 2, 0, 1, 1, 0, 2, 1, 1, 1, 0, 2, 1, 0,
      1, 2, 1, 1, 1, 0, 2, 1, 0, 1, 2, 0, 1, 1, 2, 1, 1, 1, 1;

  Eigen::ArrayXd X(35), Y(35), Z(35);
  for (int i = 0; i < 35; i++) {
    X[i] = i;
    Y[i] = i + 35;
    Z[i] = i + 70;
  }
  X = X.sin().array() + 1;
  Y = Y.sin().array() + 1;
  Z = Z.sin().array() + 1;
  auto r = prism::curve::evaluate_bernstein(X, Y, Z, short_codecs);
  CHECK(r.square().sum() == doctest::Approx(3404647.5544482986));

  auto r1 = prism::curve::evaluate_bernstein_derivative(X, Y, Z, short_codecs);
  CHECK(r1[0].square().sum() == doctest::Approx(6337101.645255285));
  CHECK(r1[1].square().sum() == doctest::Approx(4524720.611048738));
  CHECK(r1[2].square().sum() == doctest::Approx(6333786.654051635));
}

#include "cumin/inversion_check.hpp"
TEST_CASE("recursive-inversion") {
  H5Easy::File file("../python/curve/data/tetra_o9_l2b.h5");
  RowMatd bern_from_lagr_o9 = H5Easy::load<RowMatd>(file, "l2b");
  RowMati codecs_o4(35, 4), codecs_o9(220, 4);
  vec2eigen(codecs_gen(4, 3), codecs_o4);
  vec2eigen(codecs_gen(9, 3), codecs_o9);

  spdlog::set_level(spdlog::level::info);
  std::map<double, bool> tests = {{1e-1, true}, {1, false}};
  for (auto [case_num, answer] : tests) {
    RowMatd cp(35, 3);
    for (int i = 0; i < 35; i++) {
      for (int j = 0; j < 3; j++) {
        cp(i, j) = i * 3 + j;
      }
    }
    cp = codecs_o4.rightCols(3).cast<double>() -
         case_num * (cp.array().sin().matrix()).eval();

    // CHECK(prism::curve::tetrahedron_inversion_check(
    // cp, codecs_o4, codecs_o9, bern_from_lagr_o9) == answer);
  }
}
#include "cumin/curve_utils.hpp"

TEST_CASE("singularity in curve") {
  std::string filename = "../buildr/1582416.stl.h5";
  auto pc = PrismCage(filename);
  auto complete_cp = prism::curve::load_cp(filename);

  spdlog::set_level(spdlog::level::debug);
  for (auto i : {5}) {
    // for (auto i = 0; i<pc.F.size(); i++) {
    auto check = prism::curve::elevated_positive(pc.base, pc.top, {pc.F[i]},
                                                 true, {complete_cp[i]});
    if (!check) {
      spdlog::critical("i {}, F {}", i, pc.F[i]);
      exit(1);
    }
  }
}

TEST_CASE("tensor-load") {
  std::string data_path = "../python/curve/data";
  {
  H5Easy::File file(fmt::format("{}/p{}_quniform5_dxyz.h5", data_path, 3 + 1),
                    H5Easy::File::ReadOnly);
    std::vector<RowMatd> vec_dxyz;
    auto tet4_dxyz = H5Easy::load<std::vector<std::vector<std::vector<double>>>>(file, "dxyz");
    vec_dxyz.resize(tet4_dxyz[0].size());
     for (auto i1 = 0; i1 < tet4_dxyz[0].size(); i1++) {
      vec_dxyz[i1].resize(tet4_dxyz.size(), tet4_dxyz[0][0].size());
      for (auto i0 = 0; i0 < tet4_dxyz.size(); i0++)
        for (auto i2 = 0; i2 < tet4_dxyz[0][0].size(); i2++)
          vec_dxyz[i1](i0, i2) = tet4_dxyz[i0][i1][i2];
    }
    spdlog::critical("vecdxyz 0, 20 {}", vec_dxyz[10].row(2));
    spdlog::critical("vecdxyz {} x{}x{}", vec_dxyz.size(), vec_dxyz[0].rows(), vec_dxyz[0].cols());
  }
}
