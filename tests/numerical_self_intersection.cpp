#include <prism/geogram/AABB.hpp>
#include <igl/read_triangle_mesh.h>
#include <igl/remove_duplicate_vertices.h>
#include <spdlog/spdlog.h>
#include <prism/common.hpp>
#include <doctest.h>

TEST_CASE("numerical self intersection") {
  std::vector<std::string> inters = { "104563.stl"};
  spdlog::set_level(spdlog::level::trace);
  for (auto filename : inters) {
    RowMatd V;
    RowMati F;
    Eigen::VectorXi SVI, SVJ;
    igl::read_triangle_mesh(
        "/home/zhongshi/data/Thingi10K/raw_meshes/" + filename, V, F);
    RowMatd temp_V = V;  // for STL file
    igl::remove_duplicate_vertices(temp_V, 0, V, SVI, SVJ);
    for (int i = 0; i < F.rows(); i++)
      for (int j : {0, 1, 2}) F(i, j) = SVJ[F(i, j)];
    put_in_unit_box(V);
    prism::geogram::AABB tree(V, F);
    double tol = 1e-16;

    while (!tree.numerical_self_intersection(tol)) tol *= 10;
    spdlog::info("{}: self {}", filename, tol);
    REQUIRE_GT(tol, 1e-5);
  }
}