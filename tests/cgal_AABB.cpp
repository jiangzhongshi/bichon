#include "prism/geogram/AABB.hpp"
#include <doctest.h>
#include <igl/Timer.h>
#include <igl/bounding_box.h>
#include <igl/read_triangle_mesh.h>
#include <iostream>
#include <list>
#include <prism/PrismCage.hpp>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>
TEST_CASE("segment test") {
  using namespace Eigen;
  auto file =
      "/home/zhongshi/Workspace/InterSurfaceMapping/python/data/fertility.off";

  RowMatd V;
  RowMati F;
  igl::read_triangle_mesh(file, V, F);
  put_in_unit_box(V);
  for (int i = 0; i < F.rows(); i++) {
    auto [s, mt, shift] =
        tetra_split_AorB({F(i, 0), F(i, 1), F(i, 2)});
    for (int j = 0; j < 3; j++)
      F(i, j) = mt[j];
  }
  constexpr int exp_count = 1000;
  SUBCASE("CGAL") {
    prism::geogram::AABB tree(V, F);
    RowMatd P(3, 3);
    int intersect_count = 0;
    RowMatd segment_hits(exp_count, 3);
    RowMatd all_P(exp_count, 9);
    int seg_count = 0;
    segment_hits.setZero();
    srand((0));
    for (int i = 0; i < exp_count; i++) {
      P.setRandom();
      all_P.row(i) = Eigen::Map<RowVectorXd>(P.data(), 9);
      if (tree.intersects_triangle({P.row(0), P.row(1), P.row(2)})) {
        intersect_count++;
      }
      auto seg = tree.segment_query(P.row(0), P.row(1));
      if (seg) {
        segment_hits.row(i) = seg.value();
        seg_count++;
        // spdlog::info("seg {}", seg.value());
      }
    }
    spdlog::info("Finish {}/{}/{}", seg_count, intersect_count, exp_count);
    CHECK(exp_count == 1000);
    CHECK((seg_count == 132 || seg_count == 112));
    CHECK((intersect_count == 284 || intersect_count == 290));
  }
}

TEST_CASE("touching case") {
  using namespace Eigen;
  MatrixXd V(4, 3);
  V << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1;
  MatrixXi F(4, 3);
  F << 0, 1, 3, 1, 2, 3, 0, 3, 2, 0, 2, 1;
  prism::geogram::AABB tree(V, F);

  // touch one edge
  Eigen::MatrixXd P(3, 3);
  P << 0, 0, 0, 1, 0, 0, 0, -1, 0;
  CHECK(tree.intersects_triangle({P.row(0), P.row(1), P.row(2)}));

  // lift a bit
  P(2, 2) = 0.1;
  CHECK(tree.intersects_triangle({P.row(0), P.row(1), P.row(2)}));

  // Moving whole edge away
  P(0, 1) = -1;
  P(1, 1) = -1;
  CHECK_FALSE(tree.intersects_triangle({P.row(0), P.row(1), P.row(2)}));

  Eigen::MatrixXd touch_vertex(3, 3);
  touch_vertex << 0, 0, 0, -1, 0, 0, 0, -1, 0;
  CHECK(tree.intersects_triangle(
      {touch_vertex.row(0), touch_vertex.row(1), touch_vertex.row(2)}));
}

TEST_CASE("aabb singularity") {
  RowMatd V;
  RowMati F;
  igl::read_triangle_mesh("../tests/data/saddle/original.obj", V, F);
  put_in_unit_box(V);
  PrismCage pc(V, F);
  V = pc.ref.V;
  F = pc.ref.F;

  RowMatd P(2, 3);
  spdlog::set_level(spdlog::level::info);
  int num_inter = 0;
  for (int i = 0; i < 100; i++) {
    P.setRandom();
    auto inter = (pc.ref.aabb->intersects_triangle(
        {V.row(0), P.row(0), P.row(1)}, true));
    if (inter)
      num_inter++;
  }
  CHECK(num_inter == 62);
}
