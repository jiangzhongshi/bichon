#include <doctest.h>
#include <igl/edge_flaps.h>
#include <igl/max_faces_stopping_condition.h>
#include <igl/per_vertex_point_to_plane_quadrics.h>
#include <igl/qslim.h>
#include <igl/qslim_optimal_collapse_edge_callbacks.h>
#include <igl/quadric_binary_plus_operator.h>
#include <igl/read_triangle_mesh.h>
#include <igl/slice.h>
#include <igl/slice_mask.h>
#include <igl/upsample.h>
#include <igl/write_triangle_mesh.h>

#include <Eigen/Core>
#include <array>
#include <nlohmann/json.hpp>
#include <prism/PrismCage.hpp>
#include <prism/geogram/AABB.hpp>
#include <vector>

#include "prism/common.hpp"

using Quadric = std::tuple<Eigen::MatrixXd, Eigen::RowVectorXd, double>;
std::function<bool(
    const Eigen::MatrixXd &,                                         /*V*/
    const Eigen::MatrixXi &,                                         /*F*/
    const Eigen::MatrixXi &,                                         /*E*/
    const Eigen::VectorXi &,                                         /*EMAP*/
    const Eigen::MatrixXi &,                                         /*EF*/
    const Eigen::MatrixXi &,                                         /*EI*/
    const std::set<std::pair<double, int>> &,                        /*Q*/
    const std::vector<std::set<std::pair<double, int>>::iterator> &, /*Qit*/
    const Eigen::MatrixXd &,                                         /*C*/
    const int                                                        /*e*/
    )>
qslim_in_shell_handle(const PrismCage &pc, const prism::geogram::AABB &tree_B,
                      const prism::geogram::AABB &tree_T,
                      std::vector<std::set<int>> &map_track, int &v1, int &v2,
                      std::vector<Quadric> &quadrics);

TEST_CASE("qslim section") {
  std::ifstream ifs("qslim_bin.json");
  nlohmann::json jf = nlohmann::json::parse(ifs);

  int max_m = jf["max_face"];
  std::string file_in = jf["filename"];
  std::string file_out = jf["savename"];

  // Prism Initialize.
  PrismCage pc(file_in);
  RowMatd mT, mB;
  RowMati mF;
  vec2eigen(pc.base, mB);
  vec2eigen(pc.top, mT);
  vec2eigen(pc.F, mF);
  Eigen::MatrixXd VO = pc.ref.V;
  Eigen::MatrixXi FO = pc.ref.F;
  prism::geogram::AABB tree_B(mB, mF), tree_T(mT, mF);
  tree_B.num_freeze = pc.ref.aabb->num_freeze;
  tree_T.num_freeze = pc.ref.aabb->num_freeze;
  std::vector<std::set<int>> track_to_prism;
  track_to_prism.resize(FO.rows());
  for (int p = 0; p < pc.track_ref.size(); p++) {
    for (auto t : pc.track_ref[p]) track_to_prism[t].insert(p);
  }

  //// QSLIM part
  const int orig_m = FO.rows();
  int m = orig_m;
  Eigen::MatrixXd U;
  Eigen::MatrixXi G;
  Eigen::VectorXi J;
  Eigen::VectorXi I;
  Eigen::VectorXi EMAP;
  Eigen::MatrixXi E, EF, EI;
  igl::edge_flaps(FO, E, EMAP, EF, EI);
  // Quadrics per vertex
  typedef std::tuple<Eigen::MatrixXd, Eigen::RowVectorXd, double> Quadric;
  std::vector<Quadric> quadrics;
  igl::per_vertex_point_to_plane_quadrics(VO, FO, EMAP, EF, EI, quadrics);
  int v1 = -1;
  int v2 = -1;

  std::function<void(const int e, const Eigen::MatrixXd &,
                     const Eigen::MatrixXi &, const Eigen::MatrixXi &,
                     const Eigen::VectorXi &, const Eigen::MatrixXi &,
                     const Eigen::MatrixXi &, double &, Eigen::RowVectorXd &)>
      cost_and_placement;
  std::function<bool(
      const Eigen::MatrixXd &,                                         /*V*/
      const Eigen::MatrixXi &,                                         /*F*/
      const Eigen::MatrixXi &,                                         /*E*/
      const Eigen::VectorXi &,                                         /*EMAP*/
      const Eigen::MatrixXi &,                                         /*EF*/
      const Eigen::MatrixXi &,                                         /*EI*/
      const std::set<std::pair<double, int>> &,                        /*Q*/
      const std::vector<std::set<std::pair<double, int>>::iterator> &, /*Qit*/
      const Eigen::MatrixXd &,                                         /*C*/
      const int                                                        /*e*/
      )>
      pre_collapse;
  std::function<void(
      const Eigen::MatrixXd &,                                         /*V*/
      const Eigen::MatrixXi &,                                         /*F*/
      const Eigen::MatrixXi &,                                         /*E*/
      const Eigen::VectorXi &,                                         /*EMAP*/
      const Eigen::MatrixXi &,                                         /*EF*/
      const Eigen::MatrixXi &,                                         /*EI*/
      const std::set<std::pair<double, int>> &,                        /*Q*/
      const std::vector<std::set<std::pair<double, int>>::iterator> &, /*Qit*/
      const Eigen::MatrixXd &,                                         /*C*/
      const int,                                                       /*e*/
      const int,                                                       /*e1*/
      const int,                                                       /*e2*/
      const int,                                                       /*f1*/
      const int,                                                       /*f2*/
      const bool /*collapsed*/
      )>
      post_collapse;
  igl::qslim_optimal_collapse_edge_callbacks(
      E, quadrics, v1, v2, cost_and_placement, pre_collapse, post_collapse);

  pre_collapse = qslim_in_shell_handle(pc, tree_B, tree_T, track_to_prism, v1,
                                       v2, quadrics);
  // Call to greedy decimator
  bool ret =
      igl::decimate(VO, FO, cost_and_placement,
                    igl::max_faces_stopping_condition(m, orig_m, max_m),
                    pre_collapse, post_collapse, E, EMAP, EF, EI, U, G, J, I);
  // Remove phony boundary faces and clean up
  const Eigen::Array<bool, Eigen::Dynamic, 1> keep = (J.array() < orig_m);
  igl::slice_mask(Eigen::MatrixXi(G), keep, 1, G);
  igl::slice_mask(Eigen::VectorXi(J), keep, 1, J);
  Eigen::VectorXi _1, I2;
  igl::remove_unreferenced(Eigen::MatrixXd(U), Eigen::MatrixXi(G), U, G, _1,
                           I2);
  igl::slice(Eigen::VectorXi(I), I2, 1, I);

  igl::write_triangle_mesh(file_out, U, G);
}
