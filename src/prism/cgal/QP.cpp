#include "QP.hpp"

#include <CGAL/QP_functions.h>
#include <CGAL/QP_models.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>
// choose exact integral type
#ifdef CGAL_USE_GMP
#include <CGAL/Gmpzf.h>
typedef CGAL::Gmpzf ET;
#else
#include <CGAL/MP_Float.h>
typedef CGAL::MP_Float ET;
#endif
// program and solution types

namespace prism::cgal {
typedef CGAL::Quadratic_program<double> Program;
typedef CGAL::Quadratic_program_solution<ET> Solution;
Vec3d qp_normal(const RowMatd &FN, const std::vector<int> &nb) {
  Program qp(CGAL::LARGER, false, 0, false, 0);

  for (int i = 0; i < nb.size(); i++) {
    for (int j = 0; j < 3; j++)
      qp.set_a(j, i, FN(nb[i], j));
    qp.set_b(i, 1); //  n*x  >= 1
    qp.set_d(i, i, 1);
  }
  // solve the program, using ET as the exact type
  Solution s = CGAL::solve_quadratic_program(qp, ET());
  if (!s.solves_quadratic_program(qp)) {
    spdlog::error("CGAL QP failing.");
    return Vec3d(0, 0, 0);
  }

  if (s.is_infeasible()) {
    spdlog::trace("Infeasible");
    return Vec3d(0, 0, 0);
  }
  // collect and normalize
  auto it = s.variable_values_begin();
  auto s0 = (*it);
  auto s1 = *next(it);
  auto s2 = *next(next(it));
  // normalize
  auto r_sq = s0 * s0 + s1 * s1 + s2 * s2;
  auto sqrt_sign = [&r_sq](auto &s) {
    spdlog::trace("s{}", s);
    if (s * s < 1e-20)
      return 0.;
    double sign = s > 0 ? 1 : -1;
    auto squared = CGAL::to_double((s * s) / r_sq);
    return sqrt(squared) * sign;
  };
  Vec3d normal;
  normal[0] = sqrt_sign(s0);
  normal[1] = sqrt_sign(s1);
  normal[2] = sqrt_sign(s2);
  spdlog::trace("CGAL normal {}", normal);
  for (auto f : nb) {
    auto dot = normal.dot(FN.row(f));
    if (dot < 0) {
      spdlog::warn("Wrong Dot", dot);
      return Vec3d(0, 0, 0);
    }
  }
  return normal.stableNormalized();
}

} // namespace prism::cgal
#include <igl/copyleft/quadprog.h>
Vec3d prism::qp_normal_igl(const RowMatd &FN, const std::vector<int> &nb) {
  Eigen::MatrixXd G = Eigen::MatrixXd::Identity(3, 3);
  Eigen::VectorXd g0 = Eigen::VectorXd::Zero(3);
  Eigen::MatrixXd CE, CI(3, nb.size());
  Eigen::VectorXd ce0, ci0 = -Eigen::VectorXd::Ones(nb.size());
  for (int i = 0; i < nb.size(); i++) {
    for (int j = 0; j < 3; j++)
      CI(j, i)=FN(nb[i], j);
  }
  Eigen::VectorXd x;
  // CI'x >= - ci0
  bool flag = igl::copyleft::quadprog(G,g0,CE,ce0,CI,ci0, x);
  if (!flag) return Vec3d(0,0,0);
  return x.stableNormalized();
}