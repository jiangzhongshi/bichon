// This file is an implementation for Phong projection with bilinear and so on. 
// Deprecated on Oct 22, 2019 for PL prism

// for the following routines,
// coefficients are arranged as high degree first (not so conventional).

#include "projection.hpp"
#include <igl/flip_avoiding_line_search.h>
#include <math.h>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <iostream>
#include <list>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include "../predicates/inside_octahedron.hpp"

// The coefficients to the quadratic that sovles alpha, with b=beta substituted.
void alpha_linear_coef(const Eigen::MatrixXd& p, const Eigen::MatrixXd& n,
                       double b, double* result0, double* result1);

// The coefficients to the cubic that sovles beta.
//
void beta_cubic_coef(double p11 /*==0*/, double p12, double p13 /*==0*/,
                     double p21, double p22, double p23 /*==0*/, double p31,
                     double p32, double p33, double n11, double n12, double n13,
                     double n21, double n22, double n23, double n31, double n32,
                     double n33, double* result0, double* result1,
                     double* result2, double* result3);

namespace root_find {
std::vector<double> solvePoly(const std::array<double, 4>&, double lower = 0,
                              double upper = 1);
std::vector<double> solvePoly(const std::array<double, 3>&, double lower = 0,
                              double upper = 1);
std::vector<double> solvePoly(const std::array<double, 2>&, double lower = 0,
                              double upper = 1);
}  // namespace root_find

void alignment(const Eigen::Matrix3d& V, const Eigen::Matrix3d& N,
               const Eigen::RowVector3d& point, Eigen::Matrix3d& matP,
               Eigen::Matrix3d& matN) {
  using namespace Eigen;
  Eigen::Vector3d e0 = V.row(0) - V.row(2), e1 = V.row(1) - V.row(2);
  auto n = e0.cross(e1).normalized();
  Eigen::Vector3d axis(0, 0, 1);
  if (abs(n[1]) > 1e-10 || abs(n[0]) > 1e-10)
    axis = Eigen::Vector3d(n[1], -n[0], 0).normalized();
  auto rot3d = AngleAxisd(std::acos(n[2]), axis).toRotationMatrix();
  Eigen::Vector3d r0 = rot3d * e0, r1 = rot3d * e1;
  Eigen::Matrix3d rot2d;
  Eigen::Vector2d n_r0 = Eigen::Vector2d(r0[0], r0[1]).normalized();
  rot2d << n_r0[1], -n_r0[0], 0, n_r0[0], n_r0[1], 0, 0, 0, 1;
  Eigen::Matrix3d rot = rot2d * rot3d;

  matP << V.row(0) - V.row(2), V.row(1) - V.row(2), V.row(2) - point;
  matN << N.row(0) - N.row(2), N.row(1) - N.row(2), N.row(2);
  matP = matP * rot.transpose();
  matN = matN * rot.transpose();
  assert(abs(matP(0, 0)) < 1e-10 && abs(matP(0, 2)) < 1e-10 &&
         abs(matP(1, 2)) <
             1e-10);  // geometric transform to eliminate some cases.
  matP(0, 0) = 0;
  matP(0, 2) = 0;
  matP(1, 2) = 0;
}

bool prism::nonlinear::phong_projection(const Eigen::Matrix3d& V, const Eigen::Matrix3d& N,
                             const Eigen::RowVector3d& point,
                             std::array<double, 3>& tuple) {
  using namespace Eigen;
  Matrix3d matP, matN;
  std::array<double, 4> cubic_coef;
  std::array<double, 2> linear_coef;
  alignment(V, N, point, matP, matN);
  beta_cubic_coef(matP(0, 0), matP(0, 1), matP(0, 2), matP(1, 0), matP(1, 1),
                  matP(1, 2), matP(2, 0), matP(2, 1), matP(2, 2), matN(0, 0),
                  matN(0, 1), matN(0, 2), matN(1, 0), matN(1, 1), matN(1, 2),
                  matN(2, 0), matN(2, 1), matN(2, 2), &cubic_coef[0],
                  &cubic_coef[1], &cubic_coef[2], &cubic_coef[3]);
  auto beta_sols = root_find::solvePoly(cubic_coef, 0., 1.);
    spdlog::info("beta sols {}", beta_sols);
  for (auto b : beta_sols) {
    alpha_linear_coef(matP, matN, b, &linear_coef[0], &linear_coef[1]);
    auto alpha_sols = root_find::solvePoly(linear_coef, 0., 1 - b);
    spdlog::info("alpha sols {}", alpha_sols);
    // validation and get t
    for (auto a : alpha_sols) {
      auto bc = Eigen::RowVector3d(a, b, 1);
      Eigen::Vector3d Nx = bc * matN,
                      Px_q =
                          bc *
                          matP;  // Px_q (Px-q) is the vector from query to Px.
      double cross = (Px_q).cross(Nx).norm();
      if (cross < 1e-7) {
        tuple[0] = a;
        tuple[1] = b;
        tuple[2] =
            -Px_q.dot(Nx) / Nx.array().pow(2).sum();  // t Nx.Nx + (Px -q).Nx= 0
        return true;
      }
    }
  }
  return false;
}

void alpha_linear_coef(const Eigen::MatrixXd& p, const Eigen::MatrixXd& n,
                       double b, double* result0, double* result1) {
  *result1 = -((b * n(1, 2) + n(2, 2)) * (b * p(1, 0) + p(2, 0))) +
             (b * n(1, 0) + n(2, 0)) * p(2, 2);
  *result0 = -b * (n(0, 2) * p(1, 0)) - n(0, 2) * p(2, 0) + n(0, 0) * p(2, 2);
}

void beta_cubic_coef(double p11 /*==0*/, double p12, double p13 /*==0*/,
                     double p21, double p22, double p23 /*==0*/, double p31,
                     double p32, double p33, double n11, double n12, double n13,
                     double n21, double n22, double n23, double n31, double n32,
                     double n33, double* result0, double* result1,
                     double* result2, double* result3) {
  double x0 = pow(n13, 2);
  double x1 = n21 * p22;
  double x2 = n22 * p21;
  double x3 = n12 * p21;
  double x4 = n11 * p22;
  double x5 = n21 * p12;
  double x6 = x4 + x5;
  double x7 = -x3 + x6;
  double x8 = n13 * p12 * p33;
  double x9 = n31 * p22;
  double x10 = 2 * x2;
  double x11 = n23 * p31;
  double x12 = n33 * p21;
  double x13 = x3 - x4 + x5;
  double x14 = n11 * p32;
  double x15 = n31 * p12;
  double x16 = -2 * x3 + x6;
  double x17 = n22 * p31;
  double x18 = 2 * n32 * p21 - x9;
  double x19 = pow(p33, 2);
  double x20 = n12 * p31;
  double x21 = -x14 + x15 + x20;
  double x22 = x11 + x12;
  double x23 = x14 + x20;
  double x24 = n11 * n32;
  *result0 =
      p21 * x8 * (n11 * pow(n23, 2) * p12 - n13 * n23 * x7 + x0 * (x1 - x2));
  *result1 =
      x8 * (n11 * n23 * (p12 * (x11 + 2 * x12) - p33 * x13) -
            n13 * (n23 * (p21 * x14 + p21 * x15 + p31 * x16) -
                   p33 * (n11 * x10 + pow(n21, 2) * p12 - n21 * (x3 + x4)) +
                   x12 * x7) +
            x0 * (n21 * p21 * p32 - n32 * pow(p21, 2) + p21 * x9 + p31 * x1 -
                  p31 * x10));
  *result2 =
      -x8 *
      (n11 * (-n33 * p12 * (2 * x11 + x12) + p33 * (n23 * x21 + n33 * x13) +
              x19 * (n11 * n22 - n12 * n21)) +
       n13 *
           (-p31 * (n12 * x11 - n33 * x16) -
            p33 * (n11 * (2 * x17 + x18) - n21 * (-2 * x15 + x23) - n31 * x3) +
            x14 * x22 + x15 * x22) +
       x0 * (p31 * (x17 + x18) - p32 * (n21 * p31 + n31 * p21)));
  *result3 =
      x8 * (-n11 * (-pow(n33, 2) * p12 * p31 + n33 * p33 * x21 +
                    x19 * (-n12 * n31 + x24)) -
            n13 * (n33 * p31 * (x14 + x15 - x20) -
                   p33 * (pow(n31, 2) * p12 - n31 * x23 + 2 * p31 * x24)) +
            p31 * x0 * (n31 * p32 - n32 * p31));
}

namespace root_find {
// template <>
std::vector<double> solvePoly(const std::array<double, 4>& coef, double lower,
                              double upper) {  // cubic
  auto a = coef[0], b = coef[1], c = coef[2], d = coef[3];
  if (abs(a) < 1e-10)
    return solvePoly(std::array<double, 3>{b, c, d}, lower, upper);
  Eigen::Matrix3d companion;
  companion << 0, 0, -d / a, 1, 0, -c / a, 0, 1, -b / a;
  Eigen::Vector3cd ev = companion.eigenvalues();
  std::vector<double> res;
  for (int i = 0; i < 3; i++) {
    if (abs(ev[i].imag()) > 1e-10) continue;
    double t = ev[i].real();
    if (t >= lower && t <= upper) res.push_back(t);
  }
  return res;
};

// template <>
std::vector<double> solvePoly(const std::array<double, 3>& coef, double lower,
                              double upper) {  // quadratic
  auto a = coef[0], b = coef[1], c = coef[2];
  if (abs(a) < 1e-10)
    return solvePoly(std::array<double, 2>{b, c}, lower, upper);
  b = b / a;
  c = c / a;
  double delta = b * b - 4 * c;
  if (delta < 0) return {};  // no solution
  if (delta < 1e-10) {
    double t = -b / 2;
    if (t >= lower && t <= upper) return {t};
    return {};
  } else {
    std::vector<double> result;
    double t1 = (-b - sqrt(delta) * (b < 0 ? -1 : 1)) / 2;
    double t2 = c / t1;
    for (auto& t : {t1, t2})
      if (t >= lower && t <= upper) result.push_back(t);
    return result;
  }
};

// template <>
std::vector<double> solvePoly(const std::array<double, 2>& coef, double lower,
                              double upper) {  // linear
  auto a = coef[0], b = coef[1];
  double sol = -1;
  if (abs(a) >= 1e-10) {
    sol = -b / a;
    if (sol >= lower && sol <= upper) return {sol};
  }
  return {};
}
}  // namespace root_find
/*
int prism::search_projection(const RowMatd& V0,
                             const RowMatd& V1,
                             const RowMati& F,
                             const RowMati& TT,
                             const Eigen::RowVector3d& point, int guess,
                             std::array<double, 3>& tuple) {
  std::vector<bool> visited(TT.rows(), false);
  std::list<int> queue{guess};
  while (!queue.empty()) {
    // visit
    auto f = queue.front();
    queue.pop_front();
    visited[f] = true;

    // traverse body: test if in s in two steps
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> baseV(3, 3), topV(3, 3);
    baseV << V0.row(F(f, 0)), V0.row(F(f, 1)), V0.row(F(f, 2));
    topV << V1.row(F(f, 0)), V1.row(F(f, 1)), V1.row(F(f, 2));
    std::array<std::array<double, 3>, 3> top, base;
    for (int i = 0; i < 3; i++) {
      std::memcpy(top[i].data(), topV.row(i).data(), 3 * sizeof(double));
      std::memcpy(base[i].data(), baseV.row(i).data(), 3 * sizeof(double));
    }

    if (prism::maybe_inside_octahedron(base, top, {point[0], point[1], point[2]}))
      if (prism::inside_general_prism(baseV.row(0), baseV.row(1), baseV.row(2),
                                      topV.row(0), topV.row(1), topV.row(2),
                                      point)) {
        {
        prism::phong_projection(baseV, topV - baseV, point, tuple);
        return f;
      }

    // push neighbors
    for (int i = 0; i < 3; i++) {
      auto n = TT(f, i);
      if (n != -1 && !visited[n]) queue.push_back(n);
    }
  }
  return -1;
}
*/