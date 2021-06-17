#include "prism_quality.hpp"

#include <igl/grad.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <Eigen/Dense>
#include <iostream>
#include <igl/polar_svd.h>

DECLARE_DIFFSCALAR_BASE();

namespace prism::energy {

// The following grad matrix is tabulated with
// test with igl::grad
// ```python
// np.set_printoptions(precision=16)
// prismV = np.array([[0,0,0],
//               [1,0,0],
//               [1/2,np.sqrt(3)/2,0],
//               [0,0,1],
//               [1,0,1],
//               [1/2,np.sqrt(3)/2,1]
//               ])
// tetsT = np.array([[0,2,3,4],
//   [0,3,4,5],
//   [0,1,5,3],
//   [0,1,2,3],
//   [1,2,3,4],
//   [0,1,2,4],
//   [0,1,5,4],
//   [1,3,4,5],
//   [1,2,3,5],
//   [0,2,5,4],
//   [0,1,2,5],
//   [2,3,4,5] ])
// G = igl.grad(prismV, tetsT).A
// Gx, Gy, Gz = np.split(G,3)
// np.hstack([Gx,Gy,Gz]).reshape(-1,6).flatten() # interlace Gxyz to alternate rows
// ```
// Then replace 0.5773502692 (ISR3) and 1.1547005384(2*ISR3)

const double ISR3 = 1/sqrt(3);
const Eigen::Matrix<double, 36, 6> POSITIVE_TETRA_GRAD =
    (Eigen::Matrix<double, 36, 6>() << 0, 0, 0, -1, 1, 0, -2*ISR3,
     0, 2*ISR3, ISR3, -ISR3, 0, -1, 0, 0, 1, 0, 0,
     0, 0, 0, -1, 1, 0, 0, 0, 0, -ISR3, -ISR3,
     2*ISR3, -1, 0, 0, 1, 0, 0, -1, 1, 0, 0, 0, 0,
     ISR3, -ISR3, 0, -2*ISR3, 0, 2*ISR3, -1, 0,
     0, 1, 0, 0, -1, 1, 0, 0, 0, 0, -ISR3, -ISR3,
     2*ISR3, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 1, 0,
     0, -2*ISR3, 2*ISR3, -ISR3, ISR3, 0, 0, -1,
     0, 0, 1, 0, -1, 1, 0, 0, 0, 0, -ISR3, -ISR3,
     2*ISR3, 0, 0, 0, 0, -1, 0, 0, 1, 0, -1, 1, 0, 0, 0, 0,
     -ISR3, ISR3, 0, 0, -2*ISR3, 2*ISR3, 0, -1,
     0, 0, 1, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, -ISR3,
     -ISR3, 2*ISR3, 0, -1, 0, 0, 1, 0, 0, 1, -1, -1, 0,
     1, 0, -ISR3, ISR3, -ISR3, 0, ISR3, 0,
     0, -1, 0, 0, 1, -1, 0, 1, 0, 1, -1, -ISR3, 0,
     ISR3, 0, -ISR3, ISR3, 0, 0, -1, 0, 0, 1,
     -1, 1, 0, 0, 0, 0, -ISR3, -ISR3, 2*ISR3, 0,
     0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, -1, 1, 0, 0, 0, 0,
     -ISR3, -ISR3, 2*ISR3, 0, 0, -1, 0, 0, 1)
        .finished();

double prism_full_quality(const std::array<Vec3d, 6>& corners, const Eigen::RowVector3d& dimscale, QualityType qt) {
  double quality = 0;

  auto cornersMat = Eigen::Map<const RowMatd>(corners[0].data(), 6, 3);
  Eigen::Matrix<double, 36, 3, Eigen::RowMajor> jacobian =
      POSITIVE_TETRA_GRAD * cornersMat;
  if (qt == QualityType::SYMMETRIC_DIRICHLET) {
    for (int i = 0; i < 12; i++) {
      Eigen::Matrix3d jac =
          dimscale.asDiagonal()*Eigen::Map<Eigen::Matrix3d>(jacobian.data() + 9 * i);

      auto invf2 = jac.inverse().cwiseAbs2().sum();
      auto frob2 = jac.cwiseAbs2().sum();
      quality += frob2 + invf2;
    }
    quality = quality / 12 - 6;
  } else if (qt == QualityType::MIPS_3D) {
    for (int i = 0; i < 12; i++) {
      Eigen::Matrix3d jac =
          dimscale.asDiagonal()*Eigen::Map<Eigen::Matrix3d>(jacobian.data() + 9 * i);
      if (jac.determinant() <= 0)
        return std::numeric_limits<double>::infinity();
      auto invf2 = jac.inverse().cwiseAbs2().sum();
      auto frob2 = jac.cwiseAbs2().sum();
      quality += frob2 * invf2;
    }
    quality = quality / 12 - 9;
  }

  return quality;
}

// corners[id_with_grad] is DScalar
DScalar prism_full_quality(const std::array<Vec3d, 6>& corners, const Eigen::RowVector3d& dimscale, QualityType qt,
                           int id_with_grad) {
  DiffScalarBase::setVariableCount(3);
  DScalar quality(0);

  Eigen::Matrix<DScalar, 6, 3> cornersMat =
      Eigen::Map<const RowMatd>(corners[0].data(), 6, 3).cast<DScalar>();

  std::array<DScalar, 3> point_with_grad;
  for (int d = 0; d < 3; d++) {
    cornersMat(id_with_grad, d) = DScalar(d, corners[id_with_grad][d]);
  }

  Eigen::Matrix<DScalar, 36, 3, Eigen::RowMajor> jacobian =
      POSITIVE_TETRA_GRAD.cast<DScalar>() * cornersMat;

  if (qt == QualityType::SYMMETRIC_DIRICHLET) {
    for (int i = 0; i < 12; i++) {
      Eigen::Matrix<DScalar,3,3> jac =
          dimscale.cast<DScalar>().asDiagonal()*Eigen::Map<Eigen::Matrix<DScalar, 3, 3>>(jacobian.data() + 9 * i);
      auto invf2 = jac.inverse().cwiseAbs2().sum();
      auto frob2 = jac.cwiseAbs2().sum();
      quality += frob2 + invf2;
    }
    quality = quality / 12 - 6;
  } else if (qt == QualityType::MIPS_3D) {
    for (int i = 0; i < 12; i++) {
      Eigen::Matrix<DScalar,3,3> jac =
          dimscale.cast<DScalar>().asDiagonal()*Eigen::Map<Eigen::Matrix<DScalar, 3, 3>>(jacobian.data() + 9 * i);
      auto invf2 = jac.inverse().cwiseAbs2().sum();
      auto frob2 = jac.cwiseAbs2().sum();
      quality += frob2 * invf2;
    }
    quality = quality / 12 - 9;
  }
  DiffScalarBase::setVariableCount(0);
  return quality;
}

double prism_one_ring_quality(const std::vector<Vec3d>& base,
                              const std::vector<Vec3d>& top,
                              const std::vector<Vec3i>& F,
                              const std::vector<int>& nb,
                              const std::vector<int>& nbi,
                              double target_height,
                              const std::vector<double>& areas,
                              const std::pair<Vec3d, Vec3d>& modification) {
  auto quality_type = prism::energy::QualityType::MIPS_3D;
  auto [dir_base, dir_top] = modification;
  double value = 0;
  for (int index = 0; index < nb.size(); index++) {
    int v_id = nbi[index];
    auto face = F[nb[index]];
    std::array<Vec3d, 6> verts;

    for (int i = 0; i < 3; i++) {
      verts[i] = base[face[i]];
      verts[i + 3] = top[face[i]];
    }

    verts[v_id] += dir_base;
    verts[v_id + 3] += dir_top;
    Vec3d dimscale(1/sqrt(areas[index]),1/sqrt(areas[index]), 1/target_height);
    value += prism::energy::prism_full_quality(verts, dimscale, quality_type);
  }
  return value;
}

double prism_one_ring_quality(const std::vector<Vec3d>& base,
                              const std::vector<Vec3d>& top,
                              const std::vector<Vec3i>& F,
                              const std::vector<int>& nb,
                              const std::vector<int>& nbi,
                              double target_height,
                              const std::vector<double>& areas,
                              const std::pair<bool, Vec3d>& modification) {
  auto quality_type = prism::energy::QualityType::MIPS_3D;
  auto [on_base, dir] = modification;
  auto newmod = std::make_pair(Vec3d(0, 0, 0), Vec3d(0, 0, 0));
  if (on_base)
    newmod.first = dir;
  else
    newmod.second = dir;
  return prism_one_ring_quality(base, top, F, nb, nbi, target_height, areas, newmod);
}

std::tuple<double, Vec3d> prism_one_ring_quality(
    const std::vector<Vec3d>& base, const std::vector<Vec3d>& top,
    const std::vector<Vec3i>& F, const std::vector<int>& nb,
    const std::vector<int>& nbi, 
    double target_height,
    const std::vector<double>& areas,
    bool on_base, int v_with_grad) {
  auto quality_type = prism::energy::QualityType::MIPS_3D;
  Vec3d grad(0, 0, 0);
  Eigen::Matrix3d hess = Eigen::Matrix3d::Zero();
  double value = 0;
  for (int index = 0; index < nb.size(); index++) {
    auto face = F[nb[index]];
    std::array<Vec3d, 6> verts;

    for (int i = 0; i < 3; i++) {
      verts[i] = base[face[i]];
      verts[i + 3] = top[face[i]];
    }
    Vec3d dimscale(1/sqrt(areas[index]),1/sqrt(areas[index]), 1/target_height);
    if (v_with_grad != -1) {
      int v_id = nbi[index];
      if (!on_base) v_id += 3;
      auto quality =
          prism::energy::prism_full_quality(verts,  dimscale, quality_type, v_id);
      grad += quality.getGradient();
      value += quality.getValue();
    } else {  // no grad needed
      value += prism::energy::prism_full_quality(verts, dimscale, quality_type);
    }
  }

  return std::tuple(value, grad);
}

double triangle_quality(const std::array<Vec3d, 3>& vertices) {
  Vec3d e1 = vertices[1] - vertices[0];
  Vec3d e2 = vertices[2] - vertices[0];
  auto e1_len = e1.norm();
  auto e2_x = e1.dot(e2) / e1_len;
  auto e2_y = (e2 - e2_x * e1 / e1_len).norm();
  Eigen::Matrix2d tri;
  tri << e1_len, e2_x, 0, e2_y;
  Eigen::Matrix2d ref;
  ref << 1, 0.5, 0, sqrt(3) / 2;
  Eigen::Matrix2d jac = tri * ref.inverse();
  // auto invf2 = jac.inverse().cwiseAbs2().sum();
  auto det = jac.determinant();
  auto frob2 = jac.cwiseAbs2().sum();
  return frob2 / det;
}

DScalar triangle_quality(const std::array<Vec3d, 3>& vertices,
                         int id_with_grad) {
  DiffScalarBase::setVariableCount(3);
  Eigen::Matrix<DScalar, 3, 3> verticesMat =
      Eigen::Map<const RowMatd>(vertices[0].data(), 3, 3).cast<DScalar>();
  std::array<DScalar, 3> point_with_grad;
  for (int d = 0; d < 3; d++) {
    verticesMat(id_with_grad, d) = DScalar(d, vertices[id_with_grad][d]);
  }
  auto e1 = verticesMat.row(1) - verticesMat.row(0);
  auto e2 = verticesMat.row(2) - verticesMat.row(0);
  auto e1_len = e1.norm();
  auto e2_x = e1.dot(e2) / e1_len;
  auto e2_y = (e2 - e2_x * e1 / e1_len).norm();
  // return ((std::pow(e1_len,2) + std::pow(e2_x - e2_y/sqrt(3), 2) + std::pow(e2_y,2)*(4/3.))/(e1_len * 2*e2_y/std::sqrt(3)));
  Eigen::Matrix<DScalar, 2, 2> tri;
  tri << e1_len, e2_x, DScalar(0), e2_y;
  Eigen::Matrix2d invref;
  invref << 1, -1 / sqrt(3), 0, 2 / sqrt(3);
  Eigen::Matrix<DScalar, 2, 2> jac = tri * (invref).cast<DScalar>();
  auto det = jac.determinant();
  auto frob2 = jac.cwiseAbs2().sum();
  return frob2 / det;
}

std::tuple<double, Vec3d> triangle_one_ring_quality(
    const std::vector<Vec3d>& mid, const std::vector<Vec3i>& F,
    const std::vector<int>& nb, const std::vector<int>& nbi,
    bool with_grad, Vec3d modification) {
  Vec3d grad = Vec3d::Zero();
  Eigen::Matrix3d hess = Eigen::Matrix3d::Zero();
  double value = 0;
  for (int index = 0; index < nb.size(); index++) {
    int v_id = nbi[index];
    auto face = F[nb[index]];
    std::array<Vec3d, 3> verts;

    for (int i = 0; i < 3; i++) {
      verts[i] = mid[face[i]];
    }
    verts[nbi[index]] += modification;

    if (with_grad) {
      auto quality = prism::energy::triangle_quality(verts, v_id);
      grad += quality.getGradient();
      value += quality.getValue();
      hess += quality.getHessian();
    } else {  // no grad needed
      value += prism::energy::triangle_quality(verts);
    }
  }
    if (with_grad) { // projected newton
    Eigen::Matrix3d R,T,U,V;
    Vec3d S;
    igl::polar_svd(hess, R,T,U,S,V);
    for (int j=0;j<3; j++) 
      S[j] = 1/std::max(1e-6, S[j]);
    // h' = g' * H^-T = g'*(USV')^-T = g' * (V S U')^-1 = g' * U * S^-1 * V'
    grad = grad*U*S.asDiagonal()*V.transpose();
  }
  return std::tuple(value, grad);
}

}  // namespace prism::energy