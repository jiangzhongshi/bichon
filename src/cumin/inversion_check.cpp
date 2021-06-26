#include "inversion_check.hpp"

#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <Eigen/Dense>
#include <queue>

#include "bernstein_eval.hpp"
#include "curve_utils.hpp"
inline const RowMati sub_tetras =
    (RowMati(8, 4) << 3, 6, 9, 8, 0, 4, 5, 6, 1, 4, 8, 7, 2, 5, 7, 9, 4, 5, 6,
     8, 5, 6, 8, 9, 4, 5, 8, 7, 5, 7, 9, 8)
        .finished();

inline const RowMati sub_verts = (RowMati(10, 2) << 0, 0, 1, 1, 2, 2, 3, 3, 0,
                                  1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3)
                                     .finished();

bool prism::curve::tetrahedron_recursive_positive_check(
    const Eigen::VectorXd& input_cp, const RowMatd& bern_from_lag,
    const RowMati& short_codecs) {
  const int max_level = 4;
  int num_cp = input_cp.size();
  auto order = short_codecs(0, 0);

  std::queue<std::tuple<RowMatd, Eigen::VectorXd>> q;
  int level_count = 0;
  q.push({(RowMatd(4, 3) << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1).finished(),
          Eigen::VectorXd(input_cp)});

  RowMatd inp_bern = bern_from_lag * input_cp;

  q.push({RowMatd(), Eigen::VectorXd()});
  while (!q.empty()) {
    spdlog::trace("level {}. size {}", level_count, q.size());
    auto [std_vert_pos, lag_cp] = q.front();
    q.pop();
    if (std_vert_pos.size() == 0) {
      spdlog::trace("level indicator {}", level_count);
      ++level_count;
      if (level_count >= max_level) {
        spdlog::trace("{} pass max level {}", level_count, max_level);
        return false;  // empty indicator
      }
      if (q.empty()) return true;
      q.push({RowMatd(), Eigen::VectorXd()});
      continue;
    }
    if (lag_cp.minCoeff() <= 0) {
      spdlog::trace("lag negative {}: {}", level_count, lag_cp.minCoeff());
      return false;  // Lagrange Negative -> must negative
    }
    Eigen::VectorXd bern_cp = bern_from_lag * lag_cp;
    if (bern_cp.minCoeff() > 0) {
      spdlog::trace("Bern Positive {}", bern_cp.minCoeff());
      continue;  // Bern Positive -> all positive
    }

    // subdivide
    RowMatd sub_verts_pos(10, 3);
    for (int i = 0; i < sub_verts.rows(); i++) {
      sub_verts_pos.row(i) = (std_vert_pos.row(sub_verts(i, 0)) +
                              std_vert_pos.row(sub_verts(i, 1))) /
                             2;
    }
    spdlog::trace("splitting {}", level_count);
    for (auto ti = 0; ti < 8; ti++) {
      auto t = sub_tetras.row(ti);
      auto sub_tet_verts = RowMatd(4, 3);
      for (int j = 0; j < 4; j++)
        sub_tet_verts.row(j) = sub_verts_pos.row(t[j]);
      RowMatd sub_params = short_codecs.cast<double>() * sub_tet_verts / order;
      auto basis_val =
          prism::curve::evaluate_bernstein(sub_params.col(0), sub_params.col(1),
                                           sub_params.col(2), short_codecs);
      Eigen::VectorXd sub_lagr = basis_val.matrix() * inp_bern;
      if (sub_lagr.minCoeff() <= 0) {
        spdlog::trace("sub min {}", sub_lagr.minCoeff());
        return false;
      }
      q.emplace(sub_tet_verts, sub_lagr);
      spdlog::trace("push {} {}", ti, sub_lagr.minCoeff());
    }
  }
  return true;
}

bool prism::curve::tetrahedron_inversion_check(const RowMatd& cp) {
  auto& helper = prism::curve::magic_matrices();
  return tetrahedron_inversion_check(cp, helper.volume_data.vol_codec,
                                     helper.volume_data.vol_jac_codec,
                                     helper.volume_data.vol_bern_from_lagr,
                                     helper.volume_data.vol_jac_bern_from_lagr);
}

bool prism::curve::tetrahedron_inversion_check(
    const RowMatd& cp, const Eigen::Matrix<int, -1, 4>& codecs_o4,
    const Eigen::Matrix<int, -1, 4>& codecs_o9,
    const RowMatd& bern_from_lagr_o4, const RowMatd& bern_from_lagr_o9) {
  // ANCHOR: this function is 50% of the profile bottleneck.
  int order = codecs_o4(0, 0);
  int high_order = codecs_o9(0, 0);
  assert(high_order == (order - 1) * 3);
  auto& helper = prism::curve::magic_matrices();
  if (helper.inversion_helper.cache != high_order) { // TODO: potential not thread-safe
    // 3v x 35b x 220s
    helper.inversion_helper.bernstein_derivatives_checker =
        prism::curve::evaluate_bernstein_derivative(
            codecs_o9.col(1).cast<double>() / high_order,
            codecs_o9.col(2).cast<double>() / high_order,
            codecs_o9.col(3).cast<double>() / high_order, codecs_o4);
    helper.inversion_helper.cache = high_order;
  }
  auto& r1 = helper.inversion_helper.bernstein_derivatives_checker;

  RowMatd b_cp = bern_from_lagr_o4 * cp;
  // 3v x 220s x 3d
  std::vector<RowMatd> var_dim_sample;
  for (auto r : r1) {
    var_dim_sample.emplace_back(r.matrix().transpose() * b_cp);
  }
  Eigen::VectorXd lagr(codecs_o9.rows());
  for (int i = 0; i < codecs_o9.rows(); i++) {
    Eigen::Matrix3d temp;
    for (int j = 0; j < 3; j++) temp.row(j) = var_dim_sample[j].row(i);
    lagr[i] = temp.determinant();
  }
  return prism::curve::tetrahedron_recursive_positive_check(
      lagr, bern_from_lagr_o9, codecs_o9);
}