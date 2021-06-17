#include "bernstein_eval.hpp"

#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include "curve_common.hpp"

auto factorial = [](int n) {
  auto res = 1UL;
  for (int i = 1; i <= n; i++) {
    res *= i;
  }
  assert(res > 0);
  return res;
};
auto multinomial_gen = [](auto order, const RowMati& short_codecs) {
  std::vector<int> multinomial;
  auto faco = factorial(order);
  for (int i = 0; i < short_codecs.rows(); i++) {
    auto r = faco;
    for (int j = 0; j < short_codecs.cols(); j++) {
      r /= factorial(short_codecs(i, j));
    }
    multinomial.emplace_back(r);
  }
  return std::move(multinomial);
};

auto precompute_powers = [](auto order, auto& X, auto& Y, auto& Z) {
  std::vector<Eigen::ArrayX4d> stored_powers(order + 1);
  stored_powers[0] = Eigen::ArrayX4d::Ones(X.size(), 4);
  stored_powers[1] = Eigen::ArrayX4d::Ones(X.size(), 4);
  stored_powers[1].col(0) -= (X + Y + Z).array();
  stored_powers[1].col(1) = X;
  stored_powers[1].col(2) = Y;
  stored_powers[1].col(3) = Z;
  for (int i = 2; i < order + 1; i++) {
    stored_powers[i] = stored_powers[i - 1] * stored_powers[1];
  }
  return std::move(stored_powers);
};
Eigen::ArrayXXd prism::curve::evaluate_bernstein(const Eigen::VectorXd& X,
                                                 const Eigen::VectorXd& Y,
                                                 const Eigen::VectorXd& Z,
                                                 const RowMati& short_codecs) {
  assert((short_codecs.cols() == 4 || short_codecs.cols() == 3) &&
         "use the fixed length version of codec");
  int order = short_codecs(0, 0);
  assert(order == short_codecs.maxCoeff() && "Short Codecs convention.");
  auto multinomial = multinomial_gen(order, short_codecs);

  auto stored_powers = precompute_powers(order, X, Y, Z);

  Eigen::ArrayXXd res(X.size(), multinomial.size());
  for (auto ci = 0; ci < multinomial.size(); ci++) {
    auto cod = short_codecs.row(ci);
    res.col(ci) = Eigen::ArrayXd::Constant(X.size(), multinomial[ci]);
    for (auto i = 0; i < cod.size(); i++) {
      auto c = cod[i];
      res.col(ci) *= stored_powers[c].col(i);
    }
  }
  return res;
}

std::vector<Eigen::ArrayXXd> prism::curve::evaluate_bernstein_derivative(
    const Eigen::VectorXd& X, const Eigen::VectorXd& Y,
    const Eigen::VectorXd& Z, const RowMati& short_codecs) {
  assert(short_codecs.cols() == 4 && "use the fixed length version of codec");
  int order = short_codecs.maxCoeff();
  auto multinomial = multinomial_gen(order, short_codecs);

  auto stored_powers = precompute_powers(order, X, Y, Z);

  std::vector<Eigen::ArrayXXd> dxdydz(3);
  double debug_sum = 0;
  for (auto d = 0; d < 3; d++) {
    dxdydz[d].resize(multinomial.size(), X.size());
    for (auto ci = 0; ci < multinomial.size(); ci++) {
      auto mn = multinomial[ci];
      auto cod = short_codecs.row(ci);
      Eigen::ArrayXd part1 = Eigen::ArrayXd::Ones(X.size()) * cod[d + 1];
      for (int i = 0; i < cod.size(); i++) {
        auto e = cod[i];
        if (i != d + 1) {  // normal powers
          part1 *= stored_powers[e].col(i);
        } else if (e == 0) {
          part1 *= 0;
        } else {  // when d+1 == i and e != 0
          part1 *= stored_powers[e - 1].col(i);
        }
      }  // dx part
      if (cod[0] != 0) {
        auto e0 = cod[0];
        Eigen::ArrayXd part2 = e0 * stored_powers[e0 - 1].col(0);
        for (int i = 1; i < cod.size(); i++)
          part2 *= stored_powers[cod[i]].col(i);

        part1 -= part2;
      }  // dw part
      debug_sum += part1.square().sum();
      dxdydz[d].row(ci) = multinomial[ci] * part1;
    }  // over all basis
  }    // over all variables (dx,dy,dz)
  return dxdydz;
}