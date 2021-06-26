#ifndef CUMIN_BERNSTEIN_EVAL_HPP
#define CUMIN_BERNSTEIN_EVAL_HPP
#include <prism/common.hpp>
namespace prism::curve {
//
//
// (number of basis) by (number of samples)
Eigen::ArrayXXd evaluate_bernstein(const Eigen::VectorXd& X,
                                   const Eigen::VectorXd& Y,
                                   const Eigen::VectorXd& Z,
                                   const RowMati& short_codecs);

//
//
// vector of 3: (number of basis) by (number of samples)
std::array<Eigen::ArrayXXd,3> evaluate_bernstein_derivative(
    const Eigen::VectorXd& X, const Eigen::VectorXd& Y,
    const Eigen::VectorXd& Z, const RowMati& short_codecs);
}  // namespace prism::curve

#endif