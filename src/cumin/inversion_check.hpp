#ifndef PRISM_CUMIN_INVERSION_CHECK_HPP
#define PRISM_CUMIN_INVERSION_CHECK_HPP

#include <prism/common.hpp>
namespace prism::curve {
bool tetrahedron_recursive_positive_check(const Eigen::VectorXd& controlpts,
                                          const RowMatd& bern_from_lag,
                                          const Eigen::MatrixX4i& short_codecs);
bool tetrahedron_inversion_check(const RowMatd& cp, const Eigen::MatrixX4i& codecs_o4,
                                 const Eigen::MatrixX4i& codec_o9,
                                 const RowMatd& bern_from_lagr_o4,
                                 const RowMatd& bern_from_lagr_o9);
bool tetrahedron_inversion_check(const RowMatd& cp);
}  // namespace prism::curve

#endif