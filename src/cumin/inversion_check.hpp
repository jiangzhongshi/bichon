#ifndef PRISM_CUMIN_INVERSION_CHECK_HPP
#define PRISM_CUMIN_INVERSION_CHECK_HPP

#include <prism/common.hpp>
namespace prism::curve {
bool tetrahedron_recursive_positive_check(const Eigen::VectorXd& controlpts,
                                          const RowMatd& bern_from_lag,
                                          const RowMati& short_codecs);
bool tetrahedron_inversion_check(const RowMatd& cp, const Eigen::Matrix<int,-1,4>& codecs_o4,
                                 const  Eigen::Matrix<int,-1,4>& codec_o9,
                                 const RowMatd& bern_from_lagr_o4,
                                 const RowMatd& bern_from_lagr_o9);
bool tetrahedron_inversion_check(const RowMatd& cp);
}  // namespace prism::curve

#endif