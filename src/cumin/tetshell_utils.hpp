#pragma once


#include <prism/common.hpp>

namespace prism::tetshell {
RowMatd one_side_extrusion(RowMatd& V, const RowMati& F, RowMatd& VN, bool out);
void tetshell_fill(
    const RowMatd& ext_base,
    const RowMatd& shell_base,
    const RowMatd& shell_top,
    const RowMatd& ext_top,
    const RowMati& F_sh,
    Eigen::MatrixXd& V_out,
    Eigen::MatrixXi& T_out,
    Eigen::VectorXi& label_output);
} // namespace prism::tetshell