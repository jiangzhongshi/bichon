#ifndef CUMIN_STITCH_SURFACE_HPP
#define CUMIN_STITCH_SURFACE_HPP

#include <prism/common.hpp>

////////
// Prerequisite: Vmsh is ordered as Vbase, Vin
// 1. Process Vmsh, to become [Vbase, _, Vin], leaving empty index margin.
// 2. Assemble tuples for shell, and msh.
//  
// the final vertex indices are aranged as V_base, V_top, V_in

namespace prism::curve {
    bool stitch_surface_to_volume(
    const RowMatd &base, const RowMatd &top, const RowMati &F_sh,
    const std::vector<RowMatd> &complete_cp,
    const Eigen::MatrixXd &Vmsh, const Eigen::MatrixXi &Tmsh,
    RowMatd& output_nodes, RowMati& p4T);
}

#endif