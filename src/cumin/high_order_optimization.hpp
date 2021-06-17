#ifndef CUMIN_HIGH_ORDER_OPTIMIZATION_HPP
#define CUMIN_HIGH_ORDER_OPTIMIZATION_HPP

#include "curve_common.hpp"

namespace prism::curve {
bool InversionCheck(const RowMatd &lagr, const RowMati &p4T,
                    const RowMati &codec_fixed, const RowMati &codec9_fixed,
                    const RowMatd &bern_from_lagr_o4,
                    const RowMatd &bern_from_lagr_o9);

Eigen::VectorXd energy_evaluation(RowMatd &lagr, RowMati &p4T,
                                  const std::vector<RowMatd> &vec_dxyz);

// lagr is unique here per nodes. not the duplicated version.
void vertex_star_smooth(RowMatd &lagr, RowMati &p4T, int, int);

int edge_collapsing(RowMatd &lagr, RowMati &p4T, double stop_energy);

int edge_swapping(RowMatd &lagr, RowMati &p4T, double stop_energy);

int cutet_collapse(RowMatd &lagr, RowMati &p4T, double stop_energy);
int cutet_swap(RowMatd &lagr, RowMati &p4T, double stop_energy);
}  // namespace prism::curve

#endif