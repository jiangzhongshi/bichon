#ifndef CUMIN_CURVE_VALIDITY_HPP
#define CUMIN_CURVE_VALIDITY_HPP

#include <any>
#include <vector>

#include "curve_utils.hpp"

struct PrismCage;
namespace prism::local {
struct RemeshOptions;
}
namespace prism::curve {

// curve checker function handles
// input pc is for experimental with features
std::pair<std::any, std::any> curve_func_handles(
    std::vector<RowMatd> &complete_cp, const PrismCage &pc, const prism::local::RemeshOptions &option,
    int tri_order);

// local smoother to optimize each curve
// looks constant but their is a function in option that owns global
// controlpoints value
void localcurve_pass(const PrismCage &pc,
                     const prism::local::RemeshOptions &option);
}  // namespace prism::curve
#endif