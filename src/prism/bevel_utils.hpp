#ifndef PRIISM_BEVEL_UTILS_HPP
#define PRIISM_BEVEL_UTILS_HPP
#include <prism/common.hpp>
namespace prism::bevel_utils {
void adaptive_doo_sabin(const RowMatd& V, const RowMati& F, const RowMatd& VN,
                        double eps, RowMatd& dsV, RowMati& dsF, RowMatd& dsVN,
                        std::vector<int>& face_parent);

// additional splits after the normal bevel, around singularities to make
// sections
void singularity_special_bevel(RowMatd& V, RowMati& F, int num_cons,
                               RowMatd& VN, std::vector<int>& face_parent);

// simplified bevel that use edge based overlap check, and allows edge markers
void edge_based_bevel(const RowMatd& V, const RowMati& F, const RowMatd& VN,
                      const RowMati& feat_vv, RowMatd& dsV, RowMati& dsF,
                      RowMatd& dsVN, std::vector<int>& face_parent);

bool verify_edge_bevel(const RowMatd& V, const RowMati& F, const RowMatd& VN,
                       const RowMati& feat_vv);
}  // namespace prism::bevel_utils

#endif