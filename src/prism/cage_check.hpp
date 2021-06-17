#ifndef PRISM_CAGE_CHECK_HPP
#define PRISM_CAGE_CHECK_HPP
#include <set>
#include <vector>

#include "common.hpp"
struct PrismCage;
namespace prism::local {
struct RemeshOptions;
}
namespace prism::cage_check {
bool cage_is_positive(const PrismCage& pc);
bool cage_is_away_from_ref(const PrismCage& pc);
bool verify_bijection(const PrismCage& pc, const std::vector<Vec3d>& V,
                      const std::vector<Vec3i>& F,
                      const std::vector<std::set<int>>& track_to_prism);
bool verify_edge_based_track(const PrismCage& pc,
                             const prism::local::RemeshOptions& option,
                             std::vector<std::set<int>>& track_to_prism);

void initial_trackee_reconcile(PrismCage& pc,
                               double);
}  // namespace prism::cage_check
#endif