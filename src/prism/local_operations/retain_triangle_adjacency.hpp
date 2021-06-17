#ifndef PRISM_LOCAL_OPERATIONS_RETAIN_TRIANGLE_ADJACENCY_HPP
#define PRISM_LOCAL_OPERATIONS_RETAIN_TRIANGLE_ADJACENCY_HPP

#include <set>
#include "../common.hpp"
namespace prism::local {
std::pair<std::vector<Vec3i>, std::vector<Vec3i>> triangle_triangle_adjacency(
    const std::vector<Vec3i>& F);
// lazy person's book-keeping: Burn the books and buy again.
void retain_triangle_adjacency(const std::set<int>& delete_id,
                               const std::vector<Vec3i>& new_F,
                               std::vector<Vec3i>& F, std::vector<Vec3i>& TT,
                               std::vector<Vec3i>& TTi);
}  // namespace prism::local

#endif