#ifndef PRISM_COLORING_MESH_HPP
#define PRISM_COLORING_MESH_HPP

#include <prism/common.hpp>

namespace prism::local {

// greedy graph coloring algorithm
// [https://www.geeksforgeeks.org/graph-coloring-set-2-greedy-algorithm/]
void vertex_coloring(const RowMati& F, std::vector<std::vector<int>>& group);

void red_green_coloring(const RowMati& F, const RowMati& FF,
                        Eigen::VectorXi& colors);

}  // namespace prism::local

#endif
