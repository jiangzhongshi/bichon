#ifndef PRISM_POLYHEDRON_SELF_INTERSECT_HPP
#define PRISM_POLYHEDRON_SELF_INTERSECT_HPP

#include <Eigen/Core>

namespace prism {
namespace cgal {
// wrapper for CGAL self intersect detection
bool polyhedron_self_intersect(const Eigen::MatrixXd& V,
                               const Eigen::MatrixXi& F);

// get all the fe that is responsible. For middle surface extraction since that
// is the only case.
bool polyhedron_self_intersect_edges(
    const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
    std::vector<std::pair<int, int>>& fe_pairs);

void polyhedron_all_self_intersections(const Eigen::MatrixXd& V,
                                       const Eigen::MatrixXi& F);
bool tetrashell_self_intersect(const Eigen::MatrixXd& base,
                               const Eigen::MatrixXd& top,
                               const Eigen::MatrixXi& F,
                               const std::vector<bool>& mask,
                               std::vector<std::pair<int, int>>& pairs);
}  // namespace cgal
}  // namespace prism

#endif