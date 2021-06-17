#ifndef PRISM_CGAL_QP_HPP
#define PRISM_CGAL_QP_HPP

#include <Eigen/Core>
#include <prism/common.hpp>

namespace prism::cgal {
  Vec3d qp_normal(const RowMatd& FN, const std::vector<int>& nb);
}
namespace prism{
  Vec3d qp_normal_igl(const RowMatd& FN, const std::vector<int>& nb);
}

#endif