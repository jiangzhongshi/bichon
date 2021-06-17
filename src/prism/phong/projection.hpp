#ifndef PRISM_PHONG_PROJECTION_HPP
#define PRISM_PHONG_PROJECTION_HPP

#include "../common.hpp"
namespace prism::phong {
// for one prism, tuple record a,b,t
bool phong_projection(const std::array<Vec3d, 6>& stacked_V, const Vec3d& point,
                      bool tetra_split_way, std::array<double, 3>& tuple);

bool phong_projection_uv(const std::array<Vec3d, 3>& base_v,
                         const std::array<Vec3d, 3>& mid_v,
                         const std::array<Vec3d, 3>& top_v, const Vec3d& point,
                         bool tetra_split_way, std::array<double, 2>& uv);
// get the endpoints of a single fiber (prism slab, so 4 points)
void fiber_endpoints(const std::array<Vec3d, 6>& stacked_V,
                     bool tetra_split_way, double u, double v,
                     std::array<Vec3d, 4>& endpoints);

// aggreagated function
void fiber_endpoints(const std::array<Vec3d, 9>& stacked_V,
                     bool tetra_split_way, double u, double v,
                     std::array<Vec3d, 7>& endpoints);
}  // namespace prism::phong

namespace prism::nonlinear {
bool phong_projection(const Eigen::Matrix3d& V, const Eigen::Matrix3d& N,
                      const Eigen::RowVector3d& point,
                      std::array<double, 3>& tuple);
}
#endif