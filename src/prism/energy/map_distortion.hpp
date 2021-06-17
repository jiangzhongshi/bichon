#ifndef PRISM_ENERGY_MAP_DISTORTION_HPP
#define PRISM_ENERGY_MAP_DISTORTION_HPP

#include "../common.hpp"
namespace prism::energy {
enum class DistortionType { SYMMETRIC_DIRICHLET, ARAP };

double map_max_distortion(
    const Vec3d& pillar, const std::array<Vec3d, 3>& source_tri,
    const std::array<Vec3d, 3>& target_tri,
    DistortionType qt = DistortionType::SYMMETRIC_DIRICHLET);


bool map_max_angle_bound(const Vec3d& ez,  // pillar
                         const std::array<Vec3d, 3>& target_tri,
                         double angle_bound = 0.1);

double map_max_cos_angle(const Vec3d& ez,  // pillar
                                    const std::array<Vec3d, 3>& target_tri);
}  // namespace prism::energy
#endif