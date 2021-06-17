#ifndef PRISM_PHONG_TANGENT_ORIENTATION_HPP
#define PRISM_PHONG_TANGENT_ORIENTATION_HPP

#include "../common.hpp"
namespace prism::phong {
bool tangent_orientation(const std::array<Vec3d,6>& stacked_V,
                        const std::array<Vec3d,3>& triangle);
}

#endif