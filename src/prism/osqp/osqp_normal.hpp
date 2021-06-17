#pragma once

#include "../common.hpp"
namespace prism {
    Vec3d osqp_normal(const RowMatd &N, const std::vector<int>&nb);
}