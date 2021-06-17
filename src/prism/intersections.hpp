#pragma once
#include <optional>

#include "common.hpp"
namespace igl {
struct Hit;
}
namespace prism{
    struct Hit{
        int gid, id;
        double u,v,t;
    };
}
namespace prism::intersections {
std::optional<Vec3d> segment_triangle_intersection_inexact(
    const std::array<Vec3d, 2> &seg, const std::array<Vec3d, 3> &tri);

bool segment_triangle_hit(const std::array<Vec3d, 2> &seg,
                          const std::array<Vec3d, 3> &tri, prism::Hit &hit);
bool segment_triangle_hit_cgal(const std::array<Vec3d, 2> &seg,
                          const std::array<Vec3d, 3> &tri, prism::Hit &hit);

}  // namespace prism::intersections
