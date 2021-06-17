#include "intersections.hpp"

#include <igl/Hit.h>
extern "C" {
#include "igl/raytri.c"
}
#include "cgal/triangle_triangle_intersection.hpp"
#include <geogram/basic/geometry_nd.h>
#include <geogram/mesh/mesh_AABB.h>
#include <geogram/mesh/mesh_geometry.h>
#include <igl/barycentric_coordinates.h>
std::optional<Vec3d>
prism::intersections::segment_triangle_intersection_inexact(
    const std::array<Vec3d, 2> &seg, const std::array<Vec3d, 3> &tri) {
      assert(false && "This function is not correct. Temporary disabled;");

      return {};
  double t, u, v;
  auto [v0, v1, v2] = tri;
  auto s_d = seg[0];
  Vec3d dir = seg[1] - seg[0];
  intersect_triangle1(s_d.data(), dir.data(), v0.data(), v1.data(), v2.data(),
                      &t, &u, &v);
  if (t > 1 || t < 0)
    return {};
  else
    return s_d * (1 - t) + seg[1] * t;
}

bool prism::intersections::segment_triangle_hit(const std::array<Vec3d, 2> &seg,
                                                const std::array<Vec3d, 3> &tri,
                                                prism::Hit &hit) {
  hit.u = -1;
  hit.v = -1;
  hit.t = -1;
  auto [v0, v1, v2] = tri;
  auto s_d = seg[0];
  Vec3d dir = seg[1] - seg[0];
  auto flag = intersect_triangle1(s_d.data(), dir.data(), v0.data(), v1.data(),
                                  v2.data(), &hit.t, &hit.u, &hit.v);
  return (flag == 1 && hit.t >= 0);
}

bool prism::intersections::segment_triangle_hit_cgal(
    const std::array<Vec3d, 2> &seg, const std::array<Vec3d, 3> &tri,
    prism::Hit &hit) {
  auto inter = prism::cgal::segment_triangle_intersection(seg, tri);
  if (!inter) return false;
  Vec3d bc;
  igl::barycentric_coordinates(inter.value(), tri[0], tri[1], tri[2], bc);
  hit.u = bc[1];
  hit.v = bc[2];
  return true;
}