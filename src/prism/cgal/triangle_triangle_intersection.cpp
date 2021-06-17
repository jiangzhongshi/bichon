#include "triangle_triangle_intersection.hpp"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <gmp.h>
#include <spdlog/spdlog.h>

namespace prism::cgal {

bool triangle_triangle_overlap(const std::array<Vec3d, 3> &tri0,
                               const std::array<Vec3d, 3> &tri1) {
  typedef ::CGAL::Exact_predicates_inexact_constructions_kernel K;
  std::array<K::Point_3, 6> cgal_points;
  for (int i = 0; i < tri0.size(); i++) {
    cgal_points[i] = K::Point_3(tri0[i][0], tri0[i][1], tri0[i][2]);
    cgal_points[i + 3] = K::Point_3(tri1[i][0], tri1[i][1], tri1[i][2]);
  }
  K::Triangle_3 t0(cgal_points[0], cgal_points[1], cgal_points[2]);
  K::Triangle_3 t1(cgal_points[3], cgal_points[4], cgal_points[5]);
  return CGAL::do_intersect(t0, t1);
}

bool segment_triangle_overlap(const std::array<Vec3d, 2> &seg,
                              const std::array<Vec3d, 3> &tri) {
  typedef ::CGAL::Exact_predicates_inexact_constructions_kernel K;
  std::array<K::Point_3, 5> cgal_points;
  for (int i = 0; i < 3; i++)
    cgal_points[i] = K::Point_3(tri[i][0], tri[i][1], tri[i][2]);
  for (int i = 0; i < 2; i++)
    cgal_points[i + 3] = K::Point_3(seg[i][0], seg[i][1], seg[i][2]);
  K::Triangle_3 t0(cgal_points[0], cgal_points[1], cgal_points[2]);
  K::Segment_3 s0(cgal_points[3], cgal_points[4]);
  if (t0.is_degenerate()) {
    spdlog::debug("degenerate triangle");
    spdlog::dump_backtrace();
    exit(1);
    return false;
  }
  return CGAL::do_intersect(t0, s0);
}

std::optional<Vec3d> segment_triangle_intersection(
    const std::array<Vec3d, 2> &seg, const std::array<Vec3d, 3> &tri) {
  typedef ::CGAL::Exact_predicates_inexact_constructions_kernel K;
  std::array<K::Point_3, 5> cgal_points;
  for (int i = 0; i < 3; i++)
    cgal_points[i] = K::Point_3(tri[i][0], tri[i][1], tri[i][2]);
  for (int i = 0; i < 2; i++)
    cgal_points[i + 3] = K::Point_3(seg[i][0], seg[i][1], seg[i][2]);
  K::Triangle_3 t0(cgal_points[0], cgal_points[1], cgal_points[2]);
  K::Segment_3 s0(cgal_points[3], cgal_points[4]);
  auto inter = CGAL::intersection(t0, s0);
  if (inter) {
    if (inter.value().which() != 0) {
      return {};
    }
    const K::Point_3 *point = &boost::get<K::Point_3>((inter).value());
    return Vec3d(CGAL::to_double(point->x()), CGAL::to_double(point->y()),
                 CGAL::to_double(point->z()));
  }
  return {};
}

std::optional<Vec3d> gmp_segment_triangle_intersection(
    const std::array<Vec3d, 2> &seg, const std::array<Vec3d, 3> &tri) {
  return {};
}
}  // namespace prism::cgal