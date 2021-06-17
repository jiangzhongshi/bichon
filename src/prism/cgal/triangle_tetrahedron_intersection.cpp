#include "triangle_tetrahedron_intersection.hpp"
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

namespace prism::cgal {

bool triangle_tetrahedron_intersection(const std::array<Vec3d, 3>& triangle,
                                       const std::array<Vec3d, 4>& tetrahedron,
                                       bool wireless) {
  typedef ::CGAL::Exact_predicates_inexact_constructions_kernel K;
  std::array<K::Point_3, 3> tri_points;
  std::array<K::Point_3, 4> tet_points;
  for (int i = 0; i < 3; i++) {
    tri_points[i] = K::Point_3(triangle[i][0], triangle[i][1], triangle[i][2]);
  }

  for (int i = 0; i < 4; i++) {
    tet_points[i] =
        K::Point_3(tetrahedron[i][0], tetrahedron[i][1], tetrahedron[i][2]);
  }

  K::Triangle_3 tri(tri_points[0], tri_points[1], tri_points[2]);
  K::Tetrahedron_3 tet(tet_points[0], tet_points[1], tet_points[2],
                       tet_points[3]);
  if (tet.volume() <= 0) return false;
  if (!CGAL::do_overlap(tri.bbox(), tet.bbox())) return false;
  for (int i = 0; i < 3; i++)
    if (tet.bounded_side(tri_points[i]) !=
        CGAL::Bounded_side::ON_UNBOUNDED_SIDE) {
      return true;
    }

  if (CGAL::do_intersect(
          K::Triangle_3(tet_points[0], tet_points[1], tet_points[2]), tri))
    return true;
  if (CGAL::do_intersect(
          K::Triangle_3(tet_points[0], tet_points[1], tet_points[3]), tri))
    return true;
  if (CGAL::do_intersect(
          K::Triangle_3(tet_points[0], tet_points[2], tet_points[3]), tri))
    return true;
  // if (CGAL::do_intersect(K::Triangle_3(tet_points[1], tet_points[2],
  // tet_points[3]), tri)) return true;

  // The following is not enough
  // if (CGAL::do_intersect(K::Segment_3(tet_points[0], tet_points[1]), tri))
  // return true; if (CGAL::do_intersect(K::Segment_3(tet_points[0],
  // tet_points[2]), tri)) return true; if
  // (CGAL::do_intersect(K::Segment_3(tet_points[0], tet_points[3]), tri))
  // return true; if (CGAL::do_intersect(K::Segment_3(tet_points[1],
  // tet_points[2]), tri)) return true; if
  // (CGAL::do_intersect(K::Segment_3(tet_points[1], tet_points[3]), tri))
  // return true; if (CGAL::do_intersect(K::Segment_3(tet_points[2],
  // tet_points[2]), tri)) return true;

  return false;

  // Now, all three vertices are outside.
}
}  // namespace prism::cgal

#include <spdlog/spdlog.h>
namespace prism::cgal {

bool triangle_tetrahedron_intersection_wireless(
    const std::array<Vec3d, 3>& triangle,
    const std::array<Vec3d, 4>& tetrahedron) {
  typedef ::CGAL::Exact_predicates_exact_constructions_kernel K;
  std::array<K::Point_3, 3> tri_points;
  std::array<K::Point_3, 4> tet_points;
  for (int i = 0; i < 3; i++) {
    tri_points[i] = K::Point_3(triangle[i][0], triangle[i][1], triangle[i][2]);
  }

  for (int i = 0; i < 4; i++) {
    tet_points[i] =
        K::Point_3(tetrahedron[i][0], tetrahedron[i][1], tetrahedron[i][2]);
  }

  K::Triangle_3 tri(tri_points[0], tri_points[1], tri_points[2]);
  K::Tetrahedron_3 tet(tet_points[0], tet_points[1], tet_points[2],
                       tet_points[3]);
  if (!CGAL::do_overlap(tri.bbox(), tet.bbox())) return false;
  std::array<bool, 3> on_boundary{false, false, false};
  int num_on = 0;
  for (int i = 0; i < 3; i++) {
    auto b = tet.bounded_side(tri_points[i]);
    if (b == CGAL::Bounded_side::ON_BOUNDED_SIDE) {
      // spdlog::debug("i in {}", i);
      return true;  // any in
    } else if (b == CGAL::Bounded_side::ON_BOUNDARY) {
      on_boundary[i] = true;
      num_on++;
    }
  }
  // spdlog::debug("num on {}", num_on);
  if (num_on == 3) return true;  // all three on

  std::array<K::Triangle_3, 4> tet_faces{
      {{tet_points[1], tet_points[2], tet_points[3]},
       {tet_points[0], tet_points[2], tet_points[3]},
       {tet_points[0], tet_points[1], tet_points[3]},
       {tet_points[0], tet_points[1], tet_points[2]}}};
  std::array<bool, 4> need_compute_intersection{true, true, true, true};

  if (CGAL::do_intersect(tet_faces[1], tri) ||
      CGAL::do_intersect(tet_faces[2], tri) ||
      CGAL::do_intersect(tet_faces[3], tri))
    ;  // continue
  else
    return false;  // no intersection

  // spdlog::debug("tri-tri tells go on");
  std::vector<K::Point_3> all_intersections;
  for (int i = 0; i < 4; i++) {
    auto inter = CGAL::intersection(tet_faces[i], tri);
    if (!inter) continue;
    // spdlog::debug("face{}, case {}", i, inter.get().which());
    switch (inter.get().which()) {
      case 0:  // point, continue to the next check.
        all_intersections.push_back(boost::get<K::Point_3>(inter.get()));
        break;

      case 1:  // segment
        all_intersections.push_back(
            boost::get<K::Segment_3>(inter.get()).source());
        all_intersections.push_back(
            boost::get<K::Segment_3>(inter.get()).target());
        // non-edge segment -> real intersect
        break;
      default:
        return true;  // intersection is a polygon.
    }
  }

  // Finally, check if all the intersections belong to the same triangle
  for (int i = 0; i < 4; i++) {
    auto all_on = true;
    auto plane = tet_faces[i].supporting_plane();
    for (auto& p : all_intersections) {
      if (!plane.has_on(p)) {
        all_on = false;
        break;
      }
    }
    if (all_on) return false;  // borderline intersect
  }
  return true;
}

}  // namespace prism::cgal