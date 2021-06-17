#include "tetrahedron_tetrahedron_intersection.hpp"
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

namespace prism::cgal {
bool tetrahedron_tetrahedron_intersection(
    const CGAL::Epick::Tetrahedron_3& Atet,
    const CGAL::Epick::Tetrahedron_3& Btet) {
  using K = CGAL::Epick;
  if (Atet.is_degenerate() || Btet.is_degenerate()) return false;
  if (!CGAL::do_overlap(Atet.bbox(), Btet.bbox())) return false;

  for (int i = 0; i < 4; i++) {
    if (Btet.bounded_side(Atet.vertex(i)) !=
        CGAL::Bounded_side::ON_UNBOUNDED_SIDE) {  // on or in
      return true;
    }
    if (Atet.bounded_side(Btet.vertex(i)) !=
        CGAL::Bounded_side::ON_UNBOUNDED_SIDE) {  // on or in
      return true;
    }
  }

  if (CGAL::do_intersect(K::Triangle_3(Atet.vertex(0), Atet.vertex(1), Atet.vertex(2)), Btet))
    return true;
  if (CGAL::do_intersect(K::Triangle_3(Atet.vertex(0), Atet.vertex(1), Atet.vertex(3)), Btet))
    return true;
  if (CGAL::do_intersect(K::Triangle_3(Atet.vertex(0), Atet.vertex(2), Atet.vertex(3)), Btet))
    return true;
  if (CGAL::do_intersect(K::Triangle_3(Atet.vertex(1), Atet.vertex(2), Atet.vertex(3)), Btet))
    return true;

  return false;
}

bool tetrahedron_tetrahedron_intersection(const std::array<Vec3d, 4>& tetA,
                                          const std::array<Vec3d, 4>& tetB) {
  typedef ::CGAL::Exact_predicates_inexact_constructions_kernel K;
  std::array<K::Point_3, 4> A_points;
  std::array<K::Point_3, 4> B_points;

  for (int i = 0; i < 4; i++) {
    A_points[i] = K::Point_3(tetA[i][0], tetA[i][1], tetA[i][2]);
    B_points[i] = K::Point_3(tetB[i][0], tetB[i][1], tetB[i][2]);
  }

  K::Tetrahedron_3 Atet(A_points[0], A_points[1], A_points[2], A_points[3]);
  K::Tetrahedron_3 Btet(B_points[0], B_points[1], B_points[2], B_points[3]);
  return tetrahedron_tetrahedron_intersection(Atet, Btet);

  // Now, all three vertices are outside.
}
}  // namespace prism::cgal
