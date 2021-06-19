#ifndef PRISM_CGAL_AABB_HPP
#define PRISM_CGAL_AABB_HPP

#include <memory>
#include <optional>
#include <prism/common.hpp>

namespace GEO {
class MeshFacetsAABB;
class Mesh;
} // namespace GEO
namespace prism{
  struct Hit;
}
namespace prism::geogram {

struct AABB {
  // if `enabled = False`, non of the tests are active,
  // this allows for a non-intrusive implementation for disabling AABB
  AABB(const RowMatd &V, const RowMati &F, bool enabled = true);
  bool intersects_triangle(const std::array<Vec3d, 3> &P,
                           bool use_freeze = false) const;
  // if there are multiple intersection, the function will return 
  std::optional<Vec3d> segment_query(const Vec3d &start,
                                     const Vec3d &end) const;
  bool segment_query(const Vec3d &start, const Vec3d &end, int &face_id,
                     Vec3d &finalpoint) const;
  bool segment_hit(const Vec3d &start, const Vec3d &end, prism::Hit &hit) const;
  double ray_length(const Vec3d &start, const Vec3d &dir, double max_step,
                    int ignore_v) const;

  // test the numerical separation of a mesh.
  // For each point, construct a bounding box with edge=2*tol.
  // Then for each intersected primitive (with the box), check if it is
  // neighbor.
  bool numerical_self_intersection(double tol) const;
  bool self_intersections(std::vector<std::pair<int,int>>& pairs);

  std::shared_ptr<GEO::MeshFacetsAABB> geo_tree_ptr_;
  std::shared_ptr<GEO::Mesh> geo_polyhedron_ptr_;
  std::vector<int> geo_vertex_ind;
  std::vector<int> geo_face_ind;
  int num_freeze = 0;
  const bool enabled = true;
};

} // namespace prism::geogram

#endif