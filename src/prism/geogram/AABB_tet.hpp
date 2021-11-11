#ifndef PRISM_GEOGRAM_AABB_TET_HPP
#define PRISM_GEOGRAM_AABB_TET_HPP

#include <memory>
#include <optional>
#include <prism/common.hpp>

namespace GEO {
class MeshCellsAABB;
class Mesh;
}  // namespace GEO

namespace prism::geogram {
struct AABB_tet {
  AABB_tet(const RowMatd &V, const RowMati &T);
  std::tuple<int, Eigen::RowVector4d> point_query(const Vec3d &) const;
  std::vector<int> geo_cell_ind, geo_vertex_ind;
  std::shared_ptr<GEO::MeshCellsAABB> geo_tree_ptr_;
  std::shared_ptr<GEO::Mesh> geo_polyhedron_ptr_;
};
}  // namespace prism::geogram
#endif