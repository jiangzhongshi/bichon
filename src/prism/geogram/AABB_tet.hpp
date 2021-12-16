#ifndef PRISM_GEOGRAM_AABB_TET_HPP
#define PRISM_GEOGRAM_AABB_TET_HPP

#include <memory>
#include <optional>
#include <prism/common.hpp>
#include "prism/geogram/AABB.hpp"

namespace GEO {
class MeshCellsAABB;
class Mesh;
} // namespace GEO

namespace prism::geogram {
struct AABB_tet
{
    AABB_tet(const RowMatd& V, const RowMati& T);
    std::tuple<int, Eigen::RowVector4d> point_query(const Vec3d&) const;

    AABB_tet(const AABB_tet&) = delete; // disable delete until figure out the pointers.
    // used for internal reordering (Morton coding used by geogram),
    std::vector<int> geo_cell_ind, geo_vertex_ind;

    // internal data structure.
private:
    std::shared_ptr<GEO::MeshCellsAABB> geo_tree_ptr_;
    std::shared_ptr<GEO::Mesh> geo_polyhedron_ptr_;
};
} // namespace prism::geogram
#endif