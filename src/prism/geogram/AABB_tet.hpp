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
    AABB_tet(const AABB_tet&) = delete; // disable delete until figure out the pointers.

public:
    /**
     * Given a point, find tetra that contains it and bary-coord.
     *
     * @return Tuple(tet-id(int) and bary-coord(vec4d))
     * @return -1 if not found.
     */
    std::tuple<int, Eigen::RowVector4d> point_query(const Vec3d&) const;

    /**
     * @brief Find **all** elements from input that overlaps with the query
     * Note that the overlap checks orient_3d so query needs to be oriented correctly.
     * @return std::vector<size_t>
     */
    std::vector<size_t> overlap_tetra(const std::array<Vec3d, 4>&) const;
    std::vector<size_t> overlap_tri(const std::array<Vec3d, 3>&) const;

private:
    // used for internal reordering (Morton coding used by geogram),
    std::vector<int> geo_cell_ind, geo_vertex_ind;

    // internal data structure.
private:
    std::shared_ptr<GEO::MeshCellsAABB> geo_tree_ptr_;
    std::shared_ptr<GEO::Mesh> geo_polyhedron_ptr_;
};
} // namespace prism::geogram
#endif