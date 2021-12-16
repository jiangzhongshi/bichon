#include "AABB_tet.hpp"

#include <geogram/basic/geometry_nd.h>
#include <geogram/basic/numeric.h>
#include <geogram/mesh/mesh_AABB.h>
#include <geogram/mesh/mesh_geometry.h>
#include <geogram/mesh/mesh_reorder.h>
#include <geogram/mesh/mesh_repair.h>
#include <geogram/numerics/predicates.h>
#include <igl/Hit.h>
#include <igl/Timer.h>
#include <igl/barycentric_coordinates.h>
#include <igl/boundary_facets.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <iostream>
#include <limits>
#include <list>
#include <prism/geogram/geogram_utils.hpp>
#include <prism/intersections.hpp>

#include "prism/predicates/tetrahedron_overlap.hpp"
#include "prism/predicates/triangle_triangle_intersection.hpp"

prism::geogram::AABB_tet::AABB_tet(const RowMatd& V, const RowMati& T)
{
    assert(T.cols() == 4);
    geo_polyhedron_ptr_ = std::make_unique<GEO::Mesh>();
    prism::geo::to_geogram_mesh(V, T, *geo_polyhedron_ptr_);
    geo_tree_ptr_ = std::make_unique<GEO::MeshCellsAABB>(*geo_polyhedron_ptr_, true);

    geo_vertex_ind.resize(V.rows());
    GEO::Attribute<int> original_indices(geo_polyhedron_ptr_->vertices.attributes(), "vertex_id");
    for (int i = 0; i < original_indices.size(); i++) geo_vertex_ind[i] = original_indices[i];

    geo_cell_ind.resize(T.rows());
    GEO::Attribute<int> cell_indices(geo_polyhedron_ptr_->cells.attributes(), "cell_id");
    for (int i = 0; i < cell_indices.size(); i++) geo_cell_ind[i] = cell_indices[i];
}

std::tuple<int, Eigen::RowVector4d> prism::geogram::AABB_tet::point_query(const Vec3d& points) const
{
    auto p = GEO::vec3(points(0), points(1), points(2));
    auto tet_id = geo_tree_ptr_->containing_tet(p);
    Eigen::RowVector4d bc = Eigen::RowVector4d::Zero();
    if (tet_id == GEO::MeshCellsAABB::NO_TET) {
        return {-1, bc};
    }
    auto& pp = geo_polyhedron_ptr_;
    assert(pp->cells.nb_vertices(tet_id) == 4);
    auto v0 = pp->cells.vertex(tet_id, 0), v1 = pp->cells.vertex(tet_id, 1),
         v2 = pp->cells.vertex(tet_id, 2), v3 = pp->cells.vertex(tet_id, 3);
    spdlog::trace(pp->vertices.point(v0));

    auto g2e = [](auto& v) { return Vec3d(v[0], v[1], v[2]); };
    igl::barycentric_coordinates(
        points,
        g2e(pp->vertices.point(v0)),
        g2e(pp->vertices.point(v1)),
        g2e(pp->vertices.point(v2)),
        g2e(pp->vertices.point(v3)),
        bc);
    return {geo_cell_ind[tet_id], bc};
}

std::vector<size_t> prism::geogram::AABB_tet::overlap_tetra(const std::array<Vec3d, 4>& P) const
{
    std::vector<size_t> result;
    using namespace GEO;
    Box in_box;
    for (int i = 0; i < 3; i++) {
        in_box.xyz_max[i] = std::max({P[0][i], P[1][i], P[2][i], P[3][i]});
        in_box.xyz_min[i] = std::min({P[0][i], P[1][i], P[2][i], P[3][i]});
    }

    bool found = false;
    auto action = [&P,&result, &pp=geo_polyhedron_ptr_, &face_map=geo_cell_ind](GEO::index_t cell_id){
        index_t c = pp->cells.corners_begin(cell_id);
        Vec4i ind;
        std::array<Vec3d, 4> g_tetra;
        for (auto k=0; k<4;k++) {
          ind[k] = pp->cell_corners.vertex(c+k);
          auto pt = Geom::mesh_vertex(*pp, ind[k]);
          g_tetra[k] = Vec3d(pt.x, pt.y, pt.z);
        }
        if (prism::predicates::tetrahedron_tetrahedron_overlap(g_tetra, P)) {
          spdlog::trace("found overlap tetra");
          result.push_back(face_map[cell_id]);
        }
    };
    geo_tree_ptr_->compute_bbox_cell_bbox_intersections(in_box, action);
    return result;
}