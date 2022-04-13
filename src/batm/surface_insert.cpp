#include "surface_insert.hpp"

#include "Rational.h"
#include "batm/tetra_logger.hpp"
#include "batm/tetra_utils.hpp"
#include "oneapi/tbb/concurrent_map.h"
#include "oneapi/tbb/concurrent_vector.h"
#include "prism/predicates/inside_prism_tetra.hpp"
#include "wmtk/TetMesh.h"
#include "wmtk/utils/InsertTriangleUtils.hpp"

#include <geogram/basic/geometry.h>
#include <geogram/numerics/predicates.h>

auto degenerate_config(
    const std::vector<prism::tet::VertAttr>& tetv,
    const Vec4i& tet,
    const Vec3d& pt) -> std::array<int, 3>
{
    using GEO::PCK::orient_3d;
    using GEO::PCK::points_are_colinear_3d;
    std::array<bool, 4> colinear{false, false, false, false};
    for (auto i = 0; i < 4; i++) {
        if (pt == tetv[tet[i]].pos) return {tet[i], -1, -1}; // vertex
    }
    for (auto i = 0; i < 4; i++) {
        if (orient_3d(
                pt.data(),
                tetv[tet[(i + 1) % 4]].pos.data(),
                tetv[tet[(i + 2) % 4]].pos.data(),
                tetv[tet[(i + 3) % 4]].pos.data()) == 0) {
            for (auto j = 0; j < 3; j++) {
                if (points_are_colinear_3d(
                        pt.data(),
                        tetv[tet[(i + 1 + j) % 4]].pos.data(),
                        tetv[tet[(i + 1 + (j + 1) % 3) % 4]].pos.data()))
                    return {tet[(i + 1 + j) % 4], tet[(i + 1 + (j + 1) % 3) % 4], -1}; // edge
            }
            return {tet[(i + 1) % 4], tet[(i + 2) % 4], tet[(i + 3) % 4]}; // face
        }
    }
    return {-1, -1, -1}; // general
}

void prism::tet::insert_all_points(
    prism::tet::tetmesh_t& tetmesh,
    const std::vector<int>& hint_tid,
    const std::vector<Vec3d>& points)
{
    std::map<int, std::set<int>> split_maps;
    auto& [tetv, tett, vt] = tetmesh;

    std::function<int(size_t, const Vec3d&)> find_containing_tet;
    find_containing_tet = [&split_maps, &tetv = tetv, &tett = tett, &find_containing_tet](
                              size_t tid,
                              const Vec3d& pt) -> int {
        auto it = split_maps.find(tid);
        if (it == split_maps.end()) { // leaf
            auto tet = tett[tid].conn;
            if (prism::predicates::point_in_tetrahedron(
                    pt,
                    tetv[tet[0]].pos,
                    tetv[tet[1]].pos,
                    tetv[tet[2]].pos,
                    tetv[tet[3]].pos))
                return tid;
        } else {
            for (auto v : it->second) {
                auto res = find_containing_tet(v, pt);
                if (res != -1) return res;
            }
        }
        return -1;
    };

    std::vector<int> new_vid(points.size());
    for (auto i = 0; i < points.size(); i++) {
        auto pt = points[i];
        auto tid = find_containing_tet(hint_tid[i], pt); // the final tid

        auto config = degenerate_config(tetv, tett[tid].conn, pt);
        auto ux = tetv.size();
        prism::tet::logger().trace("insert {} with config {}", i, config);
        if (config[0] != -1) {
            if (config[1] == -1) { // point degenerate
                // vertex
                new_vid[i] = config[0];
                continue;
            } else if (config[2] == -1) {
                // edge
                prism::tet::sec::split_edge(tetv, tett, vt, config[0], config[1]);
                tetv.back().pos = pt;
                new_vid[i] = ux;
            } else {
                // face
                prism::tet::sec::split_face(tetv, tett, vt, config[0], config[1], config[2]);
                tetv.back().pos = pt;
                new_vid[i] = ux;
            }
        } else {
            // general position, insert the single point
            auto num_tet = tett.size();
            prism::tet::sec::divide_tetra(tetv, tett, vt, tid, pt);
            auto& s = split_maps[tid];
            for (auto n = num_tet; n < tett.size(); n++) {
                s.insert(n);
            }
            new_vid[i] = ux;
        }
    }
}


void match_tet_faces_to_triangles(
    const prism::tet::tetmesh_t& m,
    const std::vector<std::array<size_t, 3>>& faces,
    tbb::concurrent_vector<bool>& is_matched,
    tbb::concurrent_map<std::array<size_t, 3>, std::vector<int>>& tet_face_tags)
{
    is_matched.resize(faces.size(), false);

    std::map<std::array<size_t, 3>, size_t> map_surface;
    for (size_t i = 0; i < faces.size(); i++) {
        auto f = faces[i];
        std::sort(f.begin(), f.end());
        map_surface[f] = i;
    }
    auto& [verts, tets, vt] = m;
    for (auto t : tets) {
        auto vs = t.conn;
        for (int j = 0; j < 4; j++) {
            std::array<size_t, 3> f = {
                {(size_t)vs[(j + 1) % 4], (size_t)vs[(j + 2) % 4], (size_t)vs[(j + 3) % 4]}};
            std::sort(f.begin(), f.end());
            auto it = map_surface.find(f);
            if (it != map_surface.end()) {
                auto fid = it->second;
                tet_face_tags[f].push_back(fid);
                is_matched[fid] = true;
            }
        }
    }
}
namespace prism::tet {
struct TetShell : public wmtk::TetMesh
{
};
} // namespace prism::tet


auto internal_insert_single_triangle(
    wmtk::TetMesh& m,
    std::vector<prism::tet::VertAttr>& m_vertex_attribute,
    const std::vector<Eigen::Vector3d>& vertices,
    const std::array<size_t, 3>& face,
    std::vector<std::array<size_t, 3>>& marked_tet_faces)
{
    auto vertex_pos_r = [&m_vertex_attribute](size_t i) { return m_vertex_attribute[i].pos_r; };

    const auto& [flag, intersected_tets, intersected_edges, intersected_pos] =
        wmtk::triangle_insert_prepare_info<triwild::Rational>(
            m,
            vertices,
            face,
            marked_tet_faces, // output
            [](auto&) { return true; },
            [](auto&) { return true; },
            vertex_pos_r);

    if (!flag) {
        return false;
    }

    // these are only those on edges.
    std::vector<size_t> new_edge_vids;
    std::vector<size_t> new_center_vids;
    std::vector<std::array<size_t, 4>> center_split_tets;

    ///inert a triangle
    m.triangle_insertion(
        intersected_tets,
        intersected_edges,
        new_edge_vids,
        new_center_vids,
        center_split_tets);

    assert(new_center_vids.size() == center_split_tets.size());
    for (auto i = 0; i < new_center_vids.size(); i++) {
        auto vid = new_center_vids[i];
        auto& vs = center_split_tets[i];
        m_vertex_attribute[vid].pos_r =
            (m_vertex_attribute[vs[0]].pos_r + m_vertex_attribute[vs[1]].pos_r +
             m_vertex_attribute[vs[2]].pos_r + m_vertex_attribute[vs[3]].pos_r) /
            4;
    }
    assert(new_edge_vids.size() == intersected_pos.size());

    for (auto i = 0; i < intersected_pos.size(); i++) {
        m_vertex_attribute[new_edge_vids[i]].pos_r = intersected_pos[i];
    }

    return true;
};

#include <wmtk/TetMesh.h>
#include <wmtk/utils/InsertTriangleUtils.hpp>

void prism::tet::insert_triangles(
    prism::tet::tetmesh_t& tetmesh,
    const std::vector<std::array<size_t, 3>>& tris)
{
    tbb::concurrent_vector<bool> is_matched;
    tbb::concurrent_map<std::array<size_t, 3>, std::vector<int>> tet_face_tags;

    auto& [tetv, tett, vt] = tetmesh;
    auto m = TetShell();
    std::vector<std::array<size_t, 4>> tets(tett.size());
    for (auto i = 0; i < tets.size(); i++) {
        for (auto j = 0; j < 4; j++) tets[i][j] = tett[i].conn[j];
    }

    m.init(tetv.size(), tets);

    std::vector<Eigen::Vector3d> verts(tetv.size());
    for (auto i = 0; i < tetv.size(); i++) {
        verts[i] = tetv[i].pos;
        assert(tetv[i].rounded);
    }

    for (auto i = 0; i < tris.size(); i++) {
        std::vector<std::array<size_t, 3>> marked_tet_faces;
        auto success = internal_insert_single_triangle(m, tetv, verts, tris[i], marked_tet_faces);
        if (!success) continue;
        for (auto& f : marked_tet_faces) tet_face_tags[f].push_back(i);
    }
}