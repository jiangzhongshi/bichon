#include <geogram/numerics/predicates.h>
#include "AdaMesh.hpp"

#include <batm/tetra_logger.hpp>
#include <vector>
#include "Rational.h"
#include "oneapi/tbb/concurrent_vector.h"
#include "prism/common.hpp"
#include "prism/predicates/inside_prism_tetra.hpp"
#include "wmtk/utils/GeoUtils.h"
#include "wmtk/utils/InsertTriangleUtils.hpp"

auto replace = [](auto& arr, size_t a, size_t b) {
    for (auto i = 0; i < arr.size(); i++) {
        if (arr[i] == a) {
            arr[i] = b;
            return i;
        }
    }
    assert(false);
    return -1;
};


auto degenerate_config =
    [](const auto& tetv, const auto& tet, const Vec3d& pt) -> std::array<int, 3> {
    using GEO::PCK::orient_3d;
    using GEO::PCK::points_are_colinear_3d;
    std::array<bool, 4> colinear{false, false, false, false};
    for (auto i = 0; i < 4; i++) {
        if (pt == tetv[tet[i]].pos) return {int(tet[i]), -1, -1}; // vertex
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
                    return {
                        (int)tet[(i + 1 + j) % 4],
                        (int)tet[(i + 1 + (j + 1) % 3) % 4],
                        -1}; // edge
            }
            return {(int)tet[(i + 1) % 4], (int)tet[(i + 2) % 4], (int)tet[(i + 3) % 4]}; // face
        }
    }
    return {-1, -1, -1}; // general
};


auto internal_insert_single_triangle(
    wmtk::prism::AdaMesh& m,
    wmtk::prism::AdaMesh::VertAttCol& vertex_attrs,
    const std::array<size_t, 3>& face,
    std::vector<std::array<size_t, 3>>& marked_tet_faces)
{
    auto vertex_pos_r = [&vertex_attrs](size_t i) { return vertex_attrs[i].pos_r; };

    const auto& [flag, intersected_tets, intersected_edges, intersected_pos] =
        wmtk::triangle_insert_prepare_info<apps::Rational>(
            m,
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
        vertex_attrs[vid].pos_r = (vertex_attrs[vs[0]].pos_r + vertex_attrs[vs[1]].pos_r +
                                   vertex_attrs[vs[2]].pos_r + vertex_attrs[vs[3]].pos_r) /
                                  4;
        vertex_attrs[vid].rounded = false;
    }
    assert(new_edge_vids.size() == intersected_pos.size());

    for (auto i = 0; i < intersected_pos.size(); i++) {
        vertex_attrs[new_edge_vids[i]].pos_r = intersected_pos[i];
        vertex_attrs[new_edge_vids[i]].rounded = false;
    }

    return true;
};


namespace wmtk::prism {
AdaMesh::AdaMesh(const RowMatd& V, const RowMati& T)
{
    vertex_attrs.resize(V.rows());
    tet_attrs.resize(T.rows());

    std::vector<std::array<size_t, 4>> tets(T.rows());
    for (auto i = 0; i < tets.size(); i++) {
        for (auto j = 0; j < 4; j++) tets[i][j] = T(i, j);
    }
    init(V.rows(), tets);

    // attrs
    for (auto i = 0; i < V.rows(); i++) {
        vertex_attrs[i] = VertexAttributes(Vec3d(V(i, 0), V(i, 1), V(i, 2)));
    }
}


struct DivideTet : public TetMesh::OperationBuilder
{
    const AdaMesh& m;
    TetMesh::Tuple tet;
    size_t ux;

    DivideTet(const AdaMesh& _m)
        : m(_m)
    {}
    std::vector<size_t> removed_tids(const TetMesh::Tuple& t)
    {
        tet = t;
        return {t.tid(m)};
    }
    int request_vert_slots() { return 1; }
    std::vector<std::array<size_t, 4>> replacing_tets(const std::vector<size_t>& slots)
    {
        assert(slots.size() == 1);
        ux = slots.front();

        std::array<size_t, 4> t_conn;
        auto vs = m.oriented_tet_vertices(tet);
        for (auto i = 0; i < 4; i++) t_conn[i] = vs[i].vid(m);
        auto new_tets = std::vector<std::array<size_t, 4>>(4, t_conn);
        for (auto i = 0; i < 4; i++) new_tets[i][i] = ux;

        return new_tets;
    }
};


struct SplitEdge : public TetMesh::OperationBuilder
{
    const AdaMesh& m;
    std::vector<size_t> affected;
    std::array<size_t, 2> edge_verts;
    size_t ux;

    SplitEdge(const AdaMesh& _m)
        : m(_m)
    {}
    std::vector<size_t> removed_tids(const TetMesh::Tuple& t)
    {
        auto incidents = m.get_incident_tets_for_edge(t);
        for (auto i : incidents) {
            affected.push_back(i.tid(m));
        }
        edge_verts = {t.vid(m), t.switch_vertex(m).vid(m)};

        return affected;
    }
    int request_vert_slots() { return 1; }
    std::vector<std::array<size_t, 4>> replacing_tets(const std::vector<size_t>& slots)
    {
        assert(slots.size() == 1);
        ux = slots.front();

        auto new_tets = std::vector<std::array<size_t, 4>>();

        new_tets.reserve(2 * affected.size());
        for (auto i = 0; i < affected.size(); i++) {
            auto t_conn = m.oriented_tet_vids(m.tuple_from_tet(affected[i]));
            for (auto j = 0; j < 2; j++) {
                new_tets.push_back(t_conn);
                replace(new_tets.back(), edge_verts[j], ux);
            }
        }
        return new_tets;
    }
};

struct SplitFace : public TetMesh::OperationBuilder
{
    const AdaMesh& m;
    std::vector<size_t> affected;
    std::array<size_t, 3> tri_verts;
    size_t ux;

    SplitFace(const AdaMesh& _m)
        : m(_m)
    {}
    std::vector<size_t> removed_tids(const TetMesh::Tuple& t)
    {
        auto oppo = t.switch_face(m);
        affected = {t.tid(m), oppo.tid(m)};

        return affected;
    }
    int request_vert_slots() { return 1; }
    std::vector<std::array<size_t, 4>> replacing_tets(const std::vector<size_t>& slots)
    {
        assert(slots.size() == 1);
        ux = slots.front();

        auto new_tets = std::vector<std::array<size_t, 4>>();

        new_tets.reserve(2 * 3);
        for (auto i = 0; i < 2; i++) {
            auto t_conn = m.oriented_tet_vids(m.tuple_from_tet(affected[i]));
            for (auto j = 0; j < 3; j++) {
                new_tets.push_back(t_conn);
                replace(new_tets.back(), tri_verts[j], ux);
            }
        }
        return new_tets;
    }
};

void AdaMesh::insert_all_points(const std::vector<Vec3d>& points, const std::vector<int>& hint_tid)
{
    std::function<int(size_t, const Vec3d&)> find_containing_tet;
    std::map<int, std::set<int>> split_maps;
    auto& m = *this;
    find_containing_tet = [&m, &split_maps, &tetv = vertex_attrs, &find_containing_tet](
                              size_t tid,
                              const Vec3d& pt) -> int {
        auto it = split_maps.find(tid);
        if (it == split_maps.end()) { // leaf
            auto vs = m.oriented_tet_vertices(m.tuple_from_tet(tid));
            if (::prism::predicates::point_in_tetrahedron(
                    pt,
                    tetv[vs[0].vid(m)].pos,
                    tetv[vs[1].vid(m)].pos,
                    tetv[vs[2].vid(m)].pos,
                    tetv[vs[3].vid(m)].pos))
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

        auto config =
            degenerate_config(vertex_attrs, m.oriented_tet_vids(m.tuple_from_tet(tid)), pt);
        ::prism::tet::logger().trace("insert {} with config {}", i, config);

        std::vector<Tuple> new_tets;
        if (config[0] != -1) {
            if (config[1] == -1) { // point degenerate
                // vertex
                new_vid[i] = config[0];
                continue;
            } else if (config[2] == -1) {
                // edge
                auto tup = m.tuple_from_edge({(size_t)config[0], (size_t)config[1]});
                SplitEdge spl_edge(m);
                auto suc = m.customized_operation(spl_edge, tup, new_tets);

                for (auto j = 0; j < spl_edge.affected.size();
                     j++) { // this follows from the convention inside
                    auto& s = split_maps[spl_edge.affected[j]];
                    for (auto k = 0; k < 2; k++) s.insert(new_tets[j * 2 + k].tid(m));
                }
                new_vid[i] = spl_edge.ux;
            } else {
                // face
                SplitFace spl_face(m);
                spl_face.tri_verts = {(size_t)config[0], (size_t)config[1], (size_t)config[2]};
                auto [tup, fid] = m.tuple_from_face(spl_face.tri_verts);
                auto suc = m.customized_operation(spl_face, tup, new_tets);

                assert(suc);
                for (auto j = 0; j < 2; j++) { // this follows from the convention inside splitface
                    auto& s = split_maps[spl_face.affected[j]];
                    for (auto k = 0; k < 3; k++) s.insert(new_tets[j * 3 + k].tid(m));
                }
                new_vid[i] = spl_face.ux;
            }
        } else {
            // general position, insert the single point
            DivideTet divide_tet(m);
            auto tup = m.tuple_from_tet(tid);

            auto suc = m.customized_operation(divide_tet, tup, new_tets);

            assert(suc);
            auto& s = split_maps[tid];
            for (auto t : new_tets) {
                s.insert(t.tid(m));
            }
            new_vid[i] = divide_tet.ux;
        }
        m.vertex_attrs[new_vid[i]] = AdaMesh::VertexAttributes(pt);
    }
}

void AdaMesh::insert_all_triangles(const std::vector<std::array<size_t, 3>>& tris)
{
    auto& m = *this;
    tbb::concurrent_vector<bool> is_matched;

    for (auto i = 0; i < tris.size(); i++) {
        std::vector<std::array<size_t, 3>> marked_tet_faces;
        auto success =
            internal_insert_single_triangle(m, m.vertex_attrs, tris[i], marked_tet_faces);
        if (!success) continue;
        spdlog::info("success");
        for (auto& f : marked_tet_faces) tet_face_tags[f].push_back(i);
    }

    finalize_triangle_insertion(tris);
}


void AdaMesh::finalize_triangle_insertion(const std::vector<std::array<size_t, 3>>& tris)
{
    auto projected_point_in_triangle = [](const auto& c, const auto& tri) {
        std::array<Vector2r, 3> tri2d;
        int squeeze_to_2d_dir = wmtk::project_triangle_to_2d(tri, tri2d);
        auto c2d = wmtk::project_point_to_2d(c, squeeze_to_2d_dir);
        //// should exclude the points on the edges of tri2d -- NO
        return wmtk::is_point_inside_triangle(c2d, tri2d);
    };
    for (auto& i : tet_face_tags) {
        auto& vids = i.first;
        auto fids = i.second;
        if (fids.empty()) continue;

        Vector3r c = (vertex_attrs[vids[0]].pos_r + vertex_attrs[vids[1]].pos_r +
                      vertex_attrs[vids[2]].pos_r) /
                     3;

        wmtk::vector_unique(fids);

        for (int input_fid : fids) {
            std::array<Vector3r, 3> tri = {
                vertex_attrs[tris[input_fid][0]].pos_r,
                vertex_attrs[tris[input_fid][1]].pos_r,
                vertex_attrs[tris[input_fid][2]].pos_r};
            if (projected_point_in_triangle(c, tri)) {
                auto [_, global_tet_fid] = tuple_from_face(vids);
                face_attrs[global_tet_fid].m_is_surface_fs = 1;
                //
                for (auto vid : vids) {
                    vertex_attrs[vid].m_is_on_surface = true;
                }
                //
                break;
            }
        }
    }


    //// track bbox
    auto faces = get_faces();

    for (int i = 0; i < faces.size(); i++) {
        auto vs = get_face_vertices(faces[i]);
        std::array<size_t, 3> vids = {{vs[0].vid(*this), vs[1].vid(*this), vs[2].vid(*this)}};
        int on_bbox = -1;
        for (int k = 0; k < 3; k++) {
            if (vertex_attrs[vids[0]].pos_r[k] == m_params.box_min[k] &&
                vertex_attrs[vids[1]].pos_r[k] == m_params.box_min[k] &&
                vertex_attrs[vids[2]].pos_r[k] == m_params.box_min[k]) {
                on_bbox = k * 2;
                break;
            }
            if (vertex_attrs[vids[0]].pos_r[k] == m_params.box_max[k] &&
                vertex_attrs[vids[1]].pos_r[k] == m_params.box_max[k] &&
                vertex_attrs[vids[2]].pos_r[k] == m_params.box_max[k]) {
                on_bbox = k * 2 + 1;
                break;
            }
        }
        if (on_bbox < 0) continue;
        auto fid = faces[i].fid(*this);
        face_attrs[fid].m_is_bbox_fs = on_bbox;
        //
        for (size_t vid : vids) {
            vertex_attrs[vid].on_bbox_faces.push_back(on_bbox);
        }
    }

    for_each_vertex(
        [&](auto& v) { wmtk::vector_unique(vertex_attrs[v.vid(*this)].on_bbox_faces); });

    //// rounding
    std::atomic_int cnt_round(0);

    for (int i = 0; i < vertex_attrs.size(); i++) {
        if (round(i)) cnt_round++;
    }

    wmtk::logger().info("cnt_round {}/{}", cnt_round, vertex_attrs.size());
};

bool AdaMesh::triangle_insertion_before(const std::vector<Tuple>& faces)
{
    triangle_insertion_local_cache.old_face_vids.clear(); // note: reset local vars

    for (auto& loc : faces) {
        auto vs = get_face_vertices(loc);
        std::array<size_t, 3> f = {{vs[0].vid(*this), vs[1].vid(*this), vs[2].vid(*this)}};
        std::sort(f.begin(), f.end());

        triangle_insertion_local_cache.old_face_vids.push_back(f);
    }

    return true;
}

bool AdaMesh::triangle_insertion_after(const std::vector<std::vector<Tuple>>& new_faces)
{
    /// remove old_face_vids from tet_face_tags, and map tags to new faces
    // assert(new_faces.size() == triangle_insertion_local_cache.old_face_vids.size() + 1);

    for (int i = 0; i < new_faces.size(); i++) {
        if (new_faces[i].empty()) continue;

        // note: erase old tag and then add new -- old and new can be the same face
        std::vector<int> tags;
        if (i < triangle_insertion_local_cache.old_face_vids.size()) {
            auto& old_f = triangle_insertion_local_cache.old_face_vids[i];
            auto iter = tet_face_tags.find(old_f);
            if (iter != tet_face_tags.end() && !iter->second.empty()) {
                tags = iter->second;
                iter->second.clear();
            }
            if (tags.empty()) continue; // nothing to inherit to new
        } else
            tags.push_back(triangle_insertion_local_cache.face_id);


        for (auto& loc : new_faces[i]) {
            auto vs = get_face_vertices(loc);
            std::array<size_t, 3> f = {{vs[0].vid(*this), vs[1].vid(*this), vs[2].vid(*this)}};
            std::sort(f.begin(), f.end());
            tet_face_tags[f] = tags;
        }
    }

    return true;
}

} // namespace wmtk::prism