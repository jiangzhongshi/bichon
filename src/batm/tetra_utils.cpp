#include "tetra_utils.hpp"

#include <geogram/basic/geometry.h>
#include <geogram/numerics/predicates.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <limits>
#include <prism/PrismCage.hpp>
#include <prism/geogram/AABB.hpp>
#include <prism/local_operations/remesh_pass.hpp>
#include <prism/local_operations/validity_checks.hpp>
#include <vector>
#include "prism/cgal/triangle_triangle_intersection.hpp"
auto set_inter = [](auto& A, auto& B) {
    std::vector<int> vec;
    std::set_intersection(A.begin(), A.end(), B.begin(), B.end(), std::back_inserter(vec));
    return vec;
};

auto set_insert = [](auto& A, auto& a) {
    auto it = std::lower_bound(A.begin(), A.end(), a);
    A.insert(it, a);
};


namespace prism::tet {

std::tuple<
    std::vector<prism::tet::VertAttr>,
    std::vector<prism::tet::TetAttr>,
    std::vector<std::vector<int>>>
prepare_tet_info(PrismCage& pc, RowMatd& tet_v, RowMati& tet_t)
{
    auto vert_info = [&tet_v, &pc]() {
        std::vector<VertAttr> vert_attrs(tet_v.rows());
        auto nshell = pc.mid.size();
        for (auto i = 0; i < tet_v.rows(); i++) {
            vert_attrs[i].pos = tet_v.row(i);
        }
        for (auto i = 0; i < pc.mid.size(); i++) {
            assert(vert_attrs[i].pos == pc.mid[i]);
            vert_attrs[i].mid_id = i;
        }
        return vert_attrs;
    }();

    // Tet-Face (tet_t, k) -> Shell Prisms (pc.mF)
    auto tet_info = [&tet_t, &pc]() {
        std::vector<TetAttr> tet_attrs(tet_t.rows());

        std::map<std::array<int, 3>, int> cell_finder;
        for (auto i = 0; i < pc.F.size(); i++) {
            auto pri = pc.F[i];
            std::sort(pri.begin(), pri.end());
            cell_finder.emplace(pri, i);
        }
        // RowMati marker = RowMati::Constant(tet_t.rows(), 4, -1);
        for (auto i = 0; i < tet_t.rows(); i++) {
            auto& t_a = tet_attrs[i];
            for (auto j = 0; j < 4; j++) {
                t_a.conn[j] = tet_t(i, j);

                auto face = std::array<int, 3>();
                for (auto k = 0; k < 3; k++) face[k] = tet_t(i, (j + k + 1) % 4);
                std::sort(face.begin(), face.end());
                auto it = cell_finder.find(face);
                if (it != cell_finder.end()) { // found
                    t_a.prism_id[j] = it->second;
                }
            }
        }
        return tet_attrs;
    }();

    auto vert_tet_conn = [&tet_info, n_verts = vert_info.size()]() {
        auto vt_conn = std::vector<std::vector<int>>(n_verts);
        for (auto i = 0; i < tet_info.size(); i++) {
            for (auto j = 0; j < 4; j++) vt_conn[tet_info[i].conn[j]].emplace_back(i);
        }
        std::for_each(vt_conn.begin(), vt_conn.end(), [](auto& vec) {
            std::sort(vec.begin(), vec.end());
        });
        return vt_conn;
    }();

    // count how many marks
    [&tet_info, &tet_t, &pc]() {
        auto cnt = 0;
        for (auto i = 0; i < tet_t.rows(); i++) {
            for (auto j = 0; j < 4; j++) {
                if (tet_info[i].prism_id[j] != -1) cnt++;
            }
        }
        assert(cnt == pc.F.size());
    }();
    return std::tuple(vert_info, tet_info, vert_tet_conn);
};
double circumradi2(const Vec3d& p0, const Vec3d& p1, const Vec3d& p2, const Vec3d& p3)
{
    std::array<GEO::vec3, 4> geo_v;
    geo_v[0] = GEO::vec3(p0[0], p0[1], p0[2]);
    geo_v[1] = GEO::vec3(p1[0], p1[1], p1[2]);
    geo_v[2] = GEO::vec3(p2[0], p2[1], p2[2]);
    geo_v[3] = GEO::vec3(p3[0], p3[1], p3[2]);
    GEO::vec3 center = GEO::Geom::tetra_circum_center(geo_v[0], geo_v[1], geo_v[2], geo_v[3]);
    return GEO::distance2(center, geo_v[0]);
}

int tetra_validity(const std::vector<VertAttr>& vert_attrs, const Vec4i& t)
{
    return GEO::PCK::orient_3d(
        vert_attrs[t[0]].pos.data(),
        vert_attrs[t[1]].pos.data(),
        vert_attrs[t[2]].pos.data(),
        vert_attrs[t[3]].pos.data());
}

auto update_pc(
    PrismCage& pc,
    std::vector<int>& new_fid,
    std::vector<Vec3i>& new_conn,
    const std::vector<std::set<int>>& new_tracks)
{
    for (auto i = 0; i < new_fid.size(); i++) {
        auto f = new_fid[i];
        if (f >= pc.F.size()) {
            pc.F.resize(f + 1);
            pc.track_ref.resize(f + 1);
        }
        pc.F[f] = new_conn[i];
        pc.track_ref[f] = new_tracks[i];
    }
}

auto edge_adjacent_boundary_face = [](auto& tet_attrs, auto& vert_conn, int v0, int v1) {
    auto& nb0 = vert_conn[v0];
    auto& nb1 = vert_conn[v1];
    auto affected = set_inter(nb0, nb1); // removed
    std::vector<int> bnd_pris;
    for (auto t : affected) {
        for (auto j = 0; j < 4; j++) {
            if (tet_attrs[t].prism_id[j] != -1 // boundary
                && tet_attrs[t].conn[j] != v0 //  contains v0 AND v1
                && tet_attrs[t].conn[j] != v1) {
                bnd_pris.push_back(tet_attrs[t].prism_id[j]);
            }
        }
    }
    return bnd_pris;
};
auto replace = [](auto& arr, auto a, auto b) {
    for (auto i = 0; i < arr.size(); i++) {
        if (arr[i] == a) {
            arr[i] = b;
            return i;
        }
    }
    assert(false);
    return -1;
};

auto id_in_array = [](auto& v, auto& k) {
    for (auto i = 0; i < v.size(); i++) {
        if (v[i] == k) return i;
    }
    return -1;
};

void update_tetra_conn(
    const std::vector<VertAttr>& vert_attrs,
    std::vector<TetAttr>& tet_attrs,
    std::vector<std::vector<int>>& vert_conn,
    const std::vector<int>& affected,
    const std::vector<Vec4i>& new_tets,
    const std::vector<int>& modified_pids,
    const std::vector<Vec3i>& modified_tris)
{
    // update connectivity: VT
    auto n_tet = tet_attrs.size();
    vert_conn.resize(vert_attrs.size());

    // resetting prism_id.
    std::map<Vec3i, int> moved_pris_assigner;
    for (auto ti : affected) {
        tet_attrs[ti].is_removed = true;
        for (auto j = 0; j < 4; j++) {
            auto pid = tet_attrs[ti].prism_id[j];
            if (pid != -1) {
                auto face = Vec3i();
                for (auto k = 0; k < 3; k++)
                    face[k] = vert_attrs[tet_attrs[ti].conn[(j + k + 1) % 4]].mid_id;
                std::sort(face.begin(), face.end());
                assert(face.front() != -1);
                moved_pris_assigner.emplace(face, pid);
            }
        }
    }
    assert(modified_pids.size() == modified_tris.size());
    for (auto i = 0; i < modified_pids.size(); i++) {
        auto t = modified_tris[i];
        std::sort(t.begin(), t.end());
        moved_pris_assigner.emplace(t, modified_pids[i]);
    }
    spdlog::trace("sorted moved tris number {}", moved_pris_assigner);

    auto cnt_assigned_prisms = 0;
    for (auto t : new_tets) {
        for (auto i = 0; i < t.size(); i++) {
            auto diff = std::vector<int>();
            set_minus(vert_conn[t[i]], affected, diff);
            vert_conn[t[i]] = std::move(diff);

            set_insert(vert_conn[t[i]], n_tet);
            spdlog::trace("vert conn {}", vert_conn[t[i]]);
        }
        n_tet++;
        auto t_a = TetAttr();
        t_a.conn = t;

        // if face in t is a `moved_tris`, get the corresponding index (oppo vert).
        for (auto j = 0; j < 4; j++) {
            auto face = Vec3i();
            for (auto k = 0; k < 3; k++) face[k] = vert_attrs[t[(j + k + 1) % 4]].mid_id;
            std::sort(face.begin(), face.end());
            if (face.front() == -1) {
                continue;
            }
            // spdlog::trace("face {}", face);
            auto it = moved_pris_assigner.find(face);
            if (it != moved_pris_assigner.end()) {
                t_a.prism_id[j] = it->second;
                cnt_assigned_prisms++;
            }
        }

        tet_attrs.emplace_back(t_a);
    }

    spdlog::trace(
        "sorted moved tris number {} -> assigned {}",
        moved_pris_assigner.size(),
        cnt_assigned_prisms);
    // assert(cnt_assigned_prisms == sorted_moved_tris.size() && "All assigned");
    assert(n_tet == tet_attrs.size());
}


bool split_edge(
    PrismCage& pc,
    prism::local::RemeshOptions& option,
    std::vector<VertAttr>& vert_attrs,
    std::vector<TetAttr>& tet_attrs,
    std::vector<std::vector<int>>& vert_conn,
    int v0,
    int v1)
{
    auto& nb0 = vert_conn[v0];
    auto& nb1 = vert_conn[v1];
    auto affected = set_inter(nb0, nb1); // removed
    assert(!affected.empty());
    spdlog::info("Splitting...");

    std::vector<Vec4i> new_tets;
    std::vector<TetAttr> new_attrs;
    auto vx = vert_attrs.size();


    auto bnd_pris = edge_adjacent_boundary_face(tet_attrs, vert_conn, v0, v1);
    for (auto t : affected) {
        new_tets.push_back(tet_attrs[t].conn);
        new_attrs.push_back(tet_attrs[t]);
        replace(new_tets.back(), v0, vx);

        new_tets.push_back(tet_attrs[t].conn);
        new_attrs.push_back(tet_attrs[t]);
        replace(new_tets.back(), v1, vx);
    }

    vert_attrs.push_back({});
    vert_attrs.back().pos = ((vert_attrs[v0].pos + vert_attrs[v1].pos) / 2);
    auto rollback = [&]() {
        vert_attrs.pop_back();
        pc.top.pop_back();
        pc.base.pop_back();
        pc.mid.pop_back();
        return false;
    };

    auto p_vx = pc.mid.size();
    auto pv0 = vert_attrs[v0].mid_id, pv1 = vert_attrs[v1].mid_id;
    if (!bnd_pris.empty()) {
        spdlog::trace("Handling boundary edge with pris {}", bnd_pris);
        assert(bnd_pris.size() == 2);
        vert_attrs.back().mid_id = p_vx;
        assert(vert_attrs[v0].mid_id >= 0 && vert_attrs[v1].mid_id >= 0);
        pc.top.push_back((pc.top[pv0] + pc.top[pv1]) / 2);
        pc.base.push_back((pc.base[pv0] + pc.base[pv1]) / 2);
        pc.mid.push_back(vert_attrs.back().pos);
        option.target_adjustment.push_back(
            (option.target_adjustment[pv0] + option.target_adjustment[pv1]) / 2);
        spdlog::trace("pusher {} {}", pc.top.back(), pc.base.back());
    }

    for (auto t : new_tets) { // Vec4i
        if (!tetra_validity(vert_attrs, t)) return rollback();
    }

    std::vector<int> new_fid;
    std::vector<Vec3i> moved_tris;
    if (!bnd_pris.empty()) {
        auto pvx = pc.mid.size() - 1;
        assert(bnd_pris.size() == 2);
        auto& F = pc.F;
        auto f0 = bnd_pris.front(), f1 = bnd_pris.back();
        std::vector<int> old_fids = {f0, f1};
        moved_tris = std::vector<Vec3i>{F[f0], F[f1], F[f0], F[f1]};
        spdlog::trace("pvx {}", pvx);
        replace(moved_tris[0], pv0, pvx);
        replace(moved_tris[1], pv0, pvx);
        replace(moved_tris[2], pv1, pvx);
        replace(moved_tris[3], pv1, pvx);

        std::vector<std::set<int>> new_tracks;
        std::vector<RowMatd> local_cp;
        auto flag = prism::local_validity::attempt_zig_remesh(
            pc,
            pc.track_ref,
            option,
            -1,
            old_fids,
            moved_tris,
            new_tracks,
            local_cp);
        if (flag != 0) return rollback();

        // distribute and assign new_tracks.
        new_fid = {f0, f1, int(F.size()), int(F.size() + 1)};
        update_pc(pc, new_fid, moved_tris, new_tracks);
    }

    update_tetra_conn(vert_attrs, tet_attrs, vert_conn, affected, new_tets, new_fid, moved_tris);
    assert([&]() -> bool {
        auto vx = vert_conn.size() - 1;
        spdlog::trace("vx {}", vx);
        return true;
    }());
    return true;
}

bool smooth_vertex(
    PrismCage& pc,
    const prism::local::RemeshOptions& option,
    std::vector<VertAttr>& vert_attrs,
    const std::vector<TetAttr>& tet_attrs,
    const std::vector<std::vector<int>>& vert_conn,
    int v0)
{
    auto& nb = vert_conn[v0];
    const Vec3d old_pos = vert_attrs[v0].pos;
    auto neighbor_pris = std::vector<int>();
    const auto pv0 = vert_attrs[v0].mid_id;
    for (auto t : nb) {
        for (auto j = 0; j < 4; j++) {
            auto pid = tet_attrs[t].prism_id[j];
            if (pid != -1 && tet_attrs[t].conn[j] != v0) {
                neighbor_pris.push_back(pid);
            }
        }
    }
    auto get_snap_position = [&pc, &neighbor_pris, v0 = pv0]() {
        auto query = [&ref = pc.ref](
                         const Vec3d& s,
                         const Vec3d& t,
                         const std::set<int>& total_trackee) -> std::optional<Vec3d> {
            std::array<Vec3d, 2> seg_query{s, t};
            for (auto f : total_trackee) {
                auto v0 = ref.F(f, 0), v1 = ref.F(f, 1), v2 = ref.F(f, 2);
                auto mid_intersect = prism::cgal::segment_triangle_intersection(
                    seg_query,
                    {ref.V.row(v0), ref.V.row(v1), ref.V.row(v2)});
                if (mid_intersect) return mid_intersect;
            }
            return {};
        };
        std::set<int> total_trackee;
        for (auto f : neighbor_pris)
            total_trackee.insert(pc.track_ref[f].begin(), pc.track_ref[f].end());
        auto mid_intersect = query(pc.top[v0], pc.base[v0], total_trackee);
        assert(mid_intersect && "Snap project should succeed.");
        return mid_intersect.value();
    };

    Vec3d newton_pos = Vec3d::Zero();
    Vec3d old_mid = Vec3d::Zero();

    if (pv0 != -1) { // internal
        assert(!neighbor_pris.empty());
        newton_pos = get_snap_position();
        old_mid = pc.mid[pv0];
    } else {
        // newton_pos = get_newton_position();
        return false; // TODO: implement newton.
    }

    auto rollback = [&]() {
        vert_attrs[v0].pos = old_pos;
        if (pv0 != -1) pc.mid[pv0] = old_mid;
        return false;
    };

    vert_attrs[v0].pos = newton_pos;
    for (auto ti : nb) {
        auto& t = tet_attrs[ti].conn;
        if (!tetra_validity(vert_attrs, t)) {
            return rollback();
        }
    }

    if (pv0 != -1) { // boundary, attempt
        auto old_fids = std::vector<int>();
        for (auto t : nb) {
            auto& t_a = tet_attrs[t];
            for (auto j = 0; j < 4; j++) {
                if (t_a.prism_id[j] != -1 && t_a.conn[j] != v0) { // not opposite
                    old_fids.push_back(t_a.prism_id[j]);
                }
            }
        }
        assert(!old_fids.empty());
        auto moved_tris = std::vector<Vec3i>();
        for (auto f : old_fids) moved_tris.emplace_back(pc.F[f]);

        std::vector<std::set<int>> new_tracks;
        std::vector<RowMatd> local_cp;
        auto flag = prism::local_validity::attempt_zig_remesh(
            pc,
            pc.track_ref,
            option,
            1e10, // TODO: smoothing tet already captured energy, no guard here.
            old_fids,
            moved_tris,
            new_tracks,
            local_cp);
        if (flag != 0) return rollback();
        update_pc(pc, old_fids, moved_tris, new_tracks);
    }

    spdlog::trace("Vertex Snapped!!");
    // not modifying connectivity.
    return true;
}


bool swap_edge(
    const PrismCage& pc,
    const prism::local::RemeshOptions& option,
    std::vector<VertAttr>& vert_attrs,
    std::vector<TetAttr>& tet_attrs,
    std::vector<std::vector<int>>& vert_conn,
    int v1_id,
    int v2_id)
{
    // 3-2 edge to face.
    auto& nb1 = vert_conn[v1_id];
    auto& nb2 = vert_conn[v2_id];
    auto affected = set_inter(nb1, nb2);
    assert(!affected.empty());
    if (affected.size() != 3) {
        spdlog::trace("Affect Neighbor Too Many {} (>2)", affected.size());
        return false;
    }

    auto bnd_faces = edge_adjacent_boundary_face(tet_attrs, vert_conn, v1_id, v2_id);
    if (!bnd_faces.empty()) return false; // NOT handling boundary edges for now.

    auto new_tets = [&tet_attrs, v1_id, v2_id, &affected]() {
        auto t0_id = affected[0];
        auto t1_id = affected[1];
        auto t2_id = affected[2];
        auto n0_id = -1, n1_id = -1, n2_id = -1;
        for (int j = 0; j < 4; j++) {
            auto v0j = tet_attrs[t0_id].conn[j];
            if (v0j != v1_id && v0j != v2_id) {
                if (id_in_array(tet_attrs[t1_id].conn, v0j) != -1) n1_id = v0j;
                if (id_in_array(tet_attrs[t2_id].conn, v0j) != -1) n2_id = v0j;
            }
            if (id_in_array(tet_attrs[t0_id].conn, tet_attrs[t1_id].conn[j]) == -1)
                n0_id = tet_attrs[t1_id].conn[j];
        }
        assert(n0_id != n1_id && n1_id != n2_id);
        // T0 = (n1,n2,v1,v2) -> (n1,n2,v1,n0)
        // T1 = (n0, n1, v1,v2) ->  (n0, n1, n2,v2)
        // T2 = (n0,n2, v1,v2) -> (-1,-1,-1,-1)
        auto new_tets = std::vector<Vec4i>(2);
        auto new_tids = std::vector<int>({t0_id, t1_id});
        auto replace = [](auto& arr, auto v0, auto v1) {
            for (auto j = 0; j < arr.size(); j++)
                if (arr[j] == v0) arr[j] = v1;
        };
        new_tets[0] = tet_attrs[new_tids[0]].conn;
        new_tets[1] = tet_attrs[new_tids[1]].conn;

        replace(new_tets[0], v2_id, n0_id);
        replace(new_tets[1], v1_id, n2_id);
        return new_tets;
    }();

    for (auto t : new_tets) {
        if (!tetra_validity(vert_attrs, t)) return false;
    }

    update_tetra_conn(vert_attrs, tet_attrs, vert_conn, affected, new_tets, {}, {});
    return true;
}


// 2-3, internal only.
bool swap_face(
    const PrismCage& pc,
    const prism::local::RemeshOptions& option,
    std::vector<VertAttr>& vert_attrs,
    std::vector<TetAttr>& tet_attrs,
    std::vector<std::vector<int>>& vert_conn,
    int v0_id,
    int v1_id,
    int v2_id)
{
    auto& nb0 = vert_conn[v0_id];
    auto& nb1 = vert_conn[v1_id];
    auto& nb2 = vert_conn[v2_id];
    auto inter0_1 = set_inter(nb0, nb1);
    auto affected = set_inter(inter0_1, nb2);
    if (affected.size() != 2) { // has to be on boundary, or an invalid face input.
        assert(edge_adjacent_boundary_face(tet_attrs, vert_conn, v0_id, v1_id).size() > 0);
        return false;
    }


    // no top/bottom ordering of the two tets are assumed.
    auto t0 = affected.front(), t1 = affected.back();
    auto find_other = [](const Vec4i& tet, const Vec3i& tri) {
        for (auto j = 0; j < 4; j++) {
            auto ids = id_in_array(tri, tet[j]);
            if (ids == -1) {
                return tet[j];
            }
        }
        assert(false);
        return -1;
        // reorient
        // return true;
    };
    auto u0 = find_other(tet_attrs[t0].conn, {v0_id, v1_id, v2_id});
    auto u1 = find_other(tet_attrs[t1].conn, {v0_id, v1_id, v2_id});

    //
    auto new_tets = std::vector<Vec4i>{tet_attrs[t0].conn, tet_attrs[t0].conn, tet_attrs[t0].conn};
    replace(new_tets[0], v0_id, u1);
    replace(new_tets[1], v1_id, u1);
    replace(new_tets[2], v2_id, u1);

    for (auto t : new_tets) {
        if (!tetra_validity(vert_attrs, t)) return false;
    }

    update_tetra_conn(vert_attrs, tet_attrs, vert_conn, affected, new_tets, {}, {});
    return true;
}
} // namespace prism::tet