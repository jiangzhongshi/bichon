#include "tetra_utils.hpp"

#include <geogram/basic/geometry.h>
#include <geogram/numerics/predicates.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>
#include <prism/PrismCage.hpp>
#include <prism/geogram/AABB.hpp>
#include <prism/local_operations/remesh_pass.hpp>
#include <prism/local_operations/validity_checks.hpp>

auto set_inter = [](auto& A, auto& B) {
    std::vector<int> vec;
    std::set_intersection(A.begin(), A.end(), B.begin(), B.end(), std::back_inserter(vec));
    return vec;
};

auto set_insert = [](auto& A, auto& a) {
    auto it = std::lower_bound(A.begin(), A.end(), a, std::greater<int>());
    A.insert(it, a);
};


namespace prism::tet {
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


bool split_edge(
    PrismCage& pc,
    std::vector<VertAttr>& vert_attrs,
    std::vector<TetAttr>& tet_attrs,
    std::vector<std::vector<int>> vert_conn,
    int v0,
    int v1)
{
    prism::local::RemeshOptions option;
    auto& nb0 = vert_conn[v0];
    auto& nb1 = vert_conn[v1];
    auto affected = set_inter(nb0, nb1); // removed
    assert(!affected.empty());
    spdlog::info("Splitting...");

    std::vector<Vec4i> new_tets;
    std::vector<TetAttr> new_attrs;
    auto vx = vert_attrs.size();
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

    auto boundary_edge = false;
    std::vector<int> bnd_pris;
    for (auto t : affected) {
        for (auto j = 0; j < 4; j++) {
            if (tet_attrs[t].prism_id[j] != -1 // boundary
                && tet_attrs[t].conn[j] != v0 //  contains v0 AND v1
                && tet_attrs[t].conn[j] != v1) {
                boundary_edge = true;
                bnd_pris.push_back(tet_attrs[t].prism_id[j]);
            }
        }
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
    if (boundary_edge) {
        spdlog::trace("Handling boundary edge");
        assert(bnd_pris.size() == 2);
        vert_attrs.back().mid_id = p_vx;
        assert(vert_attrs[v0].mid_id >= 0 && vert_attrs[v1].mid_id >= 0);
        // vert_attrs.back().top = ((vert_attrs[v0].top + vert_attrs[v1].top) / 2);
        // vert_attrs.back().bot = ((vert_attrs[v0].bot + vert_attrs[v1].bot) / 2);
        pc.top.push_back((pc.top[v0] + pc.top[v1]) / 2);
        pc.base.push_back((pc.base[v0] + pc.base[v1]) / 2);
        pc.mid.push_back(vert_attrs.back().pos);
        spdlog::trace("pusher {} {}", pc.top.back(), pc.base.back());
    }


    for (auto t : new_tets) { // Vec4i
        if (!tetra_validity(vert_attrs, t)) return rollback();
    }

    std::vector<int> new_fid;
    std::vector<Vec3i> moved_tris;
    if (boundary_edge) {
        auto pvx = pc.mid.size() - 1;
        assert(bnd_pris.size() == 2);
        auto& F = pc.F;
        auto f0 = bnd_pris.front(), f1 = bnd_pris.back();
        std::vector<int> old_fids = {f0, f1};
        moved_tris = std::vector<Vec3i>{F[f0], F[f1], F[f0], F[f1]};
        spdlog::trace("pvx {}", pvx);
        replace(moved_tris[0], v0, pvx);
        replace(moved_tris[1], v0, pvx);
        replace(moved_tris[2], v1, pvx);
        replace(moved_tris[3], v1, pvx);

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

    for (auto ti : affected) {
        tet_attrs[ti].is_removed = true;
    }

    // update connectivity: VT TODO
    auto n_tet = tet_attrs.size();
    vert_conn.resize(vert_conn.size() + 1);
    for (auto t : new_tets) {
        for (auto i = 0; i < t.size(); i++) {
            auto diff = std::vector<int>();
            set_minus(vert_conn[t[i]], affected, diff);
            vert_conn[t[i]] = std::move(diff);

            set_insert(vert_conn[t[i]], n_tet);
        }
        n_tet++;
        auto t_a = TetAttr();
        t_a.conn = t;

        // if face in t is a `moved_tris`, get the corresponding index (oppo vert).
        for (auto j = 0; j < 4; j++) {
            auto face = Vec3i();
            for (auto k = 0; k < 3; k++) face[k] = t[(j + k + 1) % 4];
            std::sort(face.begin(), face.end());
            for (auto mi = 0; mi < moved_tris.size(); mi++) {
                if (face == moved_tris[mi]) {
                    t_a.prism_id[j] = new_fid[mi];
                }
            }
        }

        tet_attrs.emplace_back(t_a);
    }
    assert(n_tet == tet_attrs.size());
    // remove `affected`, append `new_tid`
    return true;
}
} // namespace prism::tet