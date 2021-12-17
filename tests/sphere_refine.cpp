#include <doctest.h>
#include <geogram/basic/geometry.h>
#include <igl/Timer.h>
#include <igl/avg_edge_length.h>
#include <igl/barycenter.h>
#include <igl/combine.h>
#include <igl/read_triangle_mesh.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <batm/tetra_utils.hpp>
#include <highfive/H5Easy.hpp>
#include <iterator>
#include <memory>
#include <numeric>
#include <prism/common.hpp>
#include <prism/geogram/AABB.hpp>
#include <prism/geogram/geogram_utils.hpp>
#include <prism/local_operations/remesh_pass.hpp>
#include <prism/spatial-hash/AABB_hash.hpp>
#include <prism/spatial-hash/self_intersection.hpp>
#include <queue>
#include <tuple>
#include "prism/PrismCage.hpp"
#include "spdlog/common.h"

auto reload = [](std::string filename, auto& pc) {
    H5Easy::File file(filename, H5Easy::File::ReadOnly);
    auto tet_v = H5Easy::load<RowMatd>(file, "tet_v");
    auto tet_t = H5Easy::load<RowMati>(file, "tet_t");
    Eigen::VectorXi tet_v_pid = -Eigen::VectorXi::Ones(tet_v.rows());
    if (file.exist("tet_v_pid")) {
        tet_v_pid = H5Easy::load<Eigen::VectorXi>(file, "tet_v_pid");
    } else {
        for (auto i = 0; i < pc.mid.size(); i++) tet_v_pid[i] = i;
        spdlog::info("Initial Loading, no vid pointer");
    }
    spdlog::info("Loading v {}, t {} ", tet_v.rows(), tet_t.rows());
    return prism::tet::prepare_tet_info(pc, tet_v, tet_t, tet_v_pid);
};


auto set_inter = [](auto& A, auto& B) {
    std::vector<int> vec;
    std::set_intersection(A.begin(), A.end(), B.begin(), B.end(), std::back_inserter(vec));
    return vec;
};


auto construct_collapse_queue = [](const auto& vert_info, const auto& tet_info) {
    auto local_edges =
        std::array<std::array<int, 2>, 6>{{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}};
    auto edge_set = std::set<std::tuple<int, int>>();
    for (auto tet : tet_info) {
        if (tet.is_removed) continue;
        for (auto e : local_edges) {
            auto v0 = tet.conn[e[0]], v1 = tet.conn[e[1]];
            // if (v0 > v1) continue;
            edge_set.emplace(v0, v1);
        }
    }
    auto edge_queue = std::priority_queue<std::tuple<double, int, int>>(); // max queue, but we want
                                                                           // smaller first.
    for (auto [v0, v1] : edge_set) {
        auto len = (vert_info[v0].pos - vert_info[v1].pos).squaredNorm();
        edge_queue.emplace(-len, v0, v1);
    }
    return edge_queue;
};


// iterate all edges
auto construct_edge_queue = [](const auto& vert_info, const auto& tet_info) {
    auto local_edges =
        std::array<std::array<int, 2>, 6>{{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}};
    auto edge_set = std::set<std::tuple<int, int>>();
    for (auto tet : tet_info) {
        if (tet.is_removed) continue;
        for (auto e : local_edges) {
            auto v0 = tet.conn[e[0]], v1 = tet.conn[e[1]];
            if (v0 > v1) continue;
            edge_set.emplace(v0, v1);
        }
    }
    auto edge_queue = std::priority_queue<std::tuple<double, int, int>>();
    for (auto [v0, v1] : edge_set) {
        auto len = (vert_info[v0].pos - vert_info[v1].pos).squaredNorm();
        edge_queue.emplace(len, v0, v1);
    }
    return edge_queue;
};

// iterate all edges
auto construct_face_queue = [](const auto& vert_info, const auto& tet_info) {
    auto local_faces =
        std::array<std::array<int, 3>, 4>{{{0, 1, 2}, {1, 2, 3}, {2, 3, 0}, {3, 0, 1}}};
    auto face_queue = std::priority_queue<std::tuple<double, int, int, int>>();
    auto face_set = std::set<std::array<int, 3>>();
    for (auto tet : tet_info) {
        if (tet.is_removed) continue;
        for (auto e : local_faces) {
            auto tri = std::array<int, 3>{tet.conn[e[0]], tet.conn[e[1]], tet.conn[e[2]]};
            std::sort(tri.begin(), tri.end());
            face_set.insert(tri);
        }
    }
    for (auto f : face_set) {
        face_queue.emplace(0, f[0], f[1], f[2]);
    }
    return face_queue;
};

auto faceswap_pass = [](auto& pc,
                        auto& option,
                        auto& vert_info,
                        auto& tet_info,
                        auto& vert_tet_conn,
                        auto sizing) {
    auto face_queue = construct_face_queue(vert_info, tet_info);
    spdlog::info("Face Swap: queue size {}", face_queue.size());
    spdlog::info("Size {} {}", vert_info.size(), tet_info.size());
    while (!face_queue.empty()) {
        auto [len, v0, v1, v2] = face_queue.top();
        face_queue.pop();
        prism::tet::swap_face(pc, option, vert_info, tet_info, vert_tet_conn, v0, v1, v2, sizing);
    }
    spdlog::info("Size {} {}", vert_info.size(), tet_info.size());
};
auto edgeswap_pass =
    [](auto& pc, auto& option, auto& vert_info, auto& tet_info, auto& vert_tet_conn, auto sizing) {
        auto edge_queue = construct_edge_queue(vert_info, tet_info);
        spdlog::info("Edge Swap: queue size {}", edge_queue.size());
        while (!edge_queue.empty()) {
            auto [len, v0, v1] = edge_queue.top();
            edge_queue.pop();
            prism::tet::swap_edge(pc, option, vert_info, tet_info, vert_tet_conn, v0, v1, sizing);
        }
        spdlog::info("Size {} {}", vert_info.size(), tet_info.size());
    };
auto collapse_pass = [](auto& pc,
                        auto& option,
                        auto& vert_info,
                        auto& tet_info,
                        auto& vert_tet_conn,
                        auto sizing) {
    auto edge_queue = construct_collapse_queue(vert_info, tet_info);
    spdlog::info("Edge Collapse: queue size {}", edge_queue.size());
    while (!edge_queue.empty()) {
        auto [len, v0, v1] = edge_queue.top();
        edge_queue.pop();
        {
            auto& nb1 = vert_tet_conn[v0];
            auto& nb2 = vert_tet_conn[v1];
            auto affected = set_inter(nb1, nb2);
            if (affected.empty()) continue;
        }
        prism::tet::collapse_edge(pc, option, vert_info, tet_info, vert_tet_conn, v0, v1, sizing);
    }
    spdlog::info("Size {} {}", vert_info.size(), tet_info.size());
};

auto vertexsmooth_pass =
    [](auto& pc, auto& option, auto& vert_info, auto& tet_info, auto& vert_tet_conn, auto sizing) {
        std::vector<bool> snap_flag(pc.mid.size(), false);
        auto total_cnt = 0;
        {
            for (auto v0 = 0; v0 < vert_info.size(); v0++) {
                auto& nb = vert_tet_conn[v0];
                if (nb.empty()) continue;
                total_cnt++;
                auto flag = prism::tet::smooth_vertex(
                    pc,
                    option,
                    vert_info,
                    tet_info,
                    vert_tet_conn,
                    v0,
                    sizing);
                if (vert_info[v0].mid_id != -1) {
                    if (flag) snap_flag[vert_info[v0].mid_id] = true;
                }
            }
        }
        auto cnt_snap = 0;
        for (auto f : snap_flag) {
            if (f) cnt_snap++;
        }
        spdlog::info("Snapped {} / {}", cnt_snap, total_cnt);
    };


auto serializer = [](std::string filename, auto& pc, auto& vert_info, auto& tet_info) {
    auto convert_to_VT = [](auto& vert_info, auto& tet_info) {
        RowMatd V(vert_info.size(), 3);
        RowMati T(tet_info.size(), 4);
        Eigen::VectorXi V_pid(V.rows());
        for (auto i = 0; i < vert_info.size(); i++) {
            V.row(i) = vert_info[i].pos;
            V_pid[i] = vert_info[i].mid_id;
        }
        auto actual_tets = 0;
        for (auto& t : tet_info) {
            if (t.is_removed) continue;
            auto& tet = t.conn;
            T.row(actual_tets) << tet[0], tet[1], tet[2], tet[3];
            actual_tets++;
        }
        T.conservativeResize(actual_tets, 4);
        return std::tuple(V, T, V_pid);
    };

    pc.serialize(
        filename,
        std::function<void(HighFive::File&)>(
            [&vert_info = vert_info, &tet_info = tet_info, &convert_to_VT](HighFive::File& file) {
                RowMatd V;
                Eigen::VectorXi V_pid;
                RowMati T;
                std::tie(V, T, V_pid) = convert_to_VT(vert_info, tet_info);
                Eigen::VectorXd S(T.rows());
                auto Q = S;
                for (auto i = 0; i < T.rows(); i++) {
                    S[i] = prism::tet::circumradi2(
                        V.row(T(i, 0)),
                        V.row(T(i, 1)),
                        V.row(T(i, 2)),
                        V.row(T(i, 3)));
                    Q[i] = prism::tet::tetra_quality(
                        V.row(T(i, 0)),
                        V.row(T(i, 1)),
                        V.row(T(i, 2)),
                        V.row(T(i, 3)));
                }
                spdlog::info("Saving V {} T {}", V.rows(), T.rows());
                H5Easy::dump(file, "tet_v", V);
                H5Easy::dump(file, "tet_t", T);
                H5Easy::dump(file, "tet_v_pid", V_pid);
                H5Easy::dump(file, "tet_size", S);
                H5Easy::dump(file, "tet_qual", Q);
            }));
};


TEST_CASE("amr-sphere-prepare")
{
    using namespace prism::tet;
    std::string filename = "../tests/data/sphere_40.obj.h5";
    PrismCage pc(filename);
    spdlog::info("Shell size v{}, f{}", pc.base.size(), pc.F.size());
    auto [vert_info, tet_info, vert_tet_conn] = reload("../tests/data/sphere_40.obj.h5", pc);


    spdlog::set_level(spdlog::level::trace);


    prism::local::RemeshOptions option(pc.mid.size(), 0.1);
    auto check_sorted = [&vert_tet_conn = vert_tet_conn]() -> bool {
        for (auto& arr : vert_tet_conn)
            for (auto i = 0; i < arr.size() - 1; i++)
                if (arr[i] > arr[i + 1]) return false;
        return true;
    };
    assert(check_sorted());
    split_edge(pc, option, vert_info, tet_info, vert_tet_conn, 0, 1);
    spdlog::info("Size {} {}", vert_info.size(), tet_info.size());
    assert(check_sorted());
    assert(check_sorted());
    // smooth_vertex(pc, option, vert_info, tet_info, vert_tet_conn, 46);
    // spdlog::info("Size {} {}", vert_info.size(), tet_info.size());
}

TEST_CASE("sphere-tet-swap")
{
    std::string filename = "../tests/data/sphere_40.obj.h5";
    PrismCage pc(filename);
    auto [vert_info, tet_info, vert_tet_conn] = reload("../tests/data/sphere_40.obj.h5", pc);
    prism::local::RemeshOptions option(pc.mid.size(), 0.1);


    auto edge_queue = construct_edge_queue(vert_info, tet_info);
    auto face_queue = construct_face_queue(vert_info, tet_info);
    spdlog::info("edge queue size {}", edge_queue.size());
    spdlog::info("face queue size {}", face_queue.size());

    // spdlog::set_level(spdlog::level::trace);
    spdlog::info("Size {} {}", vert_info.size(), tet_info.size());
    while (!face_queue.empty()) {
        auto [len, v0, v1, v2] = face_queue.top();
        face_queue.pop();
        prism::tet::swap_face(pc, option, vert_info, tet_info, vert_tet_conn, v0, v1, v2, 10);
    }
    spdlog::info("Size {} {}", vert_info.size(), tet_info.size());
    for (auto t : tet_info) {
        if (!t.is_removed) CHECK(prism::tet::tetra_validity(vert_info, t.conn));
    }
}

TEST_CASE("split-pass")
{
    std::string filename = "../tests/data/sphere_40.obj.h5";
    PrismCage pc(filename);
    auto [vert_info, tet_info, vert_tet_conn] = reload("../tests/data/sphere_40.obj.h5", pc);
    prism::local::RemeshOptions option(pc.mid.size(), 0.1);

    // sizing field always 1e-2
    double sizing = 1e-2;
    auto tet_sizeof = [](auto& vert_info, auto& t) {
        return prism::tet::circumradi2(
            vert_info[t[0]].pos,
            vert_info[t[1]].pos,
            vert_info[t[2]].pos,
            vert_info[t[3]].pos);
    };
    auto tet_marker = std::vector<bool>(tet_info.size(), false);
    for (auto i = 0; i < tet_info.size(); i++) {
        auto r = tet_sizeof(vert_info, tet_info[i].conn);
        if (r > sizing) tet_marker[i] = true;
    }

    auto local_edges =
        std::array<std::array<int, 2>, 6>{{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}};
    auto construct_split_queue =
        [&local_edges](const auto& vert_info, const auto& tet_info, const auto& marker) {
            assert(marker.size() == tet_info.size());
            auto edge_set = std::set<std::tuple<int, int>>();
            for (auto i = 0; i < marker.size(); i++) {
                if (marker[i] == false) continue;
                auto& tet = tet_info[i];
                for (auto e : local_edges) {
                    auto v0 = tet.conn[e[0]], v1 = tet.conn[e[1]];
                    if (v0 > v1) continue;
                    edge_set.emplace(v0, v1);
                }
            }
            auto edge_queue = std::priority_queue<std::tuple<double, int, int>>(); // max queue
            for (auto [v0, v1] : edge_set) {
                auto len = (vert_info[v0].pos - vert_info[v1].pos).squaredNorm();
                edge_queue.emplace(len, v0, v1);
            }
            return edge_queue;
        };

    auto edge_queue = construct_split_queue(vert_info, tet_info, tet_marker);
    REQUIRE_FALSE(edge_queue.empty());
    auto all_distinct_verts = [&vert_info = vert_info]() -> bool {
        std::set<std::array<__int128, 3>> verts;
        for (auto v : vert_info)
            verts.emplace(std::array<__int128, 3>{
                static_cast<__int128>(v.pos[0] * 1e20),
                static_cast<__int128>(v.pos[1] * 1e20),
                static_cast<__int128>(v.pos[2] * 1e20)});
        return verts.size() == vert_info.size();
    };
    auto smallest_pairwise = [&vert_info = vert_info]() -> double {
        auto min_gap = 1e3; // HACK: should be limit, but we are already unit.
        for (auto i = 0; i < vert_info.size(); i++) {
            for (auto j = i + 1; j < vert_info.size(); j++) {
                auto gap = (vert_info[i].pos - vert_info[j].pos).squaredNorm();
                min_gap = std::min(gap, min_gap);
            }
        }
        return min_gap;
    };
    spdlog::enable_backtrace(50);
    auto timestamp = 0;
    // TODO: timestamp.
    // TODO: use tet marker here.
    while (!edge_queue.empty()) {
        auto [len, v0, v1] = edge_queue.top();
        spdlog::trace("Edge Queue {}", edge_queue.size());
        edge_queue.pop();
        // v0,v1 might been outdated?

        auto& nb0 = vert_tet_conn[v0];
        auto& nb1 = vert_tet_conn[v1];
        auto affected = set_inter(nb0, nb1); // removed
        if (affected.empty()) {
            spdlog::trace("outdated edge {} v {}-{}", len, v0, v1);
            continue;
        }

        // spdlog::info("len {}, gap {}", len, smallest_pairwise());
        auto oversize =
            [&affected, &sizing, &tet_sizeof, &vert_info = vert_info, &tet_info = tet_info]() {
                for (auto t : affected) {
                    if (tet_info[t].is_removed) assert(false);
                    auto r = tet_sizeof(vert_info, tet_info[t].conn);
                    if (r > sizing) return true;
                }
                return false;
            }();
        if (!oversize) {
            spdlog::trace("size ok {} v {}-{}", len, v0, v1);
            continue;
        }
        auto old_tet_cnt = tet_info.size();
        auto flag = prism::tet::split_edge(pc, option, vert_info, tet_info, vert_tet_conn, v0, v1);
        if (flag) {
            spdlog::trace("Success len {}, v {}-{}", len, v0, v1);
        } else {
            spdlog::info("Fail {} v {}-{}", len, v0, v1);
        }

        auto new_tet_ids = std::vector<int>{}; // HACK: best to return this info from split edge.
        for (auto i = old_tet_cnt; i < tet_info.size(); i++) {
            new_tet_ids.push_back(i);
        }
        for (auto t : new_tet_ids) {
            assert(tet_info[t].is_removed == false);

            auto r = tet_sizeof(vert_info, tet_info[t].conn);
            if (r > sizing) // add new edges only if tetra is large.
                for (auto e : local_edges) {
                    auto v0 = tet_info[t].conn[e[0]], v1 = tet_info[t].conn[e[1]];
                    auto len = (vert_info[v0].pos - vert_info[v1].pos).squaredNorm();
                    edge_queue.emplace(len, v0, v1);
                }
        }

        // spdlog::info("ts {} gap {}", timestamp, smallest_pairwise());

        if (!all_distinct_verts()) {
            spdlog::dump_backtrace();
            assert(false);
            return;
        }
        timestamp++;
    }

    std::vector<int> remains;
    for (auto i = 0; i < tet_info.size(); i++) {
        if (tet_info[i].is_removed) continue;
        auto r = tet_sizeof(vert_info, tet_info[i].conn);
        if (r > sizing) remains.push_back(i);
    }
    spdlog::info("Remain at large {}", remains);

    REQUIRE(prism::tet::tetmesh_sanity(tet_info, vert_info, vert_tet_conn, pc));
    serializer("../buildr/debug0.h5", pc, vert_info, tet_info);
}


TEST_CASE("reload-swap")
{
    std::string filename = "../buildr/debug0.h5";
    PrismCage pc(filename);
    auto [vert_info, tet_info, vert_tet_conn] = reload(filename, pc);

    prism::local::RemeshOptions option(pc.mid.size(), 0.1);
    double sizing = 1e-2;
    if (true) faceswap_pass(pc, option, vert_info, tet_info, vert_tet_conn, sizing);
    if (true) edgeswap_pass(pc, option, vert_info, tet_info, vert_tet_conn, sizing);
    if (true) collapse_pass(pc, option, vert_info, tet_info, vert_tet_conn, sizing);

    serializer("../buildr/debug1.h5", pc, vert_info, tet_info);

    serializer("../buildr/debug2.h5", pc, vert_info, tet_info);
}


TEST_CASE("sphere-coarsen")
{
    std::string filename = "../buildr/debug0.h5";
    PrismCage pc(filename);
    auto [vert_info, tet_info, vert_tet_conn] = reload(filename, pc);

    prism::local::RemeshOptions option(pc.mid.size(), 0.1);

    auto count_face_marks = [&tet_info = tet_info]() {
        auto cnt = 0;
        for (auto i = 0; i < tet_info.size(); i++) {
            if (tet_info[i].is_removed) continue;
            for (auto j = 0; j < 4; j++) {
                if (tet_info[i].prism_id[j] != -1) cnt++;
            }
        }
        return cnt;
    };
    auto count_pc_faces = [](const auto& pc) {
        auto cnt = 0;
        for (auto f : pc.F) {
            if (f[0] == -1) continue;
            cnt++;
        }
        return cnt;
    };
    auto sizing = 1.;
    option.collapse_quality_threshold = 15;
    REQUIRE_EQ(count_face_marks(), count_pc_faces(pc));
    collapse_pass(pc, option, vert_info, tet_info, vert_tet_conn, sizing);
    REQUIRE_EQ(count_face_marks(), count_pc_faces(pc));
}


TEST_CASE("sphere-coarsen-aggresive")
{
    std::string filename = "../buildr/debug0.h5";
    PrismCage pc(filename);
    auto [vert_info, tet_info, vert_tet_conn] = reload(filename, pc);

    prism::local::RemeshOptions option(pc.mid.size(), 0.1);
    auto sizing = 1e2;
    option.collapse_quality_threshold = 150;
    //  spdlog::set_level(spdlog::level::trace);
    REQUIRE(prism::tet::tetmesh_sanity(tet_info, vert_info, vert_tet_conn, pc));
    spdlog::enable_backtrace(100);
    for (auto i = 0; i < 5; i++) {
        vertexsmooth_pass(pc, option, vert_info, tet_info, vert_tet_conn, sizing);
        faceswap_pass(pc, option, vert_info, tet_info, vert_tet_conn, sizing);
        prism::tet::compact_tetmesh(vert_info, tet_info, vert_tet_conn, &pc);
        option.target_adjustment.resize(pc.mid.size());
        collapse_pass(pc, option, vert_info, tet_info, vert_tet_conn, sizing);
        edgeswap_pass(pc, option, vert_info, tet_info, vert_tet_conn, sizing);
        collapse_pass(pc, option, vert_info, tet_info, vert_tet_conn, sizing);
        vertexsmooth_pass(pc, option, vert_info, tet_info, vert_tet_conn, sizing);
        prism::tet::compact_tetmesh(vert_info, tet_info, vert_tet_conn, &pc);
        option.target_adjustment.resize(pc.mid.size());
    }
    serializer("../buildr/coarse.h5", pc, vert_info, tet_info);
}

auto edge_split_pass_with_sizer =
    [](auto& pc, auto& option, auto& vert_info, auto& tet_info, auto& vert_tet_conn, auto& sizer) {
        auto oversize = [](const auto& vert_info, const auto& tet, const auto& sizer) {
            std::array<Vec3d, 4> stack_pos;
            for (auto j = 0; j < 4; j++) {
                stack_pos[j] = vert_info[tet[j]].pos;
            }
            auto cur_size =
                prism::tet::circumradi2(stack_pos[0], stack_pos[1], stack_pos[2], stack_pos[3]);
            auto size_con = sizer->find_size_bound(stack_pos);
            return size_con < cur_size;
        };

        // split_pass();
        std::vector<bool> split_due(tet_info.size(), false);

        for (auto i = 0; i < tet_info.size(); i++) {
            if (tet_info[i].is_removed) continue;
            if (oversize(vert_info, tet_info[i].conn, sizer)) split_due[i] = true;
        }
        assert(split_due.size() == tet_info.size());

        auto queue = construct_edge_queue(vert_info, tet_info);
        auto timestamp = 0;
        REQUIRE(prism::tet::tetmesh_sanity(tet_info, vert_info, vert_tet_conn, pc));
        while (!queue.empty()) {
            auto [len, v0, v1] = queue.top();
            queue.pop();
            if ((vert_info[v0].pos - vert_info[v1].pos).squaredNorm() != len) continue;
            auto nb = set_inter(vert_tet_conn[v0], vert_tet_conn[v1]);
            if (nb.empty()) continue;
            auto size_due = [&nb, &split_due]() {
                for (auto s : nb) {
                    if (split_due[s]) return true;
                }
                return false;
            }();

            if (!size_due) continue;
            auto old_tid = tet_info.size();
            ////// edge splitting
            prism::tet::split_edge(pc, option, vert_info, tet_info, vert_tet_conn, v0, v1);
            auto new_tid = std::vector<int>();
            for (auto i = old_tid; i < tet_info.size(); i++) {
                new_tid.push_back(i);
            }
            split_due.resize(tet_info.size(), false);

            auto local_edges =
                std::array<std::array<int, 2>, 6>{{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}};
            auto edge_set = std::set<std::tuple<int, int>>();
            for (auto i : new_tid) {
                auto& tet = tet_info[i];
                for (auto e : local_edges) {
                    auto v0 = tet.conn[e[0]], v1 = tet.conn[e[1]];
                    edge_set.emplace(std::min(v0, v1), std::max(v0, v1));
                }
                if (oversize(vert_info, tet_info[i].conn, sizer)) split_due[i] = true;
            }
            for (auto [v0, v1] : edge_set) {
                auto len = (vert_info[v0].pos - vert_info[v1].pos).squaredNorm();
                queue.emplace(len, v0, v1);
            }
            timestamp++;
            if (timestamp % 1000 == 0) {
                spdlog::info("Split Going {}", timestamp);
            }
        }
    };
TEST_CASE("graded-sphere")
{
    std::string filename = "../buildr/coarse.h5";
    PrismCage pc(filename);
    auto [vert_info, tet_info, vert_tet_conn] = reload(filename, pc);
    prism::local::RemeshOptions option(pc.mid.size(), 0.1);

    auto sizer = std::unique_ptr<prism::tet::SizeController>(nullptr);
    {
        H5Easy::File file("../tests/data/cube_tetra_10.h5", H5Easy::File::ReadOnly);
        auto bgV = H5Easy::load<RowMatd>(file, "V");
        auto bgT = H5Easy::load<RowMati>(file, "T");
        Eigen::VectorXd sizes(bgT.rows());
        sizes.setOnes();
        // assign size
        RowMatd BC;
        igl::barycenter(bgV, bgT, BC);
        for (auto i = 0; i < bgT.rows(); i++) {
            if (BC(i, 0) > 0.5) sizes[i] = 1.0;
            if (BC(i, 0) < 0.2) sizes[i] = 5e-3;
        }
        sizer.reset(new prism::tet::SizeController(bgV, bgT, sizes));
    }

    edge_split_pass_with_sizer(pc, option, vert_info, tet_info, vert_tet_conn, sizer);

    serializer("../buildr/split_once.h5", pc, vert_info, tet_info);
}