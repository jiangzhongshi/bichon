#include "tetra_remesh_pass.hpp"

#include "prism/PrismCage.hpp"
#include "prism/local_operations/remesh_pass.hpp"
#include "tetra_logger.hpp"
#include "tetra_utils.hpp"

#include <highfive/H5Easy.hpp>
#include <queue>

namespace prism::tet {

auto size_constraint = [](const auto& vert_info, const auto& tet, auto& sizer) -> double {
    // return 0.1;
    std::array<Vec3d, 4> stack_pos;
    for (auto j = 0; j < 4; j++) {
        stack_pos[j] = vert_info[tet[j]].pos;
    }
    auto size_con = sizer->find_size_bound(stack_pos);
    return size_con;
};


std::priority_queue<std::tuple<double, int, int>> construct_collapse_queue(
    const prism::tet::vert_info_t& vert_info,
    const prism::tet::tet_info_t& tet_info)
{
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
std::priority_queue<std::tuple<double, int, int>> construct_edge_queue(
    const prism::tet::vert_info_t& vert_info,
    const prism::tet::tet_info_t& tet_info)
{
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
std::priority_queue<std::tuple<double, int, int, int>> construct_face_queue(
    const prism::tet::vert_info_t& vert_info,
    const prism::tet::tet_info_t& tet_info)
{
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

int faceswap_pass(
    PrismCage* pc,
    prism::local::RemeshOptions& option,
    prism::tet::vert_info_t& vert_info,
    prism::tet::tet_info_t& tet_info,
    std::vector<std::vector<int>>& vert_tet_conn,
    double sizing)
{
    auto face_queue = construct_face_queue(vert_info, tet_info);
    auto cnt = 0;
    prism::tet::logger().info("Face Swap: mesh size {} {}", vert_info.size(), tet_info.size());
    while (!face_queue.empty()) {
        auto [len, v0, v1, v2] = face_queue.top();
        face_queue.pop();
        auto flag = prism::tet::swap_face(
            pc,
            option,
            vert_info,
            tet_info,
            vert_tet_conn,
            v0,
            v1,
            v2,
            sizing);
        if (flag) cnt++;
    }
    prism::tet::logger().info("After Swap Size {} {}", vert_info.size(), tet_info.size());
    return cnt;
};

int edgeswap_pass(
    PrismCage* pc,
    prism::local::RemeshOptions& option,
    prism::tet::vert_info_t& vert_info,
    prism::tet::tet_info_t& tet_info,
    std::vector<std::vector<int>>& vert_tet_conn,
    double sizing)
{
    auto edge_queue = construct_edge_queue(vert_info, tet_info);
    prism::tet::logger().info("Edge Swap: queue size {}", edge_queue.size());
    auto cnt = 0;
    while (!edge_queue.empty()) {
        auto [len, v0, v1] = edge_queue.top();
        edge_queue.pop();
        auto bnd_faces = edge_adjacent_boundary_face(tet_info, vert_tet_conn, v0, v1);
        auto flag = false;
        if (bnd_faces.empty())
            flag = prism::tet::swap_edge(
                pc,
                option,
                vert_info,
                tet_info,
                vert_tet_conn,
                v0,
                v1,
                sizing);
        else {
            assert(bnd_faces.size() == 2);
            flag = prism::tet::flip_edge_sf(
                pc,
                option,
                vert_info,
                tet_info,
                vert_tet_conn,
                v0,
                v1,
                sizing);
        }

        if (flag) cnt++;
    }
    prism::tet::logger().info("After E-swap, size {} {}", vert_info.size(), tet_info.size());
    return cnt;
}

int edge_split_pass_for_dof(
    PrismCage* pc,
    prism::local::RemeshOptions& option,
    prism::tet::vert_info_t& vert_attrs,
    prism::tet::tet_info_t& tet_attrs,
    std::vector<std::vector<int>>& vert_conn)
{
    // pick the middle of minimum (3) with max-collapsing allowed.
    const auto bad_quality = (option.collapse_quality_threshold + 3) / 2;

    auto cnt = 0;
    std::vector<bool> quality_due(tet_attrs.size(), false);
    auto local_edges =
        std::array<std::array<int, 2>, 6>{{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}};

    std::set<std::pair<int, int>> all_edges;
    for (auto i = 0; i < tet_attrs.size(); i++) {
        if (tet_attrs[i].is_removed) continue;
        auto& tet = tet_attrs[i].conn;
        auto qual = prism::tet::tetra_quality(
            vert_attrs[tet[0]].pos,
            vert_attrs[tet[1]].pos,
            vert_attrs[tet[2]].pos,
            vert_attrs[tet[3]].pos);
        if (qual > bad_quality) {
            for (auto [i0, i1] : local_edges) {
                auto v0 = tet[i0], v1 = tet[i1];
                all_edges.emplace(std::min(v0, v1), std::max(v0, v1));
            }
            quality_due[i] = true;
        }
    }

    // push queue
    auto queue = std::priority_queue<std::tuple<double, int, int>>();
    for (auto [v0, v1] : all_edges) {
        auto len = (vert_attrs[v0].pos - vert_attrs[v1].pos).squaredNorm();
        queue.emplace(len, v0, v1);
    }

    while (!queue.empty()) {
        auto [len, v0, v1] = queue.top();
        queue.pop();
        if ((vert_attrs[v0].pos - vert_attrs[v1].pos).squaredNorm() != len) continue;

        assert(!set_inter(vert_conn[v0], vert_conn[v1]).empty());
        auto flag = prism::tet::split_edge(pc, option, vert_attrs, tet_attrs, vert_conn, v0, v1);
        if (flag) {
            cnt++;
        }
    }

    return cnt;
}

int collapse_pass(
    PrismCage* pc,
    prism::local::RemeshOptions& option,
    prism::tet::vert_info_t& vert_info,
    prism::tet::tet_info_t& tet_info,
    std::vector<std::vector<int>>& vert_tet_conn,
    const std::unique_ptr<prism::tet::SizeController>& sizer)
{
    auto edge_queue = construct_collapse_queue(vert_info, tet_info);
    prism::tet::logger().info("Edge Collapse: queue size {}", edge_queue.size());
    auto cnt = 0;
    auto previous = std::array<int, 2>{-1, -1};
    while (!edge_queue.empty()) {
        auto [len, v0, v1] = edge_queue.top();
        edge_queue.pop();
        // TODO: this does not prevent double testing-rejection. Slightly more cost.
        if (-(vert_info[v0].pos - vert_info[v1].pos).squaredNorm() != len) continue;
        if (previous[0] == v0 && previous[1] == v1) continue;
        {
            auto sizing2 = 1.0;
            auto& nb1 = vert_tet_conn[v0];
            auto& nb2 = vert_tet_conn[v1];
            auto affected = set_inter(nb1, nb2);
            if (affected.empty()) continue;
            for (auto t : affected) {
                auto t_size = size_constraint(vert_info, tet_info[t].conn, sizer);
                sizing2 = std::min(t_size, sizing2);
            }
            if (std::abs(len) >= 0.8 * 0.8 * sizing2) continue; // only collapse over-short edges.
            // spdlog::info("sizing {}, {}", len, sizing);
        }
        previous = {v0, v1};
        // erase v0
        auto suc =
            prism::tet::collapse_edge(pc, option, vert_info, tet_info, vert_tet_conn, v0, v1, 1.0);
        if (!suc) continue;
        cnt++;
        std::set<std::pair<int, int>> new_edges;
        for (auto t : vert_tet_conn[v1]) {
            for (auto j = 0; j < 4; j++) {
                auto vx = tet_info[t].conn[j];
                if (vx != v1) {
                    new_edges.emplace(v1, vx);
                    new_edges.emplace(vx, v1);
                }
            }
        }
        // push to queue
        for (auto [vi, vj] : new_edges) {
            auto len = (vert_info[vi].pos - vert_info[vj].pos).squaredNorm();
            edge_queue.emplace(-len, vi, vj);
        }

        if (cnt % 1000 == 0) {
            prism::tet::logger().info("Collapsed {}", cnt);
        }
    }
    prism::tet::logger().info("Size {} {}", vert_info.size(), tet_info.size());
    return cnt;
}


int vertexsmooth_pass(
    PrismCage* pc,
    prism::local::RemeshOptions& option,
    prism::tet::vert_info_t& vert_info,
    prism::tet::tet_info_t& tet_info,
    std::vector<std::vector<int>>& vert_tet_conn,
    double thick)
{
    std::vector<bool> snap_flag(pc->mid.size(), false);
    auto total_cnt = 0;
    for (auto v0 = 0; v0 < vert_info.size(); v0++) {
        auto& nb = vert_tet_conn[v0];
        if (nb.empty()) continue;
        total_cnt++;
        auto smooth_types =
            std::vector<prism::tet::SmoothType>{prism::tet::SmoothType::kInteriorNewton};
        if (vert_info[v0].mid_id != -1)
            smooth_types = {
                prism::tet::SmoothType::kShellPan,
                prism::tet::SmoothType::kShellRotate,
                prism::tet::SmoothType::kShellZoom,
                prism::tet::SmoothType::kSurfaceSnap};
        for (auto st : smooth_types) {
            option.target_thickness = thick;
            auto flag = prism::tet::smooth_vertex(
                pc,
                option,
                vert_info,
                tet_info,
                vert_tet_conn,
                st,
                v0,
                1.0);
            if (flag && (st == prism::tet::SmoothType::kSurfaceSnap ||
                         st == prism::tet::SmoothType::kShellPan)) {
                snap_flag[vert_info[v0].mid_id] = true;
            }
        }
    }
    auto cnt_snap = 0;
    for (auto f : snap_flag) {
        if (f) cnt_snap++;
    }
    prism::tet::logger().info("Snapped {} / {}", cnt_snap, total_cnt);
    return cnt_snap;
}


void serializer(
    std::string filename,
    const PrismCage* pc,
    const prism::tet::vert_info_t& vert_info,
    const prism::tet::tet_info_t& tet_info)
{
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

    auto saver = std::function<void(HighFive::File&)>(
        [&vert_info = vert_info, &tet_info = tet_info, &convert_to_VT](HighFive::File& file) {
            RowMatd V;
            Eigen::VectorXi V_pid;
            RowMati T;
            std::tie(V, T, V_pid) = convert_to_VT(vert_info, tet_info);
            Eigen::VectorXd S(T.rows());
            auto Q = S;
            for (auto i = 0; i < T.rows(); i++) {
                S[i] = prism::tet::diameter2(
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
            prism::tet::logger().info("Saving V {} T {}", V.rows(), T.rows());
            H5Easy::dump(file, "tet_v", V);
            H5Easy::dump(file, "tet_t", T);
            H5Easy::dump(file, "tet_v_pid", V_pid);
            H5Easy::dump(file, "tet_size", S);
            H5Easy::dump(file, "tet_qual", Q);
        });
    if (pc != nullptr) {
        pc->serialize(filename, saver);
    } else {
        auto file = H5Easy::File(filename, H5Easy::File::Overwrite);
        saver(file);
    }
}


void edge_split_pass_with_sizer(
    PrismCage* pc,
    prism::local::RemeshOptions& option,
    prism::tet::vert_info_t& vert_info,
    prism::tet::tet_info_t& tet_info,
    std::vector<std::vector<int>>& vert_tet_conn,
    const std::unique_ptr<prism::tet::SizeController>& sizer)
{
    // split_pass();
    std::vector<double> split_due(tet_info.size(), 1.);
    for (auto i = 0; i < tet_info.size(); i++) {
        if (tet_info[i].is_removed) continue;
        split_due[i] = size_constraint(vert_info, tet_info[i].conn, sizer);
    }
    assert(split_due.size() == tet_info.size());

    auto queue = construct_edge_queue(vert_info, tet_info);
    auto timestamp = 0;
    assert(prism::tet::tetmesh_sanity(tet_info, vert_info, vert_tet_conn, pc));
    auto minimum_edge = [&]() {
        auto mini = 1.0;
        auto local_edges =
            std::array<std::array<int, 2>, 6>{{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}};
        for (auto i = 0; i < tet_info.size(); i++) {
            auto conn = tet_info[i].conn;
            for (auto [v0, v1] : local_edges) {
                mini = std::min(
                    mini,
                    (vert_info[conn[v0]].pos - vert_info[conn[v1]].pos).squaredNorm());
            }
        }
        return mini;
    };
    while (!queue.empty()) {
        auto [len2, v0, v1] = queue.top();
        queue.pop();
        if ((vert_info[v0].pos - vert_info[v1].pos).squaredNorm() != len2) continue;
        auto nb = set_inter(vert_tet_conn[v0], vert_tet_conn[v1]);
        if (nb.empty()) continue;
        auto size_due = [&nb, &len2 = len2, &split_due]() {
            for (auto s : nb) {
                if (split_due[s] < len2 * (16 / 9.)) return true;
            }
            return false;
        }();

        if (!size_due) continue;
        auto old_tid = tet_info.size();
        ////// edge splitting
        // prism::tet::logger().trace(
        //     "[{}] len {} v{}-{} #T{} #F{}: MIN {}",
        //     timestamp,
        //     len,
        //     v0,
        //     v1,
        //     tet_info.size(),
        //     pc->F.size(),
        //     minimum_edge());
        auto suc = prism::tet::split_edge(pc, option, vert_info, tet_info, vert_tet_conn, v0, v1);
        if (!suc) continue;
        auto new_tid = std::vector<int>();
        for (auto i = old_tid; i < tet_info.size(); i++) {
            new_tid.push_back(i);
        }
        split_due.resize(tet_info.size(), 1.0);

        auto local_edges =
            std::array<std::array<int, 2>, 6>{{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}};
        auto edge_set = std::set<std::tuple<int, int>>();
        for (auto i : new_tid) {
            auto& tet = tet_info[i];
            auto size_con = (size_constraint(vert_info, tet_info[i].conn, sizer));
            split_due[i] = size_con;
        }
        for (auto i : new_tid) {
            auto& tet = tet_info[i];
            for (auto e : local_edges) {
                auto v0 = tet.conn[e[0]], v1 = tet.conn[e[1]];
                auto len2 = (vert_info[v0].pos - vert_info[v1].pos).squaredNorm();
                if (len2 > split_due[i]) edge_set.emplace(std::min(v0, v1), std::max(v0, v1));
            }
        }
        for (auto [v0, v1] : edge_set) {
            auto len2 = (vert_info[v0].pos - vert_info[v1].pos).squaredNorm();
            queue.emplace(len2, v0, v1);
        }
        timestamp++;
        if (timestamp % 1000 == 0) {
            prism::tet::logger().info("Split Going {}", timestamp);
        }
    }
}


} // namespace prism::tet