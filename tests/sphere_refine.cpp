#include <doctest.h>
#include <geogram/basic/geometry.h>
#include <igl/Timer.h>
#include <igl/avg_edge_length.h>
#include <igl/combine.h>
#include <igl/read_triangle_mesh.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <batm/tetra_utils.hpp>
#include <highfive/H5Easy.hpp>
#include <iterator>
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

auto prepare = [](auto& pc) {
    std::string filename = "../tests/data/sphere_40.obj.h5";
    H5Easy::File file(filename, H5Easy::File::ReadOnly);
    auto tet_v = H5Easy::load<RowMatd>(file, "tet_v");
    auto tet_t = H5Easy::load<RowMati>(file, "tet_t");
    spdlog::info("Loading v {},t {} ", tet_v.rows(), tet_t.rows());
    return prism::tet::prepare_tet_info(pc, tet_v, tet_t);
};

TEST_CASE("amr-sphere-prepare")
{
    using namespace prism::tet;
    std::string filename = "../tests/data/sphere_40.obj.h5";
    PrismCage pc(filename);
    spdlog::info("Shell size v{}, f{}", pc.base.size(), pc.F.size());
    auto [vert_info, tet_info, vert_tet_conn] = prepare(pc);


    spdlog::set_level(spdlog::level::trace);

    // for (auto i = 0; i < tet_t.rows(); i++) {
    //     auto r = circumradi2(
    //         tet_v.row(tet_t(i, 0)),
    //         tet_v.row(tet_t(i, 1)),
    //         tet_v.row(tet_t(i, 2)),
    //         tet_v.row(tet_t(i, 3)));
    // }

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

// iterate all edges
auto construct_edge_queue = [](const auto& vert_info, const auto& tet_info) {
    auto local_edges =
        std::array<std::array<int, 2>, 6>{{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}};
    auto edge_set = std::set<std::tuple<int, int>>();
    for (auto tet : tet_info) {
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


TEST_CASE("sphere-tet-swap")
{
    std::string filename = "../tests/data/sphere_40.obj.h5";
    PrismCage pc(filename);
    auto [vert_info, tet_info, vert_tet_conn] = prepare(pc);
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
        prism::tet::swap_face(pc, option, vert_info, tet_info, vert_tet_conn, v0, v1, v2);
    }
    spdlog::info("Size {} {}", vert_info.size(), tet_info.size());
    for (auto t : tet_info) {
        if (!t.is_removed) CHECK(prism::tet::tetra_validity(vert_info, t.conn));
    }
}