#include <doctest.h>

#include <batm/tetra_logger.hpp>
#include <batm/tetra_utils.hpp>
#include <prism/common.hpp>
#include <prism/geogram/AABB.hpp>
#include <prism/geogram/geogram_utils.hpp>
#include <prism/local_operations/remesh_pass.hpp>
#include <prism/spatial-hash/AABB_hash.hpp>
#include <prism/spatial-hash/self_intersection.hpp>
#include "batm/tetra_remesh_pass.hpp"
#include "prism/PrismCage.hpp"
#include "prism/cage_check.hpp"

#include <geogram/basic/geometry.h>
#include <igl/barycenter.h>
#include <igl/read_triangle_mesh.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <condition_variable>
#include <highfive/H5Easy.hpp>
#include <iterator>
#include <memory>
#include <numeric>
#include <queue>
#include <tuple>


auto checker_in_main = [](const auto& pc, const auto& option, bool enable) {
    if (!enable) return;
    auto require = [&](bool b) {
        if (b == false) {
            spdlog::dump_backtrace();
            pc->serialize("temp.h5");
            throw std::runtime_error("Checker in main");
        }
    };
    require(prism::cage_check::cage_is_positive(*pc));
    require(prism::cage_check::cage_is_away_from_ref(*pc));
    require(prism::cage_check::verify_edge_based_track(*pc, option, pc->track_ref));
    require(prism::spatial_hash::self_intersections(pc->base, pc->F).empty());
    require(prism::spatial_hash::self_intersections(pc->top, pc->F).empty());
    spdlog::info("Verifier: Done Checking");
};

TEST_CASE("amr-sphere-prepare")
{
    using namespace prism::tet;
    std::string filename = "../tests/data/sphere_40.obj.h5";
    auto pc = new PrismCage(filename);
    spdlog::info("Shell size v{}, f{}", pc->base.size(), pc->F.size());
    auto [vert_info, tet_info, vert_tet_conn] =
        prism::tet::reload("../tests/data/sphere_40.obj.h5", pc);


    spdlog::set_level(spdlog::level::trace);

    prism::local::RemeshOptions option(pc->mid.size(), 0.1);
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
    auto pc = new PrismCage(filename);
    auto [vert_info, tet_info, vert_tet_conn] =
        prism::tet::reload("../tests/data/sphere_40.obj.h5", pc);
    prism::local::RemeshOptions option(pc->mid.size(), 0.1);


    auto edge_queue = prism::tet::construct_edge_queue(vert_info, tet_info);
    auto face_queue = prism::tet::construct_face_queue(vert_info, tet_info);
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
    auto pc = new PrismCage(filename);
    auto [vert_info, tet_info, vert_tet_conn] =
        prism::tet::reload("../tests/data/sphere_40.obj.h5", pc);
    prism::local::RemeshOptions option(pc->mid.size(), 0.1);

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
    prism::tet::serializer("../buildr/debug0.h5", pc, vert_info, tet_info);
}

TEST_CASE("sphere-coarsen-aggresive")
{
    std::string filename = "../buildr/debug0.h5";
    auto pc = new PrismCage(filename);
    auto [vert_info, tet_info, vert_tet_conn] = prism::tet::reload(filename, pc);

    prism::local::RemeshOptions option(pc->mid.size(), 0.1);
    auto sizing = 1e2;
    option.collapse_quality_threshold = 150;
    //  spdlog::set_level(spdlog::level::trace);
    REQUIRE(prism::tet::tetmesh_sanity(tet_info, vert_info, vert_tet_conn, pc));
    spdlog::enable_backtrace(100);
    for (auto i = 0; i < 5; i++) {
        prism::tet::vertexsmooth_pass(pc, option, vert_info, tet_info, vert_tet_conn, sizing);
        prism::tet::faceswap_pass(pc, option, vert_info, tet_info, vert_tet_conn, sizing);
        prism::tet::compact_tetmesh(vert_info, tet_info, vert_tet_conn, pc);
        option.target_adjustment.resize(pc->mid.size());
        // collapse_pass(&pc, option, vert_info, tet_info, vert_tet_conn, sizing);
        prism::tet::edgeswap_pass(pc, option, vert_info, tet_info, vert_tet_conn, sizing);
        // collapse_pass(&pc, option, vert_info, tet_info, vert_tet_conn, sizing);
        prism::tet::vertexsmooth_pass(pc, option, vert_info, tet_info, vert_tet_conn, sizing);
        prism::tet::compact_tetmesh(vert_info, tet_info, vert_tet_conn, pc);
        option.target_adjustment.resize(pc->mid.size());
    }
    REQUIRE(prism::tet::tetmesh_sanity(tet_info, vert_info, vert_tet_conn, pc));
    prism::tet::serializer("../buildr/coarse.h5", pc, vert_info, tet_info);
    delete pc;

    {
        std::string filename = "../buildr/coarse.h5";
        auto pc = new PrismCage(filename);
        auto [vert_info, tet_info, vert_tet_conn] = prism::tet::reload(filename, pc);
        prism::local::RemeshOptions option(pc->mid.size(), 0.1);
        REQUIRE(prism::tet::tetmesh_sanity(tet_info, vert_info, vert_tet_conn, pc));
        delete pc;
    }
}

int edge_split_pass_for_dof(
    PrismCage* pc,
    prism::local::RemeshOptions& option,
    prism::tet::vert_info_t& vert_attrs,
    prism::tet::tet_info_t& tet_attrs,
    std::vector<std::vector<int>>& vert_conn)
{
    const auto bad_quality = 10.;

    auto cnt = 0;
    std::vector<bool> quality_due(tet_attrs.size(), false);
    auto local_edges =
        std::array<std::array<int, 2>, 6>{{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}};

    std::set<std::pair<int, int>> all_edges;
    for (auto i = 0; i < tet_attrs.size(); i++) {
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

        auto flag = prism::tet::split_edge(pc, option, vert_attrs, tet_attrs, vert_conn, v0, v1);
        if (flag) cnt++;
    }

    return cnt;
}

auto barycentric_sizer_constructor = [](const auto& func) {
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
            sizes[i] = func(BC.row(i));
            // if (BC(i, 0) < 0.1) sizes[i] = std::pow(5e-2, 2);
        }
        sizer.reset(new prism::tet::SizeController(bgV, bgT, sizes));
    }
    return std::move(sizer);
};

TEST_CASE("graded-sphere")
{
    std::string filename = "../buildr/coarse.h5";
    auto pc = new PrismCage(filename);
    auto [vert_info, tet_info, vert_tet_conn] = prism::tet::reload(filename, pc);
    prism::local::RemeshOptions option(pc->mid.size(), 0.1);
    option.collapse_quality_threshold = 10;
    prism::tet::logger().enable_backtrace(100);

    auto sizer = barycentric_sizer_constructor(
        [](const auto& bc) { return (bc[0] < 0.1) ? std::pow(5e-2, 2) : 1.0; });

    auto sizer2 = barycentric_sizer_constructor([](const auto& bc) { return 1.0; });

    for (auto i = 0; i < 5; i++) {
        prism::tet::edge_split_pass_with_sizer(
            pc,
            option,
            vert_info,
            tet_info,
            vert_tet_conn,
            sizer);
        prism::tet::vertexsmooth_pass(pc, option, vert_info, tet_info, vert_tet_conn, 0.1);
        prism::tet::edgeswap_pass(pc, option, vert_info, tet_info, vert_tet_conn, 1.);
        prism::tet::faceswap_pass(pc, option, vert_info, tet_info, vert_tet_conn, 1.);
        edge_split_pass_for_dof(pc, option, vert_info, tet_info, vert_tet_conn);
        prism::tet::vertexsmooth_pass(pc, option, vert_info, tet_info, vert_tet_conn, 0.1);
        collapse_pass(pc, option, vert_info, tet_info, vert_tet_conn, sizer2);
        prism::tet::compact_tetmesh(vert_info, tet_info, vert_tet_conn, pc);
    }
    prism::tet::serializer("../buildr/left.h5", pc, vert_info, tet_info);
    return;
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
            if (BC(i, 0) > 0.9) sizes[i] = 5e-3;
        }
        sizer.reset(new prism::tet::SizeController(bgV, bgT, sizes));
    }
    option.collapse_quality_threshold = 8;
    for (auto i = 0; i < 5; i++) {
        prism::tet::edge_split_pass_with_sizer(
            pc,
            option,
            vert_info,
            tet_info,
            vert_tet_conn,
            sizer);
        prism::tet::vertexsmooth_pass(pc, option, vert_info, tet_info, vert_tet_conn, 1e-1);
        prism::tet::edgeswap_pass(pc, option, vert_info, tet_info, vert_tet_conn, 1.);
        prism::tet::faceswap_pass(pc, option, vert_info, tet_info, vert_tet_conn, 1.);
        if (i < 3) collapse_pass(pc, option, vert_info, tet_info, vert_tet_conn, sizer);
        prism::tet::compact_tetmesh(vert_info, tet_info, vert_tet_conn, pc);
    }
    prism::tet::serializer("../buildr/right.h5", pc, vert_info, tet_info);
}

TEST_CASE("loose-size")
{
    std::string filename = "../buildr/left.h5";
    auto pc = new PrismCage(filename);
    auto [vert_info, tet_info, vert_tet_conn] = prism::tet::reload(filename, pc);
    prism::local::RemeshOptions option(pc->mid.size(), 0.1);
    option.collapse_quality_threshold = 150;
    prism::tet::logger().enable_backtrace(100);

    auto sizer = std::unique_ptr<prism::tet::SizeController>(nullptr);
    {
        H5Easy::File file("../tests/data/cube_tetra_10.h5", H5Easy::File::ReadOnly);
        auto bgV = H5Easy::load<RowMatd>(file, "V");
        auto bgT = H5Easy::load<RowMati>(file, "T");
        Eigen::VectorXd sizes(bgT.rows());
        sizes.setConstant(1.);
        // assign size
        RowMatd BC;
        igl::barycenter(bgV, bgT, BC);
        for (auto i = 0; i < bgT.rows(); i++) {
            if (BC(i, 0) > 0.9) sizes[i] = std::pow(5e-2, 2);
        }
        sizer.reset(new prism::tet::SizeController(bgV, bgT, sizes));
    }
    auto improves_quality =
        [&, &vert_info = vert_info, &tet_info = tet_info, &vert_tet_conn = vert_tet_conn]() {
            edge_split_pass_for_dof(pc, option, vert_info, tet_info, vert_tet_conn);
            option.collapse_quality_threshold = -1;
            prism::tet::collapse_pass(pc, option, vert_info, tet_info, vert_tet_conn, sizer);
            option.collapse_quality_threshold = 150; // aggressive
            prism::tet::vertexsmooth_pass(pc, option, vert_info, tet_info, vert_tet_conn, 0.1);
            prism::tet::edgeswap_pass(pc, option, vert_info, tet_info, vert_tet_conn, 1.);
            prism::tet::faceswap_pass(pc, option, vert_info, tet_info, vert_tet_conn, 1.);
        };
    for (auto i = 0; i < 6; i++) {
        improves_quality();
        prism::tet::collapse_pass(pc, option, vert_info, tet_info, vert_tet_conn, sizer);
        improves_quality();
        prism::tet::collapse_pass(pc, option, vert_info, tet_info, vert_tet_conn, sizer);
        prism::tet::compact_tetmesh(vert_info, tet_info, vert_tet_conn, pc);
        prism::tet::serializer("../buildr/temp.h5", pc, vert_info, tet_info);
    }
    prism::tet::serializer("../buildr/loose-coarse.h5", pc, vert_info, tet_info);
}

TEST_CASE("continue-coarsen")
{
    std::string filename = "../buildr/loose-coarse.h5";
    auto pc = new PrismCage(filename);
    auto [vert_info, tet_info, vert_tet_conn] = prism::tet::reload(filename, pc);
    prism::local::RemeshOptions option(pc->mid.size(), 0.1);
    option.collapse_quality_threshold = 150;
    prism::tet::logger().enable_backtrace(100);
    auto sizer = std::unique_ptr<prism::tet::SizeController>(nullptr);
    {
        H5Easy::File file("../tests/data/cube_tetra_10.h5", H5Easy::File::ReadOnly);
        auto bgV = H5Easy::load<RowMatd>(file, "V");
        auto bgT = H5Easy::load<RowMati>(file, "T");
        Eigen::VectorXd sizes(bgT.rows());
        sizes.setConstant(1e2);
        // assign size
        RowMatd BC;
        igl::barycenter(bgV, bgT, BC);
        for (auto i = 0; i < bgT.rows(); i++) {
            // if (BC(i, 0) > 0.9) sizes[i] = 5e-3;
        }
        sizer.reset(new prism::tet::SizeController(bgV, bgT, sizes));
    }
    // edge_split_pass_for_dof(pc, option, vert_info, tet_info, vert_tet_conn);
    prism::tet::vertexsmooth_pass(pc, option, vert_info, tet_info, vert_tet_conn, 0.1);
    prism::tet::vertexsmooth_pass(pc, option, vert_info, tet_info, vert_tet_conn, 0.1);
    prism::tet::vertexsmooth_pass(pc, option, vert_info, tet_info, vert_tet_conn, 0.1);
    prism::tet::vertexsmooth_pass(pc, option, vert_info, tet_info, vert_tet_conn, 0.1);
    // prism::tet::logger().set_level(spdlog::level::trace);

    prism::tet::collapse_pass(pc, option, vert_info, tet_info, vert_tet_conn, sizer);
    prism::tet::serializer("../buildr/temp.h5", pc, vert_info, tet_info);
}

TEST_CASE("shell-disable")
{
    std::string filename = "../buildr/loose-coarse.h5";
    auto pc = std::shared_ptr<PrismCage>(new PrismCage(filename));
    auto [vert_info, tet_info, vert_tet_conn] = prism::tet::reload(filename, pc.get());
    // pc.reset();
    prism::local::RemeshOptions option(0, 0.1);

    option.collapse_quality_threshold = 150;
    prism::tet::logger().enable_backtrace(100);
    auto sizer = std::unique_ptr<prism::tet::SizeController>(nullptr);
    {
        H5Easy::File file("../tests/data/cube_tetra_10.h5", H5Easy::File::ReadOnly);
        auto bgV = H5Easy::load<RowMatd>(file, "V");
        auto bgT = H5Easy::load<RowMati>(file, "T");
        Eigen::VectorXd sizes(bgT.rows());
        sizes.setConstant(1e-1);
        // assign size
        RowMatd BC;
        igl::barycenter(bgV, bgT, BC);
        for (auto i = 0; i < bgT.rows(); i++) {
            // if (BC(i, 0) > 0.9) sizes[i] = 5e-3;
        }
        sizer.reset(new prism::tet::SizeController(bgV, bgT, sizes));
    }
    prism::tet::collapse_pass(pc.get(), option, vert_info, tet_info, vert_tet_conn, sizer);
}

TEST_CASE("shell-only")
{
    std::string filename = "../buildr/loose-coarse.h5";
    auto pc = std::shared_ptr<PrismCage>(new PrismCage(filename));
    prism::local::RemeshOptions option(pc->mid.size(), /*target_edge_length=*/0.5);
    option.collapse_quality_threshold = 150;
    option.target_thickness = 0.1;
    option.parallel = false;
    checker_in_main(pc, option, true);

    for (auto repeat = 0; repeat < 5; repeat++) prism::local::localsmooth_pass(*pc, option);

    pc->serialize("../buildr/after_smooth.h5");
    // spdlog::set_level(spdlog::level::trace);
    prism::local::wildcollapse_pass(*pc, option);
    prism::local::wildcollapse_pass(*pc, option);
    for (auto repeat = 0; repeat < 5; repeat++) prism::local::localsmooth_pass(*pc, option);
    prism::local::wildcollapse_pass(*pc, option);
    pc->serialize("../buildr/after_collapse.h5");
}

TEST_CASE("single-smoother-debug")
{
    std::string filename = "../buildr/after_collapse.h5";
    auto pc = std::shared_ptr<PrismCage>(new PrismCage(filename));
    prism::local::RemeshOptions option(pc->mid.size(), /*target_edge_length=*/0.5);
    option.collapse_quality_threshold = 150;
    option.target_thickness = 0.1;
    option.parallel = false;
    spdlog::set_level(spdlog::level::trace);
    for (auto repeat = 0; repeat < 1; repeat++) prism::local::localsmooth_pass(*pc, option);
}