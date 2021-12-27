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

namespace prism::tet {
bool flip_edge_sf(
    const PrismCage* pc,
    const prism::local::RemeshOptions& option,
    std::vector<VertAttr>& vert_attrs,
    std::vector<TetAttr>& tet_attrs,
    std::vector<std::vector<int>>& vert_conn,
    int v0,
    int v1,
    double size_control);
}; // namespace prism::tet

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
    auto pc = std::make_shared<PrismCage>(filename);
    spdlog::info("Shell size v{}, f{}", pc->base.size(), pc->F.size());
    auto [vert_info, tet_info, vert_tet_conn] =
        prism::tet::reload("../tests/data/sphere_40.obj.h5", pc.get());


    spdlog::set_level(spdlog::level::trace);

    prism::local::RemeshOptions option(pc->mid.size(), 0.1);
    auto check_sorted = [&vert_tet_conn = vert_tet_conn]() -> bool {
        for (auto& arr : vert_tet_conn)
            for (auto i = 0; i < arr.size() - 1; i++)
                if (arr[i] > arr[i + 1]) return false;
        return true;
    };
    assert(check_sorted());
    split_edge(pc.get(), option, vert_info, tet_info, vert_tet_conn, 0, 1);
    spdlog::info("Size {} {}", vert_info.size(), tet_info.size());
    assert(check_sorted());
    assert(check_sorted());
    // smooth_vertex(pc.get(), option, vert_info, tet_info, vert_tet_conn, 46);
    // spdlog::info("Size {} {}", vert_info.size(), tet_info.size());
}

TEST_CASE("sphere-coarsen-aggresive")
{
    std::string filename = "../buildr/debug0.h5";
    auto pc = std::make_shared<PrismCage>(filename);
    auto [vert_info, tet_info, vert_tet_conn] = prism::tet::reload(filename, pc.get());

    prism::local::RemeshOptions option(pc->mid.size(), 0.1);
    auto sizing = 1e2;
    option.collapse_quality_threshold = 150;
    //  spdlog::set_level(spdlog::level::trace);
    REQUIRE(prism::tet::tetmesh_sanity(tet_info, vert_info, vert_tet_conn, pc.get()));
    spdlog::enable_backtrace(100);
    for (auto i = 0; i < 5; i++) {
        prism::tet::vertexsmooth_pass(pc.get(), option, vert_info, tet_info, vert_tet_conn, sizing);
        prism::tet::faceswap_pass(pc.get(), option, vert_info, tet_info, vert_tet_conn, sizing);
        prism::tet::compact_tetmesh(vert_info, tet_info, vert_tet_conn, pc.get());
        option.target_adjustment.resize(pc->mid.size());
        // collapse_pass(&pc, option, vert_info, tet_info, vert_tet_conn, sizing);
        prism::tet::edgeswap_pass(pc.get(), option, vert_info, tet_info, vert_tet_conn, sizing);
        // collapse_pass(&pc, option, vert_info, tet_info, vert_tet_conn, sizing);
        prism::tet::vertexsmooth_pass(pc.get(), option, vert_info, tet_info, vert_tet_conn, sizing);
        prism::tet::compact_tetmesh(vert_info, tet_info, vert_tet_conn, pc.get());
        option.target_adjustment.resize(pc->mid.size());
    }
    REQUIRE(prism::tet::tetmesh_sanity(tet_info, vert_info, vert_tet_conn, pc.get()));
    prism::tet::serializer("../buildr/coarse.h5", pc.get(), vert_info, tet_info);

    {
        std::string filename = "../buildr/coarse.h5";
        auto pc = std::make_shared<PrismCage>(filename);
        auto [vert_info, tet_info, vert_tet_conn] = prism::tet::reload(filename, pc.get());
        prism::local::RemeshOptions option(pc->mid.size(), 0.1);
        REQUIRE(prism::tet::tetmesh_sanity(tet_info, vert_info, vert_tet_conn, pc.get()));
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
    auto pc = std::make_shared<PrismCage>(filename);
    auto [vert_info, tet_info, vert_tet_conn] = prism::tet::reload(filename, pc.get());
    prism::local::RemeshOptions option(pc->mid.size(), 0.1);
    option.collapse_quality_threshold = 10;
    prism::tet::logger().enable_backtrace(100);

    auto sizer = barycentric_sizer_constructor(
        [](const auto& bc) { return (bc[0] < 0.1) ? std::pow(5e-2, 2) : 1.0; });

    auto sizer2 = barycentric_sizer_constructor([](const auto& bc) { return 1.0; });

    for (auto i = 0; i < 5; i++) {
        prism::tet::edge_split_pass_with_sizer(
            pc.get(),
            option,
            vert_info,
            tet_info,
            vert_tet_conn,
            sizer);
        prism::tet::vertexsmooth_pass(pc.get(), option, vert_info, tet_info, vert_tet_conn, 0.1);
        prism::tet::edgeswap_pass(pc.get(), option, vert_info, tet_info, vert_tet_conn, 1.);
        prism::tet::faceswap_pass(pc.get(), option, vert_info, tet_info, vert_tet_conn, 1.);
        edge_split_pass_for_dof(pc.get(), option, vert_info, tet_info, vert_tet_conn);
        prism::tet::vertexsmooth_pass(pc.get(), option, vert_info, tet_info, vert_tet_conn, 0.1);
        collapse_pass(pc.get(), option, vert_info, tet_info, vert_tet_conn, sizer2);
        prism::tet::compact_tetmesh(vert_info, tet_info, vert_tet_conn, pc.get());
    }
    prism::tet::serializer("../buildr/left.h5", pc.get(), vert_info, tet_info);
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
            pc.get(),
            option,
            vert_info,
            tet_info,
            vert_tet_conn,
            sizer);
        prism::tet::vertexsmooth_pass(pc.get(), option, vert_info, tet_info, vert_tet_conn, 1e-1);
        prism::tet::edgeswap_pass(pc.get(), option, vert_info, tet_info, vert_tet_conn, 1.);
        prism::tet::faceswap_pass(pc.get(), option, vert_info, tet_info, vert_tet_conn, 1.);
        if (i < 3) collapse_pass(pc.get(), option, vert_info, tet_info, vert_tet_conn, sizer);
        prism::tet::compact_tetmesh(vert_info, tet_info, vert_tet_conn, pc.get());
    }
    prism::tet::serializer("../buildr/right.h5", pc.get(), vert_info, tet_info);
}

TEST_CASE("loose-size")
{
    std::string filename = "../buildr/left.h5";
    auto pc = std::make_shared<PrismCage>(filename);
    auto [vert_info, tet_info, vert_tet_conn] = prism::tet::reload(filename, pc.get());
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
    auto improves_quality = [&,
                             &vert_info = vert_info,
                             &tet_info = tet_info,
                             &vert_tet_conn = vert_tet_conn]() {
        edge_split_pass_for_dof(pc.get(), option, vert_info, tet_info, vert_tet_conn);
        option.collapse_quality_threshold = -1;
        prism::tet::collapse_pass(pc.get(), option, vert_info, tet_info, vert_tet_conn, sizer);
        option.collapse_quality_threshold = 150; // aggressive
        prism::tet::vertexsmooth_pass(pc.get(), option, vert_info, tet_info, vert_tet_conn, 0.1);
        prism::tet::edgeswap_pass(pc.get(), option, vert_info, tet_info, vert_tet_conn, 1.);
        prism::tet::faceswap_pass(pc.get(), option, vert_info, tet_info, vert_tet_conn, 1.);
    };
    for (auto i = 0; i < 6; i++) {
        improves_quality();
        prism::tet::collapse_pass(pc.get(), option, vert_info, tet_info, vert_tet_conn, sizer);
        improves_quality();
        prism::tet::collapse_pass(pc.get(), option, vert_info, tet_info, vert_tet_conn, sizer);
        prism::tet::compact_tetmesh(vert_info, tet_info, vert_tet_conn, pc.get());
        prism::tet::serializer("../buildr/temp.h5", pc.get(), vert_info, tet_info);
    }
    prism::tet::serializer("../buildr/loose-coarse.h5", pc.get(), vert_info, tet_info);
}

TEST_CASE("continue-coarsen")
{
    std::string filename = "../buildr/loose-coarse.h5";
    auto pc = std::make_shared<PrismCage>(filename);
    auto [vert_info, tet_info, vert_tet_conn] = prism::tet::reload(filename, pc.get());
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
    // edge_split_pass_for_dof(pc.get(), option, vert_info, tet_info, vert_tet_conn);
    prism::tet::vertexsmooth_pass(pc.get(), option, vert_info, tet_info, vert_tet_conn, 0.1);
    prism::tet::vertexsmooth_pass(pc.get(), option, vert_info, tet_info, vert_tet_conn, 0.1);
    prism::tet::vertexsmooth_pass(pc.get(), option, vert_info, tet_info, vert_tet_conn, 0.1);
    prism::tet::vertexsmooth_pass(pc.get(), option, vert_info, tet_info, vert_tet_conn, 0.1);
    // prism::tet::logger().set_level(spdlog::level::trace);

    prism::tet::collapse_pass(pc.get(), option, vert_info, tet_info, vert_tet_conn, sizer);
    prism::tet::serializer("../buildr/temp.h5", pc.get(), vert_info, tet_info);
}

TEST_CASE("shell-disable")
{
    std::string filename = "../buildr/loose-coarse.h5";
    auto pc = std::make_shared<PrismCage>(filename);
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
    auto pc = std::make_shared<PrismCage>(filename);
    prism::local::RemeshOptions option(pc->mid.size(), /*target_edge_length=*/0.5);
    option.collapse_quality_threshold = 150;
    option.target_thickness = 0.1;
    option.parallel = false;
    checker_in_main(pc.get(), option, true);

    for (auto repeat = 0; repeat < 5; repeat++) prism::local::localsmooth_pass(*pc, option);

    pc->serialize("../buildr/after_smooth.h5");
    // spdlog::set_level(spdlog::level::trace);
    prism::local::wildcollapse_pass(*pc, option);
    prism::local::wildcollapse_pass(*pc, option);
    for (auto repeat = 0; repeat < 5; repeat++) prism::local::localsmooth_pass(*pc, option);
    prism::local::wildcollapse_pass(*pc, option);
    pc->serialize("../buildr/after_collapse.h5");
}
