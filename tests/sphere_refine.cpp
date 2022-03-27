#include <doctest.h>

#include <batm/tetra_logger.hpp>
#include <batm/tetra_utils.hpp>
#include <nlohmann/json_fwd.hpp>
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
#include <igl/predicates/predicates.h>
#include <igl/read_triangle_mesh.h>
#include <igl/volume.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <condition_variable>
#include <highfive/H5Easy.hpp>
#include <iterator>
#include <memory>
#include <nlohmann/json.hpp>
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


TEST_CASE("sphere-coarsen-aggresive")
{
    std::string filename = "../buildr/debug0.h5";
    auto pc = std::make_shared<PrismCage>(filename);
    auto tetmesh = prism::tet::reload(filename, pc.get());

    prism::local::RemeshOptions option(pc->mid.size(), 0.1);
    auto sizing = 1e2;
    option.collapse_quality_threshold = 150;
    //  spdlog::set_level(spdlog::level::trace);
    REQUIRE(prism::tet::tetmesh_sanity(tetmesh, pc.get()));
    spdlog::enable_backtrace(100);
    for (auto i = 0; i < 5; i++) {
        prism::tet::vertexsmooth_pass(pc.get(), option, tetmesh, nullptr, 0.1);
        prism::tet::faceswap_pass(pc.get(), option, tetmesh, nullptr);
        prism::tet::compact_tetmesh(tetmesh, pc.get());
        option.target_adjustment.resize(pc->mid.size());
        // collapse_pass(&pc, option, tetmesh, sizing);
        prism::tet::edgeswap_pass(pc.get(), option, tetmesh, nullptr);
        // collapse_pass(&pc, option, tetmesh, sizing);
        prism::tet::vertexsmooth_pass(pc.get(), option, tetmesh, nullptr, 0.1);
        prism::tet::compact_tetmesh(tetmesh, pc.get());
        option.target_adjustment.resize(pc->mid.size());
    }
    REQUIRE(prism::tet::tetmesh_sanity(tetmesh, pc.get()));
    prism::tet::serializer("../buildr/coarse.h5", pc.get(), tetmesh);

    {
        std::string filename = "../buildr/coarse.h5";
        auto pc = std::make_shared<PrismCage>(filename);
        auto tetmesh = prism::tet::reload(filename, pc.get());
        prism::local::RemeshOptions option(pc->mid.size(), 0.1);
        REQUIRE(prism::tet::tetmesh_sanity(tetmesh, pc.get()));
    }
}


auto log_sizer_constructor = []() {
    auto sizer = std::unique_ptr<prism::tet::SizeController>(nullptr);
    {
        H5Easy::File file("../tests/data/cube_log_size.h5", H5Easy::File::ReadOnly);
        auto bgV = H5Easy::load<RowMatd>(file, "V");
        auto bgT = H5Easy::load<RowMati>(file, "T");
        Eigen::VectorXd sizes = H5Easy::load<Eigen::VectorXd>(file, "size");
        sizes = sizes.unaryExpr([](double d) { return std::pow(d, 2); });
        sizer.reset(new prism::tet::SizeController(bgV, bgT, sizes));
    }
    return std::move(sizer);
};

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
    std::string filename = "";
    nlohmann::json config;
    {
        auto file = (std::ifstream("../tests/graded-sphere-test-config.json"));
        file >> config;
        filename = config["filename"];
    }
    spdlog::info("Read filename {}", filename);
    auto pc = std::make_shared<PrismCage>(filename);
    auto tetmesh = prism::tet::reload(filename, pc.get());
    prism::local::RemeshOptions option(pc->mid.size(), 0.1);
    option.collapse_quality_threshold = 8;
    option.distortion_bound = 1e-3;
    prism::tet::logger().enable_backtrace(100);

    auto sizer = log_sizer_constructor();
    sizer->surface_only = true;

    std::string prefix = filename + "-";
    auto saver = [&tetmesh = tetmesh, &pc, prefix, &sizer](int i, std::string name) {
        auto stats = prism::tet::size_progress(sizer.get(), tetmesh);
        spdlog::info(
            "Progress: Volume {}. Over {}. OverOver {}  LogError {}",
            stats[2],
            stats[0] / stats[2],
            stats[1] / stats[2],
            stats[3]);
        prism::tet::serializer(fmt::format("{}_{}_{}.h5", prefix, i, name), pc.get(), tetmesh);
    };

    auto swapper = [&]() {
        prism::tet::edgeswap_pass(pc.get(), option, tetmesh, nullptr);
        prism::tet::faceswap_pass(pc.get(), option, tetmesh, nullptr);
    };
    auto print_snap_progress = [&]() {
        auto [dist2, weight] = prism::tet::snap_progress(tetmesh, pc.get());
        auto integral = sqrt(dist2.dot(weight) / weight.sum());
        spdlog::info("Snap Progress::: INT {}, MAX {}, ", integral, sqrt(dist2.maxCoeff()));
    };
    auto print_size_progress = [&]() {
        auto stats = prism::tet::size_progress(sizer.get(), tetmesh);
        spdlog::info(
            "Progress: Volume {}. Over {}. OverOver {}  LogError {}",
            stats[2],
            stats[0] / stats[2],
            stats[1] / stats[2],
            stats[3]);
    };
    print_snap_progress();

    auto improves_quality = [&, &tetmesh = tetmesh]() {
        edge_split_pass_for_dof(pc.get(), option, tetmesh);
        option.collapse_quality_threshold = -1;
        prism::tet::collapse_pass(pc.get(), option, tetmesh, sizer.get(), true);
        option.collapse_quality_threshold = 8;
        prism::tet::vertexsmooth_pass(pc.get(), option, tetmesh, nullptr, 0.1);
        prism::tet::edgeswap_pass(pc.get(), option, tetmesh, nullptr);
        prism::tet::faceswap_pass(pc.get(), option, tetmesh, nullptr);
    };

    for (auto i = 0; i < 10; i++) {
        prism::tet::edge_split_pass_with_sizer(pc.get(), option, tetmesh, sizer.get(), 1.);
        print_size_progress();
        saver(i, "split");
        improves_quality();
        saver(i, "qual");
        swapper();
        saver(i, "swap");
        print_snap_progress();
        improves_quality();
        saver(i, "qual2");
        print_snap_progress();
        prism::tet::collapse_pass(pc.get(), option, tetmesh, sizer.get(), true);
        swapper();
        saver(i, "swap2");
        prism::tet::compact_tetmesh(tetmesh, pc.get());
        saver(i, "collapse");
    }
    

    auto num_split = 1;
    while (num_split > 0) {
        num_split =
            prism::tet::edge_split_pass_with_sizer(pc.get(), option, tetmesh, sizer.get(), 1.);
        spdlog::info("Splits {}", num_split);
        print_size_progress();
    }
    for (auto i = 0; i < 3; i++) {
        improves_quality();
        print_size_progress();
    }
    saver(10, "strict-size");
    return;
    option.collapse_quality_threshold = 8;
    for (auto i = 0; i < 5; i++) {
        prism::tet::edge_split_pass_with_sizer(pc.get(), option, tetmesh, sizer.get(), 4 / 3.);
        prism::tet::vertexsmooth_pass(pc.get(), option, tetmesh, nullptr, 1e-1);
        prism::tet::edgeswap_pass(pc.get(), option, tetmesh, nullptr);
        prism::tet::faceswap_pass(pc.get(), option, tetmesh, nullptr);
        collapse_pass(pc.get(), option, tetmesh, sizer.get());
        prism::tet::compact_tetmesh(tetmesh, pc.get());
    }
    prism::tet::serializer("../buildr/sphere-right.h5", pc.get(), tetmesh);
}

TEST_CASE("strict-size")
{
    igl::predicates::exactinit();
    std::string filename = "../buildr/coarse.h5-_7_swap.h5";
    auto pc = std::make_shared<PrismCage>(filename);
    auto tetmesh = prism::tet::reload(filename, pc.get());
    prism::local::RemeshOptions option(pc->mid.size(), 0.1);
    option.collapse_quality_threshold = 8;
    prism::tet::logger().enable_backtrace(100);

    auto sizer = log_sizer_constructor();
    // auto stats = prism::tet::size_progress(sizer.get(), tetmesh);
    // return;
    auto improves_quality = [&, &tetmesh = tetmesh]() {
        edge_split_pass_for_dof(pc.get(), option, tetmesh);
        option.collapse_quality_threshold = -1;
        prism::tet::collapse_pass(pc.get(), option, tetmesh, sizer.get(), true);
        option.collapse_quality_threshold = 8;
        prism::tet::vertexsmooth_pass(pc.get(), option, tetmesh, sizer.get(), 0.1);
        prism::tet::edgeswap_pass(pc.get(), option, tetmesh, sizer.get());
        prism::tet::faceswap_pass(pc.get(), option, tetmesh, sizer.get());
    };
    auto print_size_progress = [&]() {
        auto stats = prism::tet::size_progress(sizer.get(), tetmesh);
        spdlog::info(
            "Progress: Volume {}. Over {}. OverOver {}  LogError {}",
            stats[2],
            stats[0] / stats[2],
            stats[1] / stats[2],
            stats[3]);
    };
    for (auto i = 0; i < 3; i++) {
        auto spl =
            prism::tet::edge_split_pass_with_sizer(pc.get(), option, tetmesh, sizer.get(), 1.);
        spdlog::info("Splits {}", spl);
        print_size_progress();
    }
    prism::tet::serializer("../buildr/split-satisfy.h5", pc.get(), tetmesh);
    for (auto i = 0; i < 3; i++) {
        improves_quality();
        print_size_progress();
    }
    prism::tet::serializer("../buildr/strict-size.h5", pc.get(), tetmesh);
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
    prism::local::wildcollapse_pass(*pc, option);
    prism::local::wildcollapse_pass(*pc, option);
    for (auto repeat = 0; repeat < 5; repeat++) prism::local::localsmooth_pass(*pc, option);
    prism::local::wildcollapse_pass(*pc, option);
    pc->serialize("../buildr/after_collapse.h5");
}