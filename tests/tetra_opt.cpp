#include <doctest.h>

#include "batm/tetra_logger.hpp"
#include "batm/tetra_remesh_pass.hpp"
#include "batm/tetra_utils.hpp"
#include "cumin/tetshell_utils.hpp"
#include "prism/PrismCage.hpp"
#include "prism/geogram/AABB.hpp"
#include "prism/local_operations/remesh_pass.hpp"

#include <igl/remove_unreferenced.h>
#include <shell/Utils.h>
#include <spdlog/spdlog.h>
#include <tetwild/Logger.h>
#include <tetwild/tetwild.h>
#include <highfive/H5Easy.hpp>
TEST_CASE("prepare-bunny-tetmesh")
{
    // conforming tetmeshing.
    std::string filename = "../buildr/bunny.off.h5";
    auto pc = PrismCage(filename);
    RowMatd mT, mV, mB, vtop, vbase;
    RowMati mF;
    vec2eigen(pc.top, mT);
    vec2eigen(pc.mid, mV);
    vec2eigen(pc.base, mB);
    vec2eigen(pc.F, mF);

    // filler
    RowMatd VN = mT - mB;
    for (auto i = 0; i < VN.rows(); i++) {
        if (mT.row(i) == mB.row(i)) VN.row(i).setZero();
        VN.row(i) = VN.row(i).normalized();
    }

    vtop = prism::tetshell::one_side_extrusion(mT, mF, VN, true);
    vbase = prism::tetshell::one_side_extrusion(mV, mF, VN, false);

    Eigen::MatrixXd Vmsh;
    Eigen::MatrixXi Tmsh;
    Eigen::VectorXi labels;
    prism::tetshell::tetshell_fill(vbase, mV, mT, vtop, mF, Vmsh, Tmsh, labels);

    RowMati intTmsh;
    {
        std::vector<Eigen::RowVector4i> intTmsh_vec;
        assert(labels.size() == Tmsh.rows());
        for (auto i = 0; i < labels.size(); i++) {
            if (labels[i] == 1) intTmsh_vec.push_back(Tmsh.row(i));
        }
        vec2eigen(intTmsh_vec, intTmsh);
    }
    RowMatd new_tet_V;
    RowMati new_tet_T;
    Eigen::VectorXi map_ind;
    igl::remove_unreferenced(Vmsh, intTmsh, new_tet_V, new_tet_T, map_ind);
    for (auto i = 0; i < mV.rows(); i++) {
        REQUIRE_EQ(map_ind[i], i);
        REQUIRE_EQ(mV.row(i), new_tet_V.row(i));
    }
    pc.serialize(
        filename + ".tet.h5",
        (std::function<void(HighFive::File&)>)[&](HighFive::File & file) {
            H5Easy::dump(file, "tet_v", new_tet_V);
            H5Easy::dump(file, "tet_t", new_tet_T);
        });
}
#include <igl/barycenter.h>


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


TEST_CASE("feature-tet-coarsen")
{
    std::string modelname = "../buildr/bunny.off";
    std::string filename = modelname + ".h5.tet.h5";
    
    auto pc = std::make_shared<PrismCage>(filename);
    auto tetmesh = prism::tet::reload(filename, pc.get());

    prism::local::RemeshOptions option(pc->mid.size(), 0.1);
    option.distortion_bound = 1e-3;

    auto sizer = barycentric_sizer_constructor(
        [](const auto& bc) { return (bc[0] < 0.1) ? std::pow(5e-2, 2) : 1.0; });
    option.collapse_quality_threshold = 8;
    //  spdlog::set_level(spdlog::level::trace);
    for (auto i = 0; i < 5; i++) {
        prism::tet::edge_split_pass_with_sizer(pc.get(), option, tetmesh, sizer.get(), 4/3.);
        prism::tet::serializer("../buildr/after_split.h5", pc.get(), tetmesh);
        prism::tet::vertexsmooth_pass(pc.get(), option, tetmesh, nullptr, 0.1);
        prism::tet::serializer("../buildr/after_smooth.h5", pc.get(), tetmesh);
        prism::tet::edgeswap_pass(pc.get(), option, tetmesh, nullptr);
        prism::tet::faceswap_pass(pc.get(), option, tetmesh, nullptr);
        prism::tet::serializer("../buildr/after_swap.h5", pc.get(), tetmesh);
        prism::tet::edge_split_pass_for_dof(pc.get(), option, tetmesh);
        prism::tet::serializer("../buildr/after_dof.h5", pc.get(), tetmesh);
        prism::tet::vertexsmooth_pass(pc.get(), option, tetmesh, nullptr, 0.1);
        prism::tet::serializer("../buildr/after_smooth2.h5", pc.get(), tetmesh);
        prism::tet::collapse_pass(pc.get(), option, tetmesh, sizer.get());
        prism::tet::compact_tetmesh(tetmesh, pc.get());
        prism::tet::serializer("../buildr/after_collapse", pc.get(), tetmesh);
        prism::tet::vertexsmooth_pass(pc.get(), option, tetmesh, nullptr, 0.1);
        prism::tet::serializer("../buildr/after_smooth3.h5", pc.get(), tetmesh);
        REQUIRE(prism::tet::tetmesh_sanity(tetmesh, pc.get()));
    }
    prism::tet::serializer(modelname + ".left.h5", pc.get(), tetmesh);
}

TEST_CASE("feature-tet-right")
{
    std::string modelname = "../buildr/bunny.off";
    std::string filename = modelname + ".left.h5";
    auto pc = std::make_shared<PrismCage>(filename);
    auto tetmesh = prism::tet::reload(filename, pc.get());

    prism::local::RemeshOptions option(pc->mid.size(), 0.1);
    option.distortion_bound = 1e-3;

    auto sizer = barycentric_sizer_constructor(
        [](const auto& bc) { return (bc[0] > 0.85) ? std::pow(5e-2, 2) : 1.0; });
    option.collapse_quality_threshold = 20;
    auto saver = [&tetmesh = tetmesh, &pc](int i, std::string name) {
        std::string prefix = "../buildr/right/";
        prism::tet::serializer(fmt::format("{}_{}_{}.h5", prefix, i, name), pc.get(), tetmesh);
    };
    for (auto i = 0; i < 5; i++) {
        if (i >= 3) option.collapse_quality_threshold = 8;
        prism::tet::edge_split_pass_with_sizer(pc.get(), option, tetmesh, sizer.get(), 4/3.);
        saver(i, "sp");
        prism::tet::vertexsmooth_pass(pc.get(), option, tetmesh, nullptr, 0.1);
        saver(i, "sm1");
        prism::tet::edgeswap_pass(pc.get(), option, tetmesh, nullptr);
        prism::tet::faceswap_pass(pc.get(), option, tetmesh, nullptr);
        saver(i, "sw");
        prism::tet::edge_split_pass_for_dof(pc.get(), option, tetmesh);
        saver(i, "spd");
        prism::tet::vertexsmooth_pass(pc.get(), option, tetmesh, nullptr, 0.1);
        saver(i, "sm2");
        prism::tet::collapse_pass(pc.get(), option, tetmesh, sizer.get());
        prism::tet::compact_tetmesh(tetmesh, pc.get());
        saver(i, "col");
        prism::tet::vertexsmooth_pass(pc.get(), option, tetmesh, nullptr, 0.1);
        saver(i, "sm3");
        REQUIRE(prism::tet::tetmesh_sanity(tetmesh, pc.get()));
    }
    prism::tet::serializer(modelname + ".right.h5", pc.get(), tetmesh);
}