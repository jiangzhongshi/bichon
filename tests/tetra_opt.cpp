#include <doctest.h>

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
    std::string filename = "../buildr/00000006.objtest.h5";
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
    pc.serialize(filename +
        ".tet.h5",
        (std::function<void(HighFive::File&)>)[&](HighFive::File & file) {
            H5Easy::dump(file, "tet_v", new_tet_V);
            H5Easy::dump(file, "tet_t", new_tet_T);
        });
}
#include <igl/barycenter.h>

TEST_CASE("bunny-tet-coarsen")
{
    std::string filename = "../buildr/bunny-tet.h5";
    auto pc = std::make_shared<PrismCage>(filename);
    auto [vert_info, tet_info, vert_tet_conn] = prism::tet::reload(filename, pc.get());

    prism::local::RemeshOptions option(pc->mid.size(), 0.1);

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
            // if (BC(i, 0) < 0.1) sizes[i] = 5e-3;
        }
        sizer.reset(new prism::tet::SizeController(bgV, bgT, sizes));
    }
    option.collapse_quality_threshold = 150;
    //  spdlog::set_level(spdlog::level::trace);
    REQUIRE(prism::tet::tetmesh_sanity(tet_info, vert_info, vert_tet_conn, pc.get()));
    spdlog::enable_backtrace(100);
    for (auto i = 0; i < 5; i++) {
        prism::tet::vertexsmooth_pass(pc.get(), option, vert_info, tet_info, vert_tet_conn, 1.);
        prism::tet::faceswap_pass(pc.get(), option, vert_info, tet_info, vert_tet_conn, 1.);
        prism::tet::compact_tetmesh(vert_info, tet_info, vert_tet_conn, pc.get());
        option.target_adjustment.resize(pc->mid.size());
        prism::tet::collapse_pass(pc.get(), option, vert_info, tet_info, vert_tet_conn, sizer);
        prism::tet::edgeswap_pass(pc.get(), option, vert_info, tet_info, vert_tet_conn, 1.);
        prism::tet::collapse_pass(pc.get(), option, vert_info, tet_info, vert_tet_conn, sizer);
        prism::tet::vertexsmooth_pass(pc.get(), option, vert_info, tet_info, vert_tet_conn, 1);
        // prism::tet::compact_tetmesh(vert_info, tet_info, vert_tet_conn, pc.get());
        option.target_adjustment.resize(pc->mid.size());
    }
    prism::tet::serializer("../buildr/bunny-tet-coarse.h5", pc.get(), vert_info, tet_info);
}