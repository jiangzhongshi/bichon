#include <doctest.h>

#include "cumin/tetshell_utils.hpp"
#include "prism/PrismCage.hpp"
#include "prism/geogram/AABB.hpp"

#include <shell/Utils.h>
#include <tetwild/Logger.h>
#include <tetwild/tetwild.h>

#include <spdlog/spdlog.h>
#include <highfive/H5Easy.hpp>

#include <igl/remove_unreferenced.h>
TEST_CASE("prepare-bunny-tetmesh")
{
    // conforming tetmeshing.
    auto pc = PrismCage("../buildr/bunny.off.h5");
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
    for (auto i=0; i<mV.rows(); i++) {
      REQUIRE_EQ(map_ind[i], i);
      REQUIRE_EQ(mV.row(i), new_tet_V.row(i));
    }
    pc.serialize(
        "../buildr/bunny-tet.h5",
        (std::function<void(HighFive::File&)>)[&](HighFive::File & file) {
            H5Easy::dump(file, "tet_v", new_tet_V);
            H5Easy::dump(file, "tet_t", new_tet_T);
        });
}