#include "tetshell_utils.hpp"

#include <shell/Utils.h>
#include <tetwild/Logger.h>
#include <tetwild/tetwild.h>

#include <prism/cage_utils.hpp>
#include <prism/geogram/AABB.hpp>

#include <igl/vertex_triangle_adjacency.h>
#include <spdlog/spdlog.h>


namespace prism::tetshell {
RowMatd one_side_extrusion(RowMatd& V, const RowMati& F, RowMatd& VN, bool out)
{
    auto initial_step = 1e-6;
    spdlog::enable_backtrace(30);
    std::set<int> omni_singu;
    for (int i = 0; i < VN.rows(); i++) {
        if (VN.row(i).norm() < 0.5)
            omni_singu.insert(i);
        else
            break;
    }
    int pure_singularity = omni_singu.size();
    omni_singu.clear();
    prism::cage_utils::most_visible_normals(V, F, VN, omni_singu, -1.);
    for (auto i = 0; i < pure_singularity; i++) {
        VN.row(i).setZero();
        omni_singu.insert(i);
    }
    if (omni_singu.size() != pure_singularity) {
        spdlog::error("QP Computation introduces new singularity.");
        exit(1);
    }

    if (omni_singu.empty()) {
        spdlog::info("Succeessful Normals");
    } else {
        spdlog::info("Omni Singularity {} ", omni_singu.size());
        spdlog::trace("<Freeze> Omni \n {}", omni_singu);
    }
    auto tree = prism::geogram::AABB(V, F, false);
    std::vector<double> initial_steps(V.rows(), initial_step);
    auto out_steps = prism::cage_utils::volume_extrude_steps(
        V,
        F,
        out ? VN : (-VN),
        out,
        omni_singu.size(),
        initial_steps);

    // pool back to vertices
    std::vector<double> v_out(V.rows(), initial_step), v_in(V.rows(), initial_step);
    for (int i = 0; i < F.rows(); i++) {
        for (int j = 0; j < 3; j++) {
            auto vid = F(i, j);
            v_out[vid] = std::min(v_out[vid], out_steps[i]);
        }
    }

    RowMatd outer;
    if (out)
        outer = V + Eigen::Map<Eigen::VectorXd>(v_out.data(), v_out.size()).asDiagonal() * VN;
    else
        outer = V - Eigen::Map<Eigen::VectorXd>(v_out.data(), v_out.size()).asDiagonal() * VN;
    std::vector<Vec3d> mid, top;
    std::vector<Vec3i> vF;
    std::vector<std::vector<int>> VF, VFi;
    eigen2vec(V, mid);
    eigen2vec(outer, top);
    eigen2vec(F, vF);
    igl::vertex_triangle_adjacency(V, F, VF, VFi);
    if (out)
        prism::cage_utils::recover_positive_volumes(mid, top, vF, VN, VF, pure_singularity, out);
    else
        prism::cage_utils::recover_positive_volumes(top, mid, vF, VN, VF, pure_singularity, out);

    prism::cage_utils::hashgrid_shrink(mid, top, vF, VF);

    if (out)
        prism::cage_utils::recover_positive_volumes(mid, top, vF, VN, VF, pure_singularity, out);
    else
        prism::cage_utils::recover_positive_volumes(top, mid, vF, VN, VF, pure_singularity, out);
    RowMatd mtop;
    vec2eigen(top, mtop);
    return mtop;
}


/// filling

// label == 1 inside 2 outside
void tetshell_fill(
    const RowMatd& ext_base,
    const RowMatd& shell_base,
    const RowMatd& shell_top,
    const RowMatd& ext_top,
    const RowMati& F_sh,
    Eigen::MatrixXd& V_out,
    Eigen::MatrixXi& T_out,
    Eigen::VectorXi& label_output)
{
    int oneShellVertices = ext_base.rows();
    Eigen::MatrixXd V_in(4 * ext_base.rows(), 3);
    V_in << ext_base, shell_base, shell_top, ext_top;
    Eigen::MatrixXi F_in(4 * F_sh.rows(), 3);
    for (auto i = 0; i < 4; i++) {
        F_in.middleRows(i * F_sh.rows(), F_sh.rows()) = F_sh.array() + oneShellVertices * i;
    }
    Eigen::VectorXd quality_output;
    tetwild::Args args;
    args.initial_edge_len_rel = 0.25;
    args.tet_mesh_sanity_check = true;
    args.write_csv_file = false;
    args.is_quiet = true;
    args.postfix = "";
    tetwild::logger().set_level(spdlog::level::info);
    tetwild::tetrahedralization(V_in, F_in, V_out, T_out, quality_output, label_output, args);
};

} // namespace prism::tetshell