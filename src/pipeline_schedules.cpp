#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <igl/Timer.h>
#include <igl/avg_edge_length.h>
#include <igl/read_triangle_mesh.h>
#include <igl/remove_duplicate_vertices.h>
#include <igl/write_triangle_mesh.h>
#include <spdlog/spdlog.h>

#include <highfive/H5Easy.hpp>
#include <nlohmann/json.hpp>
#include <prism/cage_utils.hpp>
#include <prism/energy/prism_quality.hpp>
#include <prism/feature_utils.hpp>
#include <prism/local_operations/section_remesh.hpp>
#include <utility>

#include "cumin/curve_utils.hpp"
#include "cumin/curve_validity.hpp"
#include "prism/PrismCage.hpp"
#include "prism/cage_check.hpp"
#include "prism/geogram/AABB.hpp"
#include "prism/local_operations/remesh_pass.hpp"
#include "prism/local_operations/remesh_with_feature.hpp"
#include "prism/local_operations/retain_triangle_adjacency.hpp"
#include "prism/spatial-hash/self_intersection.hpp"

namespace prism::curve {
void localcurve_pass(const PrismCage &pc,
                     const prism::local::RemeshOptions &option);
}
auto post_collapse = [](auto &complete_cp) {
  if (complete_cp.empty())
    return;
  complete_cp.erase(std::remove_if(complete_cp.begin(), complete_cp.end(),
                                   [](auto &c) { return c(0, 0) == -1; }),
                    complete_cp.end());
};

auto reverse_feature_order = [](PrismCage &pc,
                                prism::local::RemeshOptions &option) {
  decltype(pc.meta_edges) meta;
  if (pc.meta_edges.empty())
    return;
  for (auto [a, b] : pc.meta_edges) {
    auto a1 = std::pair{a.second, a.first};
    auto b1 = b;
    b1.second = std::vector<int>(b.second.rbegin(), b.second.rend());
    meta.emplace(a1, b1);
  }
  pc.meta_edges = std::move(meta);
};

auto checker_in_main = [](const auto &pc, const auto &option, bool enable) {
  if (!enable)
    return;
  auto require = [&](bool b) {
    if (b == false) {
      spdlog::dump_backtrace();
      pc->serialize("temp.h5");
      throw std::runtime_error("Checker in main");
    }
  };
  require(prism::cage_check::cage_is_positive(*pc));
  require(prism::cage_check::cage_is_away_from_ref(*pc));
  require(
      prism::cage_check::verify_edge_based_track(*pc, option, pc->track_ref));
  require(prism::spatial_hash::self_intersections(pc->base, pc->F).empty());
  require(prism::spatial_hash::self_intersections(pc->top, pc->F).empty());
  spdlog::info("Verifier: Done Checking");
};

#include <igl/boundary_loop.h>
#include <igl/doublearea.h>
#include <igl/is_edge_manifold.h>
#include <igl/is_vertex_manifold.h>
#include <igl/per_face_normals.h>
double min_dihedral_angles(const RowMatd &V, const RowMati &F) {
  RowMatd FN;
  RowMati TT, TTi;
  double minangle = 1;
  igl::per_face_normals_stable(V, F, FN);

  igl::triangle_triangle_adjacency(F, TT, TTi);
  for (int i = 0; i < F.rows(); i++) {
    for (int j = 0; j < 3; j++) {
      if (TT(i, j) == -1)
        continue;
      double a = FN.row(TT(i, j)).dot(FN.row(i));
      minangle = std::min(minangle, a);
    }
  }
  return minangle;
};

bool check_feature_valid(const RowMatd &V, const RowMati &F,
                         const Eigen::VectorXi &corners, const RowMati &edges) {
  std::vector<std::vector<int>> VF, VFi;
  igl::vertex_triangle_adjacency(V.rows(), F, VF, VFi);
  if (corners.size() > 0 && corners.maxCoeff() >= V.rows() &&
      corners.minCoeff() < 0) {
    spdlog::critical("Wrong feature corners.");
    return false;
  }
  for (auto e = 0; e < edges.rows(); e++) {
    auto v0 = edges(e, 0), v1 = edges(e, 1);
    auto [f, j] = prism::vv2fe(v0, v1, F, VF);
    if (f == -1) {
      spdlog::critical("Non-existent feature edges.");
      return false;
    }
  }
  return true;
}

constexpr auto consistent_orientation = [](const RowMati &F) {
  std::set<std::pair<int, int>> half_edges;
  for (auto i = 0; i < F.rows(); i++) {
    for (auto j = 0; j < 3; j++) {
      auto v0v1 = std::pair(F(i, j), F(i, (j + 1) % 3));
      if (half_edges.find(v0v1) != half_edges.end())
        return false;
      half_edges.insert(v0v1);
    }
  }
  return true;
};

bool preconditions(const RowMatd &V, const RowMati &F,
                   const std::string &filename, bool shortcircuit = true) {
  spdlog::info("Preconditions: Checking consistency of input data, with short "
               "circuit {}",
               shortcircuit);
  if (F.rows() == 0) {
    spdlog::error("Precondition: Empty Mesh");
    return false;
  }
  Eigen::VectorXi BI;
  if (!igl::is_vertex_manifold(F, BI) || !igl::is_edge_manifold(F)) {
    spdlog::error("Precondition: Input {} is not manifold", filename);
    return false;
  }

  if (!consistent_orientation(F)) {
    spdlog::error("Precondition: Input {} is not oriented correctly", filename);
    return false;
  }
  RowMati B;
  igl::boundary_facets(F, B);
  if (B.size() > 0) {
    spdlog::info("Precondition: Input {} has boundary", filename);
    return false;
  }

  Eigen::VectorXd M;
  igl::doublearea(V, F, M);
  double minarea = M.minCoeff() / 2;
  if (minarea < 1e-8) {
    spdlog::error("Precondition: Input {} area small {}", filename, minarea);
    if (shortcircuit)
      return false;
  }
  double minangle = min_dihedral_angles(V, F);
  spdlog::info("Minimum Angle {:.18f}", minangle);
  if (minangle < -1 + 0.0006091729809042379) { // smaller than 2 degree
    spdlog::error("Precondition: Input {} flat angle {}", filename, minangle);
    if (shortcircuit)
      return false;
  }
  // numerical self intersection
  prism::geogram::AABB tree(V, F, true);
  if (tree.numerical_self_intersection(1e-6)) {
    spdlog::error("Precondition: Input {} N-self intersects", filename);
    if (shortcircuit)
      return false;
  }
  std::vector<Vec3d> vecV;
  std::vector<Vec3i> vecF;
  eigen2vec(V, vecV);
  eigen2vec(F, vecF);
  if (!prism::spatial_hash::self_intersections(vecV, vecF).empty()) {
    spdlog::critical("Precondition: Input {} self intersects", filename);
    return false;
  }
  return true;
}

/// filling

RowMatd one_side_extrusion(RowMatd &V, const RowMati &F, RowMatd &VN,
                           bool out) {
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
      V, F, out ? VN : (-VN), out, omni_singu.size(), initial_steps);

  // pool back to vertices
  std::vector<double> v_out(V.rows(), initial_step),
      v_in(V.rows(), initial_step);
  for (int i = 0; i < F.rows(); i++) {
    for (int j = 0; j < 3; j++) {
      auto vid = F(i, j);
      v_out[vid] = std::min(v_out[vid], out_steps[i]);
    }
  }

  RowMatd outer;
  if (out)
    outer =
        V +
        Eigen::Map<Eigen::VectorXd>(v_out.data(), v_out.size()).asDiagonal() *
            VN;
  else
    outer =
        V -
        Eigen::Map<Eigen::VectorXd>(v_out.data(), v_out.size()).asDiagonal() *
            VN;
  std::vector<Vec3d> mid, top;
  std::vector<Vec3i> vF;
  std::vector<std::vector<int>> VF, VFi;
  eigen2vec(V, mid);
  eigen2vec(outer, top);
  eigen2vec(F, vF);
  igl::vertex_triangle_adjacency(V, F, VF, VFi);
  if (out)
    prism::cage_utils::recover_positive_volumes(mid, top, vF, VN, VF,
                                                pure_singularity, out);
  else
    prism::cage_utils::recover_positive_volumes(top, mid, vF, VN, VF,
                                                pure_singularity, out);

  prism::cage_utils::hashgrid_shrink(mid, top, vF, VF);

  if (out)
    prism::cage_utils::recover_positive_volumes(mid, top, vF, VN, VF,
                                                pure_singularity, out);
  else
    prism::cage_utils::recover_positive_volumes(top, mid, vF, VN, VF,
                                                pure_singularity, out);
  RowMatd mtop;
  vec2eigen(top, mtop);
  return mtop;
}
#include <shell/Utils.h>
#include <tetwild/Logger.h>
#include <tetwild/tetwild.h>
// label == 1 inside 2 outside
void tetshell_fill(const RowMatd &ext_base, const RowMatd &shell_base,
                   const RowMatd &shell_top, const RowMatd &ext_top,
                   const RowMati &F_sh, Eigen::MatrixXd &V_out,
                   Eigen::MatrixXi &T_out, Eigen::VectorXi &label_output) {
  int oneShellVertices = ext_base.rows();
  Eigen::MatrixXd V_in(4 * ext_base.rows(), 3);
  V_in << ext_base, shell_base, shell_top, ext_top;
  Eigen::MatrixXi F_in(4 * F_sh.rows(), 3);
  for (auto i = 0; i < 4; i++) {
    F_in.middleRows(i * F_sh.rows(), F_sh.rows()) =
        F_sh.array() + oneShellVertices * i;
  }
  Eigen::VectorXd quality_output;
  tetwild::Args args;
  args.initial_edge_len_rel = 0.25;
  args.tet_mesh_sanity_check = true;
  args.write_csv_file = false;
  args.is_quiet = true;
  args.postfix = "";
  tetwild::logger().set_level(spdlog::level::info);
  tetwild::tetrahedralization(V_in, F_in, V_out, T_out, quality_output,
                              label_output, args);
};

#include <cumin/high_order_optimization.hpp>
#include <filesystem>

#include "cumin/stitch_surface_to_volume.hpp"

bool checker_inversion(const PrismCage &pc,
                       const std::vector<RowMatd> &complete_cp) {
  for (auto i = 0; i < pc.F.size(); i++) {
    auto check = prism::curve::elevated_positive(pc.base, pc.top, {pc.F[i]},
                                                 true, {complete_cp[i]});
    if (!check) {
      spdlog::critical("i {}, F {}", i, pc.F[i]);
      return false;
    };
  }
  return true;
};

void cutet_optim(RowMatd &lagr, RowMati &p4T, nlohmann::json config) {
  auto debugMode = config["debug"];
  auto passes = config["passes"];
  auto smoothingIt = config["smooth_iter"];
  auto newEnergyThres = config["energy_threshold"];
  auto smooth_only = config["smooth_only"];

  auto threadNum = -1;
  igl::Timer igl_timer;
  igl_timer.start();
  auto &helper = prism::curve::magic_matrices(-1, -1);
  const auto tri10_lv5 = helper.bern_val_samples;
  const auto elevlag_from_bern = helper.elev_lag_from_bern;
  const auto vec_dxyz = helper.volume_data.vec_dxyz;
  const auto bern_from_lagr_o4 = helper.volume_data.vol_bern_from_lagr;
  const auto bern_from_lagr_o9 = helper.volume_data.vol_jac_bern_from_lagr;
  const auto codecs_o4_bc = helper.volume_data.vol_codec;
  const auto codecs_o9_bc = helper.volume_data.vol_jac_codec;
  // magic matrices
  spdlog::info("vecdxyz[{}] {}x{}", vec_dxyz.size(), vec_dxyz[0].rows(),
               vec_dxyz[0].cols());

  // codecs: reorder matrix

  // load the stitched file
  spdlog::info("lagr size {}x{}", lagr.rows(), lagr.cols());
  spdlog::info("cell size {}x{}", p4T.rows(), p4T.cols());

  auto InversionCheckForAll = [&](std::string str) {
    if (!(prism::curve::InversionCheck(lagr, p4T, codecs_o4_bc, codecs_o9_bc,
                                       bern_from_lagr_o4, bern_from_lagr_o9))) {
      spdlog::error("Inversion check failure: " + str + "  ");
      throw std::runtime_error("Inversion check failure");
    }
  };

  auto SaveToFile = [&](auto name) {
    auto energy = prism::curve::energy_evaluation(lagr, p4T, vec_dxyz);
    auto file = H5Easy::File(name, H5Easy::File::ReadWrite);
    H5Easy::dump(file, "energy", energy);
    H5Easy::dump(file, "lagr", lagr);
    H5Easy::dump(file, "cells", p4T);
  };

  // pre-optimization check
  InversionCheckForAll("Before Optimization");
  auto e = prism::curve::energy_evaluation(lagr, p4T, vec_dxyz);
  spdlog::info(
      "Before Optimization: meanE={:.2f} minE={:.2f} maxE={:.2f} | tetCnt={}",
      e.mean(), e.minCoeff(), e.maxCoeff(), e.size());

  // optimization
  if (threadNum == -1)
    threadNum = 16;
  int col = -1, swa = -1;
  for (int pass = 1; pass <= passes; pass++) {
    spdlog::info("======== Optimization Pass {}/{} ========", pass, passes);
    if (!smooth_only) {
      col = prism::curve::cutet_collapse(lagr, p4T, newEnergyThres);
      if (debugMode)
        InversionCheckForAll(fmt::format("Pass {} after collapsing", pass));

      swa = prism::curve::cutet_swap(lagr, p4T, newEnergyThres);
      if (debugMode)
        InversionCheckForAll(fmt::format("Pass {} after swapping", pass));
    }
    prism::curve::vertex_star_smooth(lagr, p4T, smoothingIt, threadNum);
    if (debugMode)
      InversionCheckForAll(fmt::format("Pass {} after smoothing", pass));
    if (col + swa == 0)
      break;
  }

  // post-optimization check
  InversionCheckForAll("After Optimization");

  // save file
  SaveToFile(config["output_file"]);

  double stageTime = igl_timer.getElapsedTime();
  spdlog::info("Total time for curved optimization = {}s", stageTime);
}

void volume_stage(PrismCage &pc, std::vector<RowMatd> &complete_cp,
                  nlohmann::json config) {
  RowMatd mT, mB, vtop, vbase;
  RowMati mF;
  vec2eigen(pc.top, mT);
  vec2eigen(pc.base, mB);
  vec2eigen(pc.F, mF);

  RowMatd VN = mT - mB;
  for (auto i = 0; i < VN.rows(); i++) {
    if (mT.row(i) == mB.row(i))
      VN.row(i).setZero();
    VN.row(i) = VN.row(i).normalized();
  }

  if (!prism::cage_check::cage_is_positive(pc)) {
    spdlog::error("Input is not positive. Halt.");
    return;
  }

  spdlog::info("== TOP ==");
  vtop = one_side_extrusion(mT, mF, VN, true);
  spdlog::info("== BOTTOM ==");
  vbase = one_side_extrusion(mB, mF, VN, false);

  Eigen::MatrixXd Vmsh;
  Eigen::MatrixXi Tmsh;
  Eigen::VectorXi labels;
  tetshell_fill(vbase, mB, mT, vtop, mF, Vmsh, Tmsh, labels);
  spdlog::debug("Tmsh {}", Tmsh.rows());
  std::vector<Eigen::VectorXi> T1;
  for (auto l = 0; l < labels.size(); l++) {
    if (labels[l] == 1)
      T1.emplace_back(Tmsh.row(l));
  }
  vec2eigen(T1, Tmsh);

  spdlog::debug("Tmsh {}", Tmsh.rows());

  auto &helper = prism::curve::magic_matrices(-1, -1);
  const auto elevlag_from_bern = helper.elev_lag_from_bern;
  const auto vec_dxyz = helper.volume_data.vec_dxyz;

  RowMatd nodes;
  RowMati p4T;

  prism::curve::stitch_surface_to_volume(mB, mT, mF, complete_cp, Vmsh, Tmsh,
                                         nodes, p4T);

  cutet_optim(nodes, p4T, config["cutet"]);
};

#include <igl/barycentric_coordinates.h>
////////////////////////
//// This is the main entry point for the curve mesh generation program.
//// @Params:
//// Mesh path. igl reader, supports obj, stl, off, ply
//// feature (incl. constraint points) path: h5 file.
//// Serialization path: h5 file
//// Config JSON
////////////////////////
void feature_and_curve(std::string filename, std::string fgname,
                       std::string ser_file, nlohmann::json config) {
  // Options
  auto curve_cf = config["curve"];
  auto shell_cf = config["shell"];
  auto featr_cf = config["feature"];
  auto control_cfg = config["control"];
  auto order = curve_cf["order"].get<int>();
  auto dist_th = curve_cf["distance_threshold"].get<double>();
  auto normal_th = curve_cf["normal_threshold"].get<double>();
  auto pre_split_threshold = featr_cf["initial_split_edge"].get<double>();
  auto initial_thickness = shell_cf["initial_thickness"].get<double>();
  auto target_edge_length = shell_cf["target_edge_length"].get<double>();
  auto serialize_level = control_cfg["serialize_level"].get<int>();
  auto freeze_feature = control_cfg["freeze_feature"].get<bool>();

  auto dihedral_threshold = featr_cf["dihedral_threshold"].get<double>();

  auto parse_feature_file = [&](const auto &V, const auto &F,
                                std::string fgname) {
    RowMati feature_edges;
    Eigen::VectorXi feature_corners;
    Eigen::VectorXi points_fid;
    RowMatd points_bc;
    if (fgname == "") { // no feature file, use threshold mark.
      spdlog::info("Use dihedral threshold {} to mark features",
                   dihedral_threshold);
      prism::mark_feature_edges(V, F, dihedral_threshold, feature_edges);
    } else {
      auto ext = std::filesystem::path(fgname).extension();
      if (ext == ".fgraph") {
        prism::read_feature_graph(fgname, feature_corners, feature_edges);
      } else if (ext == ".h5") {
        prism::read_feature_h5(fgname, feature_corners, feature_edges,
                               points_fid, points_bc);
      }
      spdlog::info("Parse Feature N {} E {} Constraint Points {}",
                   feature_corners.size(), feature_edges.rows(),
                   points_fid.size());
    }
    return std::tuple(feature_corners, feature_edges, points_fid, points_bc);
  };
  ///////
  auto pc = std::unique_ptr<PrismCage>(nullptr);
  auto complete_cp = std::vector<RowMatd>();
  auto ext = std::filesystem::path(filename).extension();
  if (ext == ".init" || ext == ".h5") { // loading.
    pc.reset(new PrismCage(filename));
    if (ext == ".h5") {
      if (!control_cfg["reset_cp"] && control_cfg["enable_curve"])
        complete_cp = prism::curve::load_cp(filename);
    }
  }

  if (pc == nullptr) { // initialize shell.
    RowMatd V;
    RowMati F;
    {
      igl::read_triangle_mesh(filename, V, F);
      if (fgname == "") { // no feature, stl file TODO: branch can be merged,
                          // subject to futher feature cleaning.
        Eigen::VectorXi SVI, SVJ;
        RowMatd temp_V = V; // for STL file
        igl::remove_duplicate_vertices(temp_V, 0, V, SVI, SVJ);
        for (int i = 0; i < F.rows(); i++)
          for (int j : {0, 1, 2})
            F(i, j) = SVJ[F(i, j)];
      }

      spdlog::info("V={}, F={}", V.rows(), F.rows());
      put_in_unit_box(V);
      if (preconditions(V, F, filename,
                        !control_cfg["danger_relax_precondition"]) == false)
        return;
    }
    RowMati feature_edges;
    Eigen::VectorXi feature_corners;
    Eigen::VectorXi points_fid;
    RowMatd points_bc;
    std::tie(feature_corners, feature_edges, points_fid, points_bc) =
        parse_feature_file(V, F, fgname);
    if (!check_feature_valid(V, F, feature_corners, feature_edges))
      return;

    std::vector<int> face_parent(F.rows());
    for (auto fi = 0; fi < F.rows(); fi++)
      face_parent[fi] = fi;
    RowMatd origV = V;
    RowMati origF = F;
    prism::feature_pre_split(V, F, feature_edges, pre_split_threshold,
                             face_parent);
    auto assign_constraints_to_new_faces =
        [](const RowMatd &oV, const RowMati &oF, const RowMatd &nV,
           const RowMati &nF, const std::vector<int> &face_parent,
           Eigen::VectorXi &points_fid, RowMatd &points_bc) {
          auto eval_bc = [](auto &V, auto &f, auto &bc) {
            Vec3d pos = Vec3d::Zero();
            for (auto k = 0; k < 3; k++)
              pos += V.row(f[k]) * bc[k];
            return pos;
          };
          if (points_fid.size() == 0)
            return;
          std::vector<std::vector<int>> face_map(oF.rows());
          for (auto i = 0; i < face_parent.size(); i++)
            face_map[face_parent[i]].push_back(i);
          for (auto p = 0; p < points_fid.size(); p++) {
            auto f = points_fid(p);
            Vec3d bc = points_bc.row(p);
            auto &children = face_map[f];
            Vec3d pos = eval_bc(oV, oF.row(f), bc);

            Vec3d best_bc;
            auto nfid = -1;
            auto best_error = 1.;
            // lets rank them
            for (auto c : children) {
              Vec3d nbc;
              igl::barycentric_coordinates(pos, nV.row(nF(c, 0)),
                                           nV.row(nF(c, 1)), nV.row(nF(c, 2)),
                                           nbc);
              auto npos = eval_bc(nV, nF.row(c), nbc);
              auto error = ((npos - pos).squaredNorm());
              if (best_error > error) {
                best_error = error;
                nfid = c;
                best_bc = nbc;
              }
            }
            if (best_error > 1e-5 || nfid == -1) {
              spdlog::critical("Constraints Points Re-assign.");
            }

            points_fid[p] = nfid;
            points_bc.row(p) = best_bc;
          }
        };
    assign_constraints_to_new_faces(origV, origF, V, F, face_parent, points_fid,
                                    points_bc);

    pc.reset(new PrismCage(V, F, std::move(feature_edges),
                           std::move(feature_corners), std::move(points_fid),
                           std::move(points_bc), initial_thickness,
                           PrismCage::SeparateType::kShell));
    prism::cage_check::initial_trackee_reconcile(
        *pc, shell_cf["distortion_bound"].get<double>());
    control_cfg["reset_cp"] = true;
    spdlog::info("=====Initial Good. Saving.", ser_file);
    pc->serialize(ser_file + ".init");
  }
  if (control_cfg["only_initial"])
    return;

  auto chains = prism::recover_chains_from_meta_edges(pc->meta_edges);
  spdlog::info("chains size {}", chains.size());
  if (control_cfg["enable_curve"] && control_cfg["reset_cp"] &&
      complete_cp.size() != pc->F.size()) {
    spdlog::info("order {} Reset CP", order);
    complete_cp =
        prism::curve::initialize_cp(pc->mid, pc->F, codecs_gen_id(order, 2));
  }
  if (pc->ref.inpV.rows() == 0) {
    spdlog::info("resetting inpV");
    pc->ref.inpV = pc->ref.V;
  }
  prism::local::RemeshOptions option(pc->mid.size(), 0.1);
  option.use_polyshell = featr_cf["enable_polyshell"].get<bool>();
  option.dynamic_hashgrid = true;
  option.distortion_bound = shell_cf["distortion_bound"].get<double>();
  option.target_thickness = shell_cf["target_thickness"].get<double>();
  option.curve_recurse_check = curve_cf["recursive_check"].get<bool>();

  spdlog::enable_backtrace(100);
  option.collapse_quality_threshold = 30;
  option.collapse_valence_threshold = 10;
  option.parallel = false;
  option.curve_dist_bound = dist_th;
  option.curve_normal_bound = normal_th;
  option.linear_curve = true;
  if (control_cfg["enable_curve"]) {
    option.curve_checker =
        prism::curve::curve_func_handles(complete_cp, *pc, option, order);
  }

  auto checker = [&option, &pc](bool enable) {
    checker_in_main(pc, option, enable);
  };
  auto collapse = [&]() {
    option.relax_quality_threshold = 30;
    checker(serialize_level > 7);
    int col = prism::local::wildcollapse_pass(*pc, option);
    post_collapse(complete_cp);
    checker(serialize_level > 7);
    for (auto i = 0; i < 2; i++) {
      if (!freeze_feature) {
        if (option.use_polyshell)
          col += prism::local::zig_collapse_pass(*pc, option);
        else
          col += prism::local::feature_collapse_pass(*pc, option);
        post_collapse(complete_cp);
      }
      checker(serialize_level > 7);
      reverse_feature_order(*pc, option);
    }
    checker(serialize_level > 3);
    return col;
  };
  auto relax = [&]() {
    option.relax_quality_threshold = 0;
    if (!freeze_feature) {
      if (option.use_polyshell) {
        prism::local::zig_slide_pass(*pc, option);
      } else {
        prism::local::feature_slide_pass(*pc, option);
      }
    }
    checker(serialize_level > 7);
    prism::local::localsmooth_pass(*pc, option);
    checker(serialize_level > 7);
    prism::local::wildflip_pass(*pc, option);
    checker(serialize_level > 7);
    prism::curve::localcurve_pass(*pc, option);
    checker(serialize_level > 3);
  };
  auto refine = [&](auto q) {
    if (q) { // try to refine everything if quality improves.
      option.relax_quality_threshold = 0;
      option.sizing_field = [](auto &) { return 0.02; };
    } else {
      // try to refine towards target length if quality not destroy
      option.relax_quality_threshold = 30;
      // double ael = igl::avg_edge_length(
      // Eigen::Map<RowMatd>(pc->mid[0].data(), pc->mid.size(), 3),
      // Eigen::Map<RowMati>(pc->F[0].data(), pc->F.size(), 3));
      option.sizing_field = [target_edge_length](auto &) {
        return target_edge_length;
      };
    }
    auto spl = prism::local::wildsplit_pass(*pc, option);
    if (!freeze_feature) {
      if (option.use_polyshell)
        spl += prism::local::zig_split_pass(*pc, option);
      else
        spl += prism::local::feature_split_pass(*pc, option);
    }
    checker(serialize_level > 4);
    option.sizing_field = [target_edge_length](auto &) {
      return target_edge_length;
    };
    return spl;
  };

  // Start collapse schedule.
  option.sizing_field = [target_edge_length](auto &) {
    return target_edge_length;
  };
  for (int collapse_iteration = 0; collapse_iteration < 10;
       collapse_iteration++) {
    if (control_cfg["skip_collapse"])
      break;
    if (collapse_iteration == 1)
      option.linear_curve = false;
    spdlog::info("===Collapse Iteration {}", collapse_iteration);
    auto col = collapse();
    relax();
    refine(true);
    relax();
    if (col == 0)
      break;
    reverse_feature_order(*pc, option);
    if (serialize_level > 4)
      pc->serialize(fmt::format("{}_col{}.h5", ser_file, collapse_iteration),
                    prism::curve::save_cp(complete_cp));
  }

  spdlog::info("========Done with Collapse. Try Split Now.======");
  option.sizing_field = [target_edge_length](auto &) {
    return target_edge_length;
  };

  // Start split schedule.
  for (int split_iteration = 0; split_iteration < 10; split_iteration++) {
    if (control_cfg["skip_split"])
      break;
    auto spl = refine(split_iteration > 4);
    for (int inside_improve_iteration = 0; inside_improve_iteration < 3;
         inside_improve_iteration++) {
      relax();
      collapse();
      relax();
      reverse_feature_order(*pc, option);
      if (serialize_level > 8)
        pc->serialize(fmt::format("{}_spl{}_imp{}.h5", ser_file,
                                  split_iteration, inside_improve_iteration),
                      prism::curve::save_cp(complete_cp));
    }
    if (spl == 0)
      break;
    if (serialize_level > 4)
      pc->serialize(fmt::format("{}_spl{}.h5", ser_file, split_iteration),
                    prism::curve::save_cp(complete_cp));
  }
  spdlog::info("========Finalize: Save.======");
  pc->serialize(ser_file, prism::curve::save_cp(complete_cp));
  if (control_cfg["skip_volume"])
    return;
  checker_inversion(*pc, complete_cp);
  spdlog::info("========Vol Stage======");

  config["cutet"]["output_file"] = ser_file;
  volume_stage(*pc, complete_cp, config);
}

/*
import h5py
import meshio
import numpy as np
with h5py.File('bunny.off.h5.test.h5.h5','r') as fp:
    lagr, p4T  = fp['lagr'][()], fp['cells'][()]
reorder = np.array([ 0,  1,  2,  3,  4, 16,  5,  7, 18,  9,  8, 17,  6, 13, 19,
10, 15, 21, 12, 14, 20, 11, 22, 24, 23, 25, 26, 31, 27, 32, 29, 33, 28, 30, 34])
meshio.write('file.msh',meshio.Mesh(points=lagr, cells=[('tetra35',
p4T[:,reorder])]), binary=False)
*/