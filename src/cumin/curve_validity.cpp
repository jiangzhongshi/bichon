#include "curve_validity.hpp"

#include <igl/Hit.h>
#include <igl/boundary_facets.h>
#include <igl/boundary_loop.h>
#include <igl/cotmatrix.h>
#include <igl/invert_diag.h>
#include <igl/massmatrix.h>
#include <igl/matrix_to_list.h>
#include <igl/per_vertex_normals.h>
#include <igl/upsample.h>
#include <igl/vertex_triangle_adjacency.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <any>
#include <cumin/bernstein_eval.hpp>
#include <highfive/H5Easy.hpp>
#include <prism/intersections.hpp>
#include <prism/local_operations/remesh_pass.hpp>
#include <prism/phong/projection.hpp>

#include "curve_common.hpp"
#include "curve_utils.hpp"
#include "prism/PrismCage.hpp"
#include "prism/common.hpp"
#include "prism/geogram/AABB.hpp"
#include "prism/local_operations/validity_checks.hpp"
#include <prism/polyshell_utils.hpp>


auto compute_normals(const std::vector<RowMatd> &local_cp,
                     const std::array<RowMatd, 2> &duv_lv5) -> RowMatd {
  assert(duv_lv5[0].cols() == local_cp[0].rows());
  RowMatd cross = RowMatd::Zero(duv_lv5[0].rows() * local_cp.size(), 3);
  // spdlog::info("normal compute size: {} {}", cross.rows(), cross.cols());
  using MatdX3 = Eigen::Matrix<double, -1, 3, Eigen::RowMajor>;
  int fnum = local_cp.size();
  auto r = duv_lv5[0].rows();
  for (int c = 0; c < local_cp.size(); c++) { // cp  10 x 3d, du 45sample x 10
    auto &cp = local_cp[c];
    MatdX3 du = duv_lv5[0] * cp;
    MatdX3 dv = duv_lv5[1] * cp;
    for (int i = 0; i < r; i++) { // repeat 45
      cross.row(r * c + i) = du.row(i).cross(dv.row(i));
      cross.row(r * c + i) /= cross.row(r * c + i).norm();
    }
  }
  return std::move(cross);
};

void compute_amips(const std::vector<RowMatd> &local_cp,
                   const std::array<RowMatd, 2> &duv_lv5, RowMatd &amips) {
  assert(duv_lv5[0].cols() == local_cp[0].rows());
  amips.setZero(local_cp.size(), duv_lv5[0].rows());
  auto &du = duv_lv5[0], &dv = duv_lv5[1];
  int fnum = local_cp.size();
  auto r = duv_lv5[0].rows();
  using MatdX3 = Eigen::Matrix<double, -1, 3, Eigen::RowMajor>;
  for (int c = 0; c < local_cp.size(); c++) {
    auto &cp = local_cp[c];
    MatdX3 du = duv_lv5[0] * cp;
    MatdX3 dv = duv_lv5[1] * cp;
    for (int i = 0; i < r; i++) { // repeat 45
      auto e1 = du.row(i);
      auto e2 = dv.row(i);
      auto e1_len = e1.norm();
      auto e2_x = e1.dot(e2) / e1_len;
      auto e2_y = (e2 - e2_x * e1 / e1_len).norm();
      amips(c, i) = (std::pow(e1_len, 2) + std::pow(e2_x - e2_y / sqrt(3), 2) +
                     std::pow(e2_y, 2) * (4 / 3.)) /
                    (e1_len * 2 * e2_y / std::sqrt(3));
    }
  }
};

auto inverse_uv_transformer(Vec3d &uv, int poly_id, int oppo,
                            const std::vector<double> &segs) -> void {
  auto a = segs[poly_id];
  auto b = segs[poly_id + 1];
  auto o0 = oppo, o1 = (oppo + 1) % 3, o2 = (oppo + 2) % 3;
  auto u = uv[1], v = uv[2];
  uv[o1] = (1 - a) * u + (1 - b) * v;
  uv[o2] = (a)*u + (b)*v;
  uv[o0] = 1 - uv[o1] - uv[o2];
};

auto uv_transformer(Vec3d &uv, int oppo, const std::vector<double> &segs) {
  assert(segs.front() == 0. && segs.back() == 1.);
  auto u = uv[(oppo + 1) % 3], v = uv[(oppo + 2) % 3];
  auto w = uv[oppo];
  if (w >= 1) {
    // uv.setConstant(0);
    uv << 1, 0, 0;
    // uv[oppo] = 1;
    return 0;
  }
  if (v <= 0) {
    uv << w, u, 0;
    return 0;
  }
  auto v_scale = v / (1 - w);
  auto it1 = std::find_if(segs.begin(), segs.end(),
                          [v_scale](auto x) { return x >= v_scale; });
  assert(it1 != segs.begin() && ("v should not be negative"));
  auto it0 = std::prev(it1);
  auto a = *it0, b = *it1;
  assert(a < b);

  uv[1] = (b * (u + v) - v) / (b - a);
  uv[2] = (a * (u + v) - v) / (a - b);
  uv[0] = std::max(1 - uv[1] - uv[2], 0.);
  assert(uv[0] >= 0 && uv[1] >= 0 && uv[2] >= 0);
  int d = std::distance(segs.begin(), it0);
  return d;
};

auto zig_shell_sample_hit_with_uv_transform(
    const PrismCage &pc, const std::vector<Vec3i> &nbF,
    const std::vector<int> &sp_fid /*index into nb*/,
    const std::vector<Vec3d> &sp_uv, const std::set<int> &combined_track,
    std::vector<prism::Hit> &ray_hits) {
  auto f_to_list = std::vector<std::vector<int>>(nbF.size());
  assert(sp_fid.back() < nbF.size() && "only local index");
  for (auto i = 0; i < sp_fid.size(); i++) {
    auto f = sp_fid[i];
    f_to_list[f].push_back(i);
  }
  ray_hits.resize(sp_fid.size());
  std::vector<int> non_zig_shells;
  for (auto i = 0; i < nbF.size(); i++) {
    auto &f = nbF[i];
    auto [oppo_vid, rej_id, segs] =
        prism::local_validity::identify_zig(pc.meta_edges, f);
    if (oppo_vid == -1) {
      non_zig_shells.push_back(i);
      continue;
    }
    auto [v0, v1, v2] =
        std::tie(f[oppo_vid], f[(oppo_vid + 1) % 3], f[(oppo_vid + 2) % 3]);
    auto [local_base, local_mid, local_top, zig_tris, zig_shifts] =
        prism::local_validity::zig_constructor(pc, v0, v1, v2, segs, true);
    auto lens = std::vector<double>(zig_tris.size() + 1, 0.);
    for (auto i = 0; i < zig_tris.size(); i++) {
      auto s = zig_shifts[i];
      auto l0 = zig_tris[i][(4 - s) % 3], l1 = zig_tris[i][(5 - s) % 3];
      lens[i + 1] = lens[i] + (local_mid[l1] - local_mid[l0]).norm();
    }
    for (auto &l : lens)
      l /= lens.back();

    auto &local_list = f_to_list[i];
    std::vector<int> zig_f;
    std::vector<Vec3d> zig_uv;
    RowMatd db_uv0(local_list.size(), 3);
    RowMatd db_uv1(local_list.size(), 3);
    auto cnt = 0;
    for (auto local_i : local_list) {
      auto uv = sp_uv[local_i];
      auto nid = uv_transformer(uv, oppo_vid, lens);
      zig_f.push_back(nid);
      auto s = zig_shifts[nid];


      zig_uv.emplace_back(Vec3d(uv[s], uv[(s + 1) % 3], uv[(s + 2) % 3]));
      // {
      //   auto c = Vec3d(0, 0, 0);
      //   for (auto i : {0, 1, 2}) c += (local_mid[zig_tris[nid][i]] *
      //   zig_uv.back()[i]); auto c2 = Vec3d(0,0,0); for (auto i : {0, 1, 2})
      //   c2 += (pc.mid[f[i]] * sp_uv[local_i][i]); db_uv0.row(cnt ) = c;
      //   db_uv1.row(cnt ++) = c2;
      // }
    }
    std::vector<prism::Hit> hits_in_zig;
    prism::curve::sample_hit_discrete(
        local_base, local_mid, local_top, zig_tris, zig_f, zig_uv, pc.ref.V,
        pc.ref.F, *pc.ref.aabb, combined_track, hits_in_zig);
    for (int l = 0; l < local_list.size(); l++) {
      ray_hits[local_list[l]] = hits_in_zig[l];
    }
    // spdlog::info("diff {}" ,(db_uv0 -db_uv1).norm()/local_list.size());
    // if ((db_uv0 -db_uv1).norm()/local_list.size() > 100*1e-3)
    // [&]() {
    //   spdlog::dump_backtrace();
    //   H5Easy::File file("debug_query.h5", H5Easy::File::Overwrite);
    //   RowMatd mbase, mtop, mmid;
    //   RowMati mF;
    //   vec2eigen(pc.base, mbase);
    //   vec2eigen(pc.top, mtop);
    //   vec2eigen(pc.mid, mmid);
    //   vec2eigen(pc.F, mF);

    //   H5Easy::dump(file, "dbuv0", db_uv0);
    //   H5Easy::dump(file, "dbuv1", db_uv1);
    //   H5Easy::dump(file, "base", mbase);
    //   H5Easy::dump(file, "top", mtop);
    //   H5Easy::dump(file, "mid", mmid);
    //   H5Easy::dump(file, "F", mF);
    //   H5Easy::dump(file, "refV", pc.ref.V);
    //   H5Easy::dump(file, "refF", pc.ref.F);
    //   throw std::runtime_error("query failure");
    //   return false;
    // }();
  }
  for (auto f : non_zig_shells) {
    std::vector<int> cur_f;
    std::vector<Vec3d> cur_uv;
    for (auto i : f_to_list[f]) {
      cur_f.push_back(f);
      assert(f == sp_fid[i]);
      cur_uv.push_back(sp_uv[i]);
    }

    std::vector<prism::Hit> hits_in_nonzig;
    prism::curve::sample_hit_discrete(pc.base, pc.mid, pc.top, nbF, cur_f,
                                      cur_uv, pc.ref.V, pc.ref.F, *pc.ref.aabb,
                                      combined_track, hits_in_nonzig);
    auto &local_list = f_to_list[f];
    for (int i = 0; i < local_list.size(); i++) {
      ray_hits[local_list[i]] = hits_in_nonzig[i];
    }
  }
};

constexpr auto valid_curving =
    [](const PrismCage &pc, const std::vector<int> &old_ids,
       const std::vector<Vec3i> &nbF, const auto &refData,
       const MatLexMap<Eigen::VectorXi, Vec3d> &nb_cp_map,
       const auto &mat_helpers, const auto &residual_test,
       const auto& option, std::vector<RowMatd> &local_cp) -> bool {
  auto &base = pc.base, &top = pc.top, &mid = pc.mid;
  auto &F = pc.F;
  prism::geogram::AABB &reftree = *pc.ref.aabb;

  auto &[inpV, inpF, refVN, refHN, refVF] = refData;
  const auto &[tri3_cod, tri4_cod, tet4_cod] = mat_helpers;
  auto & helper = prism::curve::magic_matrices();
  auto &tri3_lv5 = helper.bern_val_samples;
  auto &tri3_duv_lv5 = helper.duv_samples;
  if (!option.linear_curve) {
    // 1. fitting
    // get foot points (as f,u,v)
    // this is intentionally in sync with the serialized tri10_lv5 file.
    const int ulevel = 3;
    std::vector<int> sp_fid;
    std::vector<Vec3d> sp_uv;
    auto [final_F, uniq_ind, allnode_map] =
        prism::curve::upsampled_uv(nbF, sp_fid, sp_uv);
    // raycast for b
    std::vector<prism::Hit> ray_hits;
    std::set<int> combined_track;
    for (auto f : old_ids) {
      for (auto i : pc.track_ref[f]) {  // temporary hack to prevent empty
                                        // raycast. It might eventually happen
                                        // on the open boundary. TODO
        for (int j = 0; j < 3; j++)
          set_add_to(refVF[inpF(i, j)], combined_track);
      }
    }
    spdlog::trace("oldids {}", old_ids);
    if (option.use_polyshell)
      zig_shell_sample_hit_with_uv_transform(pc, nbF, sp_fid, sp_uv,
                                             combined_track, ray_hits);
    else
      prism::curve::sample_hit_discrete(base, mid, top, nbF, sp_fid, sp_uv,
                                        pc.ref.V, pc.ref.F, reftree,
                                        combined_track, ray_hits);
    RowMatd ray_hit_pos(ray_hits.size(), 3);
    RowMatd ray_hit_n(ray_hits.size(), 3);

    for (int i = 0; i < ray_hits.size(); i++) {
      auto &hit = ray_hits[i];
      auto v0 = inpF(hit.id, 0), v1 = inpF(hit.id, 1), v2 = inpF(hit.id, 2);
      auto u = hit.u, v = hit.v;
      ray_hit_pos.row(i) =
          inpV.row(v0) * (1 - u - v) + inpV.row(v1) * u + inpV.row(v2) * v;
      ray_hit_n.row(i) =
          refVN.row(v0) * (1 - u - v) + refVN.row(v1) * u + refVN.row(v2) * v;
      ray_hit_n.row(i) /= ray_hit_n.row(i).norm();
    }

    RowMatd residual_in;
    RowMatd residual_out;
    auto linear_cp = prism::curve::initialize_cp(mid, nbF, tri3_cod);
    auto targ_cp = prism::curve::ring_fit(
        nbF, linear_cp, nb_cp_map, tri3_lv5, ray_hit_pos, tri3_cod,
        std::tuple(uniq_ind), residual_in, residual_out);
    assert(linear_cp.size() == targ_cp.size());
    local_cp = targ_cp;
    RowMatd cur_residual = residual_out;
    spdlog::debug("Fit: linear residual {}",
                  residual_in.rowwise().norm().mean());
    spdlog::debug("Fit: current residual {}",
                  cur_residual.rowwise().norm().mean());

    if (!residual_test(targ_cp)) {
      spdlog::debug("Fit: Fail out residual");
      return false;
    }
    spdlog::debug("Fit: Pass out residual");
    double alpha = 1.0;
    for (int trial = 0; trial < 20; trial++, alpha *= 0.8) {  // line search
      spdlog::trace("shrinking with alpha {}", alpha);
      for (int i = 0; i < nbF.size(); i++) {
        for (int j = 0; j < targ_cp[i].rows(); j++) {
          if (targ_cp[i].row(j) == linear_cp[i].row(j)) continue;
          local_cp[i].row(j) =
              targ_cp[i].row(j) * alpha + linear_cp[i].row(j) * (1 - alpha);
        }
      }
      if (!residual_test(targ_cp)) return false;
      if (prism::curve::elevated_positive(
              base, top, nbF,
              option.curve_recurse_check, local_cp)) {
        spdlog::debug("Fit: Passed");
        if (option.curve_normal_bound > -1.) {
          RowMatd est_normals = compute_normals(local_cp, tri3_duv_lv5);
          assert(tri3_duv_lv5[0].cols() == tri3_cod.rows());
          int num_samples = tri3_duv_lv5[0].rows();
          assert(allnode_map.cols() == num_samples);
          for (int i = 0; i < allnode_map.rows(); i++) {
            for (int j = 0; j < allnode_map.cols(); j++) {
              if (est_normals.row(i * num_samples + j)
                      .dot(ray_hit_n.row(allnode_map(i, j))) <
                  option.curve_normal_bound) {
                spdlog::debug("normal rejection: {}",
                              est_normals.row(i * num_samples + j)
                                  .dot(ray_hit_n.row(allnode_map(i, j))));
                return false;
              }
            }
          }
        }
        return true;
      }
    }
  } else { // stay linear.
    local_cp = prism::curve::initialize_cp(mid, nbF, tri3_cod);
    if (!residual_test(local_cp)) {
      spdlog::debug("Fit: Fail out linear");
      return false;
    }
    return true;
  }
  return false;
};

auto mean_curv_normals = [](const RowMatd &V, const RowMati &F) { // this is an inactive function for fitting normals.
  using namespace Eigen;
  RowMatd HN;
  SparseMatrix<double> L, M, Minv;
  igl::cotmatrix(V, F, L);
  igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_VORONOI, M);
  igl::invert_diag(M, Minv);
  HN = -Minv * (L * V);
  return HN;
};

std::pair<std::any, std::any>
prism::curve::curve_func_handles(std::vector<RowMatd> &complete_cp,
                                 const PrismCage &pc, const prism::local::RemeshOptions& option, int tri_order) {
  using namespace std;
  auto & helper = magic_matrices(tri_order, 3);

  RowMatd refVN;
  RowMati inpF = pc.ref.F;
  std::vector<std::vector<int>> VF, VFi;
  igl::vertex_triangle_adjacency(pc.ref.V, inpF, VF, VFi);
  igl::per_vertex_normals(pc.ref.V, pc.ref.F, refVN);
  spdlog::debug("compute VN {}", refVN.rows());
  // pre function, checker
  auto tet_order = tri_order + 1;

  auto codecs_tri = codecs_gen(tri_order, 2);
  auto tri_codec_v = RowMati();
  vec2eigen(codecs_tri, tri_codec_v);
  
  auto pre_curving =
      [&complete_cp, // complete cp is an actual input in this list
       tri3_cod = codecs_gen_id(tri_order,2),
       tri4_cod = codecs_gen_id(tet_order,2), tet4_cod = codecs_gen_id(tet_order,3),
       tri_codec_v, refVN, refVF = VF, inpV = pc.ref.inpV, inpF,
       poisson_on_face = pc.constraints_per_face, poisson_uv=pc.constraints_points_bc, &option]
      // end of the long capture list
      (const PrismCage &pc, const std::vector<int> &old_nb,
       const std::vector<Vec3i> &moved_tris,
       std::vector<RowMatd> &local_cp) -> bool {
    std::vector<Vec3i> old_tris;
    for (auto i : old_nb)
      old_tris.push_back(pc.F[i]);
    auto bnd_cp_maps = [&old_tris, &tri3_cod,
                        &old_nb](const std::vector<RowMatd> &complete_cp) {
      auto [local_fe, local_map] = global_entry_map(old_tris, tri3_cod);
      MatLexMap<Eigen::VectorXi, Vec3d> bnd_cp_maps;
      RowMati bnd_edges;
      igl::boundary_facets(
          Eigen::Map<const RowMati>(old_tris[0].data(), old_tris.size(), 3),
          bnd_edges);
      std::set<int> bnd_cp_id;
      for (int i = 0; i < bnd_edges.rows(); i++) {
        Vec3i v0 = {bnd_edges(i, 0), bnd_edges(i, 1), -1};
        for (auto ci = 0; ci < tri3_cod.rows(); ci++) {
          // for (auto c : tri3_cod) {
          auto c = tri3_cod.row(ci);
          if (c.maxCoeff() >= 2)
            continue; // internal node.
          auto cod = sort_slice(v0, c);
          auto id = local_map.at(cod);
          auto it = std::find(bnd_cp_id.begin(), bnd_cp_id.end(), id);
          if (it != bnd_cp_id.end()) // already in
            continue;
          bnd_cp_id.insert(id);
          auto [f, e] = local_fe[id];
          bnd_cp_maps[cod] = (complete_cp[old_nb[f]].row(e));
        }
      }
      return std::move(bnd_cp_maps);
    }(complete_cp);
    auto residual_test =
        [&pc, &moved_tris, &old_nb, &inpV, &tri_codec_v, &poisson_on_face, &poisson_uv,
         dist_th = option.curve_dist_bound](std::vector<RowMatd> &cp) -> bool {
      constexpr auto enable_polyshell = false;
      assert(cp.size() == moved_tris.size());
      auto inverse_project_discrete =
          [](const auto &base, const auto &mid, const auto &top, const auto &f,
             const auto &pp) -> std::optional<std::array<double, 3>> {
        auto [v0, v1, v2] = f;
        bool split = v1 > v2;
        auto tup = std::array<double, 3>();
        if (!prism::phong::phong_projection(
                {base[v0], base[v1], base[v2], mid[v0], mid[v1], mid[v2]}, pp,
                split, tup) &&
            (!prism::phong::phong_projection(
                {mid[v0], mid[v1], mid[v2], top[v0], top[v1], top[v2]}, pp,
                split, tup))) {
          // neither in top nor bottom.
          return {};
        }
        return tup;
      };
      // gather poisson points.
      std::vector<Vec3d> poisson_points_ref; // used for shell query
      std::vector<Vec3d> poisson_points_targ; // used for target compare.
      for (auto s : old_nb)
        for (auto f : pc.track_ref[s]) {
          for (auto pid : poisson_on_face[f]) {
            poisson_points_ref.push_back(Vec3d::Zero());
            poisson_points_targ.push_back(Vec3d::Zero());
            for (auto j=0; j<3; j++) {
              poisson_points_ref.back() += poisson_uv(pid,j)*pc.ref.V.row(pc.ref.F(f,j));
              poisson_points_targ.back() += poisson_uv(pid,j)*inpV.row(pc.ref.F(f,j));
            }
          }
        }

      auto cnt = 0;
      for (auto ppi = 0; ppi < poisson_points_ref.size(); ppi++) {
        auto pp = poisson_points_ref[ppi];
        bool point_checked = false;
        // iterate over all new shells.
        for (auto si = 0; si < moved_tris.size(); si++) {
          auto &f = moved_tris[si];
          auto tup = std::array<double, 3>{-1, -1, -1};

          auto oppo_vid = -1, rej_id = -1;
          auto segs = std::vector<int>();
          if (enable_polyshell)
            std::tie(oppo_vid, rej_id, segs) =
                prism::local_validity::identify_zig(pc.meta_edges, f);
          assert(oppo_vid != -10 &&
                 "this should always go after distort check.");
          if (oppo_vid >= 0) {
            // zig shell case;
            // the follow can be precomputed if needed.
            auto [v0, v1, v2] = std::tie(f[oppo_vid], f[(oppo_vid + 1) % 3],
                                         f[(oppo_vid + 2) % 3]);
            auto [local_base, local_mid, local_top, zig_tris, zig_shifts] =
                prism::local_validity::zig_constructor(pc, v0, v1, v2, segs,
                                                       true);
            auto lens = std::vector<double>(zig_tris.size() + 1, 0.);
            for (auto i = 0; i < zig_tris.size(); i++) {
              auto s = zig_shifts[i];
              auto l0 = zig_tris[i][(4 - s) % 3], l1 = zig_tris[i][(5 - s) % 3];
              lens[i + 1] = lens[i] + (local_mid[l1] - local_mid[l0]).norm();
            }
            for (auto &l : lens)
              l /= lens.back();
            // spdlog::info("zigtris {}", zig_tris);
            for (auto pid = 0; pid < zig_tris.size(); pid++) {
              auto &z = zig_tris[pid];
              auto o_tup = inverse_project_discrete(local_base, local_mid,
                                                    local_top, z, pp);
              if (o_tup) {
                tup = o_tup.value();
                auto uv = Vec3d(1 - tup[0] - tup[1], tup[0], tup[1]);
                for (auto j = 0; j < 3; j++) {
                  uv[j] = std::max(std::min(uv[j], 1.), 0.);
                }
                auto s = (3 - zig_shifts[pid]) % 3;
                uv = Vec3d(uv[s], uv[(s + 1) % 3], uv[(s + 2) % 3]);

                auto uv2 = uv;
                inverse_uv_transformer(uv, pid, oppo_vid, lens);

                tup[0] = uv[1];
                tup[1] = uv[2];
                break;
              }
            }
          } else {
            auto o_tup =
                inverse_project_discrete(pc.base, pc.mid, pc.top, f, pp);
            if (o_tup)
              tup = o_tup.value();
          }
          if (tup[0] == -1)
            continue; // go to next point
          point_checked = true;
          auto [u, v, _] = tup;
          auto bas_val = prism::curve::evaluate_bernstein(
              Eigen::VectorXd::Constant(1, u), Eigen::VectorXd::Constant(1, v),
              Eigen::VectorXd::Constant(1, 0),
              tri_codec_v); // tri10 x 1 array
          // cp[si]; // 10x3
          Vec3d curved_pos =
              (bas_val.matrix().asDiagonal() * cp[si]).colwise().sum();
          if ((curved_pos - poisson_points_targ[ppi]).squaredNorm() >
              dist_th * dist_th) {
            spdlog::trace("pp faraway {} {}", ppi, curved_pos);
            return false;
          }
          break; // pass the check for the current pp.
        }
        if (!point_checked)
          ; // ok, the point may not be covered here.
      }
      return true;
    };
    bool flag = valid_curving(
        pc, old_nb, moved_tris,
        std::forward_as_tuple(inpV, inpF, refVN, RowMatd(), refVF), bnd_cp_maps,
        std::forward_as_tuple(tri3_cod, tri4_cod, tet4_cod),
        residual_test, option, local_cp);
    if (!flag) {
      local_cp.clear();
    }
    return flag;
  };

  // post function handle. After succeeds. Notice that shift is no longer
  // necessary since it was already taken care of.
  auto post_curving = [&complete_cp](const std::vector<int> &old_fid,
                                     const vector<int> &new_fid,
                                     const vector<RowMatd> &local_cp) -> void {
    assert(new_fid.size() == local_cp.size());
    spdlog::debug("Post Assign");
    for (auto f : old_fid)
      complete_cp[f].setConstant(-1);
    for (int i = 0; i < new_fid.size(); i++) {
      if (new_fid[i] >= complete_cp.size())
        complete_cp.resize(new_fid[i] + 1);
      complete_cp[new_fid[i]] = local_cp[i];
    }
  };

  return std::pair(
      std::make_any<std::function<bool(
          const PrismCage &, const std::vector<int> &,
          const std::vector<Vec3i> &, std::vector<RowMatd> &)>>(pre_curving),
      std::make_any<function<void(const std::vector<int> &, const vector<int> &,
                                  const vector<RowMatd> &)>>(post_curving));
}