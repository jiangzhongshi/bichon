#include "curve_utils.hpp"

#include <igl/Hit.h>
#include <igl/boundary_facets.h>
#include <igl/boundary_loop.h>
#include <igl/linprog.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/slice.h>
#include <igl/upsample.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <highfive/H5Easy.hpp>
#include <map>
#include <prism/cgal/triangle_triangle_intersection.hpp>
#include <prism/geogram/AABB.hpp>
#include <prism/intersections.hpp>
#include <prism/local_operations/retain_triangle_adjacency.hpp>
#include <prism/phong/projection.hpp>
#include <prism/predicates/triangle_triangle_intersection.hpp>
#include <utility>
#include "cumin/inversion_check.hpp"
#include "curve_common.hpp"
#include "prism/cage_utils.hpp"

std::vector<RowMatd> prism::curve::initialize_cp(const std::vector<Vec3d> &mid,
                                                 const std::vector<Vec3i> &F,
                                                 const RowMati &codec) {
  std::vector<RowMatd> complete_cp(F.size(), RowMatd::Zero(codec.rows(), 3));
  // duplicate storage, per suggestion of TS
  for (int f = 0; f < F.size(); f++) {
    for (auto c = 0; c < codec.rows(); c++) {
      auto cod = codec.row(c);
      for (auto k = 0; k < cod.size(); k++)
        complete_cp[f].row(c) += mid[F[f][cod[k]]];
    }
    complete_cp[f] /= codec.cols();
  }
  return complete_cp;
}

namespace prism::curve {
constexpr auto sf_index_a_b = [](bool type){
  
};
// Given the Lagrange points for order-d triangle and the flat base
// Propogate the control points for of three d tetra that consistitutes hex
// with the heights are scaled linearly
// Input:
//    LinearBase, LinearTop, HighOrderTop, SplitType
// Output:
//    #Tet(3) x #CtPt(35 when d=4) x #Dim(3)
// Notes:
//   To be run after going from surface triangle order 3->4
std::vector<RowMatd> surface_to_decomposed_tetra(const RowMatd &base, const RowMatd &mid_ho,
                                 const RowMatd &top, bool degenerate, bool type,
                                 const RowMati &tri_cod,
                                 const RowMati &tet_cod) {
  assert(base.rows() == 3 && base.cols() == 3 && "linear base");

  assert(tri_cod.cols() == tet_cod.cols());
  // tetmesh from shell.
  auto tet = type ? TETRA_SPLIT_A : TETRA_SPLIT_B;

  RowMatd height_fraction = RowMatd::Zero(tet.size(), tet_cod.rows()); // T3 x 35
  if (!degenerate) {
  for (int i = 0; i < tet.size(); i++) {
    for (int c = 0; c < tet_cod.rows(); c++) {
      auto cod = tet_cod.row(c);
      auto h = 0;
      for (auto ki = 0; ki < cod.size(); ki++)
        h += tet[i][cod[ki]] / 3; // integer division to get top or bottom.
      height_fraction(i, c) = (h + 0.) / cod.size();
    }
  }
  } else {
    for (int i = 1; i < tet.size(); i++) {
      for (int c = 0; c < tet_cod.rows(); c++) {
        auto cod = tet_cod.row(c);
        auto h = 0;
        auto cnt_sing = 0;
        for (auto ki = 0; ki < cod.size(); ki++) {
          h += tet[i][cod[ki]] / 3;  // integer division to get top or bottom.
          if (tet[i][cod[ki]] == 0 || tet[i][cod[ki]] == 3) cnt_sing++;
        }
        if (cnt_sing < cod.size())
          height_fraction(i, c) = (h + 0.) / (cod.size() - cnt_sing);
        else 
          height_fraction(i, c) = 0.;
      }
    }
  }

  // this map locally matches base-height-top.
  CodecMap triplet2index;
  for (int i = 0; i < tri_cod.rows(); i++) {
    triplet2index.emplace(tri_cod.row(i), i);
  }
  RowMati sf_index(tet.size(), tet_cod.rows()); // T3 x 35
  for (int i = 0; i < tet_cod.rows(); i++) {
    auto tc = tet_cod.row(i);
    for (int ts = 0; ts < tet.size(); ts++) {
      Eigen::RowVectorXi tet_tc = tc;
      for (auto j = 0; j < tet_tc.size(); j++)
        tet_tc[j] = tet[ts][tet_tc[j]] % 3;
      std::sort(tet_tc.data(), tet_tc.data() + tet_tc.size());
      sf_index(ts, i) = triplet2index.at(tet_tc);
    }
  }

  // all the above computation can be offline.

  auto elev_base = linear_elevate(base, tri_cod); // 15 x 3
  RowMatd elev_dir = mid_ho - elev_base;
  auto elev_top = linear_elevate(top, tri_cod); // 15 x 3
  RowMatd elev_dir_top = mid_ho - elev_top;

  // result is the sum of
  // elev_base[sf_index] ((T3 x 35) x D3) + frac_h (T3 x 35) *
  // dir_ho[sf_index]
  assert(elev_base.cols() == 3);
  // T6 {N35.D3}
  auto result =
      std::vector<RowMatd>(6, RowMatd(sf_index.cols(), elev_base.cols()));

  for (int i = degenerate?1:0; i < 3; i++)
    for (int j = 0; j < sf_index.cols(); j++)
      for (int k = 0; k < elev_base.cols(); k++) {
        result[i](j, k) = elev_base(sf_index(i, j), k) +
                          height_fraction(i, j) * elev_dir(sf_index(i, j), k);
        result[i + 3](j, k) =
            elev_top(sf_index(i, j), k) +
            (1-height_fraction(i, j)) * elev_dir_top(sf_index(i, j), k);
      }

  if (degenerate) {
    result.erase(result.begin() + 3);
    result.erase(result.begin());
  }

  return std::move(
      result); //#tet_elev_base +
               // Eigen::Map<Eigen::VectorXd>(height_fraction.data(),height_fraction.size()).asDiagonal()
               //* tet_elev_dir;
}

void my_upsample(std::vector<Vec3d> &V, std::vector<Vec3i> &F, int level) {
  auto single_level = [](auto &V, auto &F) {
    int v_num = V.size();
    int f_num = F.size();
    F.resize(f_num * 4);
    std::map<std::pair<int, int>, int> visited;
    for (int i = 0; i < f_num; i++) {
      Vec3i eid;
      for (int j = 0; j < 3; j++) {
        auto u0 = F[i][j], u1 = F[i][(j + 1) % 3];
        if (u0 > u1)
          std::swap(u0, u1);
        auto u = std::pair(u0, u1);
        auto it = visited.lower_bound(u);
        if (it == visited.end() || it->first != u) {
          eid[j] = v_num;
          it = visited.emplace_hint(it, u, v_num);
          v_num++;
        } else {
          eid[j] = it->second;
        }
      }
      for (int k = 0; k < 3; k++)
        F[(k + 1) * f_num + i] = Vec3i({F[i][k], eid[k], eid[(k + 2) % 3]});
      F[i] = eid;
    }
    V.resize(v_num);
    for (auto [u, id] : visited) {
      V[id] = (V[u.first] + V[u.second]) / 2;
    }
  };
  for (int i = 0; i < level; i++)
    single_level(V, F);
}

std::tuple<RowMati, std::vector<std::pair<int, int>>, RowMati> upsampled_uv(
    const std::vector<Vec3i> &F, std::vector<int> &res_fid,
    std::vector<Vec3d> &res_uv) {
  auto& helper = prism::curve::magic_matrices();
  auto& [unitV, unitF, vert_id, edge_id, face_id] = helper.upsample_data;
  int num_sample = unitV.rows();
  auto [TT, TTi] = prism::local::triangle_triangle_adjacency(F);
  Eigen::Matrix<bool, -1, 3, Eigen::RowMajor> edge_flags(F.size(), 3);
  edge_flags.setConstant(false);
  std::map<int, int> vert_map;
  RowMati all_node_map(F.size(), num_sample);
  all_node_map.setConstant(-1);
  std::vector<std::pair<int, int>> indices;
  indices.reserve(num_sample * F.size());
  for (auto i = 0, cnt = 0; i < F.size(); i++) { // flags computation
    for (int j = 0; j < 3; j++) {                // through each vertex
      auto vid = F[i][j];
      auto it = vert_map.lower_bound(vid);
      if (vert_map.empty() || it->first != vid) {
        it = vert_map.emplace_hint(it, vid, indices.size());
        indices.emplace_back(i, vert_id[j]);
      }
      all_node_map(i, vert_id[j]) = it->second;
    }
    for (int j = 0; j < 3; j++) { // through each edge
      auto u0 = F[i][j], u1 = F[i][(j + 1) % 3];
      if (edge_flags(i, j) == false) { // not visited
        for (auto k : edge_id[j]) {
          all_node_map(i, k) = indices.size();
          indices.emplace_back(i, k);
        }
        edge_flags(i, j) = true;
        auto ii = TT[i][j], jj = TTi[i][j];
        if (ii != -1)
          edge_flags(ii, jj) = true;
        continue;
      }
      // query for the oppo.
      auto ii = TT[i][j], jj = TTi[i][j];
      auto oppo_edge_id = edge_id[jj];
      int sz = edge_id[j].size();
      assert(i > ii);
      for (int k = 0; k < sz; k++) {
        all_node_map(i, edge_id[j][k]) =
            all_node_map(ii, oppo_edge_id[sz - k - 1]);
      }
    }
    for (auto k : face_id) { // face nodes, all new
      all_node_map(i, k) = indices.size();
      indices.emplace_back(i, k);
    }
  }

  for (auto [i, e] : indices) {
    res_fid.emplace_back(i);
    auto &u = unitV(e, 0), &v = unitV(e, 1);
    res_uv.emplace_back(1 - u - v, u, v);
  }

  RowMati final_F(F.size() * unitF.rows(), 3);
  for (int i = 0; i < F.size(); i++) {
    for (int j = 0; j < unitF.rows(); j++) {
      for (int k = 0; k < 3; k++)
        final_F(i * unitF.rows() + j, k) = all_node_map(i, unitF(j, k));
    }
  }
  return std::tuple(final_F, indices, all_node_map);
}
}

namespace prism::curve {
bool elevated_positive(
    const std::vector<Vec3d> &base, const std::vector<Vec3d> &top,
    const std::vector<Vec3i> &nbF,
    bool recurse_check, const std::vector<RowMatd> &local_cp) {
  auto helper = prism::curve::magic_matrices();
  auto & tri15lag_from_tri10bern = helper.elev_lag_from_bern;
  auto& dxyz = helper.volume_data.vec_dxyz;
  auto tri4_cod = TRI_CODEC.at(helper.tri_order + 1);
  auto tet4_cod = TET_CODEC.at(helper.tri_order + 1);
  assert(tri15lag_from_tri10bern.rows() !=
         tri15lag_from_tri10bern.cols());  // switch to single test
  for (int i = 0; i < nbF.size(); i++) {
    auto &cp = local_cp[i];
    auto mf = nbF[i];
    RowMatd f_base(3, 3), f_top(3, 3);
    for (int j = 0; j < 3; j++) {
      f_base.row(j) = base[mf[j]];
      f_top.row(j) = top[mf[j]];
    }

    // 6T {35 x 3D}
    auto tens = surface_to_decomposed_tetra(
        f_base, tri15lag_from_tri10bern * cp, f_top, base[mf[0]] == top[mf[0]],
        mf[1] > mf[2], tri4_cod, tet4_cod);
    for (auto &d : dxyz) {
      for (auto &t : tens) {
        Eigen::Matrix3d j = d * t;
        auto det = j.determinant();
        if (det <= 0) {
          spdlog::debug("negative {}", det);
          return false;
        }
      }
    }
    if (recurse_check) {
      for (auto &t : tens) {
        // this is codec_bc.
        if (!prism::curve::tetrahedron_inversion_check(
                t, helper.volume_data.vol_codec, helper.volume_data.vol_jac_codec, helper.volume_data.vol_bern_from_lagr,
                helper.volume_data.vol_jac_bern_from_lagr)) {
          spdlog::debug("blocked by recursive {}", mf);
          return false;
        }
      }
    }
  }
  return true;
};

RowMatd sample_hit(const std::vector<Vec3d> &base,
                   const std::vector<Vec3d> &top, const std::vector<Vec3i> &F,
                   const std::vector<int> &sp_fid,
                   const std::vector<Vec3d> &sp_uv,
                   const prism::geogram::AABB &reftree) {
  std::vector<Vec3d> sampled_points(sp_fid.size(), Vec3d(0, 0, 0));
  for (int i = 0; i < sp_fid.size(); i++) {
    auto f = sp_fid[i];
    auto w = sp_uv[i];
    Vec3d bp(0, 0, 0), tp(0, 0, 0);
    for (int j = 0; j < 3; j++) {
      bp += base[F[f][j]] * w[j];
      tp += top[F[f][j]] * w[j];
    }
    int ignore = -1;
    reftree.segment_query(bp, tp, ignore, sampled_points[i]);
  }
  return RowMatd(
      Eigen::Map<RowMatd>(sampled_points[0].data(), sampled_points.size(), 3));
}
// discrete prism projection.
void sample_hit_discrete(
    const std::vector<Vec3d> &base, const std::vector<Vec3d> &mid,
    const std::vector<Vec3d> &top, const std::vector<Vec3i> &F,
    const std::vector<int> &sp_fid, const std::vector<Vec3d> &sp_uv,
    const RowMatd &refV, const RowMati &refF,
    const prism::geogram::AABB &reftree, const std::set<int> &tri_list,
    std::vector<prism::Hit> &hits) {
  auto query = [&refV, &refF, &reftree, &total_trackee = tri_list](
                   const Vec3d &s, const Vec3d &t, prism::Hit &hit) -> bool {
    std::array<Vec3d, 2> seg_query{s, t};
    for (auto f : total_trackee) { // can be accelearated by AABB.
      auto v0 = refF(f, 0), v1 = refF(f, 1), v2 = refF(f, 2);
      std::array<Vec3d, 3> tri{refV.row(v0), refV.row(v1), refV.row(v2)};

      if (prism::predicates::segment_triangle_overlap(seg_query, tri)) {
        prism::intersections::segment_triangle_hit_cgal(seg_query, tri, hit);
        hit.id = f;
        return true;
      }
    }
    return false;
  };
  hits.resize(sp_fid.size());
  for (int i = 0; i < sp_fid.size(); i++) {
    auto f = sp_fid[i];
    auto w = sp_uv[i];
    auto [v0, v1, v2] = F[f];
    auto type = v1 > v2;
    std::array<Vec3d, 7> endpts0;
    prism::phong::fiber_endpoints({base[v0], base[v1], base[v2], mid[v0],
                                   mid[v1], mid[v2], top[v0], top[v1], top[v2]},
                                  type, w[1], w[2], endpts0);
    hits[i].id = -1;
    bool checked_once = false;
    for (int j = 0; j < endpts0.size() - 1; j++) {
      if (endpts0[j] == endpts0[j + 1])
        continue;
      checked_once = true;
      auto q = query(endpts0[j], endpts0[j + 1], hits[i]);
      if (q)
        break;
    }
    if (!checked_once) {
      // special direct setting for singularity query.
      for (auto f : tri_list) {
        auto v0 = refF(f, 0), v1 = refF(f, 1), v2 = refF(f, 2);
        if (refV.row(v0) == endpts0[0]) {
          hits[i].id = f;
          hits[i].u = 0;
          hits[i].v = 0;
        }
      }
    }
    if (hits[i].id == -1) {
      [&]() {
        spdlog::dump_backtrace();
        spdlog::error("h[{}] = {}", i, f);
        H5Easy::File file("debug_query.h5", H5Easy::File::Overwrite);
        RowMatd mbase, mtop, mmid;
        RowMati mF;
        vec2eigen(base, mbase);
        vec2eigen(top, mtop);
        vec2eigen(mid, mmid);
        vec2eigen(F, mF);

        H5Easy::dump(file, "base", mbase);
        H5Easy::dump(file, "top", mtop);
        H5Easy::dump(file, "mid", mmid);
        H5Easy::dump(file, "F", mF);
        H5Easy::dump(file, "endpt",
                     RowMatd(Eigen::Map<RowMatd>(endpts0[0].data(), 7, 3)));
        H5Easy::dump(file, "refV", refV);
        H5Easy::dump(file, "refF", refF);
        H5Easy::dump(file, "tri_list",
                     std::vector<int>(tri_list.begin(), tri_list.end()));
        spdlog::warn("hit_id[{}] = fi(in oldid) {} w {} {} {}", i, f, w[0],
                     w[1], w[2]);
        spdlog::warn("Proposed F {}, {}, {}", F[f][0], F[f][1], F[f][2]);
        spdlog::warn("query failure.");
        throw std::runtime_error("query failure");
        return false;
      }();
    }
  }
}

void sample_hit(const std::vector<Vec3d> &base, const std::vector<Vec3d> &mid,
                const std::vector<Vec3d> &top, const std::vector<Vec3i> &F,
                const std::vector<int> &sp_fid, const std::vector<Vec3d> &sp_uv,
                const RowMatd &refV, const RowMati &refF,
                const prism::geogram::AABB &reftree,
                const std::set<int> &tri_list, std::vector<prism::Hit> &hits) {
  auto query = [&refV, &refF, &reftree](const Vec3d &s, const Vec3d &t,
                                        const auto &total_trackee,
                                        prism::Hit &hit) -> bool {
    if (reftree.enabled) // this can be discarded if no performance benefit is
                         // found.
      return reftree.segment_hit(s, t, hit);
    std::array<Vec3d, 2> seg_query{s, t};
    for (auto f : total_trackee) { // can be accelearated by AABB.
      auto v0 = refF(f, 0), v1 = refF(f, 1), v2 = refF(f, 2);
      std::array<Vec3d, 3> tri{refV.row(v0), refV.row(v1), refV.row(v2)};

      if (prism::predicates::segment_triangle_overlap(seg_query, tri)) {
        if (!prism::predicates::segment_triangle_overlap(seg_query, tri)) {
          [&]() {
            H5Easy::File file("segment_triangle_overlap.h5",
                              H5Easy::File::Overwrite);
            H5Easy::dump(file, "s", s);
            H5Easy::dump(file, "t", t);
            H5Easy::dump(file, "a", tri[0]);
            H5Easy::dump(file, "b", tri[1]);
            H5Easy::dump(file, "c", tri[2]);
            spdlog::warn("debugging predicate save {}", __FILE__);
          }();
          spdlog::error("wrong Predicate");
          exit(1);
        }
        prism::intersections::segment_triangle_hit_cgal(seg_query, tri, hit);
        hit.id = f;

        return true;
      }
    }
    return false;
  };
  hits.resize(sp_fid.size());
  for (int i = 0; i < sp_fid.size(); i++) {
    auto f = sp_fid[i];
    auto w = sp_uv[i];
    Vec3d bp(0, 0, 0), tp(0, 0, 0), mp(0, 0, 0);
    for (int j = 0; j < 3; j++) {
      bp += base[F[f][j]] * w[j];
      tp += top[F[f][j]] * w[j];
      mp += mid[F[f][j]] * w[j];
    }
    hits[i].t = -1;

    auto q = query(bp, mp, tri_list, hits[i]);
    if (!q)
      q = query(mp, tp, tri_list, hits[i]);
    if (!q) {
      [&]() {
        spdlog::dump_backtrace();
        H5Easy::File file("debug_query.h5", H5Easy::File::Overwrite);
        RowMatd mbase, mtop;
        RowMati mF;
        vec2eigen(base, mbase);
        vec2eigen(top, mtop);
        vec2eigen(F, mF);

        H5Easy::dump(file, "base", mbase);
        H5Easy::dump(file, "top", mtop);
        H5Easy::dump(file, "F", mF);
        H5Easy::dump(file, "bp", bp);
        H5Easy::dump(file, "mp", mp);
        H5Easy::dump(file, "tp", tp);
        H5Easy::dump(file, "refV", refV);
        H5Easy::dump(file, "refF", refF);
        H5Easy::dump(file, "tri_list",
                     std::vector<int>(tri_list.begin(), tri_list.end()));
        spdlog::warn("hit_id[{}] = fi(in oldid) {} w {} {} {}", i, f, w[0],
                     w[1], w[2]);
        spdlog::warn("Proposed F {}, {}, {}", F[f][0], F[f][1], F[f][2]);
        spdlog::warn("query failure.");
        throw std::runtime_error("query failure");
        return false;
      }();
    }
  }
}

void sample_hit(const std::vector<Vec3d> &base, const std::vector<Vec3d> &top,
                const std::vector<Vec3i> &F, const std::vector<int> &sp_fid,
                const std::vector<Vec3d> &sp_uv,
                const prism::geogram::AABB &reftree,
                std::vector<prism::Hit> &hits) {
  hits.resize(sp_fid.size());
  for (int i = 0; i < sp_fid.size(); i++) {
    auto f = sp_fid[i];
    auto w = sp_uv[i];
    Vec3d bp(0, 0, 0), tp(0, 0, 0);
    for (int j = 0; j < 3; j++) {
      bp += base[F[f][j]] * w[j];
      tp += top[F[f][j]] * w[j];
    }
    reftree.segment_hit(bp, tp, hits[i]);
  }
}

auto mat_construct(const std::vector<Vec3i> &F, const RowMatd &tri10_lv5,
                   const RowMati &tri_codec,
                   const std::vector<std::pair<int, int>> &uniq_id,
                   const CodecMap &entry_map) {
  Eigen::SparseMatrix<double> A;
  std::vector<Eigen::Triplet<double>> trips;
  int sample_cnt = 0;
  for (auto [f, j] : uniq_id) {
    for (int k = 0; k < tri_codec.rows(); k++) {
      auto cp = entry_map.at(sort_slice(F[f], tri_codec.row(k)));
      auto val = tri10_lv5(k, j);
      trips.emplace_back(sample_cnt, cp, val);
    }
    sample_cnt++;
  }
  A.resize(uniq_id.size(), entry_map.size());
  A.setFromTriplets(trips.begin(), trips.end());
  return A;
};

//  High order fitting over one ring (several triangles)
//
//  nbF: vertex indices forming the triangles {{v0,v1,v2}, {u0,u1,u2}}
//  linear_cp: (can be replaced by initial guess) here is a linear starting
//  point.
//
auto ring_fit(const std::vector<Vec3i> &nbF, std::vector<RowMatd> &linear_cp,
              const MatLexMap<Eigen::VectorXi, Vec3d> &nb_cp_map,
              const RowMatd &tri10_lv5, const RowMatd &b,
              const RowMati &tri10_codec, const std::any &additional,
              RowMatd &residual_in, RowMatd &residual_out)
    -> std::vector<RowMatd> {
  assert(tri10_codec.rows() == tri10_lv5.rows());
  // assert(b.rows() == tri10_lv5.cols() * nbF.size());
  // relevant control points: cp map.
  auto [local_fe, local_map] = global_entry_map(nbF, tri10_codec);
  auto [uniq_id] =
      std::any_cast<std::tuple<std::vector<std::pair<int, int>>>>(additional);

  // construct matrix A
  auto A = mat_construct(nbF, tri10_lv5, tri10_codec, uniq_id, local_map);

  // set fixed constraints. (some cp are known)
  std::vector<int> bnd_cp;
  for (auto [cod, i] : local_map) {
    auto it = nb_cp_map.find(cod);
    auto [f, e] = local_fe[i];
    if (it != nb_cp_map.end()) { // found
      linear_cp[f].row(e) = it->second;
      bnd_cp.push_back(i);
    }
  }
  spdlog::trace("bndcp size {}", bnd_cp.size());
  spdlog::trace("nb_cp_map size {}", nb_cp_map.size());
  // assert(bnd_cp.size() == nb_cp_map.size()); TODO. this requires a look at.
  RowMatd bnd_sol(bnd_cp.size(), 3);
  for (int i = 0; i < bnd_cp.size(); i++) {
    auto [f, e] = local_fe[bnd_cp[i]];
    bnd_sol.row(i) = linear_cp[f].row(e);
  }

  RowMatd orig_sol(local_fe.size(), 3);
  for (int i = 0; i < local_fe.size(); i++) {
    auto [f, e] = local_fe[i];
    orig_sol.row(i) = linear_cp[f].row(e);
  }
  igl::min_quad_with_fixed_data<double> mqf;
  double alpha = 1e-5;
  Eigen::SparseMatrix<double> AtA =
      (A.transpose() * A); //+ alpha * (A.transpose() * L.transpose() * L * A);
  RowMatd Atb =
      -(A.transpose() * b); //- alpha * (A.transpose() * L.transpose() * hn);
  igl::min_quad_with_fixed_precompute(
      AtA, Eigen::Map<Eigen::VectorXi>(bnd_cp.data(), bnd_cp.size()),
      Eigen::SparseMatrix<double>(), true, mqf);
  RowMatd local_sol;
  igl::min_quad_with_fixed_solve(mqf, Atb, bnd_sol, RowMatd(), local_sol);
  for (int i = 0; i < bnd_cp.size(); i++) {
    local_sol.row(bnd_cp[i]) = bnd_sol.row(i);
  }
  [&]() {
    H5Easy::File file("debug.h5", H5Easy::File::Overwrite);
    H5Easy::dump(file, "A", RowMatd(A.toDense()));
    H5Easy::dump(file, "Ax", RowMatd(A * orig_sol));
    H5Easy::dump(file, "Ax1", RowMatd(A * local_sol));
    H5Easy::dump(file, "b", b);
    H5Easy::dump(file, "bnd", bnd_sol);
    spdlog::warn("debugging serialize");
  };
  residual_in = (A * orig_sol - b);
  residual_out = (A * local_sol - b);
  spdlog::debug("old Ax-b {}", (residual_in).rowwise().norm().mean());
  spdlog::debug("new Ax-b {}", (residual_out).rowwise().norm().mean());

  std::vector<RowMatd> duplicate_cp(nbF.size(),
                                    RowMatd::Zero(tri10_codec.rows(), 3));
  for (int i = 0; i < nbF.size(); i++) {
    for (int e = 0; e < tri10_codec.rows(); e++) {
      auto cod = tri10_codec.row(e);
      auto ind = local_map.at(sort_slice(nbF[i], cod));
      duplicate_cp[i].row(e) = local_sol.row(ind);
      linear_cp[i].row(e) = orig_sol.row(ind);
    }
  }
  return duplicate_cp;
};

} // namespace prism::curve

prism::curve::HelperTensors::UpsampleData local_upsample_arrange_data(int level) {
  RowMatd unitV;
  RowMati unitF;
  Vec3i vert_id;
  std::array<std::vector<int>, 3> edge_id;
  std::vector<int> face_id;

  if (unitV.rows() == 0) {
    unitV.resize(3, 2);
    unitV << 0, 0, 1, 0, 0, 1;
    unitF.resize(1, 3);
    unitF << 0, 1, 2;
    igl::upsample(unitV, unitF,
                  level); // turns out this is a time-consuming routine
    Eigen::VectorXi unitE;
    igl::boundary_loop(unitF, unitE);
    assert(unitE(0) == 0);
    int num_e = unitE.size();
    assert(num_e % 3 == 0);
    {
      int num_edge = num_e / 3;
      for (int j = 0; j < 3; j++) {
        vert_id[j] = unitE[j * num_edge];
        assert(unitE[j * num_edge] == j);
        edge_id[j].resize(num_e / 3 - 1);
        for (auto k = 1; k < num_e / 3; k++)
          edge_id[j][k - 1] = unitE(j * num_edge + k);
      }
      std::vector<bool> interior_flag(unitV.rows(), true);
      for (int i = 0; i < unitE.size(); i++)
        interior_flag[unitE[i]] = false;
      for (auto i = 0; i < interior_flag.size(); i++) {
        if (interior_flag[i])
          face_id.push_back(i);
      }
    }
  }
  return {unitV, unitF, vert_id, edge_id, face_id};
}

std::shared_ptr<prism::curve::HelperTensors> prism::curve::HelperTensors::tensors_;
void prism::curve::HelperTensors::init(int tri_order, int level) {
  #ifdef CUMIN_MAGIC_DATA_PATH
  std::string data_path = CUMIN_MAGIC_DATA_PATH;
  #else 
  std::string data_path = "../python/curve/data";
  #endif
  RowMatd tri10_lv5, tri15lag_from_tri10bern, lag_from_bern;
  std::array<RowMatd, 2> duv_lv5;
  std::vector<RowMatd> vec_dxyz;
  {
    H5Easy::File file(
        fmt::format("{}/tri_o{}_lv{}.h5", data_path, tri_order, level),
        H5Easy::File::ReadOnly);
    tri10_lv5 = H5Easy::load<RowMatd>(file, "bern");
    tri15lag_from_tri10bern =
        H5Easy::load<RowMatd>(file, "bern2elevlag").transpose();
    lag_from_bern = H5Easy::load<RowMatd>(file, "bern2lag").transpose();
    duv_lv5[0] = H5Easy::load<RowMatd>(file, "deri_u");  // (sample) 45 x 10
    duv_lv5[1] = H5Easy::load<RowMatd>(file, "deri_v");
  }
  {
    H5Easy::File file(
        fmt::format("{}/p{}_quniform5_dxyz.h5", data_path, tri_order + 1),
        H5Easy::File::ReadOnly);
    auto tet4_dxyz = H5Easy::load<std::vector<std::vector<std::vector<double>>>>(file, "dxyz");
    vec_dxyz.resize(tet4_dxyz[0].size());
     for (auto i1 = 0; i1 < tet4_dxyz[0].size(); i1++) {
      vec_dxyz[i1].resize(tet4_dxyz.size(), tet4_dxyz[0][0].size());
      for (auto i0 = 0; i0 < tet4_dxyz.size(); i0++)
        for (auto i2 = 0; i2 < tet4_dxyz[0][0].size(); i2++)
          vec_dxyz[i1](i0, i2) = tet4_dxyz[i0][i1][i2];
    }
  }
  RowMatd bern_from_lagr_o9, bern_from_lagr_o4;
  RowMati codecs_o4_bc, codecs_o9_bc;
  auto jac_order = tri_order* 3;
  auto tet_order = tri_order + 1;
  auto get_l2b = [data_path](std::string filename) {
    H5Easy::File file1(data_path + filename);
    return H5Easy::load<RowMatd>(file1, "l2b");
  };
  bern_from_lagr_o9 = get_l2b(fmt::format("tetra_o{}_l2b.h5", jac_order));
  bern_from_lagr_o4 = get_l2b(fmt::format("tetra_o{}_l2b.h5", tet_order));

  vec2eigen(codecs_gen(tet_order, 3), codecs_o4_bc);
  vec2eigen(codecs_gen(jac_order, 3), codecs_o9_bc);

  assert(duv_lv5[0].cols() == (tri_order + 1) * (tri_order + 2) / 2);
  assert(duv_lv5[0].rows() == tri10_lv5.cols());
  assert(duv_lv5[0].cols() == tri10_lv5.rows());
  assert(tri15lag_from_tri10bern.cols() == tri10_lv5.rows());
  assert(tri15lag_from_tri10bern.rows() ==
         (tri_order + 3) * (tri_order + 2) / 2);
  // volume checks
  assert(bern_from_lagr_o4.rows() == (tet_order+1)*(tet_order+2)*(tet_order+3)/6);
  assert(bern_from_lagr_o9.rows() == (jac_order+1)*(jac_order+2)*(jac_order+3)/6);

  spdlog::info(
      "High Order Tensors ({}): bern {}x{} bern2elev_lag {}x{} "
      "dxyz {}x{}x{} duv_lv[{}] {}x{}",
      data_path,
      tri10_lv5.rows(), tri10_lv5.cols(), tri15lag_from_tri10bern.rows(),
      tri15lag_from_tri10bern.cols(), 
      vec_dxyz.size(), vec_dxyz[0].rows(), vec_dxyz[0].cols(),
      duv_lv5.size(), duv_lv5[0].rows(), duv_lv5[0].cols());

  tensors_ = std::make_shared<prism::curve::HelperTensors>();
  tensors_->tri_order = tri_order;
  tensors_->level = level;
  tensors_->bern_val_samples = tri10_lv5;
  tensors_->elev_lag_from_bern = tri15lag_from_tri10bern;
  tensors_->duv_samples = duv_lv5;
  tensors_->upsample_data = local_upsample_arrange_data(level);
  tensors_->volume_data.vec_dxyz = vec_dxyz;
  tensors_->volume_data.vol_codec = codecs_o4_bc;
  tensors_->volume_data.vol_jac_codec = codecs_o9_bc;
  tensors_->volume_data.vol_bern_from_lagr = bern_from_lagr_o4;
  tensors_->volume_data.vol_jac_bern_from_lagr = bern_from_lagr_o9;
}