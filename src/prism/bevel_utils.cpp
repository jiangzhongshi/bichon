#include "bevel_utils.hpp"

#include <igl/adjacency_list.h>
#include <igl/boundary_facets.h>
#include <igl/boundary_loop.h>
#include <igl/per_face_normals.h>
#include <igl/topological_hole_fill.h>
#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <prism/local_operations/local_mesh_edit.hpp>

#include "local_operations/retain_triangle_adjacency.hpp"

// VN.row(i) == 0 will be automatically marked.
void identify_vertices_to_bevel(const RowMatd& V, const RowMati& F,
                                const RowMatd& VN, std::vector<bool>& bevelee) {
  bevelee.resize(0);
  bevelee.resize(V.rows(), false);
  std::vector<std::vector<int>> VF, VFi;
  igl::vertex_triangle_adjacency(V, F, VF, VFi);
  RowMatd FN;
  igl::per_face_normals_stable(V, F, FN);
  for (auto i = 0; i < F.rows(); i++) {
    auto track_ref = std::set({i});
    for (auto j : {0, 1, 2}) {
      auto v = F(i, j);
      std::merge(VF[v].begin(), VF[v].end(), track_ref.begin(), track_ref.end(),
                 std::inserter(track_ref, track_ref.begin()));
    }
    bool mark = false;
    for (auto j : {0, 1, 2}) {
      auto v = F(i, j);
      for (auto& t : track_ref)
        if (FN.row(t).dot(VN.row(v)) < 0.1) {
          mark = true;
          break;
        }
      if (mark) break;
    }
    if (mark) {
      bevelee[F(i, 0)] = true;
      bevelee[F(i, 1)] = true;
      bevelee[F(i, 2)] = true;
    }
  }
}

// this also deals with virtual faces. only output faces with real_fnum, the V,F
// passed in is augmented with virtuals.
void masked_doo_sabin(const RowMatd& V, const RowMati& F, const RowMatd& VN,
                      double eps, const std::vector<bool>& bevelee,
                      RowMatd& dsV, RowMati& dsF, RowMatd& dsVN, int real_fnum,
                      std::vector<int>& face_parent) {
  auto v_num = V.rows(), f_num = F.rows(), he_num = 3 * F.rows();
  dsV.resize(v_num + he_num + 3 * f_num, 3);
  dsF.resize(13 * F.rows(), 3);
  dsV.topRows(v_num) = V;
  dsVN.resize(dsV.rows(), 3);
  dsVN.topRows(v_num) = VN;
  std::vector<bool> has_new_vertex(v_num + he_num + 3 * f_num, true);
  RowMati FF, FFi;
  igl::triangle_triangle_adjacency(F, FF, FFi);
  for (int i = 0; i < f_num; i++) {
    auto vx = F.row(i);
    std::array<int, 3> fx, e0, e1;
    for (int j = 0; j < 3; j++) {
      fx[j] = v_num + he_num + 3 * i + j;
      e0[j] = v_num + 3 * i + j;
      auto f_oppo = FF(i, j), e_oppo = FFi(i, j);
      if (f_oppo == -1) {  // boundary case
        has_new_vertex[fx[j]] = false;
        has_new_vertex[e0[j]] = false;
        fx[j] = vx[j];
        e0[j] = vx[j];
        e1[j] = vx[(j + 1) % 3];
        continue;
      }
      assert(f_oppo != -1);
      e1[j] = v_num + 3 * f_oppo + e_oppo;
      if (!bevelee[F(i, j)]) {
        has_new_vertex[fx[j]] = false;
        has_new_vertex[e0[j]] = false;
        fx[j] = vx[j];
        e0[j] = vx[j];
      }
      if (!bevelee[F(f_oppo, e_oppo)]) {
        has_new_vertex[e1[j]] = false;
        e1[j] = F(f_oppo, e_oppo);
      }
    }  // for
    auto v0 = vx[0], v1 = vx[1], v2 = vx[2];
    auto f0 = fx[0], f1 = fx[1], f2 = fx[2];
    // if (i < real_fnum)
    dsF.middleRows(i * 13, 13) << v0, e0[0], f0, v0, f0, e1[2], e0[0], e1[0],
        f1, e0[0], f1, f0, v1, f1, e1[0], v1, e0[1], f1, e0[1], f2, f1, e0[1],
        e1[1], f2, e1[1], v2, f2, f2, v2, e0[2], f2, e0[2], f0, f0, e0[2],
        e1[2], f0, f1, f2;
    for (int e = 0; e < 3; e++) {
      if (!bevelee[F(i, e)]) continue;
      dsV.row(e0[e]) =
          (1 - eps) * V.row(F(i, e)) + eps * V.row(F(i, (e + 1) % 3));
      dsV.row(fx[e]) = (1 - eps) * V.row(F(i, e)) +
                       eps / 2 * V.row(F(i, (e + 1) % 3)) +
                       eps / 2 * V.row(F(i, (e + 2) % 3));
      dsVN.row(e0[e]) = VN.row(F(i, e));
      dsVN.row(fx[e]) = VN.row(F(i, e));
    }
  }
  for (int i = 0; i < dsF.rows(); i++) {  // mark out degenerate
    if (dsF(i, 0) == dsF(i, 1) || dsF(i, 0) == dsF(i, 2) ||
        dsF(i, 1) == dsF(i, 2))
      dsF.row(i) << -1, -1, -1;
  }

  std::vector<bool> referenced(dsV.rows(), false);
  for (int i = 0; i < dsF.rows(); i++)
    for (int j = 0; j < 3; j++) {
      if (dsF(i, j) != -1) referenced[dsF(i, j)] = true;
    }
  for (int i = 0; i < dsV.rows(); i++) {
    if (!referenced[i]) has_new_vertex[i] = false;
  }

  std::vector<int> NJ;                  // newindex -> old_index
  std::vector<int> NI(dsV.rows(), -1);  // old -> new
  for (int i = 0; i < has_new_vertex.size(); i++) {
    if (has_new_vertex[i]) {
      NJ.push_back(i);
      NI[i] = NJ.size() - 1;
    }
  }
  spdlog::info("NJ {}", NJ.size());

  for (int i = v_num; i < NJ.size(); i++) {
    dsV.row(i) = dsV.row(NJ[i]);
    dsVN.row(i) = dsVN.row(NJ[i]);
  }
  int cur = 0;
  for (int i = 0; i < 13 * real_fnum; i++) {
    if (dsF(i, 0) == -1) continue;
    for (int j = 0; j < 3; j++) {
      dsF(cur, j) = NI[dsF(i, j)];
    }
    face_parent.push_back(i / 13);
    cur++;
  }
  dsF.conservativeResize(cur, 3);
  dsV.conservativeResize(NJ.size(), 3);
  dsVN.conservativeResize(NJ.size(), 3);
}

#include <igl/remove_unreferenced.h>
void prism::bevel_utils::adaptive_doo_sabin(const RowMatd& V, const RowMati& F,
                                            const RowMatd& VN, double eps,
                                            RowMatd& dsV, RowMati& dsF,
                                            RowMatd& dsVN,
                                            std::vector<int>& face_parent) {
  std::vector<bool> bevelee;  //(V.rows(),true);
  identify_vertices_to_bevel(V, F, VN, bevelee);
  if (eps < 0) {
    for (auto b : bevelee) b = false;
    spdlog::info("bevel disabled when specify eps < 0");
  }

  RowMati F1;
  std::vector<std::vector<int>> holes;
  igl::boundary_loop(F, holes);
  Eigen::VectorXi dummy;
  igl::topological_hole_fill(F, dummy, holes, F1);

  RowMatd V1 = RowMatd::Zero(V.rows() + holes.size(), 3);
  RowMatd VN1 = RowMatd::Zero(V.rows() + holes.size(), 3);
  V1.topRows(V.rows()) = V;
  VN1.topRows(V.rows()) = VN;
  bevelee.resize(V1.rows(), false);

  masked_doo_sabin(V1, F1, VN1, eps, bevelee, dsV, dsF, dsVN, F.rows(),
                   face_parent);
  Eigen::VectorXi ruI, ruJ;
  igl::remove_unreferenced(dsV.rows(), dsF, ruI, ruJ);
  std::for_each(dsF.data(), dsF.data() + dsF.size(),
                [&ruI](int& a) { a = ruI(a); });
  dsV = igl::slice(dsV, ruJ, 1);
  dsVN = igl::slice(dsVN, ruJ, 1);
}

void prism::bevel_utils::singularity_special_bevel(
    RowMatd& mV, RowMati& mF, int num_cons, RowMatd& VN,
    std::vector<int>& face_parent) {
  if (num_cons == 0) return;
  std::vector<std::vector<int>> VF, VFi;
  igl::vertex_triangle_adjacency(mV, mF, VF, VFi);
  RowMatd FN;
  igl::per_face_normals_stable(mV, mF, FN);
  std::vector<Vec3i> F;
  std::vector<Vec3d> V, N;
  eigen2vec(mF, F);
  eigen2vec(mV, V);
  eigen2vec(VN, N);
  auto [TT, TTi] = prism::local::triangle_triangle_adjacency(F);
  std::vector<std::tuple<int, int, int /*another edge vert*/>> edge_to_split;
  int total_degree = 0;
  for (auto i = 0; i < num_cons; i++) {
    int f = VF[i][0], e = VFi[i][0];
    assert(F[f][e] == i);
    std::vector<std::pair<int, int>> neighbor;
    prism::get_star_edges(F, TT, TTi, f, e, neighbor, true);
    total_degree += neighbor.size();
    for (auto fe : neighbor) {
      int f0 = fe.first, e1 = (fe.second + 1) % 3;
      auto v0 = F[f0][e1], v1 = F[f0][(e1 + 1) % 3];
      {
        auto f1 = TT[f0][fe.second];
        if (f1 != -1)
          N[v0] = (FN.row(f0) + FN.row(f1)).stableNormalized();
        else
          N[v0] = FN.row(f0);
        if (TT[f0][(e1 + 1) % 3] == -1) {
          N[F[f0][(e1 + 1) % 3]] = FN.row(f0);
        }
      }
      if (v0 < v1) {  // from edge-bevel to face-bevel. Rely on assumptions that
                      // face comes after edge, in doo-sabin.
        bool along = true;
        igl::triangle_tuple_switch_vert(f0, e1, along, F, TT, TTi);
        igl::triangle_tuple_switch_edge(f0, e1, along, F, TT, TTi);
        igl::triangle_tuple_switch_face(f0, e1, along, F, TT, TTi);
        igl::triangle_tuple_switch_edge(f0, e1, along, F, TT, TTi);
        igl::triangle_tuple_switch_vert(f0, e1, along, F, TT, TTi);
        auto v2 = igl::triangle_tuple_get_vert(f0, e1, along, F, TT, TTi);
        edge_to_split.emplace_back(fe.first, fe.second, v2);
        assert(((V[v2] - V[v0]).cross(V[v1] - V[v0])).norm() < 1e-10);
      }
    }
  }
  assert(edge_to_split.size() == total_degree / 2);

  for (auto [f, e, v2] : edge_to_split) {
    e = (e + 1) % 3;
    int v0 = F[f][e], v1 = F[f][(e + 1) % 3];
    prism::edge_split(V.size(), F, TT, TTi, f, e);
    V.emplace_back((V[v0] + V[v1]) / 2);
    N.push_back(N[v0]);
    face_parent.push_back(face_parent[f]);
    face_parent.push_back(face_parent[f]);
    assert(v2 < v1);
    N[v1] = N[v2];
  }

  vec2eigen(V, mV);
  vec2eigen(F, mF);
  vec2eigen(N, VN);
}

void prism::bevel_utils::edge_based_bevel(const RowMatd& V, const RowMati& F,
                                          const RowMatd& VN,
                                          const RowMati& feat_vv, RowMatd& dsV,
                                          RowMati& dsF, RowMatd& dsVN,
                                          std::vector<int>& face_parent) {
  RowMatd FN;
  igl::per_face_normals_stable(V, F, FN);
  double threshold = 0.1;

  std::vector<Vec3d> dV, dVN;
  std::vector<Vec3i> dF;
  eigen2vec(V, dV);
  eigen2vec(F, dF);
  eigen2vec(VN, dVN);
  for (auto i = 0; i < F.rows(); i++) face_parent.push_back(i);

  std::set<std::pair<int, int>> feat_set;
  for (auto i = 0; i < feat_vv.rows(); i++) {
    auto v0 = feat_vv(i, 0), v1 = feat_vv(i, 1);
    if (v0 > v1) std::swap(v0, v1);
    feat_set.emplace(v0, v1);
  }
  int cnt = dV.size();
  RowMati TT, TTi;
  igl::triangle_triangle_adjacency(F, TT, TTi);
  RowMati marker = -RowMati::Ones(F.rows(), 3);

 int num_singular = 0;
  for (int i=0; i<VN.rows(); i++) {
    if (VN.row(i).norm() > 0.1) break;
    num_singular ++ ;
  }
  for (auto f = 0; f < F.rows(); f++) {
    for (auto e = 0; e < 3; e++) {
      if (marker(f, e) >= 0) continue;  // already marked
      auto f1 = TT(f, e), e1 = TTi(f, e);
      auto v0 = F(f, e), v1 = F(f, (e + 1) % 3);
      auto p = std::pair(std::min(v0, v1), std::max(v0, v1));
      if (feat_set.find(p) != feat_set.end()) continue;
      auto e_oppo = (e + 2) % 3;
      if (F(f, e_oppo) < num_singular) continue;
      if (FN.row(f1).dot(VN.row(F(f, e_oppo))) < threshold) {
        marker(f, e) = cnt;
        marker(f1, e1) = cnt;
        cnt++;
        dV.emplace_back((dV[v0] + dV[v1]) / 2);
        dVN.emplace_back((FN.row(f) + FN.row(f1)).normalized());
      }
    }
  }


  auto mark_with_pattern = [&newF = dF, &newV = dV, &newVN = dVN, &TT, &TTi](
                               const RowMati& F, auto f, const auto& mark_mat, auto type, const RowMatd& FN,
                               auto& cnt) -> void {
    auto Ff = F.row(f);
    auto mark = mark_mat.row(f);
    auto fn = FN.row(f);
    assert(type != 0);
    if (type == 1) {
      auto [e0, ve] = [&mark]() {
        for (auto j = 0; j < 3; j++)
          if (mark(j) >= 0) return std::pair(j, mark(j));
        assert(false)
        return std::pair(-1, -1);
      }();
      auto v0 = Ff(e0);
      auto v1 = Ff((e0 + 1) % 3);
      auto v2 = Ff((e0 + 2) % 3);
      auto vn = cnt++;
      auto bnd = std::array<int, 4>{v0, ve, v1, v2};
      for (auto i = 0; i < bnd.size(); i++) {
        newF.emplace_back(Vec3i{vn, bnd[i], bnd[(i + 1) % 4]});
      }
      newV.emplace_back((newV[v0] + newV[v1] + newV[v2]) / 3);
      newVN.emplace_back((newVN[v0] + newVN[v1]).normalized());
      return;
    }
    if (type == 2) {
      auto e1 = [&mark]() {  // the dull one
        for (auto j = 0; j < 3; j++)
          if (mark(j) < 0) return j;
        assert(false && "should not be here");
        return -1;
      }();
      auto v0 = Ff((e1 + 2) % 3);

      auto v1 = Ff(e1);
      auto v2 = Ff((e1 + 1) % 3);
      auto va = mark((e1 + 2) % 3);
      auto vb = mark((e1 + 1) % 3);
      auto vn = cnt++;
      if (newVN[v0].norm() < 0.1) { // singularity special bevel. type 2
        spdlog::debug("special bevel {}", f);
        auto f0_1 = TT(f, (e1 + 2) % 3);
        auto f2_1 = TT(f, (e1 + 1) % 3);
        if (fn.dot(FN.row(f0_1)) < 0.1 || fn.dot(FN.row(f2_1))<0.1) {
          auto vn1 = cnt++;
          auto bnd = std::array<int, 5>{vn1, v0, va, v1, v2};
          for (auto i = 0; i < bnd.size(); i++)
            newF.emplace_back(Vec3i{vn, bnd[i], bnd[(i + 1) % bnd.size()]});
          newF.emplace_back(Vec3i{vn1, vb, v0});
          newF.emplace_back(Vec3i{vn1, v2, vb});
          newV.emplace_back((newV[v0]*2 + newV[v1]*2 + newV[v2]) / 5);
          newVN.emplace_back(newVN[va]);
          newV.emplace_back((newV[v0]*2 + newV[v1] + newV[v2]*2) / 5);
          newVN.emplace_back(newVN[vb]);
          return;
        } else {
          newVN.push_back(fn);
        }
      } else {
        newVN.emplace_back(newVN[v0]);
      }
      auto bnd = std::array<int, 5>{vb, v0, va, v1, v2};
      for (auto i = 0; i < bnd.size(); i++) {
        newF.emplace_back(Vec3i{vn, bnd[i], bnd[(i + 1) % 5]});
      }
      newV.emplace_back((newV[v0] + newV[v1] + newV[v2]) / 3);
      return;
    }
    if (type == 3) {
      auto v0 = Ff(0), v1 = Ff(1), v2 = Ff(2);
      auto singular_index = [&]() {
        for (auto j = 0; j < 3; j++) {
          if (newVN[Ff(j)].norm() < 0.1) return j;
        }
        return -1;
      }();
      if (singular_index != -1) {
        spdlog::error("Bevel type 3 Singular.");
        throw std::runtime_error("singular index");
        return;
      }
      auto va = mark(0), vb = mark(1), vc = mark(2);
      auto v00 = cnt, v11 = cnt + 1, v22 = cnt + 2;
      for (auto i = 0; i < 3; i++) {
        newF.emplace_back(Vec3i{cnt + i, Ff(i), mark(i)});
        newF.emplace_back(Vec3i{cnt + i, mark(i), cnt + (i + 1) % 3});
        newF.emplace_back(Vec3i{cnt + (i + 1) % 3, mark(i), Ff((i + 1) % 3)});
        newV.emplace_back((newV[v0] + newV[v1] + newV[v2] + 2 * newV[Ff(i)]) /
                          5);
        newVN.emplace_back(newVN[Ff(i)]);
      }
      newF.emplace_back(Vec3i{cnt, cnt+1, cnt+2});
      cnt += 3;
      return;
    }
  };
  // now, start marking.
  for (auto f = 0; f < F.rows(); f++) {
    auto type = 0;
    for (auto e = 0; e < 3; e++) {
      if (marker(f, e) >= 0) type++;
    }
    if (type == 0) continue;
    auto num0 = dF.size();
    mark_with_pattern(F, f, marker, type, FN, cnt);
    auto num1 = dF.size();
    for (auto i = num0; i < num1; i++) face_parent.emplace_back(f);
    dF[f] = Vec3i{-1, -1, -1};
  }
  assert(dF.size() == face_parent.size());
  dsF.resize(dF.size(), 3);

  vec2eigen(dV, dsV);
  vec2eigen(dVN, dsVN);
  auto fcnt = 0;
  for (auto i = 0; i < dF.size(); i++) {
    if (dF[i][0] == -1) continue;
    auto [s, mt, shift] = tetra_split_AorB(dF[i]);
    for (auto j = 0; j < 3; j++) dsF(fcnt, j) = mt[j];
    face_parent[fcnt] = face_parent[i];
    fcnt++;
  }
  dsF.conservativeResize(fcnt, 3);
  face_parent.resize(fcnt);

  spdlog::enable_backtrace(30);
  if (!verify_edge_bevel(dsV, dsF, dsVN, feat_vv)) {
    spdlog::dump_backtrace();
    spdlog::error("Bevel Verify");
    throw std::runtime_error("Bevel not v erified");
  }
}

#include <igl/is_edge_manifold.h>
#include <igl/write_triangle_mesh.h>
bool prism::bevel_utils::verify_edge_bevel(const RowMatd& V, const RowMati& F,
                                           const RowMatd& VN,
                                           const RowMati& feat_vv) {
  {
    RowMati B;
    igl::boundary_facets(F, B);
    if (B.rows() > 0) {
      spdlog::critical("bevel breaks: hole.");
      return false;
    }
    if (!igl::is_edge_manifold(F)) {
      spdlog::critical("bevel breaks: manifold.");
      return false;
    }
  }
  RowMatd FN;
  igl::per_face_normals_stable(V, F, FN);
  RowMati TT, TTi;
  double threshold = 1e-4;

  igl::triangle_triangle_adjacency(F, TT, TTi);
  std::set<std::pair<int, int>> feat_set;
  for (auto i = 0; i < feat_vv.rows(); i++) {
    auto v0 = feat_vv(i, 0), v1 = feat_vv(i, 1);
    if (v0 > v1) std::swap(v0, v1);
    feat_set.emplace(v0, v1);
  }

  int num_singular = 0;
  for (int i=0; i<VN.rows(); i++) {
    if (VN.row(i).norm() > 0.1) break;
    num_singular ++ ;
  }
  spdlog::debug("num singular {}", num_singular);
  RowMati marker = -RowMati::Ones(F.rows(), 3);
  for (auto f = 0; f < F.rows(); f++) {
    for (auto e = 0; e < 3; e++) {
      if (marker(f, e) >= 0) continue;  // already marked
      auto f1 = TT(f, e), e1 = TTi(f, e);
      auto v0 = F(f, e), v1 = F(f, (e + 1) % 3);
      auto p = std::pair(std::min(v0, v1), std::max(v0, v1));
      if (feat_set.find(p) != feat_set.end()) continue;
      auto e_oppo = (e + 2) % 3;
      spdlog::trace("f {} e {}/{}, F[f] {} F[f,e] {}", f, e, e_oppo, F.row(f), F(f,e_oppo));
      if (F(f,e_oppo) < num_singular) continue;
      auto d = FN.row(f1).dot(VN.row(F(f, e_oppo)));
      if (d < threshold) {
        spdlog::error("dot {} between {}-{} ({})",d, FN.row(f1), VN.row(F(f, e_oppo)),F(f, e_oppo));
        return false;
        // throw std::runtime_error("Bevel Not effective");
      }
    }
  }
  return true;
};