#include "prism/PrismCage.hpp"
#include "prism/cage_utils.hpp"
#include <autodiff_mitsuba.h>
#include "prism/geogram/AABB.hpp"
#include "prism/local_operations/remesh_pass.hpp"
#include "prism/phong/query_correspondence.hpp"
#include "test_common.hpp"
#include <doctest.h>
#include <fstream>
#include <geogram/mesh/mesh_AABB.h>
#include <igl/Hit.h>
#include <igl/cat.h>
#include <igl/grad.h>
#include <igl/local_basis.h>
#include <igl/polar_svd.h>
#include <igl/read_triangle_mesh.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/write_triangle_mesh.h>
#include <prism/feature_utils.hpp>
#include <prism/geogram/geogram_utils.hpp>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/spdlog.h>
#include <string>
using DScalar = DScalar2<double, Eigen::Vector2d, Eigen::Matrix2d>;

using Vec2d = Eigen::RowVector2d;
DScalar triangle_quality(const std::array<Vec2d, 3> &vertices,
                         const std::array<Vec3d, 3> &refs, int id_with_grad) {
  DiffScalarBase::setVariableCount(2);
  Eigen::Matrix<DScalar, 3, 2> verticesMat =
      Eigen::Map<const RowMatd>(vertices[0].data(), 3, 2).cast<DScalar>();
  for (int d = 0; d < 2; d++) {
    verticesMat(id_with_grad, d) = DScalar(d, vertices[id_with_grad][d]);
  }
  auto e1 = verticesMat.row(1) - verticesMat.row(0);
  auto e2 = verticesMat.row(2) - verticesMat.row(0);
  Eigen::Matrix<DScalar, 2, 2> tri;
  tri << e1, e2;
  Eigen::Matrix2d invref;
  auto r_e1 = (refs[1] - refs[0]);
  auto r_e2 = refs[2] - refs[0];
  auto r_e1_len = r_e1.norm();
  auto r_e2_x = r_e1.dot(r_e2) / r_e1_len;
  auto r_e2_y = (r_e2 - r_e2_x * r_e1 / r_e1_len).norm();
  invref << r_e1_len, r_e2_x, 0, r_e2_y;
  invref = invref.inverse().eval();
  Eigen::Matrix<DScalar, 2, 2> jac = tri * (invref).cast<DScalar>();
  auto det = jac.determinant();
  if (det <= 0) {
    spdlog::trace("Degenerate");
    return DScalar(std::numeric_limits<double>::infinity());
  }
  auto frob2 = jac.cwiseAbs2().sum();
  return frob2 * (1 + 1 / (det * det));
}

std::tuple<double, Vec3d>
triangle_one_ring_quality(const RowMatd &mid, const RowMati &F,
                          const std::array<Vec3d, 2> &basis,
                          const RowMatd &refV, const std::vector<int> &nb,
                          const std::vector<int> &nbi, bool with_grad,
                          const Vec3d &modification = Vec3d(0, 0, 0)) {
  Vec2d grad = Vec2d::Zero();
  Eigen::Matrix2d hess = Eigen::Matrix2d::Zero();
  double value = 0;
  for (int index = 0; index < nb.size(); index++) {
    int v_id = nbi[index];
    auto face = F.row(nb[index]);
    std::array<Vec3d, 3> verts_ref;
    std::array<Vec2d, 3> verts;

    for (int i = 0; i < 3; i++) {
      verts[i] << mid.row(face[i]).dot(basis[0]),
          mid.row(face[i]).dot(basis[1]);
      verts_ref[i] = refV.row(face[i]);
    }
    verts[nbi[index]] +=
        Vec2d(modification.dot(basis[0]), modification.dot(basis[1]));
    auto quality = triangle_quality(verts, verts_ref, v_id);
    spdlog::trace("q {}", quality.getValue());
    value += quality.getValue();
    if (with_grad) {
      grad += quality.getGradient();
      hess += quality.getHessian();
    }
  }
  // if (with_grad) { // projected newton
  //   Eigen::Matrix2d R, T, U, V;
  //   Vec2d S;
  //   igl::polar_svd(hess, R, T, U, S, V);
  //   for (int j = 0; j < 2; j++)
  //     S[j] = 1 / std::max(1e-6, S[j]);
  //   // h' = g' * H^-T = g'*(USV')^-T = g' * (V S U')^-1 = g' * U * S^-1 * V'
  //   grad = grad * U * S.asDiagonal() * V.transpose();
  // }
  return std::tuple(value, basis[0] * grad[0] + basis[1] * grad[1]);
}

#include "prism/phong/projection.hpp"
std::optional<std::array<double, 3>>
project_to_dbl_prism(const std::array<Vec3d, 9> &stack, bool type,
                     const Vec3d &spatial) {
  std::array<Vec3d, 6> stackV_bm{stack[0], stack[1], stack[2],
                                 stack[3], stack[4], stack[5]};
  std::array<Vec3d, 6> stackV_mt{stack[3], stack[4], stack[5],
                                 stack[6], stack[7], stack[8]};
  std::array<double, 3> tuple;
  bool hitbottom =
      prism::phong::phong_projection(stackV_bm, spatial, type, tuple);
  if (hitbottom) {
    tuple[2] = tuple[2] - 1; // suppose mid is 0
    return tuple;
  }
  bool hittop = prism::phong::phong_projection(stackV_mt, spatial, type, tuple);
  if (hittop) {
    return tuple;
  }
  spdlog::debug("No proj");
  return {};
}

bool intersect_check(const RowMatd &V, const std::vector<Vec3i> &tris,
                     const prism::geogram::AABB &tree_base,
                     const prism::geogram::AABB &tree_top) {
  for (auto [v0, v1, v2] : tris) {
    spdlog::trace("ic v {} {} {}", v0, v1, v2);
    if (tree_base.intersects_triangle({V.row(v0), V.row(v1), V.row(v2)},
                                      v0 < tree_base.num_freeze))
      return false;
    if (tree_top.intersects_triangle({V.row(v0), V.row(v1), V.row(v2)},
                                     v0 < tree_top.num_freeze))
      return false;
  }
  return true;
}

TEST_CASE("param smoothing") {
  PrismCage pc("../build_profile/block_feature.h5");
  PrismCage pc_original("../build_profile/block_collapse.h5");
  auto refV0 = pc_original.ref.V;
  auto V = pc.ref.V;
  auto F = pc.ref.F;
  Eigen::VectorXi qfid;
  RowMatd quv;
  RowMatd mid = Eigen::Map<RowMatd>(pc.mid[0].data(), pc.mid.size(), 3);
  RowMatd top = Eigen::Map<RowMatd>(pc.top[0].data(), pc.mid.size(), 3);
  RowMatd base = Eigen::Map<RowMatd>(pc.base[0].data(), pc.mid.size(), 3);
  RowMati pcF = Eigen::Map<RowMati>(pc.F[0].data(), pc.F.size(), 3);
  prism::correspond_bc(pc, mid, pcF, V, qfid, quv);
  CHECK_GE(qfid.minCoeff(), 0); // projection success.
  // squeeze all V onto mid for now, notice that this may not be free from flip,
  // so it is better to squeeze along the way of simplification. Hope all is
  // fine for this case for now.
  for (int i = 0; i < V.rows(); i++) {
    auto u = quv(i, 0), v = quv(i, 1);
    V.row(i) = mid.row(pcF(qfid[i], 0)) * (1 - u - v) +
               mid.row(pcF(qfid[i], 1)) * u + mid.row(pcF(qfid[i], 2)) * v;
  }
  prism::geogram::AABB pxtree(mid, pcF);
  prism::geogram::AABB toptree(top, pcF);
  prism::geogram::AABB bottree(base, pcF);
  // Set up Tetrahedral Query structure
  GEO::Mesh geo_tet;
  std::unique_ptr<GEO::MeshCellsAABB> tetaabb;
  {
    std::vector<Vec3d> tetV;
    std::vector<Vec4i> tetT;
    prism::cage_utils::tetmesh_from_prismcage(pc.base, pc.mid, pc.top, pc.F, 0,
                                              tetV, tetT);
    RowMatd mtetV;
    RowMati mtetT;
    vec2eigen(tetV, mtetV);
    vec2eigen(tetT, mtetT);
    prism::geo::init_geogram();
    prism::geo::to_geogram_mesh(mtetV, mtetT, geo_tet);
  }
  tetaabb.reset(new GEO::MeshCellsAABB(geo_tet, false));
  std::vector<std::vector<int>> VF, VFi;
  igl::vertex_triangle_adjacency(V, F, VF, VFi);
  for (auto outer_iter = 0; outer_iter < 10; outer_iter++)
    // for (int i : {1000}) {
    for (int i = 974; i < pc.ref.V.rows(); i++) {
      auto tet_id =
          tetaabb->containing_tet(GEO::vec3(V(i, 0), V(i, 1), V(i, 2)));
      assert(tet_id != GEO::MeshCellsAABB::NO_TET);
      auto prism_id = tet_id / 6;
      auto [v0, v1, v2] = pc.F[prism_id];
      std::array<Vec3d, 9> prism_stack{pc.base[v0], pc.base[v1], pc.base[v2],
                                       pc.mid[v0],  pc.mid[v1],  pc.mid[v2],
                                       pc.top[v0],  pc.top[v1],  pc.top[v2]};
      bool prism_type = v1 > v2;
      std::array<Vec3d, 2> local_basis{(pc.mid[v1] - pc.mid[v0]).normalized(),
                                       (pc.mid[v2] - pc.mid[v0])};
      local_basis[1] =
          (local_basis[1] - local_basis[1].dot(local_basis[0]) * local_basis[0])
              .normalized()
              .eval();
      std::vector<Vec3i> tris;
      for (auto f : VF[i])
        tris.emplace_back(Vec3i{F(f, 0), F(f, 1), F(f, 2)});
      for (auto nt_it = 0; nt_it < 1; nt_it++) {
        auto [e0, dir] = triangle_one_ring_quality(V, F, local_basis, refV0,
                                                   VF[i], VFi[i], true);
        dir = -dir;
        dir.normalize();
        spdlog::debug("energy: {}", e0);
        spdlog::debug("dir: {} {}", dir[0], dir[1]);
        Vec3d modif(0, 0, 0);
        for (auto [j, step_size] = std::tuple(0, 1.); j < 20;
             j++, step_size *= 0.8) {
          auto proj_coord = project_to_dbl_prism(prism_stack, prism_type,
                                                 V.row(i) + step_size * dir);
          if (!proj_coord)
            continue;
          Vec3d orig_i = V.row(i);
          V.row(i) = V.row(i) + step_size * dir;
          if (!intersect_check(V, tris, bottree, toptree)) {
            V.row(i) = orig_i;
            continue;
          }
          V.row(i) = orig_i;

          auto [u, v, t] = proj_coord.value();
          spdlog::debug("uvt {} {} {}", u, v, t);
          modif = step_size * dir;
          // modif = prism_stack[3] * (1 - u - v) + prism_stack[4] * u +
          //         prism_stack[5] * v - V.row(i);
          spdlog::debug("modif {} {} {}", modif[0], modif[1], modif[2]);
          auto [e1, newdir] = triangle_one_ring_quality(
              V, F, local_basis, refV0, VF[i], VFi[i], true, modif);
          constexpr auto c1 = 1e-4;
          spdlog::debug("e0->e1: {}->{}", e0, e1);
          if (e1 < e0) {
            spdlog::debug("e1: {}", e1);
            V.row(i) += modif;
            break; //+ c1*step_size*slope // armijo
          }
        }
      }

      // reject if the neigboring normals are no longer compatible.
      // what if it goes to a different prism? allow it for now.
    }
  igl::write_triangle_mesh("temp.obj", V, F);
}