#include <doctest.h>
#include <geogram/basic/geometry.h>
#include <igl/Timer.h>
#include <igl/avg_edge_length.h>
#include <igl/combine.h>
#include <igl/read_triangle_mesh.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <highfive/H5Easy.hpp>
#include <numeric>
#include <prism/geogram/AABB.hpp>
#include <prism/geogram/geogram_utils.hpp>
#include <prism/predicates/triangle_triangle_intersection.hpp>
#include <prism/spatial-hash/AABB_hash.hpp>
#include <prism/spatial-hash/self_intersection.hpp>

TEST_CASE("spatial hash") {
  RowMatd V;
  RowMati F;
  igl::read_triangle_mesh("../tests/data/bunny.off", V, F);
  put_in_unit_box(V);
  std::vector<Vec3d> vecV;
  std::vector<Vec3i> vecF;
  eigen2vec(V, vecV);
  eigen2vec(F, vecF);
  prism::HashGrid hg(vecV, vecF);
  Eigen::Matrix3d local;
  for (auto k : {0, 1, 2}) local.row(k) = V.row(F(0, k));
  auto aabb_min = local.colwise().minCoeff();
  auto aabb_max = local.colwise().maxCoeff();
  std::set<int> q;
  hg.query(aabb_min, aabb_max, q);
  CHECK(q.size() == 40);
}

TEST_CASE("hash selfintersect") {
  prism::geo::init_geogram();
  RowMatd V;
  RowMati F;
  igl::read_triangle_mesh(
      "/home/zhongshi/Workspace/libigl/tutorial/data/snail.obj", V, F);

  int orig_fnum = F.rows();
  igl::combine(std::vector<RowMatd>{V, RowMatd(V.array() + 0.1)},
               std::vector<RowMati>{F, F}, V, F);
  std::vector<Vec3d> vecV;
  std::vector<Vec3i> vecF;
  eigen2vec(V, vecV);
  eigen2vec(F, vecF);
  auto offending_cand = prism::spatial_hash::self_intersections(vecV, vecF);
  CHECK_EQ(offending_cand.size(), 157);
}

#include <igl/vertex_triangle_adjacency.h>

#include <prism/cage_utils.hpp>

#include "prism/cage_utils.hpp"
#include "prism/local_operations/retain_triangle_adjacency.hpp"
#include "prism/predicates/triangle_triangle_intersection.hpp"
TEST_CASE("hashgrid extrusion") {
  prism::geo::init_geogram();
  ;
  RowMatd V;
  RowMati F;
  igl::read_triangle_mesh("../tests/data/bunny.off", V, F);
  put_in_unit_box(V);

  RowMatd VN;
  std::set<int> omni_singu;
  bool good_normals =
      prism::cage_utils::most_visible_normals(V, F, VN, omni_singu);
  REQUIRE(good_normals);
  REQUIRE(omni_singu.empty());

  // skip topobevel
  auto steps = prism::cage_utils::volume_extrude_steps(
      V, F, VN, true, omni_singu.size(), std::vector<double>(V.rows(), 0.2));

  spdlog::info("step {}",
               std::accumulate(steps.begin(), steps.end(), 0.) / steps.size());

  std::vector<double> v_out(V.rows(), 1);
  for (int i = 0; i < F.rows(); i++) {
    for (int j = 0; j < 3; j++) {
      auto vid = F(i, j);
      v_out[vid] = std::min(v_out[vid], steps[i]);
    }
  }
  // shrink w/ hashgrid
  std::vector<Vec3d> mid;
  std::vector<Vec3d> top;
  std::vector<Vec3i> vecF;
  eigen2vec(V, mid);
  eigen2vec(F, vecF);
  RowMatd mT =
      V +
      Eigen::Map<Eigen::VectorXd>(v_out.data(), v_out.size()).asDiagonal() * VN;
  eigen2vec(mT, top);
  std::for_each(vecF.begin(), vecF.end(), [](Vec3i& f) {
    auto [ignore, f1, s] = tetra_split_AorB(f);
    f = f1;
  });
  prism::HashGrid hg(top, vecF, false);
  std::vector<std::vector<int>> VF, VFi;
  igl::vertex_triangle_adjacency(V, F, VF, VFi);

  std::vector<Vec3d> tetV;
  std::vector<Vec4i> tetT;
  prism::cage_utils::tetmesh_from_prismcage(mid, top, vecF, tetV, tetT);
  std::vector<int> affected_faces(vecF.size());
  std::iota(affected_faces.begin(), affected_faces.end(), 0);
  while (!affected_faces.empty()) {
    hg.clear();
    for (auto f : affected_faces)
      for (int j = 0; j < 3; j++) {
        auto i = 3 * f + j;
        Eigen::Matrix<double, 4, 3> local;
        for (auto k = 0; k < local.rows(); k++) local.row(k) = tetV[tetT[i][k]];
        auto aabb_min = local.colwise().minCoeff();
        auto aabb_max = local.colwise().maxCoeff();
        hg.add_element(aabb_min, aabb_max, i);
      }
    auto cand = hg.self_candidates();
    spdlog::info("cand {}", cand.size());
    std::set<std::pair<int, int>> offend_pairs;
    auto offend_handle = prism::spatial_hash::find_offending_pairs(
        vecF, tetV, tetT, offend_pairs);
    std::for_each(cand.begin(), cand.end(), offend_handle);
    spdlog::info("offending pairs {}", offend_pairs.size());

    std::set<int> masked_verts;
    for (auto [f0, f1] : offend_pairs) {
      for (auto j = 0; j < 3; j++) {
        masked_verts.insert(vecF[f0][j]);
        masked_verts.insert(vecF[f1][j]);
      }
    }
    spdlog::info("masked_verts {}", masked_verts.size());

    int vnum = mid.size();
    affected_faces.clear();
    for (auto v : masked_verts) {
      tetV[vnum + v] = (tetV[vnum + v] + 3 * tetV[v]) / 4;
      affected_faces.insert(affected_faces.end(), VF[v].begin(), VF[v].end());
    }
    std::sort(affected_faces.begin(), affected_faces.end());
    affected_faces.erase(
        std::unique(affected_faces.begin(), affected_faces.end()),
        affected_faces.end());
  }
  H5Easy::File file("temp.h5", H5Easy::File::Overwrite);
  H5Easy::dump(file, "F", F);
  H5Easy::dump(file, "tetT",
               RowMati(Eigen::Map<RowMati>(tetT[0].data(), tetT.size(), 4)));
  H5Easy::dump(file, "tetV0",
               RowMatd(Eigen::Map<RowMatd>(tetV[0].data(), tetV.size(), 3)));

  // TODO: also adding boundary?
  // H5Easy::dump(file, "tetV1",
  //              RowMatd(Eigen::Map<RowMatd>(tetV[0].data(), tetV.size(), 3)));
  // H5Easy::dump(file, "tris",
  //              std::vector<int>(affected_faces.begin(),
  //              affected_faces.end()));
}

#include "prism/PrismCage.hpp"
TEST_CASE("spatial hash integrated") {
  prism::geo::init_geogram();
  spdlog::set_level(spdlog::level::debug);
  RowMatd V;
  RowMati F;
  igl::read_triangle_mesh("../tests/data/bunny.off", V, F);
  put_in_unit_box(V);
  PrismCage pc(V, F, 0.2, 0.1, PrismCage::SeparateType::kShell);
  pc.serialize("temp.h5");
}
