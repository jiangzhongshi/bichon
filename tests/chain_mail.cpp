#include <geogram/mesh/mesh_AABB.h>
#include <geogram/mesh/mesh_geometry.h>
#include <igl/barycentric_coordinates.h>
#include <igl/boundary_facets.h>
#include <igl/doublearea.h>
#include <igl/parallel_for.h>
#include <igl/readOBJ.h>
#include <igl/read_triangle_mesh.h>
#include <igl/remove_unreferenced.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/volume.h>
#include <igl/writeMESH.h>
#include <igl/write_triangle_mesh.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <doctest.h>
#include <prism/PrismCage.hpp>
#include <prism/cage_utils.hpp>
#include <prism/cgal/polyhedron_self_intersect.hpp>
#include <prism/energy/prism_quality.hpp>
#include <prism/energy/smoother_pillar.hpp>
#include <prism/geogram/geogram_utils.hpp>
#include <prism/phong/projection.hpp>
#include <queue>
#include <igl/Hit.h>
#include "test_common.hpp"

TEST_CASE("lizard chain") {
  PrismCage pc("../buildr/animal_tri.obj.h5");
  std::vector<Vec3d> V;
  std::vector<Vec4i> T;
  prism::cage_utils::tetmesh_from_prismcage(pc.base, pc.mid, pc.top, pc.F, 0, V,
                                            T);
  RowMatd mV;
  RowMati mT, mF;
  vec2eigen(V, mV);
  vec2eigen(T, mT);
  prism::geo::init_geogram();
  GEO::Mesh geo_tet, geo_uv;
  prism::geo::to_geogram_mesh(mV, mT, geo_tet);
  GEO::MeshCellsAABB tetaabb(geo_tet, false);

  RowMatd meshV, meshUV, meshCN;
  RowMati meshF, meshTF, meshFN;
  igl::readOBJ("../tests/data/animal_tri.obj", meshV, meshUV, meshCN, meshF,
               meshTF, meshFN);
  put_in_unit_box(meshV);
  RowMatd meshUV3 = RowMatd::Zero(meshUV.rows(), 3);
  meshUV3.leftCols(2) = meshUV.leftCols(2);
  prism::geo::to_geogram_mesh(meshUV3, meshTF, geo_uv);
  GEO::MeshFacetsAABB uv_aabb(geo_uv, false);
  spdlog::info("MeshUV {}, {}", meshUV3.colwise().maxCoeff(),
               meshUV3.colwise().minCoeff());

  RowMatd in_chainV;
  RowMati chainF;
  igl::read_triangle_mesh("../tests/data/chainmail_64x100.obj", in_chainV,
                          chainF);
  RowMatd chainV = in_chainV;
  auto make_between_zero_one = [](Eigen::VectorXd r) {
    double ma = r.maxCoeff();
    double mi = r.minCoeff();
    return (r.array() - mi) / (ma - mi);
  };
  chainV.col(0) = make_between_zero_one(in_chainV.col(0));
  chainV.col(2) = make_between_zero_one(in_chainV.col(1));
  chainV.col(1) = make_between_zero_one(in_chainV.col(2));
  spdlog::info("chainV {}, {}", chainV.colwise().maxCoeff(),
               chainV.colwise().minCoeff());
  std::set<int> missed_verts;
  RowMatd new_coord = chainV * 0;
  for (int i = 0; i < chainV.rows(); i++) {
    auto u = chainV(i, 0), v = chainV(i, 1), t = chainV(i, 2);
    t *= 0.95;
    // find in UV space
    igl::Hit hit;
    {
      GEO::vec3 hitpoint;
      double dist;
      GEO::index_t face_id = uv_aabb.nearest_facet(
          GEO::vec3(u, v, 0), hitpoint, dist);
      if (dist > 1e-5) {
        missed_verts.insert(i);
        continue;
      }
      auto realf = (int)face_id;
      auto v0 = meshTF(realf, 0), v1 = meshTF(realf, 1), v2 = meshTF(realf, 2);
      double lambda1, lambda2, lambda3;
      GEO::Geom::point_triangle_squared_distance(
          GEO::vec3(u, v, 0),
          GEO::vec3(meshUV3(v0, 0), meshUV3(v0, 1), meshUV3(v0, 2)),
          GEO::vec3(meshUV3(v1, 0), meshUV3(v1, 1), meshUV3(v1, 2)),
          GEO::vec3(meshUV3(v2, 0), meshUV3(v2, 1), meshUV3(v2, 2)), hitpoint,
          lambda1, lambda2, lambda3);
      // spdlog::info("geo f {} l2{}, l3 {}", realf, lambda2, lambda3);
      hit.id = realf;
      hit.u = lambda2;
      hit.v = lambda3;
    }

    int f = hit.id;
    Vec3d spatial =
        (meshV.row(meshF(f, 0)) * (1 - hit.u - hit.v) +
         meshV.row(meshF(f, 1)) * hit.u + meshV.row(meshF(f, 2)) * hit.v);

    auto tet_id =
        tetaabb.containing_tet(GEO::vec3(spatial[0], spatial[1], spatial[2]));
    if (tet_id == GEO::MeshCellsAABB::NO_TET) {
      spdlog::warn("No tet");
      continue;
    }
    auto prism_id = tet_id / 6;
    bool bottom = (tet_id % 6) < 3;

    std::array<double, 3> bary;
    auto [v0, v1, v2] = pc.F[prism_id];

    // new_coord.row(i) = (pc.mid[v0]+  pc.mid[v1]+ pc.mid[v2])/3;continue;
    std::array<Vec3d, 6> stackV;
    if (bottom)
      stackV = {pc.base[v0], pc.base[v1], pc.base[v2],
                pc.mid[v0],  pc.mid[v1],  pc.mid[v2]};
    else
      stackV = {pc.mid[v0], pc.mid[v1], pc.mid[v2],
                pc.top[v0], pc.top[v1], pc.top[v2]};
    prism::phong::phong_projection(stackV, spatial, v1 > v2, bary);

    stackV = {pc.mid[v0], pc.mid[v1], pc.mid[v2],
              pc.top[v0], pc.top[v1], pc.top[v2]};
    std::array<Vec3d, 4> endpoints;
    prism::phong::fiber_endpoints(stackV, v1 > v2, bary[0], bary[1], endpoints);
    Vec3d seg_length(0, 0, 0);
    for (int j = 0; j < 3; j++)
      seg_length[j] = (endpoints[j + 1] - endpoints[j]).norm();
    Vec3d advect(6, 6, 6);
    double target_length = (t)*seg_length.sum();
    for (int j = 0; j < 3; j++) {
      if (target_length < seg_length(j)) {
        advect =
            target_length / seg_length(j) * (endpoints[j + 1] - endpoints[j]) +
            endpoints[j];
        break;
      }
      target_length -= seg_length(j);
    }
    new_coord.row(i) = advect;
  }
  spdlog::info("Total {}, missed{}", new_coord.rows(), missed_verts.size());
  std::vector<Vec3i> newface;
  for (int i = 0; i < chainF.rows(); i++) {
    bool flag = false;
    for (int j = 0; j < 3; j++) {
      if (missed_verts.find(chainF(i, j)) != missed_verts.end()) {
        // chainF.row(i) << -1,-1,-1;
        flag = true;
        break;
      }
    }
    if (!flag)
      newface.emplace_back(Vec3i{chainF(i, 0), chainF(i, 1), chainF(i, 2)});
  }
  RowMatd saveV;
  RowMati saveF, I, J;
  vec2eigen(newface, chainF);
  spdlog::info("minimum {}", chainF.minCoeff());
  igl::remove_unreferenced(new_coord, chainF, saveV, saveF, I, J);
  spdlog::info("minimum {}", saveF.minCoeff());
  igl::write_triangle_mesh("temp.obj", saveV, saveF);
}
