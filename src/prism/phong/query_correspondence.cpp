#include "query_correspondence.hpp"

#include <geogram/mesh/mesh_AABB.h>
#include <igl/exact_geodesic.h>
#include <igl/heat_geodesics.h>
#include <igl/read_triangle_mesh.h>
#include <igl/remove_duplicate_vertices.h>
#include <igl/writePLY.h>
#include <igl/write_triangle_mesh.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <highfive/H5Easy.hpp>
#include <prism/cage_utils.hpp>
#include <prism/common.hpp>
#include <prism/geogram/geogram_utils.hpp>
#include <prism/intersections.hpp>
#include <prism/phong/projection.hpp>

#include "prism/PrismCage.hpp"
#include "prism/geogram/AABB.hpp"

bool prism::project_to_proxy_mesh(const std::array<Vec3d, 9> &stack,
                                  const prism::geogram::AABB &pxtree, bool type,
                                  const Vec3d &spatial, prism::Hit &hit) {
  std::array<Vec3d, 6> stackV_bm{stack[0], stack[1], stack[2],
                                 stack[3], stack[4], stack[5]};
  std::array<Vec3d, 6> stackV_mt{stack[3], stack[4], stack[5],
                                 stack[6], stack[7], stack[8]};
  std::array<double, 3> tuple;
  bool bottom = false;
  bool hitbottom =
      prism::phong::phong_projection(stackV_bm, spatial, type, tuple);

  bool hittop = prism::phong::phong_projection(stackV_mt, spatial, type, tuple);
  if (!hitbottom && !hittop) {
    spdlog::error("No proj");
    return false;
  }
  std::array<Vec3d, 4> endpoints0;
  prism::phong::fiber_endpoints(stackV_bm, type, tuple[0], tuple[1],
                                endpoints0);
  for (int i = 0; i < 3; i++) {
    if (endpoints0[i] == endpoints0[i + 1]) continue;
    if (pxtree.segment_hit(endpoints0[i], endpoints0[i + 1], hit)) return true;
  }
  std::array<Vec3d, 4> endpoints1;
  prism::phong::fiber_endpoints(stackV_mt, type, tuple[0], tuple[1],
                                endpoints1);
  for (int i = 0; i < 3; i++) {
    if (endpoints1[i] == endpoints1[i + 1]) continue;
    if (pxtree.segment_hit(endpoints1[i], endpoints1[i + 1], hit)) return true;
  }
  return false;
}

void prism::correspond_bc(const PrismCage &pc, const RowMatd &pxV,
                          const RowMati &pxF, const RowMatd &queryP,
                          Eigen::VectorXi &queryF, RowMatd &queryUV) {
  prism::geogram::AABB pxtree(pxV, pxF);
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

  queryF.setZero(queryP.rows());
  queryUV.setZero(queryP.rows(), 2);
  for (int i = pc.ref.aabb->num_freeze; i < queryP.rows(); i++) {
    Vec3d spatial = queryP.row(i);
    auto tet_id =
        tetaabb->containing_tet(GEO::vec3(spatial[0], spatial[1], spatial[2]));
    if (tet_id == GEO::MeshCellsAABB::NO_TET) {
      spdlog::error("tet not found {}", spatial);
      queryF[i] = -1;
      continue;
    }
    auto prism_id = tet_id / 6;
    prism::Hit hit{-1, -1, -1, -1, -1};
    auto [v0, v1, v2] = pc.F[prism_id];
    if (!project_to_proxy_mesh(
            {pc.base[v0], pc.base[v1], pc.base[v2], pc.mid[v0], pc.mid[v1],
             pc.mid[v2], pc.top[v0], pc.top[v1], pc.top[v2]},
            pxtree, v1 > v2, spatial, hit)) {
      spdlog::error("Failed {}", i);
      queryF[i] = -2;
      continue;
    };
    queryF[i] = hit.id;
    queryUV.row(i) << hit.u, hit.v;
  }
}

bool prism::project_to_ref_mesh(
    const PrismCage &pc, const std::vector<std::set<int>> &track_to_prism,
    const std::vector<int> &tris, const Vec3d &point_value,
    Vec3d &point_on_ref) {
  int found = -1;
  std::array<double, 3> tuple;
  std::set<int> combined_tracks;
  for (auto f0 : tris)
    std::merge(track_to_prism[f0].begin(), track_to_prism[f0].end(),
               combined_tracks.begin(), combined_tracks.end(),
               std::inserter(combined_tracks, combined_tracks.begin()));
  spdlog::trace("combined tracks {}", (combined_tracks));
  for (auto t : combined_tracks) {
    auto [v0, v1, v2] = pc.F[t];
    spdlog::trace("vvv {} {} {}", v0, v1, v2);
    if (prism::phong::phong_projection({pc.base[v0], pc.base[v1], pc.base[v2],
                                        pc.mid[v0], pc.mid[v1], pc.mid[v2]},
                                       point_value, v1 > v2, tuple) ||
        prism::phong::phong_projection({pc.mid[v0], pc.mid[v1], pc.mid[v2],
                                        pc.top[v0], pc.top[v1], pc.top[v2]},
                                       point_value, v1 > v2, tuple)) {
      found = t;
      break;
    }
  }
  if (found == -1) {
    spdlog::debug("phong proj failed");
    spdlog::debug("Point {}", point_value);
    spdlog::debug("tris_size {}", tris.size());
    spdlog::trace("i {}", tris);
    for (auto i : tris) {
      spdlog::trace("track_to_prism {}", track_to_prism[i]);
    }
    return false;
  }
  auto [v0, v1, v2] = pc.F[found];
  std::array<Vec3d, 4> endpoints0;
  prism::phong::fiber_endpoints({pc.base[v0], pc.base[v1], pc.base[v2],
                                 pc.mid[v0], pc.mid[v1], pc.mid[v2]},
                                v1 > v2, tuple[0], tuple[1], endpoints0);
  for (int i = 0; i < 3; i++) {
    if (endpoints0[i] == endpoints0[i + 1]) continue;
    auto inter = pc.ref.aabb->segment_query(endpoints0[i], endpoints0[i + 1]);
    if (inter) {
      point_on_ref = inter.value();
      return true;
    }
  }
  std::array<Vec3d, 4> endpoints1;
  prism::phong::fiber_endpoints(
      {pc.mid[v0], pc.mid[v1], pc.mid[v2], pc.top[v0], pc.top[v1], pc.top[v2]},
      v1 > v2, tuple[0], tuple[1], endpoints1);
  for (int i = 0; i < 3; i++) {
    if (endpoints1[i] == endpoints1[i + 1]) continue;
    auto inter = pc.ref.aabb->segment_query(endpoints1[i], endpoints1[i + 1]);
    if (inter) {
      point_on_ref = inter.value();
      return true;
    }
  }
  spdlog::debug("Point {}", point_value);
  spdlog::debug("intersection failed, t={}, tuple={},{}", found, tuple[0],
                tuple[1]);

  return false;
}