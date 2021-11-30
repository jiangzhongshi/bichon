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
#include "prism/PrismCage.hpp"
#include <geogram/basic/geometry.h>

auto circumradi = [](
  const Vec3d& p0,
  const Vec3d& p1,
  const Vec3d& p2,
  const Vec3d& p3
)->double{
  std::array<GEO::vec3,4> geo_v; 
  geo_v[0] = GEO::vec3(p0[0], p0[1],p0[2]);
  geo_v[1] = GEO::vec3(p1[0], p1[1],p1[2]);
  geo_v[2] = GEO::vec3(p2[0], p2[1],p2[2]);
  geo_v[3] = GEO::vec3(p3[0], p3[1],p3[2]);
  GEO::vec3 center = GEO::Geom::tetra_circum_center(geo_v[0],geo_v[1],geo_v[2],geo_v[3]);
  return GEO::distance2(center, geo_v[0]);
};

TEST_CASE("amr-sphere-prepare") {
  std::string filename = "../tests/data/sphere_40.obj.h5";
    PrismCage pc(filename);

    H5Easy::File file(filename, H5Easy::File::ReadOnly);
  auto tet_v = H5Easy::load<RowMatd>(file, "tet_v");
  auto tet_t = H5Easy::load<RowMati>(file, "tet_t");
  spdlog::info("Loading v {},t {} ", tet_v.rows(), tet_t.rows());

  for (auto i=0; i<tet_t.rows(); i++) {
      auto r = circumradi(
        tet_v.row(tet_t(i,0)),
        tet_v.row(tet_t(i,1)),
        tet_v.row(tet_t(i,2)),
        tet_v.row(tet_t(i,3))
        );
        spdlog::info("radius {}", r);
  }
}