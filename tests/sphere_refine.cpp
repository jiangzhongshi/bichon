#include <doctest.h>
#include <geogram/basic/geometry.h>
#include <igl/Timer.h>
#include <igl/avg_edge_length.h>
#include <igl/combine.h>
#include <igl/read_triangle_mesh.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include "prism/PrismCage.hpp"
#include <geogram/basic/geometry.h>
#include <highfive/H5Easy.hpp>
#include <numeric>
#include <prism/geogram/AABB.hpp>
#include <prism/geogram/geogram_utils.hpp>
#include <prism/predicates/triangle_triangle_intersection.hpp>
#include <prism/spatial-hash/AABB_hash.hpp>
#include <prism/spatial-hash/self_intersection.hpp>
#include <prism/local_operations/validity_checks.hpp>
#include <prism/local_operations/remesh_pass.hpp>

auto circumradi2(const Vec3d &p0, const Vec3d &p1, const Vec3d &p2,
                 const Vec3d &p3) -> double {
  std::array<GEO::vec3, 4> geo_v;
  geo_v[0] = GEO::vec3(p0[0], p0[1], p0[2]);
  geo_v[1] = GEO::vec3(p1[0], p1[1], p1[2]);
  geo_v[2] = GEO::vec3(p2[0], p2[1], p2[2]);
  geo_v[3] = GEO::vec3(p3[0], p3[1], p3[2]);
  GEO::vec3 center =
      GEO::Geom::tetra_circum_center(geo_v[0], geo_v[1], geo_v[2], geo_v[3]);
  return GEO::distance2(center, geo_v[0]);
}

auto set_inter = [](auto &A, auto &B) {
  std::vector<int> vec;
  std::set_intersection(A.begin(), A.end(), B.begin(), B.end(),
                        std::back_inserter(vec));
  return vec;
};

namespace wmtk {
static constexpr std::array<std::array<int, 2>, 6> m_local_edges = {
    {{{0, 1}},
     {{1, 2}},
     {{2, 0}},
     {{0, 3}},
     {{1, 3}},
     {{2, 3}}}}; // local edges within a tet
static constexpr std::array<int, 6> m_map_edge2face = {{0, 0, 0, 1, 2, 1}};
static constexpr std::array<std::array<int, 3>, 6> m_local_faces = {
    {{{0, 1, 2}}, {{0, 2, 3}}, {{0, 3, 1}}, {{3, 2, 1}}}};
} // namespace wmtk

struct TetAttr {
  std::array<int, 4> prism_id = {{-1, -1, -1, -1}};
  bool is_removed = false;
  Vec4i conn;
};

struct VertAttr {
  Vec3d mid, top, bot;
  bool on_boundary = false;
};

auto tetra_validity = [](const std::vector<VertAttr>& tet_pos, const Vec4i& t){
  return GEO::PCK::orient_3d(
    tet_pos[t[0]].mid.data()
    tet_pos[t[1]].mid.data()
    tet_pos[t[2]].mid.data()
    tet_pos[t[3]].mid.data()
    );
};

bool split_edge(PrismCage& pc, std::vector<VertAttr> &tet_pos,// std::vector<Vec4i> &tet_conn,
                std::vector<TetAttr> &tet_attrs,
                std::vector<std::vector<int>> vertex_conn, int v0, int v1) {
  prism::local::RemeshOptions option;
  auto &nb0 = vertex_conn[v0];
  auto &nb1 = vertex_conn[v1];
  auto affected = set_inter(nb0, nb1); // removed

  std::vector<Vec4i> new_tets;
  std::vector<TetAttr> new_attrs;
  auto vx = tet_pos.size();
  auto replace = [](auto &arr, auto a, auto b) {
    for (auto i = 0; i < arr.size(); i++) {
      if (arr[i] == a) {
        arr[i] = b;
        return i;
      }
    }
    assert(false);
    return -1;
  };

  auto boundary_edge = false;
  std::vector<int> bnd_pris;
  for (auto t : affected) {
    for (auto j = 0; j < 4; j++) {
      if (tet_attrs[t].prism_id[j] != -1 // boundary
      && tet_attrs[t].conn[j] != v0 //  contains v0 AND v1 
      && tet_attrs[t].conn[j] != v1 )// 
      {
        boundary_edge = true;
        bnd_pris.push_back(tet_attrs[t].prism_id[j]);
      }
    }
    new_tets.push_back(tet_attrs[t].conn);
    new_attrs.push_back(tet_attrs[t]);
    replace(new_tets.back(), v0, vx);

    new_tets.push_back(tet_attrs[t].conn);
    new_attrs.push_back(tet_attrs[t]);
    replace(new_tets.back(), v1, vx);
  }

  tet_pos.push_back({});
  tet_pos.back().mid = ((tet_pos[v0].mid + tet_pos[v1].mid) / 2);
  tet_pos.back().on_boundary = boundary_edge;

  if (boundary_edge) {
    assert(bnd_pris.size() == 2);
    assert(tet_pos[v0].on_boundary && tet_pos[v1].on_boundary);
    tet_pos.back().top = ((tet_pos[v0].top + tet_pos[v1].top) / 2);
    tet_pos.back().bot = ((tet_pos[v0].bot + tet_pos[v1].bot) / 2);
  }

  auto rollback = [&]() {
    tet_pos.pop_back();
    return false;
  };
  for (auto t : new_tets) { // Vec4i
    if (!tetra_validity(tet_pos, t))
      return rollback();
  }
  if (boundary_edge) {
    assert(bnd_pris.size() == 2);
    auto &F = pc.F;
    auto f0 = bnd_pris.front(), f1 = bnd_pris.back();
    std::vector<int> old_fids = {f0, f1};
    std::vector<int> new_fid = {f0, f1, int(F.size()), int(F.size() + 1)};
    auto moved_tris = std::vector<Vec3i>{F[f0], F[f1], F[f0], F[f1]};
    replace(moved_tris[0], v0, vx);
    replace(moved_tris[1], v0, vx);
    replace(moved_tris[2], v1, vx);
    replace(moved_tris[3], v1, vx);

    std::vector<std::set<int>> new_tracks;
    std::vector<RowMatd> local_cp;
    auto flag = prism::local_validity::attempt_zig_remesh(
        pc, pc.track_ref, option, -1, old_fids, moved_tris, new_tracks,
        local_cp);
    if (flag != 0)
      return rollback();

    // distribute and assign new_tracks.
  }

  for (auto ti: affected) {
    tet_attrs[ti].is_removed = true;
  }
  // update connectivity: VT TODO
  auto n_tet = tet_attrs.size();
  for (auto t: new_tets) {
    for (auto i=0; i<t.size(); i++) {
      set_minus(vertex_conn[t[i]], affected);
      set_insert(vertex_conn[t[i]], {n_tet});
    }
    n_tet ++;
    auto t_a = TetAttr();
    t_a.conn = t;
    // TODO: boundary
    tet_attrs.emplace_back(t_a);
  }
  assert(n_tet == tet_attrs.size());
  // remove `inter`, append `new_tid`
  // assigns split TetAttrs.
  return true;
};

TEST_CASE("amr-sphere-prepare") {
  std::string filename = "../tests/data/sphere_40.obj.h5";
  PrismCage pc(filename);

  H5Easy::File file(filename, H5Easy::File::ReadOnly);
  auto tet_v = H5Easy::load<RowMatd>(file, "tet_v");
  auto tet_t = H5Easy::load<RowMati>(file, "tet_t");
  spdlog::info("Loading v {},t {} ", tet_v.rows(), tet_t.rows());

  for (auto i = 0; i < tet_t.rows(); i++) {
    auto r = circumradi2(tet_v.row(tet_t(i, 0)), tet_v.row(tet_t(i, 1)),
                         tet_v.row(tet_t(i, 2)), tet_v.row(tet_t(i, 3)));
    // spdlog::info("radius2 {}", r);
  }

  // Tet-Face (tet_t, k) -> Shell Prisms (pc.mF)
  std::map<std::array<int, 3>, int> cell_finder;
  for (auto i = 0; i < pc.F.size(); i++) {
    auto pri = pc.F[i];
    std::sort(pri.begin(), pri.end());
    cell_finder.emplace(pri, i);
  }
  RowMati marker = RowMati::Constant(tet_t.rows(), 4, -1);
  for (auto i = 0; i < tet_t.rows(); i++) {
    for (auto j = 0; j < 4; j++) {
      auto face = std::array<int, 3>();
      for (auto k = 0; k < 3; k++)
        face[k] = tet_t(i, (j + k) % 4);
      std::sort(face.begin(), face.end());
      auto it = cell_finder.find(face);
      if (it != cell_finder.end()) { // found
        marker(i, j) = it->second;
      }
    }
  }

  // count how many marks
  [&marker, &tet_t, &pc]() {
    auto cnt = 0;
    for (auto i = 0; i < tet_t.rows(); i++) {
      for (auto j = 0; j < 4; j++) {
        if (marker(i, j) != -1)
          cnt++;
      }
    }
    CHECK_EQ(cnt, pc.F.size());
  }();
}