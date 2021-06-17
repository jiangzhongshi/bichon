#include <doctest.h>
#include <igl/read_triangle_mesh.h>
#include <igl/vertex_triangle_adjacency.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <cumin/bernstein_eval.hpp>
#include <cumin/curve_common.hpp>
#include <cumin/curve_utils.hpp>
#include <cumin/curve_validity.hpp>

#include <nlohmann/json.hpp>
#include <prism/feature_utils.hpp>
#include <prism/geogram/geogram_utils.hpp>
#include <prism/phong/projection.hpp>
#include <string>

#include "prism/PrismCage.hpp"
#include "prism/cage_check.hpp"
#include "prism/geogram/AABB.hpp"
#include "prism/local_operations/remesh_pass.hpp"
#include "prism/local_operations/retain_triangle_adjacency.hpp"
#include "prism/spatial-hash/AABB_hash.hpp"
#include "prism/spatial-hash/self_intersection.hpp"
#include "test_common.hpp"

#define DATA_PATH "../build_clang/"

#include "prism/local_operations/remesh_with_feature.hpp"
namespace prism::curve {
void localcurve_pass(PrismCage &pc, const prism::local::RemeshOptions &option);
}
auto post_collapse = [](auto &complete_cp) {
  complete_cp.erase(std::remove_if(complete_cp.begin(), complete_cp.end(),
                                   [](auto &c) { return c(0, 0) == -1; }),
                    complete_cp.end());
};

auto collision_checker = [](auto &pc) {
  if (!prism::spatial_hash::self_intersections(pc.top, pc.F).empty()) {
    spdlog::error("top inter");
    return false;
  }
  if (!prism::spatial_hash::self_intersections(pc.base, pc.F).empty()) {
    spdlog::error("base inter");
    return false;
  }
  if (!prism::cage_check::cage_is_away_from_ref(pc)) {
    spdlog::error("X inter");
    return false;
  }
  return true;
};

auto reverse_feature_order = [](PrismCage &pc,
                                prism::local::RemeshOptions &option) {
  decltype(pc.meta_edges) meta;
  for (auto [a, b] : pc.meta_edges) {
    auto a1 = std::pair{a.second, a.first};
    auto b1 = b;
    b1.second = std::vector<int>(b.second.rbegin(), b.second.rend());
    meta.emplace(a1, b1);
  }
  pc.meta_edges = std::move(meta);
  auto &crt = option.chain_reject_trackee;
  for (int i = 0; i < crt.size(); i += 2) {
    std::swap(crt[i], crt[i + 1]);
  }
};

#include "prism/energy/prism_quality.hpp"
double total_energy(const std::vector<Vec3d> &V, const std::vector<Vec3i> &F) {
  std::set<int> low_quality_vertices;
  double total_quality = 0;
  double max_quality = 0;
  for (auto [v0, v1, v2] : F) {
    auto q = prism::energy::triangle_quality({V[v0], V[v1], V[v2]});
    total_quality += q;
    max_quality = std::max(max_quality, q);
  }

  spdlog::info("Total Q {} fnum {}, avg {}, max {}", total_quality, F.size(),
               total_quality / F.size(), max_quality);
  return max_quality;
};

//std::tuple<RowMatd, RowMatd, RowTenXd<3UL>, std::array<RowMatd, 2UL>,
//           std::tuple<RowMatd, RowMati, Vec3i,
//                      std::array<std::vector<int>, 3UL>, std::vector<int>>>
//magic_matrices(int tri_order, int level);
std::tuple<RowMatd, RowMatd, std::vector<RowMatd>, std::array<RowMatd, 2UL>, std::tuple<RowMatd, RowMati, Vec3i, std::array<std::vector<int>, 3UL>, std::vector<int>>> magic_matrices(int tri_order, int level);

#include <prism/intersections.hpp>
auto max_distance_error = [](const PrismCage &pc,
                             const std::vector<RowMatd> &cp) {
  prism::geogram::AABB tree(pc.ref.V, pc.ref.F);
  auto [tri10_lv5, elevlag_from_bern, tet4_dxyz, duv_lv5, upsample_helper] =
      magic_matrices(3, 3);

  int sample_size = tri10_lv5.cols();
  RowMatd high_order_pos = RowMatd::Zero(pc.F.size() * sample_size, 3);
  for (int f = 0; f < pc.F.size(); f++) {
    high_order_pos.middleRows(sample_size * f, sample_size) =
        tri10_lv5.transpose() * cp[f];
  }
  auto ulevel = 3ul;
  std::vector<int> sp_fid;
  std::vector<Vec3d> sp_uv;
  auto [unitV, unitF, vert_id, edge_id, face_id] = upsample_helper;
  for (int f = 0; f < pc.F.size(); f++) {
    for (int s = 0; s < sample_size; s++) {
      sp_fid.push_back(f);
      auto &u = unitV(s, 0), &v = unitV(s, 1);
      sp_uv.emplace_back(1 - u - v, u, v);
    }
  }

  std::vector<prism::Hit> ray_hits;
  std::set<int> combined_track;
  prism::curve::sample_hit_discrete(pc.base, pc.mid, pc.top, pc.F, sp_fid,
                                    sp_uv, pc.ref.V, pc.ref.F, tree,
                                    combined_track, ray_hits);
  RowMatd ray_hit_pos(ray_hits.size(), 3);
  auto &inpF = pc.ref.F;
  auto &inpV = pc.ref.inpV;
  for (int i = 0; i < ray_hits.size(); i++) {
    auto hit = ray_hits[i];
    auto v0 = inpF(hit.id, 0), v1 = inpF(hit.id, 1), v2 = inpF(hit.id, 2);
    auto u = hit.u, v = hit.v;
    ray_hit_pos.row(i) =
        inpV.row(v0) * (1 - u - v) + inpV.row(v1) * u + inpV.row(v2) * v;
  }
  spdlog::info((high_order_pos - ray_hit_pos).rowwise().norm().maxCoeff());
};