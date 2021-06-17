#include "self_intersection.hpp"

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

#include <algorithm>

#include "AABB_hash.hpp"
#include "prism/cage_utils.hpp"
#include "prism/predicates/tetrahedron_overlap.hpp"
#include "prism/local_operations/retain_triangle_adjacency.hpp"
#include "prism/predicates/triangle_triangle_intersection.hpp"
#include <highfive/H5Easy.hpp>


constexpr auto share_vertex = [](const auto &f0, auto &f1, int &s0, int &s1) {
  auto cnt = 0;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      if (f0[i] == f1[j]) {
        cnt++;
        s0 = i;
        s1 = j;
      };
  assert(cnt < 3);  // unless all the same
  return cnt;
};
constexpr auto intersection = [](const auto &V, const auto &f0, auto &f1) {
  auto &[p0, p1, p2] = f0;
  auto &[q0, q1, q2] = f1;
  return prism::predicates::triangle_triangle_overlap({V[p0], V[p1], V[p2]},
                                                      {V[q0], V[q1], V[q2]});
};
constexpr auto reduced_intersection = [](const auto &V, const auto &f0,
                                         auto &f1) {
  auto &[p0, p1, p2] = f0;
  auto &[q0, q1, q2] = f1;
  assert(V[p0] == V[q0] && "first vertex coincide");
  bool o1 = prism::predicates::segment_triangle_overlap({V[p1], V[p2]},
                                                        {V[q0], V[q1], V[q2]});
  bool o2 = prism::predicates::segment_triangle_overlap({V[q1], V[q2]},
                                                        {V[p0], V[p1], V[p2]});
  return o1 || o2;
};

auto prism::spatial_hash::self_intersections(const std::vector<Vec3d> &vecV,
                                             const std::vector<Vec3i> &vecF)
    -> std::vector<std::pair<int, int>> {
  prism::HashGrid hg(vecV, vecF);
  auto cand = hg.self_candidates();
  std::vector<std::pair<int, int>> offending_cand;
  std::for_each(
      cand.begin(), cand.end(),
      [&offending_cand, &vecV, &vecF](const auto &f01) {
        auto [f0, f1] = f01;
        auto verts0 = Vec3i({vecF[f0][0], vecF[f0][1], vecF[f0][2]});
        auto verts1 = Vec3i({vecF[f1][0], vecF[f1][1], vecF[f1][2]});
        int s0 = -1, s1 = -1;
        auto cnt = share_vertex(verts0, verts1, s0, s1);
        auto flag = false;
        if (cnt == 0) {
          flag = intersection(vecV, verts0, verts1);
        } else if (cnt == 1) {
          int df0 = f0, df1 = f1;
          assert(s0 != -1 && s1 != -1);
          std::swap(verts0[0], verts0[s0]);
          std::swap(verts1[0], verts1[s1]);
          flag = reduced_intersection(vecV, verts0, verts1);
          if (flag == true) {
            spdlog::warn(
                "hypothesis: this would not happen in shell settings.");
          }
        } else {
          ;
        };
        if (flag) {
          offending_cand.emplace_back(f0, f1);
        }
      });
  return offending_cand;
}

auto prism::spatial_hash::tetrashell_self_intersections(
    const std::vector<Vec3d> &base, const std::vector<Vec3d> &top,
    const std::vector<Vec3i> &F) -> std::set<std::pair<int, int>> {
  // this is not dealing with singularity explicitly, but the degenerate tetra
  // should not interfere.

  std::vector<Vec3d> tetV;
  std::vector<Vec4i> tetT;
  prism::cage_utils::tetmesh_from_prismcage(base, top, F, tetV, tetT);

  prism::HashGrid hg(top, F, /*start empty*/ false);
  for (int i = 0; i < tetT.size(); i++) {
    Eigen::Matrix<double, 4, 3> local;
    for (auto k : {0, 1, 2, 3}) local.row(k) = tetV[tetT[i][k]];
    auto aabb_min = local.colwise().minCoeff();
    auto aabb_max = local.colwise().maxCoeff();
    hg.add_element(aabb_min, aabb_max, i);
  }

  auto cand = hg.self_candidates();
  std::set<std::pair<int, int>> offend_pairs;
  auto offend_handle =
      prism::spatial_hash::find_offending_pairs(F, tetV, tetT, offend_pairs);
  std::for_each(cand.begin(), cand.end(), offend_handle);
  return offend_pairs;
}

std::function<void(const std::pair<int, int> &)>
prism::spatial_hash::find_offending_pairs(
    const std::vector<Vec3i> &F, const std::vector<Vec3d> &tetV,
    const std::vector<Vec4i> &tetT,
    std::set<std::pair<int, int>> &offend_pairs) {
  return [&F, &tetV, &tetT,
          &offend_pairs](const std::pair<int, int> &t01) -> void {
    auto [t0, t1] = t01;
    auto f0 = t0 / 3, f1 = t1 / 3;
    if (f0 == f1) return;
    if (f0 > f1) std::swap(f0,f1);
    if (offend_pairs.find({f0,f1}) != offend_pairs.end()) return;
    {
      auto verts0 = Vec3i({F[f0][0], F[f0][1], F[f0][2]});
      auto verts1 = Vec3i({F[f1][0], F[f1][1], F[f1][2]});
      int s0 = -1, s1 = -1;
      auto cnt = share_vertex(verts0, verts1, s0, s1);
      if (cnt > 0) return;  // skip vertex touching case. TODO: prove this
    }

    std::array<Vec3d, 4> local0, local1;
    for (auto k : {0, 1, 2, 3}) local0[k] = tetV[tetT[t0][k]];
    for (auto k : {0, 1, 2, 3}) local1[k] = tetV[tetT[t1][k]];
    if (prism::predicates::tetrahedron_tetrahedron_overlap(local0, local1)) {
      offend_pairs.insert({f0, f1});
    }
   
  };
}