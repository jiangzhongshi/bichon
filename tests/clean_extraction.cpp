#include <prism/geogram/AABB.hpp>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/write_triangle_mesh.h>
#include <spdlog/spdlog.h>
#include <prism/PrismCage.hpp>
#include <prism/cgal/polyhedron_self_intersect.hpp>
#include <prism/local_operations/local_mesh_edit.hpp>
#include <prism/local_operations/retain_triangle_adjacency.hpp>
#include <prism/local_operations/validity_checks.hpp>
#include <doctest.h>
#include "test_common.hpp"

namespace prism::cgal {
bool polyhedron_self_intersect_edges(
    const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
    std::vector<std::pair<int, int>>& fe_pairs);
bool polyhedron_self_intersect(const Eigen::MatrixXd& V,
                               const Eigen::MatrixXi& F,
                               std::vector<std::pair<int, int>>& pairs);
}  // namespace prism::cgal
TEST_CASE("collision free extract") {
  PrismCage pc("/home/zhongshi/data/raw10k_ser/1097847.stl.h5");
  spdlog::set_level(spdlog::level::info);
  Eigen::MatrixXd mV;
  Eigen::MatrixXi mF;

  while (true) {
    vec2eigen(pc.mid, mV);
    vec2eigen(pc.F, mF);
    spdlog::info("Start");
    std::vector<std::pair<int, int>> fe_pair;
    prism::cgal::polyhedron_self_intersect_edges(mV, mF, fe_pair);
    spdlog::info("FE Pair {}", fe_pair.size());
    if (fe_pair.size() == 0) break;
    auto& V = pc.mid;
    auto& F = pc.F;
    auto [FF, FFi] = prism::local::triangle_triangle_adjacency(F);
    std::vector<std::tuple<int, int, int, int>> fevv;
    fevv.reserve(fe_pair.size());
    for (auto [f, e] : fe_pair) {
      fevv.emplace_back(f, e, F[f][e], F[f][(e + 1) % 3]);
    }

    for (auto [f, e, v0, v1] : fevv) {
      spdlog::info("fevv {} {} {} {}", f, e, v0, v1);

      auto u0 = F[f][e], u1 = F[f][(e + 1) % 3];
      if (auto u0_ = F[f][e], u1_ = F[f][(e + 1) % 3];
          u0_ == u1_ || u0_ != u0 ||
          u1_ != u1)  // vid changed, means the edge is outdated.
        continue;

      auto f1 = FF[f][e], e1 = FFi[f][e];
      if (f1 == -1) continue;  // boundary check
      auto f0 = f, e0 = e;
      v0 = F[f0][(e0 + 2) % 3];
      v1 = F[f1][(e1 + 2) % 3];
      std::array<Vec3d, 3> newlocation{(pc.base[u0] + pc.base[u1]) / 2,
                                       Vec3d(0, 0, 0),
                                       (pc.top[u0] + pc.top[u1]) / 2};
      auto new_mid = pc.ref.aabb->segment_query(newlocation[0], newlocation[2]);
      if (!new_mid) {
        spdlog::trace("split mid failed");
        return;
      }
      newlocation[1] = new_mid.value();

      std::tuple<std::vector<int> /*fid*/, std::vector<int>, /*shift*/
                 std::vector<std::set<int>>               /*track*/
                 >
          checker;

      auto alpha = 1.;
      auto flag = 1;
      while (flag == 1) {  // vc problem
        auto new_b = newlocation[0] * (alpha) + (1 - alpha) * newlocation[1];
        auto new_t = newlocation[2] * (alpha) + (1 - alpha) * newlocation[1];
        flag = prism::local_validity::attempt_split(
            pc.base, V, pc.top, F, *pc.ref.aabb, pc.ref.V, pc.ref.F,
            pc.track_ref, 1e10, false, f0, f1, e0, e1,
            {new_b, newlocation[1], new_t}, checker);
        alpha *= 0.8;
        if (alpha < 1e-8) break;
      }

      if (flag != 0) {
        spdlog::error("Split Attempt Failed {}-{} {}-{}, flag = {}, alpha = {}",
                      f0, e0, f1, e1, flag, alpha);
        exit(1);
        continue;
      }
      auto& [new_fid, new_shifts, new_tracks] = checker;
      prism::edge_split(V.size() - 1, F, FF, FFi, f0, e0);

      assert(new_fid.size() == new_tracks.size());
      pc.track_ref.resize(F.size());
      for (int i = 0; i < new_tracks.size(); i++) {
        pc.track_ref[new_fid[i]] = new_tracks[i];
      }
    }

    vec2eigen(V, mV);
    vec2eigen(F, mF);
  }
}

namespace prism::cgal {
bool tetrashell_self_intersect(const Eigen::MatrixXd& base,
                               const Eigen::MatrixXd& top,
                               const Eigen::MatrixXi& F,
                               const std::vector<bool>& mask,
                               std::vector<std::pair<int, int>>& pairs);
}
TEST_CASE("collision free shell") {
  PrismCage pc("../buildr/36086.stl.h5");

  spdlog::set_level(spdlog::level::info);
  std::vector<std::pair<int, int>> ff_pairs;
  RowMatd mV, mTop;
  RowMati mF;
  spdlog::info("mid {}", pc.mid.size());
  vec2eigen(pc.mid, mV);
  vec2eigen(pc.base, mTop);
  vec2eigen(pc.F, mF);
  spdlog::info("mF {} {}", pc.F.size());
  std::vector<double> steps(pc.mid.size(), 1);
  RowMatd curV = mTop;

  ff_pairs.emplace_back(-1, -1);
  while (ff_pairs.size() > 0) {
    ff_pairs.clear();
    for (int v = 0; v < steps.size(); v++) {
      curV.row(v) = mV.row(v) * (1 - steps[v]) + steps[v] * mTop.row(v);
    }
    std::vector<bool> mask(mF.rows(), true);
    prism::cgal::tetrashell_self_intersect(mV, curV, mF, mask, ff_pairs);
    spdlog::info("shell {}", ff_pairs.size());
    std::set<int> responsible_vertices;
    for (auto [f0, f1] : ff_pairs) {
      for (int j = 0; j < 3; j++) {
        responsible_vertices.insert(mF(f0, j));
        responsible_vertices.insert(mF(f1, j));
      }
    }
    spdlog::info("responsible {}", (responsible_vertices));
    for (auto v : responsible_vertices) {
      double alpha = steps[v] * 0.8;
      curV.row(v) = mV.row(v) * (1 - alpha) + alpha * mTop.row(v);
    }
    bool intersect_input = false;
    for (int i = 0; i < mF.rows(); i++) {
      if (pc.ref.aabb->intersects_triangle(
              {curV.row(mF(i, 0)), curV.row(mF(i, 1)), curV.row(mF(i, 2))}))
        intersect_input = true;
      break;
    }
    if (!intersect_input)
      for (auto v : responsible_vertices) steps[v] *= 0.8;
    else
      spdlog::error("Hit: Split!");
  }
}