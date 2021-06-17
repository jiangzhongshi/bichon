#include "extraction.hpp"

#include <prism/geogram/AABB.hpp>
#include <igl/boundary_facets.h>
#include <igl/parallel_for.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/write_triangle_mesh.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/spdlog.h>

#include <prism/PrismCage.hpp>
#include <prism/cgal/polyhedron_self_intersect.hpp>
#include <prism/energy/smoother_pillar.hpp>
#include <prism/local_operations/local_mesh_edit.hpp>
#include <prism/local_operations/mesh_coloring.hpp>
#include <prism/local_operations/retain_triangle_adjacency.hpp>
#include <prism/local_operations/validity_checks.hpp>

#include "common.hpp"
namespace prism {

bool shell_shrinker(const RowMatd& mV, const RowMati& mF,
                    const prism::geogram::AABB& tree, RowMatd& curV,
                    std::vector<bool>& mask, std::vector<int>& f_to_split) {
  constexpr auto count_true = [](const std::vector<bool>& m) {
    int cnt = 0;
    for (auto i : m)
      if (i) cnt++;
    spdlog::info("mask:{}/{}", cnt, m.size());
  };
  std::vector<std::vector<int>> VF, VFi;
  std::vector<std::pair<int, int>> ff_pairs;
  igl::vertex_triangle_adjacency(mV.rows(), mF, VF, VFi);

  ff_pairs.clear();
  ff_pairs.emplace_back(-1, -1);
  int responsible = 0;
  while (ff_pairs.size() > 0) {
    ff_pairs.clear();

    spdlog::info("preparing shell intersect info...");
    count_true(mask);
    prism::cgal::tetrashell_self_intersect(mV, curV, mF, mask, ff_pairs);
    spdlog::info("offending pairs {}", ff_pairs.size());

    if (ff_pairs.size() == 0) break;
    std::set<int> responsible_vertices;
    std::set<int> responsible_faces;
    for (auto [f0, f1] : ff_pairs) {
      responsible_faces.insert(f0);
      responsible_faces.insert(f1);
      for (int j = 0; j < 3; j++) {
        responsible_vertices.insert(mF(f0, j));
        responsible_vertices.insert(mF(f1, j));
      }
    }
    spdlog::info("resp faces {} vert {}", responsible_faces.size(),
                 responsible_vertices.size());
    mask = std::vector<bool>(mF.rows(), false);
    for (auto v : responsible_vertices) {
      for (auto f : VF[v]) mask[f] = true;
    }
    count_true(mask);

    spdlog::debug("faces {}", (responsible_faces));
    spdlog::debug("vertices {}", (responsible_vertices));
    if (responsible == responsible_faces.size() + responsible_vertices.size()) {
      spdlog::warn("Same Responsible");
    }
    responsible = responsible_faces.size() + responsible_vertices.size();
    Eigen::VectorXi face_mover(mF.rows());
    face_mover.setZero();
    for (auto v : responsible_vertices) {
      if (v < tree.num_freeze) {
        for (auto i : VF[v]) face_mover[i]++;
        continue;  // avoiding changing it to ensure bitwise exact
      }
      Vec3d record = curV.row(v);
      curV.row(v) = mV.row(v) * 0.2 + curV.row(v) * 0.8;

      bool shrink_success = true;
      for (auto i : VF[v]) {
        if (tree.intersects_triangle(
                {curV.row(mF(i, 0)), curV.row(mF(i, 1)), curV.row(mF(i, 2))},
                mF(i, 0) < tree.num_freeze)) {
          curV.row(v) = record;  // restore
          shrink_success = false;
          spdlog::debug("{} fail to shrink colliding tri{}", v, i);
          break;
        }
      }
      if (shrink_success) {
        spdlog::debug("{} succeed", v);
        for (auto i : VF[v]) face_mover[i]++;
      }
    }

    bool progress_on_face = false;
    for (int i : responsible_faces) {
      if (face_mover[i] != 3) {
        f_to_split.push_back(i);
        progress_on_face = true;
      }
    }
    if (!f_to_split.empty()) return false;
  }

  return true;
}

bool shell_extraction(PrismCage& pc, bool base) {
  std::vector<bool> intersection_candidates(pc.F.size(), true);
  while (true) {
    RowMatd mV, mTop;
    RowMati mF;

    spdlog::info("mid {}", pc.mid.size());
    vec2eigen(pc.mid, mV);
    vec2eigen(pc.F, mF);
    spdlog::info("mF {}", pc.F.size());

    vec2eigen(base ? pc.base : pc.top, mTop);
    std::vector<int> f_to_split;
    

    if (shell_shrinker(mV, mF, *pc.ref.aabb, mTop, intersection_candidates,
                       f_to_split)) {
      eigen2vec(mTop, base ? pc.base : pc.top);
      return true;
    }
    eigen2vec(mTop, base ? pc.base : pc.top);
    // splits
    RowMati FF, FFi;
    igl::triangle_triangle_adjacency(mF, FF, FFi);
    Eigen::VectorXi colors = Eigen::VectorXi::Zero(mF.rows());
    // colors.setConstant(2);
    std::vector<std::vector<int>> VF, VFi;
    igl::vertex_triangle_adjacency(mV.rows(), mF, VF, VFi);
    for (auto f : f_to_split) {  // 2 red, 1 green, 0 white
      colors[f] = 2;
    }
    RowMati edge_vert(mF.rows(), 3);
    while (true) {
      prism::local::red_green_coloring(mF, FF, colors);
      // see if greens are splittable
      std::vector<int> nonsplittables;
      edge_vert.setConstant(-1);
      int count_greens = 0;
      for (int i = 0; i < mF.rows(); i++) {
        if (colors[i] != 1) continue;  // only deal with green here
        count_greens++;
        int e = -1;
        for (int j = 0; j < 3; j++) {
          if (FF(i, j) != -1 && colors[FF(i, j)] == 2) {
            e = j;
            break;
          }  // find red neighbor
        }
        assert(e != -1);
        // test volume after split (i,e)
        std::array<Vec3d, 3> newloc{
            (pc.base[mF(i, e)] + pc.base[mF(i, (e + 1) % 3)]) / 2,
            Vec3d(0, 0, 0),
            (pc.top[mF(i, e)] + pc.top[mF(i, (e + 1) % 3)]) / 2};
        auto mid = pc.ref.aabb->segment_query(newloc[0], newloc[2]);
        if (!mid) {
          spdlog::error("Intersection Failed,");
          exit(1);
        }
        newloc[1] = mid.value();
        pc.base.push_back(newloc[0]);
        pc.mid.push_back(newloc[1]);
        pc.top.push_back(newloc[2]);

        int ux = pc.mid.size() - 1;
        edge_vert(i, e) = ux;
        edge_vert(FF(i, e), FFi(i, e)) = ux;
        auto v0 = mF(i, e), v1 = mF(i, (e + 1) % 3), v2 = mF(i, (e + 2) % 3);
        std::vector<Vec3i> new_tris{Vec3i{v0, ux, v2}, Vec3i{v1, v2, ux}};
        for (int ii : {0, 1}) {
          auto [_t, face, _s] = tetra_split_AorB(new_tris[ii]);
          new_tris[ii] = (face);
        }
        pc.F.push_back(new_tris[0]);
        pc.F.push_back(new_tris[1]);

        pc.track_ref.push_back(pc.track_ref[i]);
        pc.track_ref.push_back(pc.track_ref[i]);
        auto vc = prism::local_validity::volume_check(pc.mid, pc.top, new_tris,
                                                      pc.ref.aabb->num_freeze);
        if (!vc) {
          nonsplittables.push_back(i);
          colors[i] = 2;
        }
      }
      spdlog::info("Spl size {}/{}", nonsplittables.size(), count_greens);
      if (nonsplittables.size() == 0)
        break;
      else {  // reset
        pc.base.resize(mV.rows());
        pc.mid.resize(mV.rows());
        pc.top.resize(mV.rows());
        pc.F.resize(mF.rows());
        pc.track_ref.resize(mF.rows());
      }
    }  // while
    for (int i = 0; i < colors.size(); i++) {
      if (colors[i] != 2) continue;
      for (int e = 0; e < 3; e++) {
        if (edge_vert(i, e) != -1) continue;  // assigned
        // assignment
        edge_vert(i, e) = pc.base.size();
        edge_vert(FF(i, e), FFi(i, e)) = pc.base.size();
        std::array<Vec3d, 3> newloc{
            (pc.base[mF(i, e)] + pc.base[mF(i, (e + 1) % 3)]) / 2,
            Vec3d(0, 0, 0),
            (pc.top[mF(i, e)] + pc.top[mF(i, (e + 1) % 3)]) / 2};
        newloc[1] = pc.ref.aabb->segment_query(newloc[0], newloc[2])
                        .value_or(Vec3d(0, 0, 0));
        pc.base.push_back(newloc[0]);
        pc.mid.push_back(newloc[1]);
        pc.top.push_back(newloc[2]);
      }
      auto v0 = mF(i, 0), v1 = mF(i, 1), v2 = mF(i, 2);
      auto e0 = edge_vert(i, 0), e1 = edge_vert(i, 1), e2 = edge_vert(i, 2);
      pc.F.emplace_back(Vec3i{v0, e0, e2});
      pc.F.emplace_back(Vec3i{v1, e1, e0});
      pc.F.emplace_back(Vec3i{v2, e2, e1});
      auto [type, face, shift] = tetra_split_AorB(Vec3i{e0, e1, e2});
      pc.F.emplace_back(face);
      for (int j = 0; j < 4; j++) pc.track_ref.push_back(pc.track_ref[i]);
    }  // insert red faces
    for (int i = 0; i < colors.size(); i++) {
      if (colors[i] > 0) {  // removed 
        pc.F[i] = pc.F.back();
        pc.F.pop_back();
        pc.track_ref[i] = pc.track_ref.back();
        pc.track_ref.pop_back();
        intersection_candidates[i] = true;
      }
    }

    intersection_candidates.resize(pc.F.size(), true); // mask for collision check

  }
}

}  // namespace prism