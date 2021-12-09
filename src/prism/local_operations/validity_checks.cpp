#include "validity_checks.hpp"

#include <geogram/numerics/predicates.h>
#include <igl/Timer.h>
#include <igl/euler_characteristic.h>
#include <igl/parallel_for.h>
#include <igl/remove_unreferenced.h>
#include <igl/volume.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <highfive/H5Easy.hpp>
#include <limits>
#include <prism/predicates/inside_octahedron.hpp>
#include <prism/predicates/triangle_triangle_intersection.hpp>
#include <queue>
#include <stdexcept>
#include <vector>

#include "../energy/map_distortion.hpp"
#include "../energy/prism_quality.hpp"
#include "../geogram/AABB.hpp"
#include "../predicates/inside_prism_tetra.hpp"
#include "../predicates/positive_prism_volume_12.hpp"
#include "../spatial-hash/AABB_hash.hpp"
#include "local_mesh_edit.hpp"
#include "prism/PrismCage.hpp"
#include "prism/common.hpp"
#include "prism/polyshell_utils.hpp"
#include "remesh_pass.hpp"

namespace prism::local_validity {

constexpr auto quality_on_tris = [](const auto &base, const auto &mid,
                                    const auto &top, const auto &moved_tris) {
  double quality = 0;

  for (auto [v0, v1, v2] : moved_tris) {
    auto q = prism::energy::triangle_quality({mid[v0], mid[v1], mid[v2]});
    quality += q;
  }
  return quality;
};

double max_quality_on_tris(const std::vector<Vec3d> &base,
                           const std::vector<Vec3d> &mid,
                           const std::vector<Vec3d> &top,
                           const std::vector<Vec3i> &moved_tris) {
  double quality = 0;

  for (auto [v0, v1, v2] : moved_tris) {
    // auto q = prism::energy::prism_full_quality(
    //              {base[v0], base[v1], base[v2], mid[v0], mid[v1], mid[v2]}) +
    //          prism::energy::prism_full_quality(
    //              {mid[v0], mid[v1], mid[v2], top[v0], top[v1], top[v2]});
    auto q = prism::energy::triangle_quality({mid[v0], mid[v1], mid[v2]});
    if (std::isnan(q))
      return std::numeric_limits<double>::infinity();
    quality = std::max(quality, q);
  }
  return quality;
}

bool dynamic_intersect_check(
    const std::vector<Vec3d> &base, const std::vector<Vec3i> &F,
    const std::vector<int>
        &vec_removed, // proposed removal face_id to be ignored in the test.
    const std::vector<Vec3i> &tris, // proposed addition triangles
    const prism::HashGrid &grid) {
  spdlog::trace("In DIC 2x{}", tris.size());
  std::set<int> removed(
      vec_removed.begin(),
      vec_removed.end()); // important, this has to be sorted or set for
                          // std::difference to work
  constexpr auto share_vertex_id = [](const auto &Fc, auto &f) {
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        if (Fc[i] == f[j])
          return true;
    return false; // TODO prove this complete, one vertex touch requires
                  // reduced collision check.
  };
  constexpr auto intersection = [](const auto &V, const auto &Fc, auto &f) {
    auto &[p0, p1, p2] = Fc;
    auto &[q0, q1, q2] = f;
    return prism::predicates::triangle_triangle_overlap({V[p0], V[p1], V[p2]},
                                                        {V[q0], V[q1], V[q2]});
  };
  igl::Timer timer;
  timer.start();
  for (int i = 0; i < tris.size(); i++) {
    for (int j = i + 1; j < tris.size(); j++) {
      if ((!share_vertex_id(tris[i], tris[j])) &&
          intersection(base, tris[i], tris[j]))
        return false;
    }
  }
  for (auto f : tris) {
    RowMat3d local;
    for (auto k : {0, 1, 2})
      local.row(k) = base[f[k]];
    std::set<int> candidates;
    grid.query(local.colwise().minCoeff(), local.colwise().maxCoeff(),
               candidates);
    std::vector<int> result;
    std::set_difference(candidates.begin(), candidates.end(), removed.begin(),
                        removed.end(), std::back_inserter(result));
    if (result.empty()) {
      spdlog::trace("f {}", f);
      spdlog::trace("cand {}", candidates);
      spdlog::trace("removed {}", removed);
      // when the check is always one ring, this should be an unreachable
      // branch. but when the check is large, then this is possible if the shell
      // of interest is an interior triangle of the patch. And in this case,
      // check intersection (quadratic/2) inside the patch.
      spdlog::trace(
          "Empty candidate for spatial hash: this should not happen unless the "
          "triangle is isolated from its neighbor.");
    }
    for (auto c : result) { // candidate faces to test.
      if ((!share_vertex_id(F[c], f)) && intersection(base, F[c], f))
        return false;
    }
  }
  auto elapsed = timer.getElapsedTimeInMicroSec();
  spdlog::trace("DIC true {}", elapsed);
  return true; // safe operation, no intersection
}

bool prism_positivity_with_numerical(const std::array<Vec3d, 6> &verts,
                                     const std::array<bool, 3> &constrained) {
  auto matV = Eigen::Map<const RowMatd>(verts[0].data(), 6, 3);
  for (int p = 0; p < 3; p++) {
    if (constrained[p])
      continue;
    Eigen::VectorXd vol_index;
    igl::volume(matV,
                Eigen::Map<const RowMati>(TWELVE_TETRAS[4 * p].data(), 4, 4),
                vol_index);
    if (vol_index.minCoeff() <= 0) {
      spdlog::trace(vol_index.format(Eigen::IOFormat(Eigen::FullPrecision)));
      return false;
    }
  }
  return prism::predicates::positive_prism_volume(verts, constrained);
};

bool volume_check(const std::vector<Vec3d> &base, const std::vector<Vec3d> &mid,
                  const std::vector<Vec3d> &top, const std::vector<Vec3i> &tris,
                  int num_cons) {
  //
  spdlog::trace("In VC");
  igl::Timer timer;
  timer.start();
  auto checker = [](const std::array<Vec3d, 6> &a,
                    const std::array<bool, 3> &c) {
    return prism::predicates::positive_prism_volume(a, c);
    // return prism::predicates::positive_nonlinear_prism(a,c);
  };
  for (auto [v0, v1, v2] : tris) {
    // auto checker = [tc = (v1 > v2 ? TETRA_SPLIT_A : TETRA_SPLIT_B)](
    //                    const std::array<Vec3d, 6> &verts,
    //                    const std::array<bool, 3> &constrained) {
    //   for (auto &tet : tc) {
    //     if (GEO::PCK::orient_3d(verts[tet[0]].data(), verts[tet[1]].data(),
    //                             verts[tet[2]].data(),
    //                             verts[tet[3]].data()) <= 0) {
    //       // spdlog::trace("t {}", tet);
    //       return false;
    //     }
    //   }
    //   return true;
    // };
    std::array<bool, 3> cons_flag{v0 < num_cons, v1 < num_cons, v2 < num_cons};
    spdlog::trace("VC: BM \n{} {} {} {} {} {}", 
    base[v0], base[v1], base[v2],
    mid[v0], mid[v1], mid[v2]);

    if (!checker({base[v0], base[v1], base[v2], mid[v0], mid[v1], mid[v2]},
                 cons_flag))
      return false;
    spdlog::trace("VC: MT");
    if (!checker({mid[v0], mid[v1], mid[v2], top[v0], top[v1], top[v2]},
                 cons_flag))
      return false;
  }
  auto elapsed = timer.getElapsedTimeInMicroSec();
  spdlog::trace("VC true {}", elapsed);
  return true;
}

bool volume_check(const std::vector<Vec3d> &base, const std::vector<Vec3d> &top,
                  const std::vector<Vec3i> &tris, int num_cons) {
  //
  spdlog::trace("In VC");
  igl::Timer timer;
  timer.start();
  for (auto [v0, v1, v2] : tris) {
    std::array<bool, 3> cons_flag{v0 < num_cons, v1 < num_cons, v2 < num_cons};
    if (!prism::predicates::positive_prism_volume(
            {base[v0], base[v1], base[v2], top[v0], top[v1], top[v2]},
            cons_flag))
      return false;
  }
  auto elapsed = timer.getElapsedTimeInMicroSec();
  spdlog::trace("VC true {}", elapsed);
  return true;
}

bool intersect_check(const std::vector<Vec3d> &base,
                     const std::vector<Vec3d> &top,
                     const std::vector<Vec3i> &tris,
                     const prism::geogram::AABB &tree) {
  spdlog::trace("In IC 2x{}", tris.size());
  igl::Timer timer;
  timer.start();
  for (auto [v0, v1, v2] : tris) {
    if (tree.intersects_triangle({base[v0], base[v1], base[v2]},
                                 v0 < tree.num_freeze)) {
      spdlog::trace("base {} {} {}", v0, v1, v2);
      return false;
    }
    if (tree.intersects_triangle({top[v0], top[v1], top[v2]},
                                 v0 < tree.num_freeze)) {
      spdlog::trace("top {} {} {}", v0, v1, v2);
      return false;
    }
  }
  auto elapsed = timer.getElapsedTimeInMicroSec();
  spdlog::trace("IC true {}", elapsed);
  return true;
}

// this is a distort check, without nonlinear business, just plain old three
// tetra.
std::optional<std::vector<std::set<int>>> distort_check_trip(
    const std::vector<Vec3d> &base,
    const std::vector<Vec3d> &mid, // placed new verts
    const std::vector<Vec3d> &top, const std::vector<Vec3i> &tris,
    const std::set<int> &combined_trackee, // indices to ref.F tracked
    const RowMatd &refV, const RowMati &refF, double distortion_bound,
    int num_freeze, bool bundled_intersection) {
  // NormalCheck
  spdlog::trace("In NC ct#{}, tris{}", combined_trackee.size(), tris.size());
  igl::Timer timer;
  timer.start();
  assert(base.size() == top.size());
  assert(base.size() == mid.size());
  std::vector<std::set<int>> distributed_refs(tris.size());
  for (int i = 0; i < tris.size(); i++) {
    auto [v0, v1, v2] = tris[i];
    std::array<Vec3d, 3> base_vert{base[v0], base[v1], base[v2]};
    std::array<Vec3d, 3> mid_vert{mid[v0], mid[v1], mid[v2]};
    std::array<Vec3d, 3> top_vert{top[v0], top[v1], top[v2]};
    spdlog::trace("checking tris{}: {}-{}-{}", i, v0, v1, v2);
    auto oct_type_bot = (v1 > v2);
    auto oct_type_top = (v1 > v2);
    for (auto t : combined_trackee) { // for every tracked original triangle.
      std::array<Vec3d, 3> ref_tri = {
          refV.row(refF(t, 0)), refV.row(refF(t, 1)), refV.row(refF(t, 2))};
      bool intersected_prism = false;
      auto ri = [&mid_vert, &ref_tri]() {
        for (int ri = 0; ri < 3; ri++)
          for (int si = 0; si < 3; si++)
            if (mid_vert[si] == ref_tri[ri])
              return ri;
        return -1;
      }();
      if (ri != -1) {
        spdlog::trace("ri {}", ri);
        std::swap(ref_tri[0], ref_tri[ri]);
        intersected_prism = prism::pointless_triangle_intersect_tripletetra(
                                base_vert, mid_vert, oct_type_bot, ref_tri) ||
                            prism::pointless_triangle_intersect_tripletetra(
                                mid_vert, top_vert, oct_type_top, ref_tri);
        std::swap(ref_tri[0], ref_tri[ri]);
      } else
        intersected_prism =
            prism::triangle_intersect_tripletetra(
                base_vert, mid_vert, oct_type_bot, ref_tri, num_freeze > v0) ||
            prism::triangle_intersect_tripletetra(
                mid_vert, top_vert, oct_type_top, ref_tri, num_freeze > v0);
      if (!intersected_prism) {
        continue;
      }

      for (int tc = (v0 < num_freeze) ? 1 : 0; tc < 3; tc++) {
        auto pillar = top_vert[tc] - base_vert[tc];
        auto distortion = prism::energy::map_max_cos_angle(pillar, ref_tri);
        if (distortion < distortion_bound) {
          spdlog::trace("ref {} tris [{},{},{}], tc{}, distortion: {}", t, v0,
                        v1, v2, tc, distortion);
          Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, ", ", ";\n", "[",
                                   "]", "", "");
          spdlog::trace("ref pos [{}, {}, {}]", ref_tri[0].format(HeavyFmt),
                        ref_tri[1].format(HeavyFmt),
                        ref_tri[2].format(HeavyFmt));
          return {};
        }
      }
      distributed_refs[i].insert(t);
      // Test intersect between tri(t) and top/base(i)
      auto bundled_check = [&, sing = (v0 < num_freeze)]() {
        if (!bundled_intersection)
          return true;
        // this is enabled when global (AABB tree)
        // intersection is off.
        if (sing && (base_vert[0] == ref_tri[0])) {
          if (prism::predicates::segment_triangle_overlap(
                  {base_vert[1], base_vert[2]}, ref_tri))
            return false;
          if (prism::predicates::segment_triangle_overlap(
                  {top_vert[1], top_vert[2]}, ref_tri))
            return false;
          if (prism::predicates::segment_triangle_overlap(
                  {ref_tri[1], ref_tri[2]}, base_vert))
            return false;
          if (prism::predicates::segment_triangle_overlap(
                  {ref_tri[1], ref_tri[2]}, top_vert))
            return false;
        } else {
          if ((prism::predicates::triangle_triangle_overlap(base_vert,
                                                            ref_tri) ||
               prism::predicates::triangle_triangle_overlap(top_vert, ref_tri)))
            return false;
        }
        return true;
      }();
      if (!bundled_check) {
        spdlog::trace("bundled overlap detected");
        Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, ", ", ";\n", "[", "]",
                                 "", "");
        spdlog::trace("ref {} pos [{}, {}, {}]", t, ref_tri[0].format(HeavyFmt),
                      ref_tri[1].format(HeavyFmt), ref_tri[2].format(HeavyFmt));
        spdlog::trace("base pos [{}, {}, {}]", base_vert[0].format(HeavyFmt),
                      base_vert[1].format(HeavyFmt),
                      base_vert[2].format(HeavyFmt));
        spdlog::trace("top pos [{}, {}, {}]", top_vert[0].format(HeavyFmt),
                      top_vert[1].format(HeavyFmt),
                      top_vert[2].format(HeavyFmt));
        return {};
      }
    }
  }
  auto elapsed = timer.getElapsedTimeInMicroSec();
  // for (auto &s : distributed_refs) {
  //   if (s.size() == 0) {
  //     spdlog::trace("empty?");
  //     continue;
  //     // return {};
  //   }
  //   auto sliceF = RowMati(s.size(), 3);
  //   int i = 0;
  //   for (auto &a : s) sliceF.row(i++) = refF.row(a);
  //   auto I = Eigen::VectorXi(), J = Eigen::VectorXi();
  //   igl::remove_unreferenced(refV.rows(), sliceF, I, J);
  //   std::for_each(sliceF.data(), sliceF.data() + sliceF.size(),
  //                 [&I](auto &a) { a = I(a); });

  //   auto ec = igl::euler_characteristic(sliceF);
  //   if (ec != 1) {
  //     spdlog::trace("euler?");
  //     return {};
  //   }
  // }
  spdlog::trace("NC true {}", elapsed);
  // spdlog::trace("dref {}", distributed_refs);
  return distributed_refs;
}

std::optional<std::vector<std::set<int>>>
distort_check(const std::vector<Vec3d> &base,
              const std::vector<Vec3d> &mid, // placed new verts
              const std::vector<Vec3d> &top, const std::vector<Vec3i> &tris,
              const std::set<int> &combined_trackee, // indices to ref.F tracked
              const RowMatd &refV, const RowMati &refF, double distortion_bound,
              int num_freeze, bool bundled_intersection) {
  // ANCHOR: 80% bottleneck for no-curve pipeline.
  // reduce the predicates would go a long way
  // Possible improvements: 1. check walls, instead of cells and fill in
  // topologically
  // 2. Parallel
  // NormalCheck
  spdlog::trace("In NC ct#{}, tris{}", combined_trackee.size(), tris.size());
  igl::Timer timer;
  timer.start();
  assert(base.size() == top.size());
  assert(base.size() == mid.size());
  std::vector<std::set<int>> distributed_refs(tris.size());
  for (int i = 0; i < tris.size(); i++) { // over all the prism cells.
    auto [v0, v1, v2] = tris[i];
    std::array<Vec3d, 3> base_vert{base[v0], base[v1], base[v2]};
    std::array<Vec3d, 3> mid_vert{mid[v0], mid[v1], mid[v2]};
    std::array<Vec3d, 3> top_vert{top[v0], top[v1], top[v2]};
    std::array<bool, 3> oct_type_bot, oct_type_top;
    prism::determine_convex_octahedron(base_vert, mid_vert, oct_type_bot,
                                       num_freeze > v0);
    if (prism::octa_convexity(base_vert, mid_vert, oct_type_bot) == false) {
      spdlog::trace("non convex bot octahedron");
      // return {};
    }
    prism::determine_convex_octahedron(mid_vert, top_vert, oct_type_top,
                                       num_freeze > v0);
    if (prism::octa_convexity(mid_vert, top_vert, oct_type_top) == false) {
      spdlog::trace("non convex top octahedron");
      // return {};
    }

    spdlog::trace("checking tris{}: {}-{}-{}", i, v0, v1, v2);
    for (auto t : combined_trackee) { // for every tracked original triangle.
      std::array<Vec3d, 3> ref_tri = {
          refV.row(refF(t, 0)), refV.row(refF(t, 1)), refV.row(refF(t, 2))};
      bool intersected_prism = false;
      auto ri = [&mid_vert, &ref_tri]() {
        for (int ri = 0; ri < 3; ri++)
          for (int si = 0; si < 3; si++)
            if (mid_vert[si] == ref_tri[ri])
              return ri;
        return -1;
      }();
      if (ri != -1) {
        spdlog::trace("ri {}", ri);
        std::swap(ref_tri[0], ref_tri[ri]);
        intersected_prism = prism::pointless_triangle_intersect_octahedron(
                                base_vert, mid_vert, oct_type_bot, ref_tri) ||
                            prism::pointless_triangle_intersect_octahedron(
                                mid_vert, top_vert, oct_type_top, ref_tri);
        std::swap(ref_tri[0], ref_tri[ri]);
      } else
        intersected_prism =
            prism::triangle_intersect_octahedron(
                base_vert, mid_vert, oct_type_bot, ref_tri, num_freeze > v0) ||
            prism::triangle_intersect_octahedron(
                mid_vert, top_vert, oct_type_top, ref_tri, num_freeze > v0);
      if (!intersected_prism) {
        continue;
      }

      for (int tc = (v0 < num_freeze) ? 1 : 0; tc < 3; tc++) {
        auto pillar = top_vert[tc] - base_vert[tc];
        auto distortion = prism::energy::map_max_cos_angle(pillar, ref_tri);
        if (distortion < distortion_bound) {
          spdlog::trace("ref {} tris [{},{},{}], tc{}, distortion: {}", t, v0,
                        v1, v2, tc, distortion);
          Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, ", ", ";\n", "[",
                                   "]", "", "");
          spdlog::trace("ref pos [{}, {}, {}]", ref_tri[0].format(HeavyFmt),
                        ref_tri[1].format(HeavyFmt),
                        ref_tri[2].format(HeavyFmt));
          return {};
        }
      }
      distributed_refs[i].insert(t);
      // Test intersect between tri(t) and top/base(i)
      auto bundled_check = [&, sing = (v0 < num_freeze)]() {
        if (!bundled_intersection)
          return true;
        // this is enabled when global (AABB tree)
        // intersection is off.
        if (sing && (base_vert[0] == ref_tri[0])) {
          if (prism::predicates::segment_triangle_overlap(
                  {base_vert[1], base_vert[2]}, ref_tri))
            return false;
          if (prism::predicates::segment_triangle_overlap(
                  {top_vert[1], top_vert[2]}, ref_tri))
            return false;
          if (prism::predicates::segment_triangle_overlap(
                  {ref_tri[1], ref_tri[2]}, base_vert))
            return false;
          if (prism::predicates::segment_triangle_overlap(
                  {ref_tri[1], ref_tri[2]}, top_vert))
            return false;
        } else {
          if ((prism::predicates::triangle_triangle_overlap(base_vert,
                                                            ref_tri) ||
               prism::predicates::triangle_triangle_overlap(top_vert, ref_tri)))
            return false;
        }
        return true;
      }();
      if (!bundled_check) {
        spdlog::trace("bundled overlap detected");
        Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, ", ", ";\n", "[", "]",
                                 "", "");
        spdlog::trace("ref pos [{}, {}, {}]", ref_tri[0].format(HeavyFmt),
                      ref_tri[1].format(HeavyFmt), ref_tri[2].format(HeavyFmt));
        spdlog::trace("base pos [{}, {}, {}]", base_vert[0].format(HeavyFmt),
                      base_vert[1].format(HeavyFmt),
                      base_vert[2].format(HeavyFmt));
        spdlog::trace("top pos [{}, {}, {}]", top_vert[0].format(HeavyFmt),
                      top_vert[1].format(HeavyFmt),
                      top_vert[2].format(HeavyFmt));
        return {};
      }
    }
  }
  auto elapsed = timer.getElapsedTimeInMicroSec();
  // for (auto &s : distributed_refs) {
  //   if (s.size() == 0) {
  //     spdlog::trace("empty?");
  //     // return {};
  //   }
  //   auto sliceF = RowMati(s.size(), 3);
  //   int i = 0;
  //   for (auto &a : s)
  //     sliceF.row(i++) = refF.row(a);
  //   auto I = Eigen::VectorXi(), J = Eigen::VectorXi();
  //   igl::remove_unreferenced(refV.rows(), sliceF, I, J);
  //   std::for_each(sliceF.data(), sliceF.data() + sliceF.size(),
  //                 [&I](auto &a) { a = I(a); });

  //   auto ec = igl::euler_characteristic(sliceF);
  //   if (ec != 1) {
  //     spdlog::trace("euler?");
  //     return {};
  //   }
  // }
  spdlog::trace("NC true {}", elapsed);
  return distributed_refs;
}

/// @brief find set of triangles to reject adjacent to the feature
template <typename T>
auto find_rejection_trackee(const RowMati &F,
                            const std::vector<std::vector<int>> &VF,
                            const std::vector<std::vector<int>> &VFi,
                            const std::vector<int> &seg,
                            T it0,
                            T it1) -> std::set<int> {
  it1--; // end to back
  assert(seg.size() >= 2);
  spdlog::trace("segs {}", seg);
  if (seg.size() == 2) {
    auto v0 = *it0, v1 = *it1;
    spdlog::trace("Single edge in the segment {}-{}", v0, v1);
    auto &nb = VF[v0];
    auto &nbi = VFi[v0];
    for (auto i = 0; i < nb.size(); i++) {
      auto f0 = nb[i];
      auto e0 = nbi[i];
      assert(F(f0, e0) == v0 && "VFi should satisfy.");
      if (F(f0, (e0 + 2) % 3) == v1) { // f0 = (v1,v0,vx), on the right of (v0,v1)
        return {f0};
      }
    }
    assert(false && "Not found the triangle opposite to feature.");
    return {};
  }

  std::set<int> reject_faces;
  it0++;
  for (; it0 != it1; it0++) {
    auto v0 = *it0;
    auto vp = *std::prev(it0);
    auto vn = *std::next(it0);
    auto &nb = VF[v0];
    auto &nbi = VFi[v0];
    auto start = -1, end = -1;
    for (auto i = 0; i < nb.size(); i++) {
      auto f0 = nb[i];
      auto e0 = nbi[i];
      if (F(f0, (e0 + 1) % 3) == vp)
        start = i;
      if (F(f0, (e0 + 2) % 3) == vn)
        end = i;
    }
    if (end < start)
      end += nb.size();
    for (auto i = start; i <= end; i++) {
      reject_faces.insert(nb[i % nb.size()]);
    }
  }
  return reject_faces;
};

template std::set<int> find_rejection_trackee(
  const RowMati &,
  const std::vector<std::vector<int>> &,
  const std::vector<std::vector<int>> &,
  const std::vector<int> &,
  std::vector<int>::iterator, std::vector<int>::iterator
);

bool feature_handled_distort_check(const PrismCage &pc,
                                   const prism::local::RemeshOptions &option,
                                   const std::vector<Vec3i> &moved_pris,
                                   const std::vector<int> &old_fid,
                                   std::vector<std::set<int>> &sub_trackee) {
  auto &base = pc.base, &top = pc.top, &mid = pc.mid;
  auto &F = pc.F;
  auto &refV = pc.ref.V;
  auto &refF = pc.ref.F;
  auto num_freeze = pc.ref.aabb->num_freeze;
  std::set<int> combined_tracks;
  for (auto f : old_fid)
    set_add_to(pc.track_ref[f], combined_tracks);
  sub_trackee.resize(moved_pris.size());

  for (auto i = 0; i < moved_pris.size(); i++) {
    auto &f = moved_pris[i];
    auto remain_track = combined_tracks;
    for (auto j = 0; j < 3; j++) {
      auto v0 = f[j], v1 = f[(j + 1) % 3];
      auto it0 = pc.meta_edges.find({v0, v1});
      auto it1 = pc.meta_edges.find({v1, v0});

      if (it0 == it1)
        continue;
      bool left = false;
      if (it0 != pc.meta_edges.end()) { // left
        left = true;
      } else { // right
        it0 = it1;
      }
      auto &seg = it0->second.second;
      auto reject = std::set<int>();
      if (left)
        reject = find_rejection_trackee(pc.ref.F, pc.ref.VF, pc.ref.VFi, seg,
                                        seg.begin(), seg.end());
      else
        reject = find_rejection_trackee(pc.ref.F, pc.ref.VF, pc.ref.VFi, seg,
                                        seg.rbegin(), seg.rend());

      auto minus_track = std::set<int>();
      set_minus(remain_track, reject, minus_track);
      remain_track = std::move(minus_track);
    }
    auto sub_ref = distort_check(base, mid, top, {f}, remain_track, refV, refF,
                                 option.distortion_bound, num_freeze,
                                 option.dynamic_hashgrid);
    if (sub_ref && sub_ref.value()[0].size() > 0) {
      sub_trackee[i] = sub_ref.value()[0];
    } else {
      return false;
    }
  }
  for (auto &s : sub_trackee)
    assert(s.size() > 0);
  auto total_dist = std::set<int>();
  for (auto &s : sub_trackee)
    set_add_to(s, total_dist);
  if (total_dist.size() != combined_tracks.size()) {
    return false;
    // TODO: this may or may not interfer with the initialization.
    // spdlog::dump_backtrace();
    // auto file = H5Easy::File("debug.h5", H5Easy::File::Overwrite);
    // pc.serialize("debug_pc.h5");
    // throw std::runtime_error("debug: distribution incomplete.");
  }
  return true;
};

} // namespace prism::local_validity

void prism::local_validity::post_operation(
    PrismCage &pc, const prism::local::RemeshOptions &option,
    const std::vector<int> &old_fids, const std::vector<int> &new_fids,
    const std::vector<std::set<int>> &new_tracks,
    std::vector<RowMatd> &local_cp) {
  if (option.curve_checker.second.has_value())
    (std::any_cast<
        std::function<void(const std::vector<int> &, const std::vector<int> &,
                           const std::vector<RowMatd> &)>>(
        option.curve_checker.second))(old_fids, new_fids, local_cp);
  if (pc.top_grid != nullptr) {
    spdlog::trace("HashGrid remove");
    for (auto f : old_fids) {
      pc.top_grid->remove_element(f);
      pc.base_grid->remove_element(f);
    }
    pc.top_grid->insert_triangles(pc.top, pc.F, new_fids);
    pc.base_grid->insert_triangles(pc.base, pc.F, new_fids);
  }

  pc.track_ref.resize(pc.F.size());
  for (int i = 0; i < new_tracks.size(); i++) {
    pc.track_ref[new_fids[i]] = new_tracks[i];
  }
}

prism::local_validity::PolyOpError prism::local_validity::attempt_zig_remesh(
    const PrismCage &pc, const std::vector<std::set<int>> &map_track,
    const prism::local::RemeshOptions &option,
    // specified infos below
    double old_quality, const std::vector<int> &old_fid,
    const std::vector<Vec3i> &moved_pris,
    std::vector<std::set<int>> &sub_trackee, std::vector<RowMatd> &local_cp) {
  using Err = PolyOpError;
  constexpr auto lets_comb_the_pillars = true;
  auto &base = pc.base, &top = pc.top, &mid = pc.mid;
  auto &F = pc.F;
  auto &refV = pc.ref.V;
  auto &refF = pc.ref.F;
  auto num_freeze = pc.ref.aabb->num_freeze;
  auto castmatd = [](auto &vec) {
    RowMatd mat;
    vec2eigen(vec, mat);
    return mat;
  };

  std::vector<Vec3i> old_tris;
  for (auto f : old_fid)
    old_tris.push_back(F[f]);

  spdlog::trace("old_pris {}", old_tris);
  auto quality_before = (old_quality >= 0)
                            ? old_quality
                            : max_quality_on_tris(base, mid, top, old_tris);
  auto quality_after = max_quality_on_tris(base, mid, top, moved_pris);
  spdlog::trace("Quality compare {} -> {}", quality_before, quality_after);
  if (std::isnan(quality_after) || !std::isfinite(quality_after))
    return Err::kQuality;
  if ((quality_after > option.relax_quality_threshold) &&
      quality_after > quality_before) // if the quality is too large, not allow
                                      // it to increase.
    return Err::kQuality;
  //  volume check
  if (!volume_check(base, mid, top, moved_pris, num_freeze)) {
    return Err::kVolume;
  }
  auto ic =
      dynamic_intersect_check(pc.base, pc.F, old_fid, moved_pris,
                              *pc.base_grid) &&
      dynamic_intersect_check(pc.top, pc.F, old_fid, moved_pris, *pc.top_grid);
  if (!ic)
    return Err::kIntersect;

  std::set<int> combined_tracks;
  for (auto f : old_fid)
    set_add_to(pc.track_ref[f], combined_tracks);
  sub_trackee.clear();
  sub_trackee.resize(moved_pris.size());

  for (auto i = 0; i < moved_pris.size(); i++) {
    auto &f = moved_pris[i];

    auto [oppo_vid, rej_id, segs] =
        prism::local_validity::identify_zig(pc.meta_edges, f);
    if (rej_id == -10)
      return Err::kTwoFeature;
    if (rej_id >= 0) {
      // rej_id = 2*chain_id +0/1
      auto remain_track = std::set<int>();
      spdlog::trace("oppo {} segs {}", oppo_vid, segs.size());
      auto [v0, v1, v2] =
          std::tie(f[oppo_vid], f[(oppo_vid + 1) % 3], f[(oppo_vid + 2) % 3]);
      // there is no left/right here, since identify_zig already handled it.
      auto reject = find_rejection_trackee(pc.ref.F, pc.ref.VF, pc.ref.VFi, segs, 
                                        segs.begin(), segs.end());

      set_minus(combined_tracks, reject, remain_track);
      spdlog::trace("rej_id {}: combined {} - reject {}", rej_id, combined_tracks, reject);

      auto [local_base, local_mid, local_top, zig_tris, _] =
          zig_constructor(pc, v0, v1, v2, segs, lets_comb_the_pillars);
      auto local_freeze = 0;
      if (f[0] < num_freeze)
        local_freeze = 1;

      if (!volume_check(local_base, local_mid, local_top, zig_tris,
                        local_freeze)) {
        return Err::kSubVolume;
      }

      // the distort check is not well bundled here.
      auto intersect_free_check = [&]() {
        for (auto rt : remain_track) {
          auto tri = std::array<Vec3d, 3>{pc.ref.V.row(pc.ref.F(rt, 0)),
                                          pc.ref.V.row(pc.ref.F(rt, 1)),
                                          pc.ref.V.row(pc.ref.F(rt, 2))};
          if (prism::predicates::triangle_triangle_overlap(
                  {pc.top[f[0]], pc.top[f[1]], pc.top[f[2]]}, tri))
            return false;
          if (prism::predicates::triangle_triangle_overlap(
                  {pc.base[f[0]], pc.base[f[1]], pc.base[f[2]]}, tri))
            return false;
        }
        return true;
      };
      if (!intersect_free_check())
        return Err::kSubIntersect;
      auto dc = distort_check(local_base, local_mid, local_top, zig_tris,
                              remain_track, pc.ref.V, pc.ref.F,
                              option.distortion_bound, local_freeze, false);
      if (!dc)
        return Err::kDistort;
      for (auto &v : dc.value())
        set_add_to(v, sub_trackee[i]);
    } else { // regular edge, without poly.
      auto dc =
          distort_check(pc.base, pc.mid, pc.top, {f}, combined_tracks, pc.ref.V,
                        pc.ref.F, option.distortion_bound, num_freeze, true);
      if (!dc)
        return Err::kDistort;
      sub_trackee[i] = dc.value()[0];
    }
  }

  std::set<int> total_sub_trackee;
  for (auto &s : sub_trackee)
    set_add_to(s, total_sub_trackee);
  spdlog::trace("total distributed {} == prev {}", total_sub_trackee.size(),
                combined_tracks.size());
  // spdlog::trace("total {}, from {}", total_sub_trackee, combined_tracks);
  assert(total_sub_trackee.size() == combined_tracks.size());
  if (option.curve_checker.first.has_value() &&
      !(std::any_cast<std::function<bool(
            const PrismCage &, const std::vector<int> &,
            const std::vector<Vec3i> &, std::vector<RowMatd> &)>>(
          option.curve_checker.first))(pc, old_fid, moved_pris, local_cp)) {
    return Err::kCurve;
  }

  if (lets_comb_the_pillars) {
    // throw std::runtime_error("post assignment since the result of combing has
    // passed.");
  }
  return Err::kSuccess;
}