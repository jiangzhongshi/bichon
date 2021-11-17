#include "cage_check.hpp"

#include <igl/facet_components.h>
#include <igl/triangle_triangle_adjacency.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/spdlog.h>
#include <optional>
#include <highfive/H5Easy.hpp>

#include "PrismCage.hpp"
#include "geogram/AABB.hpp"
#include "igl/write_triangle_mesh.h"
#include "polyshell_utils.hpp"
#include "prism/energy/map_distortion.hpp"
#include "prism/feature_utils.hpp"
#include "prism/local_operations/remesh_pass.hpp"
#include "prism/local_operations/retain_triangle_adjacency.hpp"
#include "prism/local_operations/validity_checks.hpp"
#include "prism/predicates/inside_octahedron.hpp"

namespace prism::local_validity {
template <typename T>
std::set<int> find_rejection_trackee(const RowMati &F,
                                     const std::vector<std::vector<int>> &VF,
                                     const std::vector<std::vector<int>> &VFi,
                                     const std::vector<int> &seg, T, T);
std::optional<std::vector<std::set<int>>> distort_check(
    const std::vector<Vec3d> &,
    const std::vector<Vec3d> &,  // placed new verts
    const std::vector<Vec3d> &, const std::vector<Vec3i> &shells,
    const std::set<int> &combined_trackee,  // indices to ref.F tracked
    const RowMatd &refV, const RowMati &refF, double distortion_bound,
    int num_freeze, bool bundled_intersection);

bool volume_check(const std::vector<Vec3d> &base, const std::vector<Vec3d> &mid,
                  const std::vector<Vec3d> &top, const std::vector<Vec3i> &tris,
                  int num_cons);
}  // namespace prism::local_validity
bool prism::cage_check::verify_edge_based_track(
    const PrismCage &pc, const prism::local::RemeshOptions &option,
    std::vector<std::set<int>> &map_track) {
  auto num_freeze = pc.ref.aabb->num_freeze;
  assert(map_track.size() == pc.F.size() && "use with shell pipeline");
  auto verify = [&F_sh = std::as_const(pc.F), &top = std::as_const(pc.top),
                 &base = std::as_const(pc.base), &mid = std::as_const(pc.mid),
                 &refV = pc.ref.V, &refF = pc.ref.F, &option, &pc,
                 &meta = pc.meta_edges,
                 num_freeze](int sh, int i, double distortion) -> bool {
    if (!option.use_polyshell) {  // no zig version.
      auto tracks = prism::local_validity::distort_check(
          base, mid, top, {F_sh[sh]}, std::set<int>{i}, refV, refF, distortion,
          num_freeze, option.dynamic_hashgrid);
      if (tracks) spdlog::trace("tracks {}", tracks.value());
      return (tracks && tracks.value()[0].size() > 0);
    }
    // polyshell
    auto &f = F_sh[sh];
    auto [oppo_vid, rej_id, segs] =
        prism::local_validity::identify_zig(meta, f);
    if (rej_id == -10) {
      spdlog::critical("Not allow two features in same triangle");
      throw std::runtime_error("Not allow two features in same triangle");
      return false;
    }
    if (rej_id >= 0) {
      auto [v0, v1, v2] =
          std::tie(f[oppo_vid], f[(oppo_vid + 1) % 3], f[(oppo_vid + 2) % 3]);
      auto reject = prism::local_validity::find_rejection_trackee(pc.ref.F, pc.ref.VF, pc.ref.VFi, segs, 
                                        segs.begin(), segs.end());
      if (reject.find(i) != reject.end()) {
        spdlog::debug("should have been rejected p{} t{}", sh, i);
        return false;
      }
      auto [local_base, local_mid, local_top, zig_tris, _] =
          prism::local_validity::zig_constructor(pc, v0, v1, v2, segs, true);
      auto castmatd = [](auto &vec) {
        RowMatd mat;
        vec2eigen(vec, mat);
        return mat;
      };
      auto tracks = prism::local_validity::distort_check(
          local_base, local_mid, local_top, zig_tris, std::set<int>{i}, refV,
          refF, distortion, f[0] < num_freeze ? 1 : 0, true);
      if (!tracks) return false;
      for (auto i : tracks.value())
        if (i.size() > 0) return true;
      return false;
    } else {
      auto tracks = prism::local_validity::distort_check(
          base, mid, top, {F_sh[sh]}, std::set<int>{i}, refV, refF, distortion,
          num_freeze, option.dynamic_hashgrid);
      return (tracks && tracks.value()[0].size() > 0);
    }
  }; // function verify
  // check surjective.
  {
    Eigen::VectorXi flags_per_face(pc.ref.F.rows());
    flags_per_face.setConstant(1);
    for (auto k : map_track) {
      for (auto f : k) flags_per_face[f] = 0;
    }
    for (int k = 0; k < flags_per_face.size(); k++) {
      if (flags_per_face[k] > 0) {
        spdlog::error("Not surjective tri {}", k);
        return false;
      }
    }
  }
  // pc meta edges
  for (auto [m, ch_segs] : pc.meta_edges) {
    if (m.first < 0 || m.second < 0) {
      spdlog::error("meta edges mis-reference");
      return false;
    }
  }
  // do feature
  std::set<std::pair<int, int>> orig_chains;
  auto chains = prism::recover_chains_from_meta_edges(pc.meta_edges);
  for (auto ch : chains) {
    for (auto c = ch.begin(); std::next(c) != ch.end(); c++) {
      auto v0 = *c, v1 = *std::next(c);
      if (v0 > v1) std::swap(v0, v1);
      orig_chains.insert({v0, v1});
    }
  }

  // verify that existing shell-triangle pair are valid
  // verify that an extra triangle will not be tracked
  // becomes a little different with dangling node and degree-4 triangle.
  // skip the rim check for dangler case for now.
  RowMati TT, TTi;
  igl::triangle_triangle_adjacency(pc.ref.F, TT, TTi);
  auto markedTT = TT;
  for (auto f = 0; f < pc.ref.F.rows(); f++) {
    for (auto e = 0; e < 3; e++) {
      auto v0 = pc.ref.F(f, e), v1 = pc.ref.F(f, (e + 1) % 3);
      if (v0 > v1) std::swap(v0, v1);
      if (orig_chains.find({v0, v1}) != orig_chains.end()) {
        markedTT(f, e) = -1;
      }
    }
  }
  Eigen::VectorXi occurence(pc.mid.size());
  occurence.setZero();
  for (auto [m, ch_segs] : pc.meta_edges) {
    occurence[m.first]++;
    occurence[m.second]++;
  }

  for (int si = 0; si < pc.F.size(); si++) {
    if (pc.F[si][0] == pc.F[si][1]) continue;  // uncleaned collapsed faces
    auto &si_track = map_track[si];
    spdlog::trace("track{}: {}", si, map_track[si]);
    if (map_track[si].empty()) {
      spdlog::error("Empty trace {}", si);
      return false;
    }
    // topology check

    std::set<int> extended, rims;

    for (auto tri : map_track[si]) {
      spdlog::trace("Checking: s t {} {}", si, tri);
      if (verify(si, tri, -1.) != true) {
        spdlog::error("shell {} TRACKS triangle {} but overlap NOT detected.",
                     si, tri);
        return false;
      }
      for (int j = 0; j < 3; j++) {
        if (markedTT(tri, j) != -1) extended.insert(TT(tri, j));
      }
    }
    if (occurence[pc.F[si][0]] == 1 || occurence[pc.F[si][1]] == 1 ||
        occurence[pc.F[si][2]] == 1) {
      spdlog::trace("Skip Dangling Node");
      continue;  // to next si.
    }
    spdlog::trace("extended {}", extended);
    set_minus(extended, map_track[si], rims);
    spdlog::trace("mt {} \n rims {}", map_track[si], rims);
    for (auto tri : rims) {
      spdlog::trace("s t {} not {}", si, tri);
      if (verify(si, tri, option.distortion_bound) != false) {
        spdlog::error("shell {} NOT tracking triangle {} but overlap detected",
                     si, tri);
        return false;
      }
    }
    {
      RowMati trackF(map_track[si].size(), 3);
      int i = 0;
      for (auto &tri : si_track) trackF.row(i++) << pc.ref.F.row(tri);
      Eigen::VectorXi C;
      igl::facet_components(trackF, C);
      if (C.maxCoeff() > 0) {
        spdlog::trace("connected components {}", C.maxCoeff() + 1);
        return false;
      }
    }
  }
  return true;
}

bool prism::cage_check::verify_bijection(
    const PrismCage &pc, const std::vector<Vec3d> &V,
    const std::vector<Vec3i> &F,
    const std::vector<std::set<int>> &track_to_prism) {
  assert(track_to_prism.size() == F.size() && "for use during section mesh");
  // TODO: refactor and use the section distort check directly.
  auto verify = [&F_sh = std::as_const(pc.F), &top = std::as_const(pc.top),
                 &base = std::as_const(pc.base), &mid = std::as_const(pc.mid),
                 &V, &tris = F](int sh, int i, bool require_inter) -> bool {
    auto cur_tri =
        std::array<Vec3d, 3>{V[tris[i][0]], V[tris[i][1]], V[tris[i][2]]};

    int num_freeze = 0;
    auto [v0, v1, v2] = F_sh[sh];
    std::array<Vec3d, 3> base_vert{base[v0], base[v1], base[v2]};
    std::array<Vec3d, 3> mid_vert{mid[v0], mid[v1], mid[v2]};
    std::array<Vec3d, 3> top_vert{top[v0], top[v1], top[v2]};
    std::array<bool, 3> oct_type;
    prism::determine_convex_octahedron(base_vert, top_vert, oct_type,
                                       num_freeze > v0);

    bool intersected_prism =
        prism::triangle_intersect_octahedron(base_vert, mid_vert, oct_type,
                                             cur_tri, num_freeze > v0) ||
        prism::triangle_intersect_octahedron(mid_vert, top_vert, oct_type,
                                             cur_tri, num_freeze > v0);
    if (!intersected_prism) {  // if no intersection
      if (require_inter) {
        spdlog::trace("require intersection s{} t{}", sh, i);
        return false;
      }
      return true;
    }

    if (require_inter == false) {
      spdlog::trace("unintended intersection occur s{} t{}", sh, i);
      return false;
    }
    for (int tc = (v0 < num_freeze) ? 1 : 0; tc < 3; tc++) {
      auto pillar = top_vert[tc] - base_vert[tc];
      auto distortion = prism::energy::map_max_cos_angle(pillar, cur_tri);
      if (distortion < 0.) {
        spdlog::trace("distortion fail {} occur s{} t{}", distortion, sh, i);
        return false;
      }
    }
    return true;
  };
  auto [TT, TTi] = prism::local::triangle_triangle_adjacency(pc.F);
  for (int i = 0; i < track_to_prism.size();
       i++) {  // for each triangle in section mesh
    auto &si = track_to_prism[i];
    for (auto sf : si) {
      // verify shell sf intersects triangle i;

      if (!verify(sf, i, true)) {
        spdlog::trace("verify: shell {} SHOULD intersect triangle {}", sf, i);
        return false;
      }
      for (int j = 0; j < 3; j++) {
        auto f1 = TT[sf][j];
        if (si.find(f1) != si.end()) continue;  // skip if neighbor also in.

        if (!verify(f1, i, false)) {
          spdlog::trace("verify: shell {} NOT intersect triangle {}", f1, i);
          return false;
        }
      }
    }
  }
  return true;
}

bool prism::cage_check::cage_is_positive(const PrismCage &pc) {
  auto num_cons = pc.ref.aabb->num_freeze;
  for (auto i = 0; i < pc.F.size(); i++) {
    if (!prism::local_validity::volume_check(pc.base, pc.mid, pc.top, {pc.F[i]},
                                             num_cons)) {
      spdlog::error("Negative Volume", pc.F[i]);
      return false;
    }
  }
  return true;
}
bool prism::cage_check::cage_is_away_from_ref(const PrismCage &pc) {
  auto aabb = std::optional<prism::geogram::AABB>();
  aabb.emplace(pc.ref.V, pc.ref.F);
  auto num_freeze = pc.ref.aabb->num_freeze;
  aabb->num_freeze = num_freeze;

  for (auto [v0, v1, v2] : pc.F) {
    // singular edge does not exist. Bevel always split it aggressively.
    assert(!(v1 < num_freeze && v2 < num_freeze));
    if (aabb->intersects_triangle({pc.base[v0], pc.base[v1], pc.base[v2]},
                                  v0 < num_freeze)) {
      spdlog::error("Intersect Base [{}, {}, {}]", v0, v1, v2);
      return false;
    }
    if (aabb->intersects_triangle({pc.top[v0], pc.top[v1], pc.top[v2]},
                                  v0 < num_freeze)) {
      spdlog::error("Intersect Top [{}, {}, {}]", v0, v1, v2);
      return false;
    }
  }
  return true;
}

void prism::cage_check::initial_trackee_reconcile(PrismCage &pc, double distortion_bound) {
  spdlog::enable_backtrace(100);
  auto &refV = pc.ref.V;
  auto &refF = pc.ref.F;
  for (auto f = 0; f < pc.F.size(); f++) {
    auto &track = pc.track_ref[f];
    for (auto it = track.begin(); it != track.end();) {
      auto tracks = prism::local_validity::distort_check(
          pc.base, pc.mid, pc.top, {pc.F[f]}, std::set<int>{*it}, refV, refF,
          distortion_bound, pc.ref.aabb->num_freeze, true);
      if (tracks) spdlog::trace("tracks {}", tracks.value());
      if (!tracks || tracks.value()[0].size() == 0) {
        track.erase(it++);
      } else {
        it++;
      }
    }
    if (track.empty()) {
      spdlog::dump_backtrace();
      spdlog::error("Reconcile Empty: {}", f); exit(1);}
  }
}