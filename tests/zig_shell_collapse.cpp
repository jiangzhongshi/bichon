#include <igl/vertex_triangle_adjacency.h>

#include <algorithm>
#include <highfive/H5Easy.hpp>
#include <prism/PrismCage.hpp>
#include <prism/cage_check.hpp>
#include <prism/geogram/AABB.hpp>
#include <queue>

#include "cumin/curve_validity.hpp"
#include "prism/PrismCage.hpp"
#include "prism/common.hpp"
#include "prism/local_operations/remesh_with_feature.hpp"
#include "test_common.hpp"
#include <prism/feature_utils.hpp>
#include <prism/local_operations/validity_checks.hpp>
#include <prism/polyshell_utils.hpp>

void reverse_feature_order(PrismCage &pc,
                                prism::local::RemeshOptions &option) {
  decltype(pc.meta_edges) meta;
  for (auto [a, b] : pc.meta_edges) {
    auto a1 = std::pair{a.second, a.first};
    auto b1 = b;
    b1.second = std::vector<int>(b.second.rbegin(), b.second.rend());
    meta.emplace(a1, b1);
  }
  pc.meta_edges = std::move(meta);
};
constexpr auto recover_option_etc = [](auto &pc) {
  auto option = prism::local::RemeshOptions(pc.mid.size(), 1e-1);
  option.dynamic_hashgrid = true;
  option.distortion_bound = 0.1;
  option.target_thickness = 1e-3;

  spdlog::enable_backtrace(100);
  option.relax_quality_threshold = 20;
  option.collapse_valence_threshold = 10;
  option.parallel = false;
  auto chains = prism::recover_chains_from_meta_edges(pc.meta_edges);
  option.chain_reject_trackee = [&pc, &chains]() {
    std::vector<std::set<int>> feature_region_segments;
    std::vector<std::set<int>> region_around_chain;
    prism::feature_chain_region_segments(pc.ref.F, pc.ref.V.rows(), chains,
                                         feature_region_segments,
                                         region_around_chain);
    return std::move(feature_region_segments);
  }();
  return std::move(option);
};

constexpr auto smooth_like_checker = [](auto &pc, auto &option) {
  std::vector<std::vector<int>> VF, VFi;
  igl::vertex_triangle_adjacency(
      pc.mid.size(), Eigen::Map<const RowMati>(pc.F[0].data(), pc.F.size(), 3),
      VF, VFi);
  spdlog::info("Start Static Sanity Checks.");
  for (auto vid = 0; vid < pc.mid.size(); vid++) {
    auto nb = VF[vid];
    auto &old_fid = nb;
    std::vector<Vec3i> moved_tris;
    std::vector<std::set<int>> new_tracks;
    for (auto &f : VF[vid]) {
      moved_tris.push_back(pc.F[f]);
    }
    std::vector<RowMatd> local_cp;
    auto old_quality = 1e10;
    int flag = prism::local_validity::attempt_zig_remesh(
        pc, pc.track_ref, option, old_quality, old_fid, moved_tris, new_tracks,
        local_cp);
    if (flag != 0)
      spdlog::dump_backtrace();
    REQUIRE_EQ(flag, 0);
  }
  spdlog::info("Finish Static Sanity Checks.");
};

#include "prism/local_operations/retain_triangle_adjacency.hpp"
TEST_CASE("reset-zig") {
  PrismCage pc("../buildr/anc101.obj.resetmV.h5");
  vec2eigen(pc.mid, pc.ref.V);
  vec2eigen(pc.F, pc.ref.F);
  for (auto &[m, ch_segs] : pc.meta_edges) {
    auto &[ch, segs] = ch_segs;
    segs = std::vector<int>{m.first, m.second};
  }
  [pc = &pc]() { // start out. reset track.
    // do feature
    spdlog::info("Set feature aware track");
    std::set<std::pair<int, int>> feature_edge_set;
    for (auto [m, id_chain] : pc->meta_edges) {
      auto [id, chain] = id_chain;
      for (int i = 0; i < chain.size() - 1; i++) {
        auto v0 = chain[i], v1 = chain[(i + 1)];
        if (v0 > v1)
          std::swap(v0, v1);
        feature_edge_set.insert({v0, v1});
      }
    }
    auto [TT, TTi] = prism::local::triangle_triangle_adjacency(pc->F);
    pc->track_ref.resize(TT.size());
    for (int i = 0; i < TT.size(); i++) {
      pc->track_ref[i] = {i};
      for (int j = 0; j < 3; j++) {
        auto v0 = pc->F[i][j], v1 = pc->F[i][(j + 1) % 3];
        if (v0 > v1)
          std::swap(v0, v1);
        if (feature_edge_set.find({v0, v1}) != feature_edge_set.end())
          continue;
        pc->track_ref[i].insert(TT[i][j]);
      }
    }
  }();
  // pc.ref.inpV = RowMatd();
  // pc.ref.refV = 
  auto option = recover_option_etc(pc);
  option.distortion_bound = 0.1;
  option.target_thickness = 1e-2;
  option.use_polyshell = true;
  CHECK(prism::cage_check::verify_edge_based_track(pc, option, pc.track_ref));
  for (auto ite = 0; ite < 20; ite++) {
    option.relax_quality_threshold = 20;
    auto col = 0;
    //  option.sizing_field = [](auto &) { return 0.2; };
    col += prism::local::wildcollapse_pass(pc, option);
    col += prism::local::zig_collapse_pass(pc, option);
    col += prism::local::zig_collapse_pass(pc, option);
    option.relax_quality_threshold = 0;
    prism::local::zig_slide_pass(pc, option);
    prism::local::localsmooth_pass(pc, option);
    prism::local::wildflip_pass(pc, option);
    // CHECK(prism::cage_check::verify_edge_based_track(pc, option,
    // pc.track_ref)); 
     option.sizing_field = [](auto &) { return 0.02; };
    option.relax_quality_threshold = 20;
    prism::local::wildsplit_pass(pc, option);
    prism::local::zig_split_pass(pc, option);
    option.relax_quality_threshold = 0;
    prism::local::zig_slide_pass(pc, option);
    prism::local::localsmooth_pass(pc, option);
    prism::local::wildflip_pass(pc, option);
    reverse_feature_order(pc, option);
    if (col == 0)
      break;
    pc.serialize("../buildr/debug_hybrid_comb.h5");
  }
}

TEST_CASE("snapper-check") {
  PrismCage pc("../buildr/temp.h5");
  //28062
  //28069
  auto option = recover_option_etc(pc);
  CHECK(prism::cage_check::verify_edge_based_track(pc, option, pc.track_ref));
exit(1);
  option.relax_quality_threshold = 20;
  prism::local::wildcollapse_pass(pc, option);
  prism::local::feature_collapse_pass(pc, option);
  pc.serialize("../buildr/debug0.h5");
  reverse_feature_order(pc, option);
  spdlog::set_level(spdlog::level::trace);
  prism::local::feature_collapse_pass(pc, option);
  exit(1);
  CHECK(prism::cage_check::verify_edge_based_track(pc, option, pc.track_ref));
  // pc.serialize("../buildr/debug1.h5");
}