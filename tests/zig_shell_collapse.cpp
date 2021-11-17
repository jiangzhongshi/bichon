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
  auto pc = std::shared_ptr<PrismCage>(new PrismCage("col2.h5"));
  prism::local::RemeshOptions option(pc->mid.size(), 0.1);
  // auto chains = prism::recover_chains_from_meta_edges(pc->meta_edges);
  option.use_polyshell = true;
  option.dynamic_hashgrid = true;

  spdlog::enable_backtrace(100);
  option.distortion_bound = 0.01;
  // option.target_thickness = 5e-2;
  option.collapse_quality_threshold = 30;
  option.collapse_valence_threshold = 10;
  option.parallel = false;
  option.linear_curve = true;
  option.relax_quality_threshold = 0;
  prism::reverse_feature_order(pc->meta_edges);
  prism::local::zig_slide_pass(*pc, option);
  // spdlog::set_level(spdlog::level::trace);
  // CHECK(prism::cage_check::verify_edge_based_track(*pc, option, pc->track_ref));
  pc->serialize("slide.h5");
  //  prism::local::wildflip_pass(*pc, option);
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
  prism::reverse_feature_order(pc.meta_edges);
  spdlog::set_level(spdlog::level::trace);
  prism::local::feature_collapse_pass(pc, option);
  exit(1);
  CHECK(prism::cage_check::verify_edge_based_track(pc, option, pc.track_ref));
  // pc.serialize("../buildr/debug1.h5");
}