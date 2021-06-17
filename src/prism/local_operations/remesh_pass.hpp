#ifndef PRISM_LOCAL_OPERATIONS_REMESH_PASS_HPP
#define PRISM_LOCAL_OPERATIONS_REMESH_PASS_HPP

#include <any>
#include <list>
#include <map>
#include <utility>
#include <vector>

#include "../common.hpp"

struct PrismCage;
namespace prism::geogram {
struct AABB;
};
namespace prism::local {
struct RemeshOptions {
  double distortion_bound = 0.1;
  double target_thickness = 0.01;
  double zig_thick = 1e-4;
  bool parallel = true;
  bool use_polyshell = false; // zig remesh or snapper remesh
  double relax_quality_threshold = 20;
  double collapse_quality_threshold = 30;
  double collapse_valence_threshold = -1;  // enabled if positive.
  bool split_improve_quality = true;
  bool volume_centric = false;  // volume quality etc.
  bool dynamic_hashgrid =
      false;  // use a dynamic spatial hashgrid instead of static AABB

  std::function<double(const Vec3d &)> sizing_field;
  std::vector<double> target_adjustment;
  RemeshOptions() = default;
  RemeshOptions(int v_num, double edge_len) {
    sizing_field = [edge_len](const Vec3d &) { return edge_len; };
    target_adjustment.resize(v_num, 1);
  }
  bool linear_curve = true; // postpone curving to later stages, to save time at the beginning.
  // additional
  double curve_dist_bound = 1e-2;
  double curve_normal_bound = 1e-2;
  bool curve_recurse_check = true;
  // pre and post
  std::pair<std::any, std::any> curve_checker;
  std::vector<std::set<int>> chain_reject_trackee;
};
}  // namespace prism::local

namespace prism::local {
constexpr auto shift_left = [](const auto &new_fid, const auto &new_shifts,
                               auto &F, auto &FF, auto &FFi) {
  constexpr auto roll_shift_left = [](auto &vec, int s) -> Vec3i {
    auto [a, b, c] = vec;
    if (s == 0) return {a, b, c};
    if (s == 1)
      return {b, c, a};
    else
      return {c, a, b};
  };
  for (int i = 0; i < new_shifts.size(); i++) {
    auto f = new_fid[i];
    auto s = new_shifts[i];
    // shift F,FF,FFi
    F[f] = roll_shift_left(F[f], s);
    FF[f] = roll_shift_left(FF[f], s);
    FFi[f] = roll_shift_left(FFi[f], s);
    // take care of the FFi for neighbors
    for (int j : {0, 1, 2}) {
      int f1 = FF[f][j];
      if (f1 == -1) continue;
      int e1 = FFi[f][j];
      FFi[f1][e1] = j;
    }
  }
};

int wildcollapse_pass(PrismCage &pc, RemeshOptions &);
void wildflip_pass(PrismCage &pc, const RemeshOptions &);
int wildsplit_pass(PrismCage &pc, RemeshOptions &);
void localsmooth_pass(PrismCage &pc, const RemeshOptions &);
void shellsmooth_pass(PrismCage &pc, const RemeshOptions &option);
}  // namespace prism::local
#endif