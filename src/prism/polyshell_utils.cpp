#include "polyshell_utils.hpp"
#include <prism/PrismCage.hpp>
#include <prism/cage_check.hpp>
#include <prism/geogram/AABB.hpp>
#include <prism/local_operations/validity_checks.hpp>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>
#include <highfive/H5Easy.hpp>

auto prism::local_validity::identify_zig(
    const std::map<std::pair<int, int>, std::pair<int, std::vector<int>>>
        &meta_edges,
    const Vec3i &f) -> std::tuple<int, int, std::vector<int>> {
  auto oppo_vid = -1;
  auto cid = -1;
  auto segs = std::vector<int>({});
  for (auto j = 0; j < 3; j++) {
    auto v0 = f[j], v1 = f[(j + 1) % 3];
    auto it0 = meta_edges.find({v0, v1});
    auto it1 = meta_edges.find({v1, v0});
    if (it0 == it1) // == end iterator. i.e. not found.
      continue;
    if (cid >= 0) {
      spdlog::trace("two feat would be in one triangle");
      return std::tuple(-10, -10, std::vector<int>());
    }
    if (it0 != meta_edges.end()) { // left
      cid = 2 * (it0->second.first);
      segs = it0->second.second;
    } else { // right
      cid = 2 * (it1->second.first) + 1;
      segs = it1->second.second;
      segs = std::vector<int>(segs.rbegin(), segs.rend()); // reverse order
    }
    oppo_vid = (j + 2) % 3;
  }
  return std::tuple(oppo_vid, cid, segs);
};

auto prism::local_validity::zig_constructor(const PrismCage &pc, int v0, int v1,
                                            int v2,
                                            const std::vector<int> &segs,
                                            bool lets_comb_the_pillars)
    -> std::tuple<std::vector<Vec3d>, std::vector<Vec3d>, std::vector<Vec3d>,
                  std::vector<Vec3i>, std::vector<int>> {
  auto &base = pc.base, &top = pc.top, &mid = pc.mid;
  auto &refV = pc.ref.V;
  spdlog::trace("v0 v1 v2 {},{},{}", v0, v1, v2);
  std::vector<Vec3d> local_mid(0); //(segs.size() + 1);
  local_mid.reserve(segs.size() + 3);
  auto flip_order = v1 > v2;
  auto start = mid[v1];
  auto end = mid[v2];
  auto padded_segs = segs;
  if (segs.empty()) {
    local_mid.push_back(start);
    local_mid.push_back(end);
    padded_segs = {-1, -1};
  } else {
    if (refV.row(segs.front()) != start) { // add to front.
      local_mid.push_back(start);
      padded_segs.insert(padded_segs.begin(), -1);
    }
    for (auto s : segs)
      local_mid.push_back(refV.row(s)); // 1,2,...,n
    if (refV.row(segs.back()) != end) { // add to back
      local_mid.push_back(end);
      padded_segs.push_back(-1);
    }
  }
  if (flip_order) {
    std::reverse(local_mid.begin(), local_mid.end());
    std::reverse(padded_segs.begin(), padded_segs.end());
    std::swap(v1, v2);
  }

  assert(local_mid.size() >= segs.size());
  std::vector<Vec3d> local_base(local_mid.size());
  std::vector<Vec3d> local_top(local_mid.size());
  assert(v1 < v2);
  local_base.front() = base[v1];
  local_base.back() = base[v2];
  local_top.front() = top[v1];
  local_top.back() = top[v2];
  int n = local_mid.size();
  if (lets_comb_the_pillars) {
    auto cumlength = std::vector<double>(n, 0.);
    for (auto i = 1; i < n; i++) {
      cumlength[i] =
          (cumlength[i - 1] + (local_mid[i] - local_mid[i - 1]).norm());
    }
    // spdlog::info(" base[v1] {}, {}",  base[v1],  base[v2]);
    for (auto i = 1; i < n - 1; i++) {
      auto alpha = cumlength[i] / cumlength.back();
      Vec3d offset_mid = mid[v1] * (1 - alpha) + mid[v2] * alpha - local_mid[i];
      local_base[i] = base[v1] * (1 - alpha) + base[v2] * alpha - offset_mid;
      local_top[i] = top[v1] * (1 - alpha) + top[v2] * alpha - offset_mid;
    }
  } else {
    for (auto i = 0; i < n; i++) {
      if (padded_segs[i] == -1) continue;
      local_base[i] = pc.zig_base.row(padded_segs[i]);
      local_top[i] = pc.zig_top.row(padded_segs[i]);
    }
  }

  auto v0i = 0;
  if (v0 > v2) {
    assert(v0 > v1);
    local_mid.push_back(mid[v0]);
    local_base.push_back(base[v0]);
    local_top.push_back(top[v0]);
    v0i = n;
  } else if (v0 > v1) { // v1, v0, v2
    local_mid.insert(local_mid.begin() + 1, mid[v0]);
    local_base.insert(local_base.begin() + 1, base[v0]);
    local_top.insert(local_top.begin() + 1, top[v0]);
    v0i = 1;
  } else { // v0, v1, v2
    local_mid.insert(local_mid.begin(), mid[v0]);
    local_base.insert(local_base.begin(), base[v0]);
    local_top.insert(local_top.begin(), top[v0]);
    v0i = 0;
  }
  auto zig_tris = std::vector<Vec3i>();
  if (v0i == 1) {
    if (!flip_order)
      zig_tris.emplace_back(Vec3i{1, 0, 2});
    else
      zig_tris.emplace_back(Vec3i{1, 2, 0});
  }
  for (int i = 0; i < n; i++) {
    if (v0i == i || v0i == i + 1)
      continue;
    if (!flip_order)
      zig_tris.emplace_back(Vec3i{v0i, i, i + 1});
    else
      zig_tris.emplace_back(Vec3i{v0i, i + 1, i});
  }
  if (flip_order)
    std::reverse(zig_tris.begin(), zig_tris.end());
  auto new_shifts = triangle_shifts(zig_tris);

  spdlog::trace("zig {}\n flip {}\n lm {}\n segs {}", zig_tris,
                flip_order, local_mid.size(), segs.size());
  assert(local_mid.size() == padded_segs.size() + 1);
  assert(local_mid.size() == zig_tris.size() + 2);
  if (false)
  {
    auto file = H5Easy::File("zig.h5", H5Easy::File::Overwrite);
    H5Easy::dump(file, "segs", segs);
    H5Easy::dump(file, "local_base",
                 RowMatd(Eigen::Map<const RowMatd>(local_base[0].data(),
                                                   local_base.size(), 3)));
    H5Easy::dump(file, "local_mid",
                 RowMatd(Eigen::Map<const RowMatd>(local_mid[0].data(),
                                                   local_mid.size(), 3)));
    H5Easy::dump(file, "local_top",
                 RowMatd(Eigen::Map<const RowMatd>(local_top[0].data(),
                                                   local_top.size(), 3)));
    H5Easy::dump(file, "zig_tris",
                 RowMati(Eigen::Map<const RowMati>(zig_tris[0].data(),
                                                   zig_tris.size(), 3)));
                //  GLOBAL_DANGRESOUR_SAVE_BUTTON = false;

  }
  return std::tuple(local_base, local_mid, local_top, zig_tris, new_shifts);
};

namespace prism::local_validity {
auto zig_comb(const PrismCage &pc, int v0, int v1, int v2,
              const std::vector<int> &segs) {
  auto &base = pc.base, &top = pc.top, &mid = pc.mid;
  auto &refV = pc.ref.V;
  spdlog::trace("v0 v1 v2 {},{},{}", v0, v1, v2);
  std::vector<Vec3d> local_mid(0); //(segs.size() + 1);
  local_mid.reserve(segs.size() + 3);
  auto flip_order = v1 > v2;
  auto start = mid[v1];
  auto end = mid[v2];
  auto padded_segs = segs;
  if (segs.empty()) {
    local_mid.push_back(start);
    local_mid.push_back(end);
    padded_segs = {-1, -1};
  } else {
    if (refV.row(segs.front()) != start) { // add to front.
      local_mid.push_back(start);
      padded_segs.insert(padded_segs.begin(), -1);
    }
    for (auto s : segs)
      local_mid.push_back(refV.row(s)); // 1,2,...,n
    if (refV.row(segs.back()) != end) { // add to back
      local_mid.push_back(end);
      padded_segs.push_back(-1);
    }
  }
  if (flip_order) {
    std::reverse(local_mid.begin(), local_mid.end());
    std::reverse(padded_segs.begin(), padded_segs.end());
    std::swap(v1, v2);
  }

  assert(local_mid.size() >= segs.size());
  assert(v1 < v2);
  int n = local_mid.size();
  auto cumlength = std::vector<double>(n, 0.);
  for (auto i = 1; i < n; i++) {
    cumlength[i] =
        (cumlength[i - 1] + (local_mid[i] - local_mid[i - 1]).norm());
  }
  for (auto i = 1; i < n - 1; i++) {
    auto alpha = cumlength[i] / cumlength.back();
    Vec3d offset_mid = mid[v1] * (1 - alpha) + mid[v2] * alpha - local_mid[i];
    // padded_segs[i] = base[v1] * (1 - alpha) + base[v2] * alpha - offset_mid;
    // padded_segs[i] = top[v1] * (1 - alpha) + top[v2] * alpha - offset_mid;
  }
};
} // namespace prism::local_validity