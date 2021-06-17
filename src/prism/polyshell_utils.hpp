#ifndef PRISM_POLYSHELL_UTILS_HPP
#define PRISM_POLYSHELL_UTILS_HPP

#include <list>
#include <set>
#include <vector>
#include <map>

#include "common.hpp"
struct PrismCage;
namespace prism::local_validity {
auto identify_zig(const std::map<std::pair<int, int>, std::pair<int, std::vector<int>>> &meta_edges,
                  const Vec3i &f) -> std::tuple<int, int, std::vector<int>> ;

auto zig_constructor(const PrismCage &pc, int v0, int v1, int v2,
                     const std::vector<int> &segs, bool lets_comb_the_pillars)
       -> std::tuple<std::vector<Vec3d>, std::vector<Vec3d>, std::vector<Vec3d>,
                  std::vector<Vec3i>, std::vector<int>> ;
}
#endif