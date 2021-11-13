#ifndef PRISM_POLYSHELL_UTILS_HPP
#define PRISM_POLYSHELL_UTILS_HPP

#include <list>
#include <set>
#include <vector>
#include <map>

#include "common.hpp"
struct PrismCage;
namespace prism::local_validity {
///
/// @brief   Identifies which edge in the (coarse) triangle is a poly-edge, by looking up feature info meta_edges.
/// @param   meta_edges storage structure of feature information.
///          Maps (v0,v1) from the coarse mesh to (cid, segs) where segs is a sequence of vertices from the reference mesh.
/// @param   f triangle to identify, of three vertex indices
///
/// @return  oppo_vid opposite vertex local id (inside the triangle).
/// @return   `cid`: id of the chain (oddity determines the side 2c or 2c + 1).
/// @return   `segs`: vector of "real" vertex.
/// @default:  -1,-1,{}
/// @error:  -10,-10,{} raise when `f` has two feature edges. (should be split before).
///
auto identify_zig(const std::map<std::pair<int, int>, std::pair<int, std::vector<int>>> &meta_edges,
                  const Vec3i &f) -> std::tuple<int, int, std::vector<int>> ;

///
/// @brief  ad-hoc constructor for the poly-prisms.
/// @note   this is called after `identify_zig`, and relies on v0 minimum ordering.
/// @return local_base, local_mid, local_top 
/// @return zig_tris (composed of local indices), zig_shifts (so that it is consistent during triple-tet tests)
auto zig_constructor(const PrismCage &pc, int v0, int v1, int v2,
                     const std::vector<int> &segs, bool lets_comb_the_pillars)
       -> std::tuple<std::vector<Vec3d>, std::vector<Vec3d>, std::vector<Vec3d>,
                  std::vector<Vec3i>, std::vector<int>> ;
}
#endif