#ifndef PRISM_FEATUE_UTILS_HPP
#define PRISM_FEATUE_UTILS_HPP

#include <list>
#include <set>
#include <vector>

#include "common.hpp"
namespace prism {

constexpr auto vv2fe = [](auto &v0, auto &v1, const auto &F, const auto &VF) {
  std::vector<int> common;
  std::set_intersection(VF[v0].begin(), VF[v0].end(), VF[v1].begin(),
                        VF[v1].end(), std::back_inserter(common));
  for (auto f : common) {
    for (int j = 0; j < 3; j++) {
      if (F(f, j) == v0 && F(f, (j + 1) % 3) == v1) {
        return std::pair(f, j);
      }
    }
  }
  return std::pair(-1, -1);
};

// reader from [Gao et al]
bool read_feature_graph(std::string path, Eigen::VectorXi &feat_nodes,
                        RowMati &feat_edges);

bool read_feature_h5(std::string path, Eigen::VectorXi &feat_nodes,
                     RowMati &feat_edges, Eigen::VectorXi &points_fid,
                     RowMatd &points_bc);
// compute the dot product of adjacent face normals (TT and FN)
// and mark v0-v1 if below threshold.
void mark_feature_edges(const RowMatd &V, const RowMati &F, double thre,
                        RowMati &feat);
// connect list of feature edges, and split into disjoint chains.
// feat_nodes is optional and high valence connectors are automatically
// detected. Note: Cyclic loops is not specially marked, since front == end;
bool feature_chains_from_edges(const Eigen::VectorXi &feat_nodes,
                               const RowMati &feat_edges, int vnum,
                               std::vector<std::list<int>> &all_chain);

// mark the region around the chains
// for each chain, mark the reference that should be reject by left/right.
void feature_chain_region_segments(
    const RowMati &F, int vnum, const std::vector<std::list<int>> chains,
    std::vector<std::set<int>> &feature_ignore,
    std::vector<std::set<int>> &region_around_chain);

std::vector<std::list<int>> glue_meta_together(
    const std::map<std::pair<int, int>, std::pair<int, std::vector<int>>>
        &meta);
std::vector<std::list<int>> recover_chains_from_meta_edges(
    const std::map<std::pair<int, int>, std::pair<int, std::vector<int>>>
        &meta);

// split triangles according to slice_vv
// while maintaining feature_edges correctly tagged.
std::tuple<RowMatd, RowMati, RowMati> subdivide_feature_triangles(
    const RowMatd &mV, const RowMati &mF, const RowMati &feature_edges,
    const std::vector<std::pair<int, int>> &slicer,
    std::vector<int> &face_parent);

// split the edges connecting two feature verts but not feature edge.
void split_feature_ears(RowMatd &mV, RowMati &mF, const RowMati &mE,
                        std::vector<int> &face_parent);

bool feature_sanity(const RowMatd &mV, RowMati &mE);

// this function splits extrmely long feature edges above a threshold
// to promote the grouping of feature edges, without the need of fractional.
bool feature_pre_split(RowMatd &V, RowMati &F, RowMati &feature_edges,
                       double threshold, std::vector<int> &face_parent);
}  // namespace prism

#endif