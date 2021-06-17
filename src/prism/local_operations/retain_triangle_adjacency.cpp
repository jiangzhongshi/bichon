//
//
#include "retain_triangle_adjacency.hpp"

#include <igl/triangle_triangle_adjacency.h>
#include <array>
#include <set>
namespace prism::local {
std::pair<std::vector<Vec3i>, std::vector<Vec3i>> triangle_triangle_adjacency(const std::vector<Vec3i>& F) {
  RowMati mFF, mFFi, mF;
  mF = Eigen::Map<const RowMati>(F[0].data(), F.size(), 3);

  igl::triangle_triangle_adjacency(mF, mFF, mFFi);
  std::vector<Vec3i> FF(F.size()), FFi(F.size());
  std::memcpy(FF[0].data(), mFF.data(), F.size() * sizeof(F[0]));
  std::memcpy(FFi[0].data(), mFFi.data(), F.size() * sizeof(F[0]));
  return {FF, FFi};
}
}  // namespace prism

void prism::local::retain_triangle_adjacency(
    const std::set<int>& delete_id, const std::vector<Vec3i>& new_T,
    std::vector<Vec3i>& T, std::vector<Vec3i>& TT, std::vector<Vec3i>& TTi) {
  using namespace Eigen;
  using vecV2d = std::vector<Eigen::RowVector2d>;
  using vecV3d = std::vector<Vec3d>;
  using vecV3i = std::vector<Vec3i>;
  using T_t = vecV3i;
  using TT_t = T_t;

  // input
  T_t new_tets = new_T;
  std::set<int> surround_id;
  {
    std::set<int> all_id;
    for (auto d : delete_id) {
      all_id.insert(d);
      for (auto j : {0, 1, 2}) all_id.insert(TT[d][j]);
    }
    std::set_difference(all_id.begin(), all_id.end(), delete_id.begin(),
                        delete_id.end(),
                        std::inserter(surround_id, surround_id.begin()));
  }
  std::vector<int> new_tet_map;     // map the local indices to global
  std::vector<int> voided_tets_id;  // only for num_addition < 0
  const int num_new = (int)new_T.size();
  const int num_influence = (int)surround_id.size();
  const int num_addition = int(num_new - delete_id.size());
  new_tet_map.reserve(num_new + num_influence);
  new_tet_map.insert(new_tet_map.end(), delete_id.begin(), delete_id.end());
  for (int i = 0; i < num_addition; i++) new_tet_map.push_back(i + T.size());
  if (num_addition < 0) {
    voided_tets_id.resize(-num_addition);
    for (int i = 0; i < -num_addition; i++)
      voided_tets_id[i] = (new_tet_map[i + num_new]);
    new_tet_map.resize(num_new);
  }
  new_tet_map.insert(new_tet_map.end(), surround_id.begin(), surround_id.end());

  // add influenced to new_tets
  new_tets.resize(num_new + num_influence);
  for (int i = num_new; i < num_new + num_influence; i++) {
    new_tets[i] = T[new_tet_map[i]];
  }

  auto [new_TT, new_TTi] = prism::local::triangle_triangle_adjacency(new_tets);
  // only t id are confused within the new_*'s
  // recover
  int old_size = T.size();
  if (num_addition > 0) {
    T.resize(old_size + num_addition, {-1, -1, -1});
    TT.resize(old_size + num_addition, {-1, -1, -1});
    TTi.resize(old_size + num_addition, {-1, -1, -1});
  }

  // Brute force: nullify pointers to deleted tets.
  for (auto i : delete_id) {
    for (auto j : {0, 1, 2}) {
      auto sur = TT[i][j];
      if (sur != -1) {
        auto sur_j = TTi[i][j];
        TT[sur][sur_j] = -1;
        TTi[sur][sur_j] = -1;
      }
    }
  }

  for (int i = 0; i < num_new; i++) {
    T[new_tet_map[i]] = new_tets[i];
    TTi[new_tet_map[i]] = new_TTi[i];

    for (int j = 0; j < 3; j++) {  // TT needs care because of t id
      auto adj_tet = new_TT[i][j];
      if (adj_tet == -1)
        TT[new_tet_map[i]][j] = -1;
      else
        TT[new_tet_map[i]][j] = new_tet_map[adj_tet];
    }
  }
  for (int i = num_new; i < num_new + num_influence; i++)
    for (int j = 0; j < 3; j++) {
      int adj_tet = new_TT[i][j];
      if (adj_tet == -1) continue;
      // got // some // valid // information
      TT[new_tet_map[i]][j] = new_tet_map[adj_tet];
      TTi[new_tet_map[i]][j] = new_TTi[i][j];
    }  // for, for

  if (num_addition < 0) {
    // take special care here.
    // fix ending's influence
    std::set<int> voided_set(voided_tets_id.begin(), voided_tets_id.end());
    std::set<int> ending_set;
    std::set<int> symmetric_diff;
    for (int i = 0; i < -num_addition; i++)
      ending_set.insert(ending_set.end(), old_size + num_addition + i);
    std::set_symmetric_difference(
        voided_set.begin(), voided_set.end(), ending_set.begin(),
        ending_set.end(), std::inserter(symmetric_diff, symmetric_diff.end()));
    int num_substituting = symmetric_diff.size() / 2;
    auto ending_iter = symmetric_diff.rbegin();
    auto voided_iter = symmetric_diff.begin();
    for (int i = 0; i < num_substituting; i++) {
      int t = *ending_iter;
      int vi = *voided_iter;
      for (auto f : {0, 1, 2}) {
        if (TT[t][f] == -1) continue;
        assert(TT[TT[t][f]][TTi[t][f]] == t);
        TT[TT[t][f]][TTi[t][f]] = vi;
      }
      TT[vi] = TT[t];
      TTi[vi] = TTi[t];

      ending_iter++;
      voided_iter++;
    }

    ending_iter = symmetric_diff.rbegin();
    voided_iter = symmetric_diff.begin();
    // assign back
    for (int i = 0; i < num_substituting; i++) {
      int t = *ending_iter;
      int vi = *voided_iter;
      T[vi] = T[t];

      ending_iter++;
      voided_iter++;
    }

    T.resize(old_size + num_addition);
    TT.resize(old_size + num_addition);
    TTi.resize(old_size + num_addition);
  }
}