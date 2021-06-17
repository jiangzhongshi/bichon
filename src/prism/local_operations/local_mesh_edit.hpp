#ifndef PRISM_LOCAL_OPERATIONS_HPP
#define PRISM_LOCAL_OPERATIONS_HPP

#include <vector>
#include "../common.hpp"
#include "triangle_tuple.h"

namespace prism {
bool edge_flip(std::vector<Vec3i>& F, std::vector<Vec3i>& FF,
               std::vector<Vec3i>& FFi, int f0, int e0);
bool edge_split(int ux, std::vector<Vec3i>& F, std::vector<Vec3i>& FF_vec,
                std::vector<Vec3i>& FFi_vec, int f0, int e0);

// one ring of emanating edges from the vertex F[f0][e0]
// by default, this gives *clockwise* tuples, with F[f][e] == v0
// if switch, this will gives cc tuples, with F[f][e+1] == v0
template <typename DerivedF>
inline bool get_star_edges(const DerivedF& F, const DerivedF& FF,
                           const DerivedF& FFi, int f0, int e0,
                           std::vector<std::pair<int, int>>& neighbor,
                           bool clockwise = true) {
  auto v_center = igl::triangle_tuple_get_vert(f0, e0, true, F, FF, FFi);

  if (!clockwise) {
    e0 = (e0 + 2) % 3;
  }
  auto v_start = igl::triangle_tuple_get_vert(f0, e0, !clockwise, F, FF, FFi);
  int cur_v = -1;
  bool interior = true;
  while (cur_v != v_start) {
    if (!igl::triangle_tuple_next_in_one_ring(f0, e0, clockwise, F, FF, FFi))
      interior = false;
    cur_v = igl::triangle_tuple_get_vert(f0, e0, !clockwise, F, FF, FFi);
    neighbor.emplace_back(f0, e0);
  }
  return interior;
}

bool edge_collapse(std::vector<Vec3i>& F, std::vector<Vec3i>& FF,
                   std::vector<Vec3i>& FFi, int f0, int e0);
}  // namespace prism

namespace prism {
bool edge_split(std::vector<Vec3d>& V0, std::vector<Vec3i>& F0,
                std::vector<Vec3i>& FF, std::vector<Vec3i>& FFi,
                std::vector<Eigen::Vector2i>& E,  // not v0, v1 but f,e
                std::vector<Vec3i>& EMAP,         // Fx[0,1,2] -> E
                int ue);

bool edge_collapse(std::vector<Vec3d>& V, std::vector<Vec3i>& F,
                   std::vector<Vec3i>& FF, std::vector<Vec3i>& FFi,
                   std::vector<Eigen::Vector2i>& E, std::vector<Vec3i>& EMAP,
                   int ue);

bool edge_flip(std::vector<Vec3d>& V, std::vector<Vec3i>& F,
               std::vector<Vec3i>& FF, std::vector<Vec3i>& FFi,
               std::vector<Eigen::Vector2i>& E, std::vector<Vec3i>& EMAP,
               int ue);
}  // namespace prism
#endif