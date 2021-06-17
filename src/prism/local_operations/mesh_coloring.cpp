#include "mesh_coloring.hpp"
#include <igl/adjacency_list.h>
#include <iostream>
void prism::local::vertex_coloring(const RowMati& F,
                                   std::vector<std::vector<int>>& groups) {
  // from github hankstag/progressive_embedding
  int n = F.maxCoeff() + 1;
  // meanless to color an empty mesh
  assert(n > 0);

  std::vector<std::vector<int>> N;  // Adjacency list
  igl::adjacency_list(F, N);
  // initialize color for each vertex
  Eigen::VectorXd C;
  C.setConstant(n, -1);

  size_t vm = 0;  // max valence
  for (auto l : N) vm = std::max(l.size(), vm);

  // assign color 0 for vertex 0
  C[0] = 0;
  // for every 1 ring, mark the color that has been used
  // in theory, it need (vm+1) colors at most
  Eigen::VectorXi G;
  G.setZero(vm + 1);
  int nc = 1;  // # of color used
  for (int i = 1; i < n; i++) {
    for (int k = 0; k < N[i].size(); k++) {
      if (C[N[i][k]] != -1) G(C[N[i][k]]) = 1;
    }
    for (int j = 0; j < G.rows(); j++) {
      if (G(j) == 0) {
        C[i] = j;
        break;
      }
    }
    G.setZero();
  }

  groups.resize(C.maxCoeff() + 1);
  for (auto id = 0; id < C.rows(); id++) {
    groups[C[id]].push_back(id);
  }
}

void prism::local::red_green_coloring(const RowMati& F, const RowMati& FF,
                        Eigen::VectorXi& colors) {
  int sum = 0;
  while (sum != colors.sum()) {
    sum = colors.sum();
    for (int i = 0; i < F.rows(); i++) {
      if (colors[i] == 2) continue;  // red
      int count = 0;
      for (int j = 0; j < 3; j++) {
        if (FF(i, j) != -1 && colors[FF(i, j)] == 2) count++;
      }
      if (count > 1)  // red
        colors[i] = 2;
      if (count == 1) colors[i] = 1;  // green
    }
  }
}