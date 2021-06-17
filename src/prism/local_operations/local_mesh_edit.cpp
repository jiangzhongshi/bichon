#include "local_mesh_edit.hpp"
#include "triangle_tuple.h"

#include <spdlog/spdlog.h>
#include <algorithm>
#define IGL_COLLAPSE_EDGE_NULL 0

// helper
template <typename DerivedF>
inline std::array<int, 13> init_e00_f00(const DerivedF& F, const DerivedF& FF,
                                        const DerivedF& FFi, int f0, int f1,
                                        int e0) {
  auto e1 = (FFi[f0][e0]);
  auto e01 = (e0 + 1) % 3;
  auto e02 = (e0 + 2) % 3;
  auto e11 = (e1 + 1) % 3;
  auto e12 = (e1 + 2) % 3;
  auto f01 = (FF[f0][e01]);
  auto f02 = (FF[f0][e02]);
  auto f11 = (FF[f1][e11]);
  auto f12 = (FF[f1][e12]);
  auto u1 = F[f1][e1];
  auto u0 = F[f0][e0];
  auto v0 = F[f0][e02];
  auto v1 = F[f1][e12];

  return {e1, e01, e02, e11, e12, f01, f02, f11, f12, u0, u1, v0, v1};
}

bool prism::edge_flip(std::vector<Vec3d>& V, std::vector<Vec3i>& F,
                      std::vector<Vec3i>& FF, std::vector<Vec3i>& FFi,
                      std::vector<Eigen::Vector2i>& E, std::vector<Vec3i>& EMAP,
                      int ue) {
  int f0 = E[ue][0], e0 = E[ue][1];
  int f1 = FF[f0][e0];
  if (f1 == -1) return false;
  auto [e1, e01, e02, e11, e12, f01, f02, f11, f12, u0, u1, v0, v1] =
      init_e00_f00(F, FF, FFi, f0, f1, e0);

  int er = ue;
  int er02 = EMAP[f0][e02];
  int er01 = EMAP[f0][e01];
  int er11 = EMAP[f1][e11];
  int er12 = EMAP[f1][e12];

  prism::edge_flip(F, FF, FFi, f0, e0);

  E[er] << f0, e01;
  E[er01] << f1, e1;
  E[er02] << f0, e02;
  E[er11] << f0, e0;
  E[er12] << f1, e12;
  EMAP[f0][e0] = er11;
  EMAP[f0][e01] = er;
  EMAP[f0][e02] = er02;
  EMAP[f1][e1] = er01;
  EMAP[f1][e11] = er;
  EMAP[f1][e12] = er12;
  assert(EMAP[f01][FFi[f1][e1]] == er01);
  assert(EMAP[f02][FFi[f0][e02]] == er02);
  assert(EMAP[f11][FFi[f0][e0]] == er11);
  assert(EMAP[f12][FFi[f1][e12]] == er12);

  return true;
}

bool prism::edge_flip(std::vector<Vec3i>& F, std::vector<Vec3i>& FF,
                      std::vector<Vec3i>& FFi, int f0, int e0) {
  int f1 = FF[f0][e0];
  if (f1 == -1) return false;
  auto [e1, e01, e02, e11, e12, f01, f02, f11, f12, u0, u1, v0, v1] =
      init_e00_f00(F, FF, FFi, f0, f1, e0);
  if (f02==f11 || f01 ==f12) return false; // topological error
  F[f0][e01] = v1;
  F[f1][e11] = v0;

  assert(FF[f0][e0] == f1);
  assert(FF[f0][e01] == f01);
  FF[f0][e0] = f11;
  FF[f0][e01] = f1;
  assert(FF[f1][e1] == f0);
  assert(FF[f1][e11] == f11);
  FF[f1][e1] = f01;
  FF[f1][e11] = f0;
  if (f11 != -1) FF[f11][FFi[f1][e11]] = f0;
  if (f01 != -1) FF[f01][FFi[f0][e01]] = f1;

  auto ffi01 = FFi[f0][e01];
  auto ffi02 = FFi[f0][e02];
  auto ffi11 = FFi[f1][e11];
  auto ffi12 = FFi[f1][e12];
  assert(FFi[f0][e0] == e1);
  FFi[f0][e0] = ffi11;
  assert(FFi[f1][e1] == e0);
  FFi[f1][e1] = ffi01;
  FFi[f0][e01] = e11;
  FFi[f1][e11] = e01;
  assert(FFi[f0][e02] == ffi02);
  assert(FFi[f1][e12] == ffi12);

  // use updated FFi([f0 f1],:)
  if (f11 != -1) FFi[f11][FFi[f0][e0]] = e0;
  if (f12 != -1) FFi[f12][FFi[f1][e12]] = e12;
  if (f01 != -1) FFi[f01][FFi[f1][e1]] = e1;
  if (f02 != -1) FFi[f02][FFi[f0][e02]] = e02;

  return true;
}

bool prism::edge_split(std::vector<Vec3d>& V0, std::vector<Vec3i>& F0,
                       std::vector<Vec3i>& FF, std::vector<Vec3i>& FFi,
                       std::vector<Eigen::Vector2i>& E,  // not v0, v1 but f,e
                       std::vector<Vec3i>& EMAP,         // Fx[0,1,2] -> E
                       int ue) {
  int f0 = E[ue][0], e0 = E[ue][1];
  int f1 = FF[f0][e0];
  if (f1 == -1)
    return false;  // TODO: in fact, there can be another branch to deal with
                   // boundary.
  auto [e1, e01, e02, e11, e12, f01, f02, f11, f12, u0, u1, v0, v1] =
      init_e00_f00(F0, FF, FFi, f0, f1, e0);

  F0.push_back(F0[f0]);
  F0.push_back(F0[f1]);
  EMAP.emplace_back(Vec3i{-1, -1, -1});
  EMAP.emplace_back(Vec3i{-1, -1, -1});
  int ux = V0.size();

  F0[f1][e1] = ux;
  F0[f0][e0] = ux;
  int fx1 = F0.size() - 1;
  int fx0 = F0.size() - 2;
  F0[fx1][e11] = ux;
  F0[fx0][e01] = ux;
  if (f12 != -1) {
    FF[f12][FFi[f1][e12]] = fx1;
  }
  if (f02 != -1) {
    FF[f02][FFi[f0][e02]] = fx0;
  }

  FF.push_back(FF[f0]);
  FF.push_back(FF[f1]);
  FF[f0][e02] = fx0;
  FF[f0][e0] = fx1;
  FF[f1][e12] = fx1;
  FF[f1][e1] = fx0;
  FF[fx1][e11] = f1;
  FF[fx0][e01] = f0;

  FFi.push_back(FFi[f0]);
  FFi.push_back(FFi[f1]);
  FFi[f0][e02] = e01;
  FFi[f1][e12] = e11;
  FFi[fx0][e01] = e02;
  FFi[fx1][e11] = e12;

  V0.push_back((V0[u0] + V0[u1]) / 2);

  // E, EMAP
  // 4 ring
  int er02 = EMAP[f0][e02];
  int er01 = EMAP[f0][e01];
  int er11 = EMAP[f1][e11];
  int er12 = EMAP[f1][e12];
  E[er02] << fx0, e02;
  assert(EMAP[f02][FFi[fx0][e02]] == er02);
  E[er01] << f0, e01;
  E[er11] << f1, e11;
  E[er12] << fx1, e12;

  // 4 star
  assert(EMAP[f0][e0] == ue);
  EMAP[fx1][e1] = ue;

  EMAP[fx0][e02] = EMAP[f0][e02];
  EMAP[f0][e02] = E.size();
  EMAP[fx0][e01] = E.size();
  E.push_back(Eigen::Vector2i(f0, e02));

  EMAP[fx0][e0] = E.size();
  EMAP[f1][e1] = E.size();
  E.push_back(Eigen::Vector2i(fx0, e0));

  EMAP[fx1][e12] = EMAP[f1][e12];
  EMAP[f1][e12] = E.size();
  EMAP[fx1][e11] = E.size();
  E.push_back(Eigen::Vector2i(f1, e12));

  return true;
}

bool prism::edge_split(int ux, std::vector<Vec3i>& F0, std::vector<Vec3i>& FF,
                       std::vector<Vec3i>& FFi, int f0, int e0) {
  int f1 = FF[f0][e0];
  if (f1 == -1)
    return false;  // in fact, there can be another branch to deal with
                   // boundary.
  auto [e1, e01, e02, e11, e12, f01, f02, f11, f12, u0, u1, v0, v1] =
      init_e00_f00(F0, FF, FFi, f0, f1, e0);

  F0.push_back(F0[f0]);
  F0.push_back(F0[f1]);

  F0[f0][e0] = ux;
  F0[f1][e1] = ux;
  int fx0 = F0.size() - 2;
  F0[fx0][e01] = ux;
  int fx1 = F0.size() - 1;
  F0[fx1][e11] = ux;

  if (f12 != -1) {
    FF[f12][FFi[f1][e12]] = fx1;
  }
  if (f02 != -1) {
    FF[f02][FFi[f0][e02]] = fx0;
  }

  FF.push_back(FF[f0]);
  FF.push_back(FF[f1]);
  FF[f0][e02] = fx0;
  FF[f0][e0] = fx1;
  FF[f1][e12] = fx1;
  FF[f1][e1] = fx0;
  FF[fx1][e11] = f1;
  FF[fx0][e01] = f0;

  FFi.push_back(FFi[f0]);
  FFi.push_back(FFi[f1]);
  FFi[f0][e02] = e01;
  FFi[f1][e12] = e11;
  FFi[fx0][e01] = e02;
  FFi[fx1][e11] = e12;

  return true;
}

bool prism::edge_collapse(std::vector<Vec3i>& F, std::vector<Vec3i>& FF,
                          std::vector<Vec3i>& FFi, int f0, int e0) {
  int f1 = FF[f0][e0];
  if (f1 == -1) return false;
  auto [e1, e01, e02, e11, e12, f01, f02, f11, f12, u0, u1, v0, v1] =
      init_e00_f00(F, FF, FFi, f0, f1, e0);

  // get neighbor
    std::vector<std::pair<int, int>> neighbor0, neighbor1;
  get_star_edges(F, FF, FFi, f0, e0, neighbor0);
  get_star_edges(F, FF, FFi, f1, e1, neighbor1);

  std::vector<int> nv0(neighbor0.size()), nv1(neighbor1.size());
  for (int i = 0; i < neighbor0.size(); i++) {
    auto [f,e] = neighbor0[i];
    nv0[i] = F[f][(e + 1) % 3];
  }
  for (int i = 0; i < neighbor1.size(); i++) {
    auto [f,e]  = neighbor1[i];
    nv1[i] = F[f][(e + 1) % 3];
  }

  std::sort(nv0.begin(), nv0.end());
  std::sort(nv1.begin(), nv1.end());
  decltype(nv0) inter;
  std::set_intersection(nv0.begin(), nv0.end(), nv1.begin(), nv1.end(),
                        std::back_inserter(inter));
  if (inter.size() != 2) {
    return false;  // violate link condition
  }

  // remove face pair
  // change all u0 to u1
  for (auto [f,e] : neighbor0) {
    assert(F[f][e] == u0);
    F[f][e] = u1;
  }
  F[f0] = {IGL_COLLAPSE_EDGE_NULL, IGL_COLLAPSE_EDGE_NULL,
           IGL_COLLAPSE_EDGE_NULL};
  F[f1] = {IGL_COLLAPSE_EDGE_NULL, IGL_COLLAPSE_EDGE_NULL,
           IGL_COLLAPSE_EDGE_NULL};

  // FF
  FF[f0] = {-1, -1, -1};
  FF[f1] = {-1, -1, -1};
  if (f11 != -1) {
    assert(FF[f11][FFi[f1][e11]] == f1);
    FF[f11][FFi[f1][e11]] = f12;
  }
  if (f12 != -1) {
    assert(FF[f12][FFi[f1][e12]] == f1);
    FF[f12][FFi[f1][e12]] = f11;
  }
  if (f02 != -1) {
    assert(FF[f02][FFi[f0][e02]] == f0);
    FF[f02][FFi[f0][e02]] = f01;
  }
  if (f01 != -1) {
    assert(FF[f01][FFi[f0][e01]] == f0);
    FF[f01][FFi[f0][e01]] = f02;
  }

  // FFi
  if (f11 != -1) FFi[f11][FFi[f1][e11]] = FFi[f1][e12];
  if (f12 != -1) FFi[f12][FFi[f1][e12]] = FFi[f1][e11];

  if (f01 != -1) FFi[f01][FFi[f0][e01]] = FFi[f0][e02];
  if (f02 != -1) FFi[f02][FFi[f0][e02]] = FFi[f0][e01];
  FFi[f0] = {IGL_COLLAPSE_EDGE_NULL, IGL_COLLAPSE_EDGE_NULL,
             IGL_COLLAPSE_EDGE_NULL};
  FFi[f1] = {IGL_COLLAPSE_EDGE_NULL, IGL_COLLAPSE_EDGE_NULL,
             IGL_COLLAPSE_EDGE_NULL};
  return true;
}

bool prism::edge_collapse(std::vector<Vec3d>& V, std::vector<Vec3i>& F,
                          std::vector<Vec3i>& FF, std::vector<Vec3i>& FFi,
                          std::vector<Eigen::Vector2i>& E,
                          std::vector<Vec3i>& EMAP, int ue) {
  int f0 = E[ue][0], e0 = E[ue][1];
  int f1 = FF[f0][e0];
  if (f1 == -1) return false;
  auto [e1, e01, e02, e11, e12, f01, f02, f11, f12, u0, u1, v0, v1] =
      init_e00_f00(F, FF, FFi, f0, f1, e0);

  // get neighbor
  std::vector<std::pair<int, int>> neighbor0, neighbor1;
  get_star_edges(F, FF, FFi, f0, e0, neighbor0);
  get_star_edges(F, FF, FFi, f1, e1, neighbor1);

  std::vector<int> nv0(neighbor0.size()), nv1(neighbor1.size());
  for (int i = 0; i < neighbor0.size(); i++) {
    auto [f,e] = neighbor0[i];
    nv0[i] = F[f][(e + 1) % 3];
  }
  for (int i = 0; i < neighbor1.size(); i++) {
    auto [f,e] = neighbor1[i];
    nv1[i] = F[f][(e + 1) % 3];
  }

  std::sort(nv0.begin(), nv0.end());
  std::sort(nv1.begin(), nv1.end());
  decltype(nv0) inter;
  std::set_intersection(nv0.begin(), nv0.end(), nv1.begin(), nv1.end(),
                        std::back_inserter(inter));
  if (inter.size() != 2) {
    return false;  // violate link condition
  }

  // remove face pair
  // change all u0 to u1
  for (auto [f,e] : neighbor0) {
    assert(F[f][e] == u0);
    F[f][e] = u1;
  }
  F[f0] = {IGL_COLLAPSE_EDGE_NULL, IGL_COLLAPSE_EDGE_NULL,
           IGL_COLLAPSE_EDGE_NULL};
  F[f1] = {IGL_COLLAPSE_EDGE_NULL, IGL_COLLAPSE_EDGE_NULL,
           IGL_COLLAPSE_EDGE_NULL};

  // update EMAP and E
  int er1 = EMAP[f1][e12];      // connecting v1-u1
  int er0 = EMAP[f0][e01];      // connecting v0-u1
  E[er0] << f01, FFi[f0][e01];  // use FF before
  E[er1] << f12, FFi[f1][e12];
  EMAP[f02][FFi[f0][e02]] = er0;
  EMAP[f11][FFi[f1][e12]] = er1;
  EMAP[f0] = {-1, -1, -1};
  EMAP[f1] = {-1, -1, -1};

  // FF
  FF[f0] = {-1, -1, -1};
  FF[f1] = {-1, -1, -1};
  assert(FF[f11][FFi[f1][e11]] == f1);
  FF[f11][FFi[f1][e11]] = f12;
  assert(FF[f12][FFi[f1][e12]] == f1);
  FF[f12][FFi[f1][e12]] = f11;
  assert(FF[f02][FFi[f0][e02]] == f0);
  FF[f02][FFi[f0][e02]] = f01;
  assert(FF[f01][FFi[f0][e01]] == f0);
  FF[f01][FFi[f0][e01]] = f02;

  // FFi
  FFi[f11][FFi[f1][e11]] = FFi[f1][e12];
  FFi[f12][FFi[f1][e12]] = FFi[f1][e11];

  FFi[f01][FFi[f0][e01]] = FFi[f0][e02];
  FFi[f02][FFi[f0][e02]] = FFi[f0][e01];
  FFi[f0] = {IGL_COLLAPSE_EDGE_NULL, IGL_COLLAPSE_EDGE_NULL,
             IGL_COLLAPSE_EDGE_NULL};
  FFi[f1] = {IGL_COLLAPSE_EDGE_NULL, IGL_COLLAPSE_EDGE_NULL,
             IGL_COLLAPSE_EDGE_NULL};
  return true;
}
