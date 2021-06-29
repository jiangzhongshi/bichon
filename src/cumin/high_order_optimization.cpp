#include "high_order_optimization.hpp"

#include <igl/boundary_facets.h>
#include <igl/tet_tet_adjacency.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <highfive/H5Easy.hpp>
#include <prism/common.hpp>
#include <queue>
#include <optional>

#include "curve_common.hpp"
#include "curve_utils.hpp"
#include "inversion_check.hpp"

auto verts_inside_volume = [](auto &p4T) {
  RowMati bf;
  RowMati p1T = p4T.leftCols(4);
  igl::boundary_facets(p1T, bf);

  std::set<int> all_verts(p1T.data(), p1T.data() + p1T.size());
  std::set<int> sf_verts(bf.data(), bf.data() + bf.size());
  std::set<int> inside_verts;
  set_minus(all_verts, sf_verts, inside_verts);
  spdlog::debug("p4T rows={}  inside_verts={}", p4T.rows(),
                inside_verts.size());
  return inside_verts;
};

auto vertex_node_adjacency = [](int n, auto &p4T, auto &codec_n) {
  // for P4, the affected nodes of a vertex, excluding the link, which is
  // supposed to be fixed during optimization

  // Args:
  //     n (int): number of vertices (not nodes).
  //     pT (ndarray): Face/Cell Array
  //     codec_n ([type]): sum codec for the ordering (fixed lenght version)

  // Returns:
  //     list of list: nodes around each vertex one ring (exclude vertices)
  assert(codec_n.cols() == 4);  // short codec. first row (order,0,0,0)
  std::vector<std::vector<int>>
      local_affected;  // for each vert, find associated node.
  for (int i = 0; i < 4; i++) {
    local_affected.push_back({});
    for (int j = 0; j < codec_n.rows(); j++) {
      if (codec_n(j, i) > 0) {
        local_affected.back().emplace_back(j);
      }
    }
  }
  std::vector<std::vector<int>> VN(n);
  for (int i = 0; i < n; i++) VN[i].push_back(i);

  for (int ti = 0; ti < p4T.rows(); ti++) {
    for (int i = 0; i < 4; i++) {
      for (auto &a : local_affected[i]) VN[p4T(ti, i)].push_back(p4T(ti, a));
    }
  }
  for (auto &vec : VN) {
    std::sort(vec.begin(), vec.end());
    vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
  }
  return VN;
};

auto vertex_tetra_adjacency = [](int n, auto &p4T) {
  // Adjacency from vertex to (possibly high order) tetras
  //   Warning: this function assumes vertex ordered first

  //   Args:
  //       n (int): number of vertices
  //       pT (ndarray, T by node_total):
  //       dim (int, optional): Defaults to 3.

  //   Returns:
  //       list of list: tetras around each vertex
  std::vector<std::vector<int>> VT(n);
  // assert(p4T.rows() == 1067);
  for (int i = 0; i < p4T.rows(); i++) {
    for (int j = 0; j < 4; j++) {
      assert(i < p4T.rows());
      VT[p4T(i, j)].push_back(i);
    }
  }

#ifndef NDEBUG
  for (int i = 0; i < n; i++) {
    for (const auto nn : VT[i]) {
      assert(nn < p4T.rows());
    }
  }

#endif
  return VT;
};

using RowMatX3d = Eigen::Matrix<double, -1, 3, Eigen::RowMajor>;
auto mips_energy(const RowMatX3d &nodes, const std::vector<RowMatd> &dxyz,
                 bool with_grad = false) {
  using Scalar = double;
  auto mips = 0.;
  Eigen::Matrix<Scalar, -1, 3, Eigen::RowMajor> grad;
  grad.setZero(nodes.rows(), 3);
  auto cnt = 0;
  for (auto &d : dxyz) {
    Eigen::Matrix<Scalar, 3, 3> jac = d * nodes;
    if (jac.determinant() <= 0) {
      return std::tuple(Scalar(1e100), grad);
    }
    Eigen::Matrix<Scalar, 3, 3> jinv = jac.inverse();
    auto frob2 = jac.squaredNorm();
    auto invf2 = jinv.squaredNorm();
    mips += frob2 * invf2;
    if (with_grad)
      grad += (2 * invf2 * d.transpose() * jac -
               2 * frob2 * (jinv * d).transpose() * jinv * jinv.transpose());
    cnt++;
  }
  mips /= cnt;
  grad /= cnt;
  return std::tuple(mips, grad);
  // J = D X, Ji = inv(J)
  // 2*sqnorm(Ji)*D'*J - 2*sqnorm(J)*(Ji D)'*Ji*Ji'
};

constexpr auto gradient_descent = [](const auto &closure, const auto &x0,
                                     auto iter, RowMatX3d &newnode) -> int {
  auto [val0, g0] = closure(x0, true);

  // normalize g0
  if (g0.norm() > 1) g0.normalize();
  // double maxE = std::fabs(g0.maxCoeff());
  // double minE = std::fabs(g0.minCoeff());
  // double normalizeE = (maxE > minE) ? maxE : minE;
  // g0 = g0 / normalizeE;

  // find a valid step-length
  for (auto it = 0; it < iter; it++) {
    auto x1 = x0 - g0 * std::pow(0.5, it);
    auto [val1, _] = closure(x1, false);
    // spdlog::trace("val1 {}", val1);
    if (val1 < val0) {
      if (double(val0 - val1) / (val0) < 0.001) {
        // gradient descent won't help much
        return 2;
      }
      // spdlog::info("succeed {}->{}", val0, val1);
      // good solution, return
      newnode = x1;
      return 1;
    }
  }

  return 3;
};

auto e_stat = [](auto &e) {
  return fmt::format(" meanE={:.2f} minE={:.2f} maxE={:.2f} | tCnt={}",
                     e.mean(), e.minCoeff(), e.maxCoeff(), e.size());
};

namespace prism::curve {

bool InversionCheck(const RowMatd &lagr, const RowMati &p4T,
                    const RowMati &codec_fixed, const RowMati &codec9_fixed,
                    const RowMatd &bern_from_lagr_o4,
                    const RowMatd &bern_from_lagr_o9) {
  auto n_node = bern_from_lagr_o4.rows();
  RowMatX3d nodes35(n_node, 3);
  for (int i = 0; i < p4T.rows(); i++) {
    for (int j = 0; j < n_node; j++) {
      nodes35.row(j) = lagr.row(p4T(i, j));
    }
    if (!prism::curve::tetrahedron_inversion_check(
            nodes35, codec_fixed, codec9_fixed, bern_from_lagr_o4,
            bern_from_lagr_o9)) {
      spdlog::critical("Tet id = {} | p4T rows = {}", i, p4T.rows());
      return false;
    }
  }
  return true;
}

Eigen::VectorXd energy_evaluation(RowMatd &lagr, RowMati &p4T,
                                  const std::vector<RowMatd> &vec_dxyz) {
  auto elem_size = p4T.cols();
  auto total_val = 0.;
  Eigen::VectorXd energies(p4T.rows());
  energies.setZero();
  for (auto i = 0; i < p4T.rows(); i++) {
    auto t = i;
    RowMatX3d nodes35(elem_size, 3);
    for (auto j = 0; j < elem_size; j++) {
      nodes35.row(j) = lagr.row(p4T(t, j));
    }
    auto [val, grad] = mips_energy(nodes35, vec_dxyz, false);
    energies[i] = val;  /// vec_dxyz.size();
  }
  return energies;
}

void one_ring_vertex_coloring(const int n_v, const std::set<int> &inside_verts,
                              const std::vector<std::vector<int>> &VT,
                              const RowMati &p4T, std::vector<int> &colors) {
  colors.resize(n_v);
  std::fill(colors.begin(), colors.end(), -1);
  std::vector<bool> available(n_v, true);

  std::vector<int> ring;
  std::set<int>::const_iterator it = inside_verts.begin();
  colors[*it] = 0;
  ++it;

  for (; it != inside_verts.end(); ++it) {
    const int v = *it;

    ring.clear();
    for (const auto &t : VT[v]) {
      for (int j = 0; j < 4; ++j) ring.push_back(p4T(t, j));
    }
    std::sort(ring.begin(), ring.end());
    ring.erase(std::unique(ring.begin(), ring.end()), ring.end());

    for (const auto n : ring) {
      if (colors[n] != -1) available[colors[n]] = false;
    }

    int first_available_col;
    for (first_available_col = 0; first_available_col < available.size();
         first_available_col++) {
      if (available[first_available_col]) break;
    }

    assert(available[first_available_col]);

    colors[v] = first_available_col;

    for (const auto n : ring) {
      if (colors[n] != -1) available[colors[n]] = true;
    }
  }
}

void one_ring_vertex_sets(const int n_v, const std::set<int> &inside_verts,
                          const std::vector<std::vector<int>> &VT,
                          const RowMati &p4T, const int threshold,
                          std::vector<std::vector<int>> &concurrent_sets,
                          std::vector<int> &serial_set) {
  std::vector<int> colors;
  one_ring_vertex_coloring(n_v, inside_verts, VT, p4T, colors);
  int max_c = -1;
  for (const auto c : colors) max_c = std::max(max_c, c);

  concurrent_sets.clear();
  concurrent_sets.resize(max_c + 1);
  serial_set.clear();

  for (size_t i = 0; i < colors.size(); ++i) {
    const int col = colors[i];
    // skipped vertex
    if (col >= 0) concurrent_sets[col].push_back(i);
  }

  for (int i = concurrent_sets.size() - 1; i >= 0; --i) {
    if (concurrent_sets[i].size() < threshold) {
      serial_set.insert(serial_set.end(), concurrent_sets[i].begin(),
                        concurrent_sets[i].end());
      concurrent_sets.erase(concurrent_sets.begin() + i);
    }
  }
}

class ElementInQueue {
 public:
  std::array<int, 2> v_ids;
  double weight;

  ElementInQueue() {}
  ElementInQueue(const std::array<int, 2> &ids, double w)
      : v_ids(ids), weight(w) {}
};
struct cmp_s {
  bool operator()(const ElementInQueue &e1, const ElementInQueue &e2) {
    if (e1.weight == e2.weight) return e1.v_ids < e2.v_ids;
    return e1.weight > e2.weight;
  }
};
struct cmp_l {
  bool operator()(const ElementInQueue &e1, const ElementInQueue &e2) {
    if (e1.weight == e2.weight) return e1.v_ids > e2.v_ids;
    return e1.weight < e2.weight;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// lagr is unique here per nodes. not the duplicated version.
void vertex_star_smooth(RowMatd &lagr, RowMati &p4T, int total_iteration,
                        int threadNum) {
  auto &helper = prism::curve::magic_matrices();
  auto &codec_fixed = helper.volume_data.vol_codec;
  auto &vec_dxyz = helper.volume_data.vec_dxyz;
  auto &codec9_fixed = helper.volume_data.vol_jac_codec;
  auto &bern_from_lagr_o4 = helper.volume_data.vol_bern_from_lagr;
  auto &bern_from_lagr_o9 = helper.volume_data.vol_jac_bern_from_lagr;
  auto inside_verts =
      verts_inside_volume(p4T);  // exclude vertices on the boundary facet
  auto VN = vertex_node_adjacency(lagr.rows(), p4T, codec_fixed);
  const auto VT = vertex_tetra_adjacency(lagr.rows(), p4T);

  auto elem_size = p4T.cols();
  assert(4 == codec_fixed.cols());
  if (elem_size == 35) assert(4 == codec_fixed(0, 0));

  auto serious_check = [&codec_fixed, &codec9_fixed, elem_size,
                        &bern_from_lagr_o4, &bern_from_lagr_o9, &lagr,
                        &p4T](auto &newvars, auto &tetras,
                              auto &marker) {  // serious check.
    for (auto i = 0; i < tetras.size(); i++) {
      auto t = tetras[i];
      RowMatX3d nodes35(elem_size, 3);
      for (auto j = 0; j < elem_size; j++) {
        if (marker(i, j) < 0)
          nodes35.row(j) = lagr.row(p4T(t, j));
        else
          nodes35.row(j) = newvars.row(marker(i, j));  // lagr.row(p4T(t, j));
      }
      if (!prism::curve::tetrahedron_inversion_check(
              nodes35, codec_fixed, codec9_fixed, bern_from_lagr_o4,
              bern_from_lagr_o9)) {
        return false;
      }
    }
    return true;
  };

  auto one_ring_smoother = [&](int v) {
    auto &nodes = VN[v];  // free variable nodes around vertex v
    auto &tetras = VT[v];
    // marker
    auto marker = RowMati(tetras.size(), elem_size);
    RowMatX3d free_vars = RowMatX3d::Zero(nodes.size(), 3);
    marker.setConstant(-1);
    for (auto i = 0; i < tetras.size(); i++) {
      for (auto j = 0; j < elem_size; j++) {
        auto t = tetras[i];
        auto it = std::find(nodes.begin(), nodes.end(), p4T(t, j));
        if (it != nodes.end()) {
          marker(i, j) = std::distance(nodes.begin(), it);  // local index.
          free_vars.row(marker(i, j)) = lagr.row(*it);
        }
      }
    }

    auto free_num = nodes.size();

    auto closure = [free_num = nodes.size(), &tetras, &p4T, &lagr, elem_size,
                    &vec_dxyz, &marker, &codec_fixed, &codec9_fixed,
                    &bern_from_lagr_o9](const auto &x, auto with_grad) {
      RowMatX3d free_nodes_grad = RowMatX3d::Zero(free_num, 3);
      auto total_val = 0.;
      for (auto i = 0; i < tetras.size(); i++) {
        auto t = tetras[i];
        RowMatX3d nodes35(elem_size, 3);

        for (auto j = 0; j < elem_size; j++) {
          if (marker(i, j) < 0)
            nodes35.row(j) = lagr.row(p4T(t, j));
          else
            nodes35.row(j) = x.row(marker(i, j));  // lagr.row(p4T(t, j));
        }

        auto [val, grad] = mips_energy(nodes35, vec_dxyz, with_grad);
        total_val += val;
        if (with_grad)
          for (auto j = 0; j < elem_size; j++)
            if (marker(i, j) >= 0)
              free_nodes_grad.row(marker(i, j)) += grad.row(j);
      }
      return std::tuple(total_val, free_nodes_grad);
    };

    // auto newnode = std::optional<RowMatX3d>({});
    RowMatX3d newnode;
    int returnCode;
    for (auto i = 0; i < 10; i++) {
      returnCode = gradient_descent(closure, free_vars, 50, newnode);
      if (returnCode == 1) {  // on success
        free_vars = newnode;
      } else {  // either we failed to find a newnode, or the change of energy
                // is too small
        if (i == 0) {
          return false;  // gradient_descent failed
        } else {
          newnode = free_vars;
          break;  // use last round's result
        }
      }
    }

    // to next}
    auto &newvars = newnode;
    if (!serious_check(
            newvars, tetras,
            marker)) {  // expensive check, do not put into gradient descent
      return false;
    }
    for (auto nid = 0; nid < nodes.size(); nid++) {  // official assign
      lagr.row(nodes[nid]) = newvars.row(nid);
    }
    return true;
  };  // one_ring_smoother

  auto ser_energy = [&](auto name) {
    auto energy = energy_evaluation(lagr, p4T, vec_dxyz);
    auto file = H5Easy::File(name, H5Easy::File::Overwrite);
    H5Easy::dump(file, "energy", energy);
    H5Easy::dump(file, "lagr", lagr);
    H5Easy::dump(file, "cells", p4T);
  };

  // optimization loop
  std::vector<std::vector<int>> concurrent_sets;
  std::vector<int> serial_set;
  int threshold = threadNum * 2;
  if (inside_verts.empty()) {
    auto energy = energy_evaluation(lagr, p4T, vec_dxyz);
    spdlog::info("Smoothing skipped {}/{}| Energy {} | vCnt={}",
                 total_iteration, total_iteration, e_stat(energy),
                 inside_verts.size());
    return;
  }
  one_ring_vertex_sets(lagr.rows(), inside_verts, VT, p4T, threshold,
                       concurrent_sets, serial_set);
  for (int it = 0; it < total_iteration; it++) {
    for (const auto &s : concurrent_sets) {
      igl::parallel_for(
          s.size(), [&](size_t i) { one_ring_smoother(s[i]); }, 1);
    }

    for (size_t v_id : serial_set) one_ring_smoother(v_id);

    auto energy = energy_evaluation(lagr, p4T, vec_dxyz);
    spdlog::info("Smoothing it={}/{}  | Energy {} | vCnt={}", it + 1,
                 total_iteration, e_stat(energy), inside_verts.size());
  }
  // ser_energy("after.h5");  // save to file
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// A generic implementation of tetrahedral mesh operations:
// Input: local
// Output: localconnect modifier map, nodemodify map.
std::optional<
    std::tuple<std::vector<std::tuple<int, Vec3d>>,
               std::vector<Eigen::VectorXi>, std::map<int, std::vector<int>>>>
local_edit(const std::vector<Vec3d> &nodes,
           const std::vector<Eigen::VectorXi> &all_tets,
           const std::vector<int> &old_ids, const std::vector<int> &new_ids,
           const std::vector<Eigen::Vector4i> &new_tets,
           const std::vector<std::vector<int>> &connect, double old_quality) {
  using veci = Eigen::VectorXi;

  auto &helper = prism::curve::magic_matrices();
  auto &codec = helper.volume_data.vol_codec;
  auto &vec_dxyz = helper.volume_data.vec_dxyz;
  RowMati linear_tet_block(old_ids.size(), 4);
  for (auto i = 0; i < old_ids.size(); i++)
    for (auto j = 0; j < 4; j++) {
      linear_tet_block(i, j) = all_tets[old_ids[i]][j];
    }
  RowMati bnd_f;
  igl::boundary_facets(linear_tet_block, bnd_f);

  // record bnd_nodes;

  const int order = codec(0, 0);
  auto construct_key = [order](const auto &tet, const auto &cod) {
    veci key(order);
    auto cnt = 0;
    for (auto j = 0; j < 4; j++) {
      for (auto k = 0; k < cod[j]; k++) {
        key[cnt++] = tet[j];
      }
    }
    std::sort(key.data(), key.data() + key.size());
    return key;
  };

  CodecMap old_codec_nodes;  //
  CodecMap bnd_codi_map;     // map from boundary nodes to original nodes.
  std::vector<Vec3d> bnd_pos;
  for (auto i : old_ids) {
    // spdlog::info("Tet {}", all_tets[i].transpose());
    for (auto n = 0; n < codec.rows(); n++) {
      auto key = construct_key(all_tets[i], codec.row(n));
      auto it = old_codec_nodes.lower_bound(key);
      if (it == old_codec_nodes.end() || it->first != key) {
        // spdlog::info("key {}", key.transpose());
        it = old_codec_nodes.emplace_hint(it, key, all_tets[i][n]);
      }
    }
  }
  for (auto i = 0; i < bnd_f.rows(); i++) {
    //  spdlog::info("face {}", bnd_f.row(i));
    for (auto n = 0; n < codec.rows(); n++) {
      if (codec(n, 3) != 0) continue;
      auto key = construct_key(bnd_f.row(i), codec.row(n));
      auto it = bnd_codi_map.lower_bound(key);
      if (it == bnd_codi_map.end() || it->first != key) {
        // spdlog::info("cid {}", codec.row(n));
        // spdlog::info("bey {}", key.transpose());
        bnd_codi_map.emplace_hint(it, key, old_codec_nodes.at(key));
      }
    }
  }

  // compute new and record
  // tuple(t,e, node_id, node_pos)
  std::vector<std::tuple<int, int, int, Vec3d>> node_map;
  std::map<Eigen::VectorXi, std::tuple<int, Vec3d>, MatLexCompi> new_nodes;
  auto new_cutets = std::vector<veci>(new_tets.size());
  auto num_nodes = nodes.size();
  // also, a map from codec_i to pos
  auto num_ele = all_tets[0].size();
  assert(num_ele == helper.volume_data.vol_codec.rows());
  for (auto i = 0; i < new_ids.size(); i++) {
    auto tet = new_tets[i];
    new_cutets[i] = veci::Zero(num_ele);
    RowMatd local_nodes = RowMatd::Zero(num_ele, 3);
    for (auto n = 0; n < num_ele; n++) {
      auto key = construct_key(tet, codec.row(n));
      auto it = bnd_codi_map.find(key);
      if (it != bnd_codi_map
                    .end()) {  // tet rules. 1. region boundary, take directly.
        local_nodes.row(n) = nodes[it->second];
        new_cutets[i][n] = it->second;
        continue;
      }
      {  // 2. internal face, linear and record. if
         // (codec.row(n).minCoeff() == 0)
        auto it = new_nodes.lower_bound(key);
        if (it == new_nodes.end() || it->first != key) {
          for (int k = 0; k < 4; k++)
            local_nodes.row(n) += codec(n, k) * nodes[tet[k]];
          local_nodes.row(n) /= codec.row(n).sum();
          it = new_nodes.emplace_hint(
              it, key, std::tuple(num_nodes++, local_nodes.row(n)));
        }
        auto &[id, p] = it->second;
        local_nodes.row(n) = p;
        new_cutets[i][n] = id;
      }
    }
    if (local_nodes.rows() == 35) {  // to be extend to higher orders.
      local_nodes.row(34) = local_nodes.topRows(34).colwise().mean().eval();
      auto key = construct_key(tet, codec.row(34));
      std::get<1>(new_nodes[key]) = local_nodes.row(34);
    }
    if (!prism::curve::tetrahedron_inversion_check(local_nodes)) {
      spdlog::trace("Invert {}", i);
      return {};
    };
    auto [q, ignore] = mips_energy(local_nodes, vec_dxyz);
    if (std::isnan(q) || q > old_quality) {
      spdlog::trace("Quality {}", i);
      return {};
    }
  }

  // if all the checks pass. real updates to be applied
  // node positions updater
  // Three types:
  //  - boundary nodes (kept),
  //  - internal old nodes,
  //  - internal new nodes.
  auto removing_nodes = std::set<int>();
  for (auto [key, id] : old_codec_nodes) {
    if (bnd_codi_map.find(key) == bnd_codi_map.end()) removing_nodes.insert(id);
  }
  // node_mod_map = std::tuple(vec_rm_nodes, new_cutets);

  // connectivity VT
  auto relevant_vert =
      std::set<int>(linear_tet_block.data(),
                    linear_tet_block.data() + linear_tet_block.size());

  std::set<int> rel_tets, nb_tets;
  std::map<int, std::vector<int>> new_vt;
  for (auto v : relevant_vert) {
    rel_tets.insert(connect[v].begin(), connect[v].end());
    new_vt.emplace(v, std::vector<int>{});
  }
  set_minus(rel_tets, old_ids, nb_tets);
  spdlog::trace("relevant_vert {}", fmt::join(relevant_vert, ","));
  spdlog::trace("nbtets {}", fmt::join(nb_tets, ","));

  for (auto i = 0; i < new_ids.size(); i++) {
    for (auto j = 0; j < 4; j++) {
      new_vt[new_tets[i][j]].push_back(new_ids[i]);
    }
  }

  for (auto t : nb_tets)
    for (auto j = 0; j < 4; j++) {
      auto it = new_vt.find(all_tets[t][j]);
      if (it != new_vt.end()) it->second.push_back(t);
    }
  std::vector<std::tuple<int, Vec3d>> node_assigner;
  for (auto n : removing_nodes)
    node_assigner.emplace_back(
        n, Vec3d::Constant(std::numeric_limits<double>::quiet_NaN()));
  for (auto &[key, val] : new_nodes) node_assigner.emplace_back(val);
  return std::optional(std::tuple(std::move(node_assigner),
                                  std::move(new_cutets), std::move(new_vt)));
};

int cutet_collapse(RowMatd &lagr, RowMati &p4T, double stop_energy) {
  auto &helper = prism::curve::magic_matrices();
  auto &codecs_o4 = helper.volume_data.vol_codec;
  auto &vec_dxyz = helper.volume_data.vec_dxyz;
  auto &codecs_o9 = helper.volume_data.vol_jac_codec;
  auto &bern_from_lagr_o4 = helper.volume_data.vol_bern_from_lagr;
  auto &bern_from_lagr_o9 = helper.volume_data.vol_jac_bern_from_lagr;

  auto inside_verts = verts_inside_volume(p4T);

  auto n_node = codecs_o4.rows();
  if (n_node != p4T.cols()) {
    spdlog::critical("tet order mismatch");
    throw 1;
  }

  std::vector<Vec3d> vertices(lagr.rows());
  std::vector<Eigen::VectorXi> tets(p4T.rows());
  for (int i = 0; i < lagr.rows(); i++) {
    vertices[i] = lagr.row(i);
  }
  for (int i = 0; i < p4T.rows(); i++) {
    tets[i] = p4T.row(i);
  }

  // conn_tets[i] = j -> i-th vertex connected with j-th tet
  std::vector<std::vector<int>> conn_tets(vertices.size());
  for (int i = 0; i < tets.size(); i++) {
    for (int j = 0; j < 4; j++) conn_tets[tets[i][j]].push_back(i);
  }
  auto find = [](auto &arr, auto &val) {
    for (auto j = 0; j < arr.size(); j++)
      if (arr[j] == val) return j;
    return -1;
  };
  assert([&]() -> bool {
    for (auto i = 0; i < tets.size(); i++)
      for (auto j = 0; j < 4; j++) {
        auto v = tets[i][j];
        if (find(conn_tets[v], i) == -1) {
          spdlog::critical("wrong conn ({}):{}", v,
                           fmt::join(conn_tets[v], ","));
          return false;
        }
      }
    return true;
  }());
  //
  std::vector<bool> v_is_removed(vertices.size(), false);
  std::vector<bool> t_is_removed(tets.size(), false);
  //
  std::vector<bool> is_surface_vs(vertices.size(), true);
  for (int v_id : inside_verts) is_surface_vs[v_id] = false;
  //
  std::vector<std::array<int, 2>> edges;
  for (int i = 0; i < tets.size(); i++) {
    const auto &t = tets[i];
    for (int j = 0; j < 3; j++) {
      std::array<int, 2> e = {{t[0], t[j + 1]}};
      if (e[0] > e[1]) std::swap(e[0], e[1]);
      edges.push_back(e);
      e = {{t[j + 1], t[(j + 1) % 3 + 1]}};
      if (e[0] > e[1]) std::swap(e[0], e[1]);
      edges.push_back(e);
    }
  }
  std::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

  std::priority_queue<ElementInQueue, std::vector<ElementInQueue>, cmp_s>
      ec_queue;
  for (auto &e : edges) {
    double l_2 = (vertices[e[0]] - vertices[e[1]]).norm();
    if (!is_surface_vs[e[0]]) ec_queue.push(ElementInQueue(e, l_2));
    if (!is_surface_vs[e[1]])
      ec_queue.push(ElementInQueue({{e[1], e[0]}}, l_2));
  }
  edges.clear();

  // Collapsing starts
  auto cnt_suc = 0;
  while (!ec_queue.empty()) {
    auto [e, old_weight] = ec_queue.top();
    ec_queue.pop();
    //
    if (v_is_removed[e[0]] || v_is_removed[e[1]]) continue;
    if (is_surface_vs[e[0]]) continue;
    double weight = (vertices[e[0]] - vertices[e[1]]).norm();
    if (weight != old_weight) continue;

    // try to collapse an edge
    auto [v1_id, v2_id] = e;
    spdlog::debug("Entering {}-{}", v1_id, v2_id);
    std::vector<int> old_tids = conn_tets[v1_id];
    std::vector<Eigen::Vector4i> new_tets;
    std::vector<int> new_tids = {};

    for (auto t : old_tids) {
      Eigen::Vector4i line_tet = tets[t].head(4);
      auto v1_i = find(line_tet, v1_id);
      auto v2_i = find(line_tet, v2_id);
      assert(v1_i >= 0);
      if (v2_i >= 0) continue;
      line_tet[v1_i] = v2_id;
      new_tets.emplace_back(line_tet);
      new_tids.push_back(t);
    }
    assert(new_tids.size() < old_tids.size());
    // link condition
    std::set<int> inter;
    std::set_intersection(conn_tets[v1_id].begin(), conn_tets[v1_id].end(),
                          conn_tets[v2_id].begin(), conn_tets[v2_id].end(),
                          std::inserter(inter, inter.end()));
    if (inter.size() != old_tids.size() - new_tids.size()) {
      spdlog::info("lk condition");
      continue;
    }

    double max_energy = stop_energy;  // max_energy of old tets
    for (auto t_id : old_tids) {
      RowMatX3d old_nodes35(n_node, 3);
      for (auto j = 0; j < codecs_o4.rows(); j++) {
        old_nodes35.row(j) = vertices[tets[t_id][j]];
      }
      auto [q, ign] = mips_energy(old_nodes35, vec_dxyz);
      max_energy = std::max(max_energy, q);
    }

    auto is_valid = local_edit(vertices, tets, old_tids, new_tids, new_tets,
                               conn_tets, max_energy);
    if (!is_valid) continue;
    spdlog::debug("[{}] Success {} {}", cnt_suc, v1_id, v2_id);
    cnt_suc++;
    auto &[node_assigner, new_cutets, new_vt] = is_valid.value();
    assert([&new_vt = new_vt]() -> bool {
      for (auto [v, vt] : new_vt) {
        if (vt.size() == 0) return true;
      }
      return false;
    }() && "V1 should be in the relevant list where vt is now empty.");

    /////////////////
    // real update //
    /////////////////
    // get new edges
    spdlog::trace("old_tids {}", fmt::join(old_tids, ","));
    spdlog::trace("new tids {}", fmt::join(new_tids, ","));
    auto n_v1_id = std::set<int>();
    for (auto t_id: old_tids) {
      for (int j = 0; j < 4; j++) {
          n_v1_id.insert(tets[t_id][j]);
      }
    }
    for (auto t_id:old_tids) {
      if (find(tets[t_id], v1_id) >= 0 && find(tets[t_id], v2_id) >= 0) { // removers.
       for (int j = 0; j < 4; j++)
        n_v1_id.erase(tets[t_id][j]);
      }
    }
    n_v1_id.erase(v1_id);
    n_v1_id.erase(v2_id);
    for (auto &[n, p] : node_assigner) {
      if (n >= vertices.size()) {
        vertices.resize(n + 1);
        v_is_removed.resize(n + 1, false);
      }
      if (p.hasNaN()) {
        v_is_removed[n] = true;
      }
      vertices[n] = p;
    }
    for (auto t : old_tids) {
      tets[t].setConstant(-1);
      t_is_removed[t] = true;
    }
    for (auto i = 0; i < new_tids.size(); i++) {
      tets[new_tids[i]] = new_cutets[i];
      t_is_removed[new_tids[i]] = false;
    }
    for (auto [v, vt] : new_vt) {
      if (vt.size() == 0) v_is_removed[v] = true;
      std::sort(vt.begin(), vt.end());
      conn_tets[v] = std::move(vt);
    }
    assert([&]() -> bool {
      for (auto i = 0; i < tets.size(); i++)
        for (auto j = 0; j < 4; j++) {
          auto v = tets[i][j];
          if (v == -1) continue;
          if (find(conn_tets[v], i) == -1) {
            spdlog::critical("wrong conn i={}, VT(v{}):{}", i, v,
                             fmt::join(conn_tets[v], ","));
            return false;
          }
        }
      return true;
    }());
    auto nb_v = std::set<int>();
    for (auto t:conn_tets[v2_id]) {
        for (auto j=0;j<4; j++) {
          nb_v.insert(tets[t][j]);
        }
    }
    nb_v.erase(v2_id);
    spdlog::debug("nv1_id {}", fmt::join(n_v1_id,"."));
    for (auto v_id : n_v1_id) {
      double l_2 = (vertices[v_id] - vertices[v2_id]).norm();
      if (!is_surface_vs[v_id])
        ec_queue.push(ElementInQueue({{v_id, v2_id}}, l_2));
      if (!is_surface_vs[v2_id])
        ec_queue.push(ElementInQueue({{v2_id, v_id}}, l_2));
    }
    assert(v_is_removed.size() == vertices.size());
    assert(t_is_removed.size() == tets.size());
    assert(([&]() -> bool {  // check energy
      for (auto i=0; i<tets.size(); i++) {
        auto t = tets[i];
        if (t_is_removed[i]) continue;
          RowMatX3d nodes35(t.size(), 3);
          for (auto j = 0; j < t.size(); j++) {
            nodes35.row(j) = vertices[t[j]];
          }
          auto [val, ign] = mips_energy(nodes35, vec_dxyz, false);
          if (std::isnan(val)) {
            spdlog::critical("PostNAN!!!");
            throw 1;
          }
          if (val > 1e8) {
            spdlog::critical("large energy");
            throw 1;
          }
      }
      return true;
    }()));
  }

  RowMatd lagr_new(std::count(v_is_removed.begin(), v_is_removed.end(), false),
                   3);
  RowMati p4T_new(std::count(t_is_removed.begin(), t_is_removed.end(), false),
                  n_node);
  std::vector<int> map_v_ids(vertices.size(), -1);
  int cnt = 0;
  for (int i = 0; i < vertices.size(); i++) {
    if (v_is_removed[i]) continue;
    lagr_new.row(cnt) = vertices[i];
    map_v_ids[i] = cnt;
    cnt++;
  }
  int cnt_t = 0;
  for (int i = 0; i < tets.size(); i++) {
    if (t_is_removed[i]) continue;
    assert(tets[i][0] != -1);
    for (int j = 0; j < tets[i].size(); j++) {
      p4T_new(cnt_t, j) = map_v_ids[tets[i][j]];
    }
    cnt_t++;
  }

  lagr = lagr_new;
  p4T = p4T_new;
  auto energy = energy_evaluation(lagr, p4T, vec_dxyz);
  spdlog::info("{} edges collapsed | Energy {}", cnt_suc, e_stat(energy));
  return cnt_suc;
}

int cutet_swap(RowMatd &lagr, RowMati &p4T, double stop_energy) {
  auto set_intersection = [](const std::vector<int> &s11,
                             const std::vector<int> &s22, std::vector<int> &v) {
    std::vector<int> s1 = s11;
    std::vector<int> s2 = s22;
    std::sort(s1.begin(), s1.end());
    std::sort(s2.begin(), s2.end());
    std::set_intersection(s1.begin(), s1.end(), s2.begin(), s2.end(),
                          std::back_inserter(v));
  };

  /////////////////////////////
  auto &helper = prism::curve::magic_matrices();
  auto &codecs_o4 = helper.volume_data.vol_codec;
  auto &vec_dxyz = helper.volume_data.vec_dxyz;
  auto &codecs_o9 = helper.volume_data.vol_jac_codec;
  auto &bern_from_lagr_o4 = helper.volume_data.vol_bern_from_lagr;
  auto &bern_from_lagr_o9 = helper.volume_data.vol_jac_bern_from_lagr;

  auto n_node = codecs_o4.rows();
  auto inside_verts = verts_inside_volume(p4T);
  std::vector<Vec3d> vertices(lagr.rows());
  std::vector<Eigen::VectorXi> tets(p4T.rows());
  for (int i = 0; i < lagr.rows(); i++) {
    vertices[i] = lagr.row(i);
  }
  for (int i = 0; i < p4T.rows(); i++) {
    tets[i] = p4T.row(i);
  }
  // conn_tets[i] = j -> i-th vertex connected with j-th tet
  std::vector<std::vector<int>> conn_tets(vertices.size());
  for (int i = 0; i < tets.size(); i++) {
    for (int j = 0; j < 4; j++) conn_tets[tets[i][j]].push_back(i);
  }
  //
  std::vector<bool> v_is_removed(vertices.size(), false);
  std::vector<bool> t_is_removed(tets.size(), false);
  //
  std::vector<bool> is_surface_vs(vertices.size(), true);
  for (int v_id : inside_verts) is_surface_vs[v_id] = false;
  //
  std::vector<std::array<int, 2>> edges;
  for (int i = 0; i < tets.size(); i++) {
    const auto &t = tets[i];
    for (int j = 0; j < 3; j++) {
      std::array<int, 2> e = {{t[0], t[j + 1]}};
      if (e[0] > e[1]) std::swap(e[0], e[1]);
      edges.push_back(e);
      e = {{t[j + 1], t[(j + 1) % 3 + 1]}};
      if (e[0] > e[1]) std::swap(e[0], e[1]);
      edges.push_back(e);
    }
  }
  std::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

  std::priority_queue<ElementInQueue, std::vector<ElementInQueue>, cmp_l>
      es_queue;
  for (auto &e : edges) {
    if (is_surface_vs[e[0]] && is_surface_vs[e[1]]) {
      std::vector<int> n12_t_ids;
      set_intersection(conn_tets[e[0]], conn_tets[e[1]], n12_t_ids);
      std::vector<int> n_v_ids;
      for (int t_id : n12_t_ids) {
        for (int j = 0; j < 4; j++) {
          if (tets[t_id][j] != e[0] && tets[t_id][j] != e[1])
            n_v_ids.push_back(tets[t_id][j]);
        }
      }
      std::sort(n_v_ids.begin(), n_v_ids.end());
      n_v_ids.erase(std::unique(n_v_ids.begin(), n_v_ids.end()), n_v_ids.end());
      if (n12_t_ids.size() != n_v_ids.size()) continue;
    }
    double l_2 = (vertices[e[0]] - vertices[e[1]]).norm();
    es_queue.push(ElementInQueue(e, l_2));
  }
  edges.clear();

  int cnt_suc = 0;
  while (!es_queue.empty()) {
    auto [e, old_weight] = es_queue.top();
    es_queue.pop();
    auto [v1_id, v2_id] = e;

    std::vector<int> old_tids;
    set_intersection(conn_tets[e[0]], conn_tets[e[1]], old_tids);
    if (old_tids.size() != 3)
      continue;  // only enables 3-2 swap https://i.imgur.com/zcmFleu.png

    auto id_in_array = [](auto &v, auto &k) {
      for (auto i = 0; i < v.size(); i++) {
        if (v[i] == k) return i;
      }
      return -1;
    };

    auto t0_id = old_tids[0];
    int t1_id = old_tids[1];
    int t2_id = old_tids[2];
    std::array<int, 2> t12_ids = {{t1_id, t2_id}};
    auto n0_id = -1, n1_id = -1, n2_id = -1;
    for (int j = 0; j < 4; j++) {
      auto v0j = tets[t0_id][j];
      if (v0j != v1_id && v0j != v2_id) {
        if (id_in_array(tets[t1_id], v0j) != -1)
          n1_id = v0j;
        if (id_in_array(tets[t2_id], v0j) != -1)
          n2_id = v0j;
      }
      if (id_in_array(tets[t0_id], tets[t1_id][j]) == -1)
        n0_id = tets[t1_id][j];
    }
    assert (n0_id != n1_id && n1_id != n2_id);
    // T0 = (n1,n2,v1,v2) -> (n1,n2,v1,n0)
    // T1 = (n0, n1, v1,v2) ->  (n0, n1, n2,v2) 
    // T2 = (n0,n2, v1,v2) -> (-1,-1,-1,-1)
    auto new_tids = std::vector<int>({t0_id, t1_id});
    auto new_tets = std::vector<Eigen::Vector4i>(2);
    auto replace = [](auto& arr, auto v0, auto v1){
      for (auto j=0 ;j<arr.size();j++) if (arr[j]==v0) arr[j] = v1;
    };
    new_tets[0] = tets[new_tids[0]].head(4);
    new_tets[1] = tets[new_tids[1]].head(4);

    replace(new_tets[0], v2_id, n0_id);
    replace(new_tets[1], v1_id, n2_id);

    double max_energy = stop_energy;
    for (int t_id : old_tids) {
      RowMatX3d old_nodes35(n_node, 3);
      for (int j = 0; j < codecs_o4.rows(); j++) {
        old_nodes35.row(j) = vertices[tets[t_id][j]];
      }
      std::tuple<double, RowMatX3d> old_q = mips_energy(old_nodes35, vec_dxyz);
      if (std::get<0>(old_q) > max_energy) max_energy = std::get<0>(old_q);
    }
    auto is_valid = local_edit(vertices, tets, old_tids, new_tids, new_tets, conn_tets, max_energy);
    if (!is_valid) continue;
    spdlog::debug("[{}] Success {} {}", cnt_suc, v1_id, v2_id);
    cnt_suc++;
    auto &[node_assigner, new_cutets, new_vt] = is_valid.value();
    // update
     for (auto &[n, p] : node_assigner) {
      if (n >= vertices.size()) {
        vertices.resize(n + 1);
        v_is_removed.resize(n + 1, false);
      }
      if (p.hasNaN()) {
        v_is_removed[n] = true;
      }
      vertices[n] = p;
    }
    for (auto t : old_tids) {
      tets[t].setConstant(-1);
      t_is_removed[t] = true;
    }
    for (auto i = 0; i < new_tids.size(); i++) {
      tets[new_tids[i]] = new_cutets[i];
      t_is_removed[new_tids[i]] = false;
    }
    for (auto [v, vt] : new_vt) {
      if (vt.size() == 0) v_is_removed[v] = true;
      std::sort(vt.begin(), vt.end());
      conn_tets[v] = std::move(vt);
    }
  }

  RowMatd lagr_new(std::count(v_is_removed.begin(), v_is_removed.end(), false),
                   3);
  RowMati p4T_new(std::count(t_is_removed.begin(), t_is_removed.end(), false),
                  n_node);
  std::vector<int> map_v_ids(vertices.size(), -1);
  int cnt = 0;
  for (int i = 0; i < vertices.size(); i++) {
    if (v_is_removed[i]) continue;
    lagr_new.row(cnt) = vertices[i];
    map_v_ids[i] = cnt;
    cnt++;
  }
  int cnt_t = 0;
  for (int i = 0; i < tets.size(); i++) {
    if (t_is_removed[i]) continue;
    for (int j = 0; j < tets[i].size(); j++) {
      p4T_new(cnt_t, j) = map_v_ids[tets[i][j]];
    }
    cnt_t++;
  }
  p4T = p4T_new;
  lagr = lagr_new;
  auto energy = energy_evaluation(lagr, p4T, vec_dxyz);
  spdlog::info("{} edges swapped   | Energy {}", cnt_suc, e_stat(energy));
  return cnt_suc;
};


int edge_collapsing(RowMatd &lagr, RowMati &p4T, double stop_energy) {
  auto &helper = prism::curve::magic_matrices();
  auto &codecs_o4 = helper.volume_data.vol_codec;
  auto &vec_dxyz = helper.volume_data.vec_dxyz;
  auto &codecs_o9 = helper.volume_data.vol_jac_codec;
  auto &bern_from_lagr_o4 = helper.volume_data.vol_bern_from_lagr;
  auto &bern_from_lagr_o9 = helper.volume_data.vol_jac_bern_from_lagr;

  auto inside_verts = verts_inside_volume(p4T);

  auto n_node = codecs_o4.rows();
  if (n_node != p4T.cols()) {
    spdlog::critical("tet order mismatch");
    throw 1;
  }

  std::vector<Vec3d> vertices(lagr.rows());
  std::vector<Eigen::VectorXi> tets(p4T.rows());
  for (int i = 0; i < lagr.rows(); i++) {
    vertices[i] = lagr.row(i);
  }
  for (int i = 0; i < p4T.rows(); i++) {
    tets[i] = p4T.row(i);
  }

  // conn_tets[i] = j -> i-th vertex connected with j-th tet
  std::vector<std::vector<int>> conn_tets(vertices.size());
  for (int i = 0; i < tets.size(); i++) {
    for (int j = 0; j < 4; j++) conn_tets[tets[i][j]].push_back(i);
  }
  //
  std::vector<bool> v_is_removed(vertices.size(), false);
  std::vector<bool> t_is_removed(tets.size(), false);
  //
  std::vector<bool> is_surface_vs(vertices.size(), true);
  for (int v_id : inside_verts) is_surface_vs[v_id] = false;
  //
  std::vector<std::array<int, 2>> edges;
  for (int i = 0; i < tets.size(); i++) {
    const auto &t = tets[i];
    for (int j = 0; j < 3; j++) {
      std::array<int, 2> e = {{t[0], t[j + 1]}};
      if (e[0] > e[1]) std::swap(e[0], e[1]);
      edges.push_back(e);
      e = {{t[j + 1], t[(j + 1) % 3 + 1]}};
      if (e[0] > e[1]) std::swap(e[0], e[1]);
      edges.push_back(e);
    }
  }
  std::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

  std::priority_queue<ElementInQueue, std::vector<ElementInQueue>, cmp_s>
      ec_queue;
  for (auto &e : edges) {
    double l_2 = (vertices[e[0]] - vertices[e[1]]).norm();
    if (!is_surface_vs[e[0]]) ec_queue.push(ElementInQueue(e, l_2));
    if (!is_surface_vs[e[1]])
      ec_queue.push(ElementInQueue({{e[1], e[0]}}, l_2));
  }
  edges.clear();

  // Collapsing starts
  auto cnt_suc = 0;
  while (!ec_queue.empty()) {
    auto [e, old_weight] = ec_queue.top();
    ec_queue.pop();
    //
    if (v_is_removed[e[0]] || v_is_removed[e[1]]) continue;
    if (is_surface_vs[e[0]]) continue;
    double weight = (vertices[e[0]] - vertices[e[1]]).norm();
    if (weight != old_weight) continue;

    // try to collapse an edge
    auto [v1_id, v2_id] = e;
    spdlog::debug("Entering {}-{}", v1_id, v2_id);
    std::vector<int> check_t_ids;
    std::vector<int> rm_t_ids;            // those containing (v1, v2)
    for (auto t_id : conn_tets[v1_id]) {  // for all tets connected with v1,
                                          // either check or delete
      if (std::find(conn_tets[v2_id].begin(), conn_tets[v2_id].end(), t_id) ==
          conn_tets[v2_id].end())  // if t_id not adjacent to v2
        check_t_ids.push_back(t_id);
      else
        rm_t_ids.push_back(t_id);  // remove all tets with v1-v2.
    }

    auto map_node_ids = [&rm_t_ids, &tets, &codecs_o4, v1_id = v1_id,
                         v2_id = v2_id]() {
      // maps from
      std::map<int, int> map_node_ids;
      for (int i = 0; i < rm_t_ids.size(); i++) {
        int t_id = rm_t_ids[i];
        auto v1_id_j = -1, v2_id_j = -1;
        for (int j = 0; j < 4; j++) {
          if (tets[t_id][j] == v1_id) v1_id_j = j;
          if (tets[t_id][j] == v2_id) v2_id_j = j;
        }
        assert(v1_id_j != -1 && v2_id_j != -1);
        std::map<std::array<int, 4>, int> local_map;  // codec -> node_id
        for (int j = 0; j < codecs_o4.rows(); j++) {
          local_map[{{codecs_o4(j, 0), codecs_o4(j, 1), codecs_o4(j, 2),
                      codecs_o4(j, 3)}}] = tets[t_id][j];
        }

        for (int j = 0; j < codecs_o4.rows(); j++) {
          if (codecs_o4(j, v1_id_j) != 0 &&
              codecs_o4(j, v2_id_j) ==
                  0) {  // ho-nodes that only depend on v1, oppo faces to v2.
            std::array<int, 4> code_j = {{codecs_o4(j, 0), codecs_o4(j, 1),
                                          codecs_o4(j, 2), codecs_o4(j, 3)}};
            code_j[v2_id_j] = code_j[v1_id_j];
            code_j[v1_id_j] = 0;  // exchange of bc between v1-v2
            int tmp_v2_id = local_map[code_j];
            map_node_ids[tets[t_id][j]] = tmp_v2_id;
          }
        }
      }
      return std::move(map_node_ids);
    }();

    std::map<int, std::pair<int, Vec3d>> map_node_pos;
    auto is_valid = true;
    double max_energy = 0;  // max_energy of old tets
    for (auto t_id : check_t_ids) {
      RowMatX3d old_nodes35(n_node, 3);
      for (auto j = 0; j < codecs_o4.rows(); j++) {
        old_nodes35.row(j) = vertices[tets[t_id][j]];
      }
      std::tuple<double, RowMatX3d> old_q = mips_energy(old_nodes35, vec_dxyz);
      if (std::get<0>(old_q) > max_energy) max_energy = std::get<0>(old_q);
    }

    for (auto i = 0; i < check_t_ids.size(); i++) {
      auto t_id = check_t_ids[i];
      std::array<Vec3d, 4>
          new_vs;        // vertex positions of the new constructed tets.
      auto v1_id_k = 0;  // tets[t_id][v1_id_k] == v1_id
      for (auto j = 0; j < 4; j++) {
        if (tets[t_id][j] == v1_id) {  // coord(v1) = v2
          new_vs[j] = vertices[v2_id];
          v1_id_k = j;
        } else
          new_vs[j] = vertices[tets[t_id][j]];
      }
      RowMatX3d nodes35 = RowMatX3d::Zero(n_node, 3);  // n_elem*3
      for (int j = 0; j < codecs_o4.rows(); j++) {
        if (codecs_o4(j, v1_id_k) ==
            0) {  // no contribution from v1 (boundary), use nodes directly.
          nodes35.row(j) = vertices[tets[t_id][j]];
          continue;
        }
        // otherwise, assigning positions based on linear lifting.
        for (int k = 0; k < 4; k++)
          nodes35.row(j) += codecs_o4(j, k) * new_vs[k];
        nodes35.row(j) /= codecs_o4.row(j).sum();
        map_node_pos[tets[t_id][j]] = std::make_pair(-1, nodes35.row(j));
      }
      auto &cur_tet = tets[t_id];
      // for the nodes in map_node_ids, record in map_node_pos.
      for (auto j = 0; j < codecs_o4.rows(); j++) {
        if (map_node_ids.find(cur_tet[j]) != map_node_ids.end()) {
          int mapped_n_id = map_node_ids[cur_tet[j]];
          nodes35.row(j) = vertices[mapped_n_id];
          map_node_pos[cur_tet[j]] =
              std::make_pair(mapped_n_id, nodes35.row(j));
        }
      }
      // the interior position is assigned the average of all.
      Vec3d mid_p(0, 0, 0);
      for (int j = 0; j < nodes35.rows() - 1; j++) {
        mid_p += nodes35.row(j);
      }

      nodes35.row(34) = mid_p / (nodes35.rows() - 1);
      map_node_pos[tets[t_id][34]] = std::make_pair(-1, nodes35.row(34));

      if (!prism::curve::tetrahedron_inversion_check(
              nodes35, codecs_o4, codecs_o9, bern_from_lagr_o4,
              bern_from_lagr_o9)) {
        is_valid = false;
        break;
      }
      std::tuple<double, RowMatX3d> q = mips_energy(nodes35, vec_dxyz);
      if (std::get<0>(q) > std::max(stop_energy, max_energy)) {
        // if (std::get<0>(q) > stop_energy) {
        is_valid = false;
        break;
      }
    }
    if (!is_valid) continue;
    spdlog::debug("[{}] Success {} {}", cnt_suc, v1_id, v2_id);
    cnt_suc++;

    /////////////////
    // real update //
    /////////////////
    // get new edges
    std::set<int> n_v1_id;
    for (int t_id : check_t_ids) {
      for (int j = 0; j < 4; j++) {
        if (tets[t_id][j] != v1_id) n_v1_id.insert(tets[t_id][j]);
      }
    }
    for (int t_id : rm_t_ids) {
      for (int j = 0; j < 4; j++) {
        if (tets[t_id][j] != v1_id && tets[t_id][j] != v2_id) {
          n_v1_id.erase(tets[t_id][j]);
        }
      }
    }
    // update
    for (int t_id : rm_t_ids) {
      t_is_removed[t_id] = true;
      int v1_id_j = -1;
      for (int j = 0; j < 4; j++) {
        if (tets[t_id][j] == v1_id) v1_id_j = j;
        //
        conn_tets[tets[t_id][j]].erase(
            std::find(conn_tets[tets[t_id][j]].begin(),
                      conn_tets[tets[t_id][j]].end(), t_id));
      }
      for (int j = 0; j < tets[t_id].size(); j++) {
        if (codecs_o4(j, v1_id_j) != 0) v_is_removed[tets[t_id][j]] = true;
      }
    }

    for (int t_id : check_t_ids) {
      for (int j = 0; j < tets[t_id].size(); j++) {
        if (map_node_pos.find(tets[t_id][j]) != map_node_pos.end()) {
          int new_v_id = map_node_pos[tets[t_id][j]].first;
          if (new_v_id >= 0) {
            tets[t_id][j] = new_v_id;
          } else {
            auto pos = map_node_pos[tets[t_id][j]].second;
            vertices[tets[t_id][j]] = pos;
          }
        }
      }
      conn_tets[v2_id].push_back(t_id);
    }
    spdlog::debug("nv1_id {}", fmt::join(n_v1_id,"."));
    for (int v_id : n_v1_id) {
      double l_2 = (vertices[v_id] - vertices[v2_id]).norm();
      if (!is_surface_vs[v_id])
        ec_queue.push(ElementInQueue({{v_id, v2_id}}, l_2));
      if (!is_surface_vs[v2_id])
        ec_queue.push(ElementInQueue({{v2_id, v_id}}, l_2));
    }
  }

  RowMatd lagr_new(std::count(v_is_removed.begin(), v_is_removed.end(), false),
                   3);
  RowMati p4T_new(std::count(t_is_removed.begin(), t_is_removed.end(), false),
                  n_node);
  std::vector<int> map_v_ids(vertices.size(), -1);
  int cnt = 0;
  for (int i = 0; i < vertices.size(); i++) {
    if (v_is_removed[i]) continue;
    lagr_new.row(cnt) = vertices[i];
    map_v_ids[i] = cnt;
    cnt++;
  }
  int cnt_t = 0;
  for (int i = 0; i < tets.size(); i++) {
    if (t_is_removed[i]) continue;
    for (int j = 0; j < tets[i].size(); j++) {
      p4T_new(cnt_t, j) = map_v_ids[tets[i][j]];
    }
    cnt_t++;
  }

  // auto ofile = H5Easy::File(in_file + "_out.h5", H5Easy::File::Overwrite);
  // H5Easy::dump(ofile, "lagr", lagr_new);
  // H5Easy::dump(ofile, "cells", p4T_new);
  lagr = lagr_new;
  p4T = p4T_new;
  auto energy = energy_evaluation(lagr, p4T, vec_dxyz);
  spdlog::info("{} edges collapsed | Energy {}", cnt_suc, e_stat(energy));
  return cnt_suc;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int edge_swapping(RowMatd &lagr, RowMati &p4T, double stop_energy) {
  auto set_intersection = [](const std::vector<int> &s11,
                             const std::vector<int> &s22, std::vector<int> &v) {
    std::vector<int> s1 = s11;
    std::vector<int> s2 = s22;
    std::sort(s1.begin(), s1.end());
    std::sort(s2.begin(), s2.end());
    std::set_intersection(s1.begin(), s1.end(), s2.begin(), s2.end(),
                          std::back_inserter(v));
  };

  /////////////////////////////
  auto &helper = prism::curve::magic_matrices();
  auto &codecs_o4 = helper.volume_data.vol_codec;
  auto &vec_dxyz = helper.volume_data.vec_dxyz;
  auto &codecs_o9 = helper.volume_data.vol_jac_codec;
  auto &bern_from_lagr_o4 = helper.volume_data.vol_bern_from_lagr;
  auto &bern_from_lagr_o9 = helper.volume_data.vol_jac_bern_from_lagr;

  auto n_node = codecs_o4.rows();
  auto inside_verts = verts_inside_volume(p4T);
  std::vector<Vec3d> vertices(lagr.rows());
  std::vector<Eigen::VectorXi> tets(p4T.rows());
  for (int i = 0; i < lagr.rows(); i++) {
    vertices[i] = lagr.row(i);
  }
  for (int i = 0; i < p4T.rows(); i++) {
    tets[i] = p4T.row(i);
  }
  // conn_tets[i] = j -> i-th vertex connected with j-th tet
  std::vector<std::vector<int>> conn_tets(vertices.size());
  for (int i = 0; i < tets.size(); i++) {
    for (int j = 0; j < 4; j++) conn_tets[tets[i][j]].push_back(i);
  }
  //
  std::vector<bool> v_is_removed(vertices.size(), false);
  std::vector<bool> t_is_removed(tets.size(), false);
  //
  std::vector<bool> is_surface_vs(vertices.size(), true);
  for (int v_id : inside_verts) is_surface_vs[v_id] = false;
  //
  std::vector<std::array<int, 2>> edges;
  for (int i = 0; i < tets.size(); i++) {
    const auto &t = tets[i];
    for (int j = 0; j < 3; j++) {
      std::array<int, 2> e = {{t[0], t[j + 1]}};
      if (e[0] > e[1]) std::swap(e[0], e[1]);
      edges.push_back(e);
      e = {{t[j + 1], t[(j + 1) % 3 + 1]}};
      if (e[0] > e[1]) std::swap(e[0], e[1]);
      edges.push_back(e);
    }
  }
  std::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

  std::priority_queue<ElementInQueue, std::vector<ElementInQueue>, cmp_l>
      es_queue;
  for (auto &e : edges) {
    if (is_surface_vs[e[0]] && is_surface_vs[e[1]]) {
      std::vector<int> n12_t_ids;
      set_intersection(conn_tets[e[0]], conn_tets[e[1]], n12_t_ids);
      std::vector<int> n_v_ids;
      for (int t_id : n12_t_ids) {
        for (int j = 0; j < 4; j++) {
          if (tets[t_id][j] != e[0] && tets[t_id][j] != e[1])
            n_v_ids.push_back(tets[t_id][j]);
        }
      }
      std::sort(n_v_ids.begin(), n_v_ids.end());
      n_v_ids.erase(std::unique(n_v_ids.begin(), n_v_ids.end()), n_v_ids.end());
      if (n12_t_ids.size() != n_v_ids.size()) continue;
    }
    double l_2 = (vertices[e[0]] - vertices[e[1]]).norm();
    es_queue.push(ElementInQueue(e, l_2));
  }
  edges.clear();

  auto map_nodes = [&](int t_id, const std::array<int, 3> &v_ids,
                       int empty_v_id,
                       std::map<std::array<int, 3>, int> &map_node_coords) {
    auto empty_v_id_j = -1;
    std::array<int, 3> v_ids_j = {-1, -1, -1};
    for (int j = 0; j < 4; j++) {
      if (tets[t_id][j] == empty_v_id)
        empty_v_id_j = j;
      else if (tets[t_id][j] == v_ids[0])
        v_ids_j[0] = j;
      else if (tets[t_id][j] == v_ids[1])
        v_ids_j[1] = j;
      else if (tets[t_id][j] == v_ids[2])
        v_ids_j[2] = j;
    }
    for (int j = 0; j < n_node; j++) {
      if (codecs_o4(j, empty_v_id_j) == 0) {
        map_node_coords[{{codecs_o4(j, v_ids_j[0]), codecs_o4(j, v_ids_j[1]),
                          codecs_o4(j, v_ids_j[2])}}] = tets[t_id][j];
      }
    }
  };

  int cnt_suc = 0;
  while (!es_queue.empty()) {
    std::array<int, 2> e = es_queue.top().v_ids;
    double old_weight = es_queue.top().weight;
    es_queue.pop();

    int v1_id = e[0];
    int v2_id = e[1];
    std::vector<int> n12_t_ids;
    set_intersection(conn_tets[e[0]], conn_tets[e[1]], n12_t_ids);
    if (n12_t_ids.size() != 3)
      continue;  // only enables 3-2 swap https://i.imgur.com/zcmFleu.png

    auto id_in_array = [](auto &v, auto &k) {
      for (auto i = 0; i < v.size(); i++) {
        if (v[i] == k) return i;
      }
      return -1;
    };

    int t0_id = n12_t_ids[0];
    std::array<int, 2> t12_ids = {{n12_t_ids[1], n12_t_ids[2]}};
    auto n0_id = -1, n1_id = -1, n2_id = -1;
    int t1_id = t12_ids[0];
    int t2_id = t12_ids[1];
    for (int j = 0; j < 4; j++) {
      if (tets[t0_id][j] == v1_id) {
        ;  // v1_id_j = j;
      } else if (tets[t0_id][j] == v2_id) {
        ;  // v2_id_j = j;
      } else {
        if (id_in_array(tets[t1_id], tets[t0_id][j]) != -1)
          n1_id = tets[t0_id][j];
        if (id_in_array(tets[t2_id], tets[t0_id][j]) != -1)
          n2_id = tets[t0_id][j];
      }
      if (id_in_array(tets[t0_id], tets[t1_id][j]) == -1)
        n0_id = tets[t1_id][j];
    }

    std::map<std::array<int, 3>, int> t0_map1_node_coords;
    std::map<std::array<int, 3>, int> t0_map2_node_coords;
    map_nodes(t0_id, {{v1_id, n1_id, v2_id}}, n2_id, t0_map1_node_coords);
    map_nodes(t0_id, {{v1_id, n2_id, v2_id}}, n1_id, t0_map2_node_coords);
    //
    std::array<std::map<int, std::pair<int, Vec3d>>, 2> maps_node_ids;

    // v1n1v2 -> n0n1v2
    // v1n2v2 -> n0n2v2
    std::map<std::array<int, 3>, int> t1_map_node_coords;
    std::map<std::array<int, 3>, int> t2_map_node_coords;
    map_nodes(t1_id, {{n0_id, n1_id, v2_id}}, v1_id, t1_map_node_coords);
    map_nodes(t2_id, {{n0_id, n2_id, v2_id}}, v1_id, t2_map_node_coords);
    for (auto &t0_map : t0_map1_node_coords) {
      maps_node_ids[0][t0_map.second] =
          std::make_pair(t1_map_node_coords[t0_map.first],
                         vertices[t1_map_node_coords[t0_map.first]]);
    }
    for (auto &t0_map : t0_map2_node_coords) {
      maps_node_ids[0][t0_map.second] =
          std::make_pair(t2_map_node_coords[t0_map.first],
                         vertices[t2_map_node_coords[t0_map.first]]);
    }

    std::map<int, int> map2_node_ids;
    // v1n1v2 -> v1n1n0
    // v1n2v2 -> v1n2n0
    t1_map_node_coords.clear();
    t2_map_node_coords.clear();
    map_nodes(t1_id, {{v1_id, n1_id, n0_id}}, v2_id, t1_map_node_coords);
    map_nodes(t2_id, {{v1_id, n2_id, n0_id}}, v2_id, t2_map_node_coords);
    for (auto &t0_map : t0_map1_node_coords) {
      maps_node_ids[1][t0_map.second] =
          std::make_pair(t1_map_node_coords[t0_map.first],
                         vertices[t1_map_node_coords[t0_map.first]]);
    }
    for (auto &t0_map : t0_map2_node_coords) {
      maps_node_ids[1][t0_map.second] =
          std::make_pair(t2_map_node_coords[t0_map.first],
                         vertices[t2_map_node_coords[t0_map.first]]);
    }

    ///////////////////////////

    double max_energy = 0;
    for (int t_id : n12_t_ids) {
      RowMatX3d old_nodes35(n_node, 3);
      for (int j = 0; j < codecs_o4.rows(); j++) {
        old_nodes35.row(j) = vertices[tets[t_id][j]];
      }
      std::tuple<double, RowMatX3d> old_q = mips_energy(old_nodes35, vec_dxyz);
      if (std::get<0>(old_q) > max_energy) max_energy = std::get<0>(old_q);
    }
    //
    bool is_valid = true;
    for (int i = 0; i < 2; i++) {
      RowMatX3d nodes35(n_node, 3);
      for (int j = 0; j < codecs_o4.rows(); j++) {
        if (maps_node_ids[i].find(tets[t0_id][j]) != maps_node_ids[i].end())
          nodes35.row(j) = vertices[maps_node_ids[i][tets[t0_id][j]].first];
        else
          nodes35.row(j) = vertices[tets[t0_id][j]];
      }
      //
      int tmp_v_id;
      if (i == 0)
        tmp_v_id = v2_id;
      else
        tmp_v_id = v1_id;
      int tmp_k =
          std::distance(tets[t0_id].data(),
                        std::find(tets[t0_id].data(), tets[t0_id].data() + 4,
                                  tmp_v_id));  // find in true verts.
      for (int j = 0; j < codecs_o4.rows(); j++) {
        if (codecs_o4(j, tmp_k) == 0 && codecs_o4(j, (tmp_k + 1) % 4) != 0 &&
            codecs_o4(j, (tmp_k + 2) % 4) != 0 &&
            codecs_o4(j, (tmp_k + 3) % 4) != 0) {
          nodes35.row(j) << 0, 0, 0;
          for (int k = 1; k < 4; k++)
            nodes35.row(j) +=
                nodes35.row((tmp_k + k) % 4) * codecs_o4(j, (tmp_k + k) % 4);
          nodes35.row(j) /= codecs_o4.row(j).sum();
          maps_node_ids[i][tets[t0_id][j]] = std::make_pair(-1, nodes35.row(j));
        }
      }
      //
      Vec3d mid_p(0, 0, 0);
      for (int j = 0; j < nodes35.rows() - 1; j++) {
        mid_p += nodes35.row(j);
      }
      nodes35.row(34) = mid_p / (nodes35.rows() - 1);
      maps_node_ids[i][tets[i == 0 ? t0_id : t1_id][34]] =
          std::make_pair(-2, nodes35.row(34));

      /////
      std::tuple<double, RowMatX3d> q = mips_energy(nodes35, vec_dxyz);
      if (std::get<0>(q) > std::max(stop_energy, max_energy)) {
        is_valid = false;
        break;
      }
      if (!prism::curve::tetrahedron_inversion_check(
              nodes35, codecs_o4, codecs_o9, bern_from_lagr_o4,
              bern_from_lagr_o9)) {
        is_valid = false;
        break;
      }
    }
    if (!is_valid) continue;
    spdlog::debug("[{}] Success {} {}", cnt_suc, v1_id, v2_id);
    cnt_suc++;

    ///////////////////////////

    // update
    std::vector<int> rm_v_ids;
    for (int t_id : t12_ids) {
      int v1_id_k, v2_id_k;
      for (int k = 0; k < 4; k++) {
        if (tets[t_id][k] == v1_id) v1_id_k = k;
        if (tets[t_id][k] == v2_id) v2_id_k = k;
      }
      for (int i = 0; i < codecs_o4.rows() - 1; i++) {
        if (codecs_o4(i, v1_id_k) != 0 && codecs_o4(i, v2_id_k) != 0) {
          v_is_removed[tets[t_id][i]] = true;
          rm_v_ids.push_back(tets[t_id][i]);
        }
      }
    }
    v_is_removed[tets[t2_id][34]] = true;
    //
    t_is_removed[t2_id] = true;
    int mid_v_id = tets[t1_id][34];
    tets[t1_id] = tets[t0_id];
    tets[t1_id][34] = mid_v_id;
    int tmp_cnt = 0;
    for (int i = 0; i < 2; i++) {
      int t_id;
      if (i == 0)
        t_id = t0_id;
      else
        t_id = t1_id;
      for (int j = 0; j < n_node; j++) {
        if (maps_node_ids[i].find(tets[t_id][j]) != maps_node_ids[i].end()) {
          const auto tmp_map = maps_node_ids[i][tets[t_id][j]];
          int v_id = tmp_map.first;
          if (v_id >= 0)
            tets[t_id][j] = v_id;
          else if (v_id == -1) {
            tets[t_id][j] = rm_v_ids[tmp_cnt++];
          }
          vertices[tets[t_id][j]] = tmp_map.second;
        }
      }
    }
    //
    t0_map1_node_coords.clear();
    t0_map2_node_coords.clear();
    map_nodes(t0_id, {{n0_id, n1_id, n2_id}}, v2_id, t0_map1_node_coords);
    map_nodes(t1_id, {{n0_id, n1_id, n2_id}}, v1_id, t0_map2_node_coords);
    std::map<int, int> map_node_id;
    for (const auto &t0_map : t0_map1_node_coords) {
      int t0_v_id = t0_map.second;
      int t1_v_id = t0_map2_node_coords[t0_map.first];
      if (t0_v_id == t1_v_id) continue;
      //            v_is_removed[t0_v_id] = true;
      v_is_removed[t1_v_id] = false;
      int j = id_in_array(tets[t0_id], t0_v_id);
      tets[t0_id][j] = t1_v_id;
    }
    conn_tets[v1_id].erase(
        std::find(conn_tets[v1_id].begin(), conn_tets[v1_id].end(), t2_id));
    conn_tets[v1_id].erase(
        std::find(conn_tets[v1_id].begin(), conn_tets[v1_id].end(), t0_id));
    conn_tets[v2_id].erase(
        std::find(conn_tets[v2_id].begin(), conn_tets[v2_id].end(), t2_id));
    conn_tets[v2_id].erase(
        std::find(conn_tets[v2_id].begin(), conn_tets[v2_id].end(), t1_id));
    //
    conn_tets[n0_id].erase(
        std::find(conn_tets[n0_id].begin(), conn_tets[n0_id].end(), t2_id));
    conn_tets[n0_id].push_back(t0_id);
    conn_tets[n2_id].erase(
        std::find(conn_tets[n2_id].begin(), conn_tets[n2_id].end(), t2_id));
    conn_tets[n2_id].push_back(t1_id);
  }

  RowMatd lagr_new(std::count(v_is_removed.begin(), v_is_removed.end(), false),
                   3);
  RowMati p4T_new(std::count(t_is_removed.begin(), t_is_removed.end(), false),
                  n_node);
  std::vector<int> map_v_ids(vertices.size(), -1);
  int cnt = 0;
  for (int i = 0; i < vertices.size(); i++) {
    if (v_is_removed[i]) continue;
    lagr_new.row(cnt) = vertices[i];
    map_v_ids[i] = cnt;
    cnt++;
  }
  int cnt_t = 0;
  for (int i = 0; i < tets.size(); i++) {
    if (t_is_removed[i]) continue;
    for (int j = 0; j < tets[i].size(); j++) {
      p4T_new(cnt_t, j) = map_v_ids[tets[i][j]];
    }
    cnt_t++;
  }
  p4T = p4T_new;
  lagr = lagr_new;
  auto energy = energy_evaluation(lagr, p4T, vec_dxyz);
  spdlog::info("{} edges swapped   | Energy {}", cnt_suc, e_stat(energy));
  return cnt_suc;
};

}  // namespace prism::curve
