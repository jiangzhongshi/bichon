#include "feature_utils.hpp"

#include <igl/per_face_normals.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/vertex_triangle_adjacency.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <fstream>
#include <highfive/H5Easy.hpp>
#include <queue>
#include <utility>

#include "common.hpp"
#include "local_operations/mesh_coloring.hpp"

void prism::mark_feature_edges(const RowMatd &V, const RowMati &F,
                               double threshold, RowMati &feat) {
  RowMati TT, TTi;
  RowMatd FN;
  std::vector<std::array<int, 2>> feat_vec;
  igl::triangle_triangle_adjacency(F, TT, TTi);
  igl::per_face_normals(V, F, FN);
  for (int i = 0; i < F.rows(); i++) {
    for (int j = 0; j < 3; j++) {
      if (FN.row(i).dot(FN.row(TT(i, j))) < threshold) {
        auto v0 = F(i, j), v1 = F(i, (j + 1) % 3);
        feat_vec.push_back({std::min(v0, v1), std::max(v0, v1)});
      }
    }
  }
  std::sort(feat_vec.begin(), feat_vec.end());
  feat_vec.erase(std::unique(feat_vec.begin(), feat_vec.end()), feat_vec.end());
  feat = Eigen::Map<RowMati>(feat_vec[0].data(), feat_vec.size(), 2);
}

bool prism::read_feature_graph(std::string path, Eigen::VectorXi &feat_nodes,
                               RowMati &feat_edges) {
  // https://github.com/gaoxifeng/Feature-Preserving-Octree-Hex-Meshing/blob/67e6d59affb13116a260e40153a81f3bd3ac7256/io.cpp#L412
  // differently from obj, this is 0 based index, directly usable in the
  // program.
  struct Mesh_Feature {
    double angle_threshold;
    int orphan_curve;
    int orphan_curve_single;
    Eigen::VectorXi IN_corners;
    RowMati IN_v_pairs;
  };
  Mesh_Feature mf;
  std::fstream f(path, std::ios::in);
  if (f.fail()) return false;
  char s[1024];
  int cnum = 0, edgenum = 0;
  f.getline(s, 1023);
  sscanf(s, "%lf %i %i", &mf.angle_threshold, &mf.orphan_curve,
         &mf.orphan_curve_single);
  f.getline(s, 1023);
  sscanf(s, "%i %i", &cnum, &edgenum);
  mf.IN_corners.resize(cnum);
  for (int i = 0; i < cnum; i++) {
    f.getline(s, 1023);
    sscanf(s, "%i", &(mf.IN_corners[i]));
  }
  mf.IN_v_pairs.resize(edgenum, 2);
  for (int i = 0; i < edgenum; i++) {
    f.getline(s, 1023);
    int v0 = -1, v1 = -1;
    sscanf(s, "%i %i", &v0, &v1);
    mf.IN_v_pairs(i, 0) = v0;
    mf.IN_v_pairs(i, 1) = v1;
  }
  f.close();
  feat_nodes = std::move(mf.IN_corners);
  feat_edges = std::move(mf.IN_v_pairs);
  return true;
}

bool prism::read_feature_h5(std::string path, Eigen::VectorXi &feat_nodes,
                            RowMati &feat_edges, Eigen::VectorXi &points_fid,
                            RowMatd &points_bc) {
  auto file = H5Easy::File(path, H5Easy::File::ReadOnly);
  if (file.exist("E")) {
    feat_edges = H5Easy::load<RowMati>(file, "E");
  }
  if (file.exist("V")) {
    feat_nodes = H5Easy::load<Eigen::VectorXi>(file, "V");
  }
  if (file.exist("P_fid")) {  // not doing Poisson Points Here. Let's
                              // surgery after init
    points_fid = H5Easy::load<Eigen::VectorXi>(file, "P_fid");
    points_bc = H5Easy::load<RowMatd>(file, "P_bc");
  }
  return true;
}

bool prism::feature_chains_from_edges(const Eigen::VectorXi &feat_nodes,
                                      const RowMati &feat_edges, int vnum,
                                      std::vector<std::list<int>> &all_chain) {
  RowMati feature_edges = feat_edges;
  // find corners
  std::map<int, int> valency;
  std::for_each(feature_edges.data(),
                feature_edges.data() + feature_edges.size(),
                [&valency](const auto &a) {
                  if (valency.find(a) == valency.end()) valency[a] = 0;
                  valency[a]++;
                });
  std::set<int> corners;
  for (auto [v, k] : valency)
    if (k != 2) corners.insert(v);
  std::for_each(feat_nodes.data(), feat_nodes.data() + feat_nodes.size(),
                [&corners](auto a) { corners.insert(a); });
  // split/duplicate corners and build connectivity
  // curve connectivity is defined by v0 -> list of adjacent (up to two).
  std::map<int, std::vector<int>> curve_conn;
  std::vector<int> extra_corner_ind;
  for (int i = 0, extra_vnum = 0; i < feature_edges.rows(); i++) {
    for (int j : {0, 1}) {
      auto &v = feature_edges(i, j);
      if (corners.find(v) !=
          corners.end()) {  // replace each occurence with a fake one.
        extra_corner_ind.push_back(v);
        v = vnum + extra_vnum++;
      }
      if (curve_conn.find(v) == curve_conn.end())
        curve_conn.emplace(v, std::vector<int>{});
    }
    auto v0 = feature_edges(i, 0);
    auto v1 = feature_edges(i, 1);
    assert(corners.find(v0) == corners.end());
    assert(corners.find(v1) == corners.end());
    curve_conn[v0].push_back(v1);
    curve_conn[v1].push_back(v0);
  }

#ifndef NDEBUG
  for (auto &[c, k] : curve_conn) {
    assert(k.size() <= 2);
    if (k.size() == 1) assert(c >= vnum);
  }
#endif

  auto trace_through = [&curve_conn](int v0, int v1, auto f) {
    while (true) {
      auto itv1 = curve_conn.find(v1);
      if (itv1 == curve_conn.end()) return true;  // cycle
      auto e1 = itv1->second;
      assert(e1.size() > 0);
      curve_conn.erase(itv1);
      if (e1.size() < 2)  // reached a corner.
        return false;
      auto tmp = v1;
      assert(e1[0] == v0 || e1[1] == v0);
      v1 = e1[0] + e1[1] - v0;
      v0 = tmp;
      *f = (v1);
    }
  };

  while (curve_conn.size() > 0) {
    auto it = curve_conn.begin();
    auto nb = it->second;
    auto v0 = it->first, v1 = nb[0];
    std::list<int> chain{v1, v0};
    if (nb.size() == 2) chain.push_back(nb[1]);
    curve_conn.erase(it);
    bool cycle = trace_through(v0, v1, std::front_inserter(chain));
    if (!cycle && nb.size() == 2) {
      assert(chain.front() >= vnum);
      v1 = nb[1];
      bool c = trace_through(v0, v1, std::back_inserter(chain));
      assert(!c);
      assert(chain.back() >= vnum);
    }
    std::for_each(chain.begin(), chain.end(),
                  [&extra_corner_ind, vnum](auto &a) {
                    if (a >= vnum) a = extra_corner_ind[a - vnum];
                  });
    if (cycle) {
      chain.pop_back();
      assert(chain.front() == chain.back());
    }
    all_chain.emplace_back(chain);
  }
  return true;
}

void feature_chain_region_segments_legacy(
    const RowMati &F, int vnum, const std::vector<std::list<int>> chains,
    std::vector<std::set<int> /*refs*/>  // 0 is left and 1 is right
        &feature_side_region) {
  std::set<int> feature_corners;
  for (auto &c : chains) {
    auto cf = c.front(), cb = c.back();
    if (cf == cb) continue;
    feature_corners.insert(cf);
    feature_corners.insert(cb);
  }
  Eigen::VectorXi VF, NI;
  RowMati TT, TTi;
  igl::vertex_triangle_adjacency(F, vnum, VF, NI);
  igl::triangle_triangle_adjacency(F, TT, TTi);
  for (int ci = 0; ci < chains.size(); ci++) {
    auto &c = chains[ci];
    std::set<int> faces_of_interest;
    std::set<int> seg0;
    std::map<std::pair<int, int>, int> edges_to_chain;
    for (auto it = std::next(c.begin()); it != c.end(); it++)
      edges_to_chain.emplace(std::pair(*std::prev(it), *it), ci);
    // first, collect all adjacent faces to the current chain
    for (auto i : chains[ci]) {
      if (feature_corners.find(i) != feature_corners.end())
        continue;  // corners are not added
      for (auto j = NI[i]; j < NI[i + 1]; j++) faces_of_interest.insert(VF[j]);
    }

    // bfs coloring, flag is for swapping left/right.
    int flag = 0;
    std::function<void(int)> bfs_segment;
    bfs_segment = [&faces_of_interest, &seg0, &TT, &F, &flag, &bfs_segment, &ci,
                   &edges_to_chain](int fi) -> void {
      auto it = faces_of_interest.find(fi);
      if (it == faces_of_interest.end()) return;
      seg0.insert(*it);
      faces_of_interest.erase(it);
      auto f = F.row(fi);
      for (auto j : {0, 1, 2}) {
        auto it0 = edges_to_chain.end();
        if (flag != -1) it0 = edges_to_chain.find({f[j], f[(j + 1) % 3]});
        if (it0 != edges_to_chain.end() && it0->second == ci) {
          assert(flag != -1);
          flag = 1;  // the found it has a following edge (is a face on the
                     // left of chain)
          continue;
        }
        if (flag != 1) it0 = edges_to_chain.find({f[(j + 1) % 3], f[j]});
        if (it0 != edges_to_chain.end() && it0->second == ci) {
          assert(flag != 1);
          flag = -1;
          continue;
        }
        bfs_segment(TT(fi, j));
      }
    };
    bfs_segment(*faces_of_interest.begin());
    assert(flag != 0);
    if (flag == -1) std::swap(faces_of_interest, seg0);
    feature_side_region.emplace_back(seg0);               // left seg
    feature_side_region.emplace_back(faces_of_interest);  // right seg
  }
  assert(feature_side_region.size() == 2 * chains.size());
}

auto one_ring_component_coloring = [](const std::set<std::pair<int, int>> &map,
                                      const std::vector<Vec3i> &nb_tris,
                                      int vid) {
  std::map<int, std::array<int, 2>> vert2face;
  auto found_in = [&map](auto &v0, auto &v1) {
    return map.find({v0, v1}) != map.end() || map.find({v1, v0}) != map.end();
  };
  for (int fi = 0; fi < nb_tris.size(); fi++) {
    auto &f = nb_tris[fi];
    int j = 0;
    for (; j < 3; j++)
      if (f[j] == vid) break;
    assert(j != 3);
    auto v0 = vid, v1 = f[(j + 1) % 3], v2 = f[(j + 2) % 3];

    for (auto vv : {v1, v2}) {
      if (found_in(v0, vv)) {
        // spdlog::trace("feat {} {}", v0, vv);
        continue;
      }
      auto it = vert2face.lower_bound(vv);
      if (it != vert2face.end() && it->first == vv)
        it->second[1] = (fi);
      else
        vert2face.emplace_hint(it, vv, std::array<int, 2>{fi, -1});
    }
  }
  spdlog::trace("nb_tris {}", nb_tris);
  spdlog::trace("v2f {}", vert2face);

  std::vector<int> separate_type(nb_tris.size(), -1);
  int color = 0;
  for (color = 0; !vert2face.empty(); color++) {
    std::deque<int> fid_queue;
    fid_queue.push_back(vert2face.begin()->first);
    while (!fid_queue.empty()) {
      auto v1 = fid_queue.front();
      fid_queue.pop_front();
      auto it = vert2face.find(v1);
      if (it != vert2face.end()) {
        for (auto f : it->second) {
          separate_type[f] = color;
          for (auto j = 0; j < 3; j++) {
            auto v2 = nb_tris[f][j];
            if (v2 != v1 && v2 != vid) fid_queue.push_back(v2);
          }
        }
        vert2face.erase(it);
      }
    }
    if (color > 2 * nb_tris.size()) spdlog::error("inf loop");
  }
  for (auto &s : separate_type) {
    if (s == -1) s = color++;
  }
  spdlog::trace("sep {}", separate_type);
  return separate_type;
};

void prism::feature_chain_region_segments(
    const RowMati &F, int vnum, const std::vector<std::list<int>> chains,
    std::vector<std::set<int> /*refs*/>  // 0 is left and 1 is right
        &feature_side_region,
    std::vector<std::set<int>> &region_around_chains) {
  std::vector<std::vector<int>> VF, VFi;
  igl::vertex_triangle_adjacency(vnum, F, VF, VFi);
  RowMati TT, TTi;
  igl::triangle_triangle_adjacency(F, TT, TTi);
  auto feature_tags = std::set<std::pair<int, int>>();
  for (auto ch : chains) {  // tear off TT at chains.
    for (auto it = ch.begin(); std::next(it) != ch.end(); it++) {
      auto v0 = *it, v1 = *std::next(it);
      feature_tags.emplace(v0, v1);
      auto &nb = VF[v0], &nbi = VFi[v0];
      for (int i = 0; i < nb.size(); i++) {
        auto f = nb[i];
        assert(F(f, nbi[i]) == v0);
        if (F(f, (nbi[i] + 1) % 3) == v1) {  // lefty
          auto &f1 = TT(f, nbi[i]), &e1 = TTi(f, nbi[i]);
          TT(f1, e1) = -1;
          TTi(f1, e1) = -1;
          f1 = -1;
          e1 = -1;
        }
      }
    }
  }
  for (int ci = 0; ci < chains.size(); ci++) {
    auto &ch = chains[ci];
    std::set<int> faces_of_interest;
    std::set<int> seg0;
    std::map<std::pair<int, int>, int> edges_to_chain;
    for (auto it = std::next(ch.begin()); it != ch.end(); it++)
      edges_to_chain.emplace(std::pair(*std::prev(it), *it), ci);
    // first, collect all adjacent faces to the current chain
    for (auto i : chains[ci]) {
      faces_of_interest.insert(VF[i].begin(), VF[i].end());
    }
    spdlog::trace("ci {} faces {}", ci, faces_of_interest);
    // bfs coloring
    // initial coloring.
    std::set<int> solid_left, solid_right;
    for (auto it = ch.begin(); std::next(it) != ch.end(); it++) {
      auto v0 = *it, v1 = *std::next(it);
      auto &nb = VF[v0], &nbi = VFi[v0];
      for (int i = 0; i < nb.size(); i++) {
        auto f = nb[i];
        assert(F(f, nbi[i]) == v0);
        if (F(f, (nbi[i] + 1) % 3) == v1) {  // lefty
          solid_left.insert(f);
        }
        if (F(f, (nbi[i] + 2) % 3) == v1) {  // righty
          solid_right.insert(f);
        }
      }
    }
    // spdlog::trace("solid left {}", solid_left);
    // spdlog::trace("solid right {}", solid_right);
    auto isin = [](auto x, auto &s) { return s.find(x) != s.end(); };
    auto left_forbid = std::set<int>();
    auto right_forbid = std::set<int>();
    for (auto vid : ch) {
      auto &nb = VF[vid];
      auto nb_tris = std::vector<Vec3i>();
      auto left_mark = -1, right_mark = -1;
      for (auto i = 0; i < nb.size(); i++) {
        auto f = nb[i];
        nb_tris.emplace_back(Vec3i{F(f, 0), F(f, 1), F(f, 2)});
        if (isin(f, solid_left)) left_mark = i;
        if (isin(f, solid_right)) right_mark = i;
      }
      // spdlog::trace("mark {} {}", left_mark ,right_mark);
      // spdlog::trace("nb {}", nb);
      assert(left_mark >= 0 && right_mark >= 0);
      auto labels = one_ring_component_coloring(feature_tags, nb_tris, vid);
      if (labels[left_mark] == labels[right_mark]) {
        spdlog::debug("Dangler {}", vid);
        left_forbid.insert(nb[right_mark]);
        right_forbid.insert(nb[left_mark]);
        continue;
      }
      auto left_lab = labels[left_mark];
      auto right_lab = labels[right_mark];
      // spdlog::trace("labs {} {}", left_lab,right_lab);
      for (auto i = 0; i < nb.size(); i++) {
        if (labels[i] == left_lab) right_forbid.insert(nb[i]);
        if (labels[i] == right_lab) left_forbid.insert(nb[i]);
      }
      // spdlog::trace("forbids {} {}", left_forbid,right_forbid);
    }
    feature_side_region.emplace_back(left_forbid);
    feature_side_region.emplace_back(right_forbid);
  }
  assert(feature_side_region.size() == 2 * chains.size());
}

std::vector<std::list<int>> prism::glue_meta_together(
    const std::map<std::pair<int, int>, std::pair<int, std::vector<int>>>
        &meta) {
  std::vector<std::vector<std::pair<int, int>>> collect;
  for (auto [m, cid_chain] : meta) {
    auto [v0, v1] = m;
    auto [cid, _c] = cid_chain;
    if (collect.size() <= cid) collect.resize(cid + 1);
    collect[cid].emplace_back(v0, v1);
  }
  auto glue_together = [](std::vector<std::pair<int, int>> &vv_groups) {
    std::map<int, std::list<int>> v0_seg;
    for (auto [v0, v1] : vv_groups) v0_seg.emplace(v0, std::list<int>{v0, v1});
    std::vector<int> reorder;
    auto it = v0_seg.begin();
    while (v0_seg.size() > 1) {
      auto v1 = it->second.back(), v0 = it->first;
      auto it1 = v0_seg.find(v1);
      if (it1 != v0_seg.end()) {
        assert(it1->second.size() >= 2);
        it->second.insert(it->second.end(), std::next(it1->second.begin()),
                          it1->second.end());
        v0_seg.erase(it1);
      } else {
        it++;
      }
    }
    return std::move(v0_seg.begin()->second);
  };
  std::vector<std::list<int>> chains;
  for (int i = 0; i < collect.size(); i++) {
    chains.emplace_back(glue_together(collect[i]));
  }
  return chains;
}

auto glue_together = [](std::vector<std::pair<int, int>> &vv_groups) {
  std::map<int, std::list<int>> v0_seg;
  for (auto [v0, v1] : vv_groups) v0_seg.emplace(v0, std::list<int>{v0, v1});
  std::vector<int> reorder;
  auto it = v0_seg.begin();
  while (v0_seg.size() > 1) {
    auto v1 = it->second.back(), v0 = it->first;
    auto it1 = v0_seg.find(v1);
    if (it1 != v0_seg.end()) {
      assert(it1->second.size() >= 2);
      it->second.insert(it->second.end(), std::next(it1->second.begin()),
                        it1->second.end());
      v0_seg.erase(it1);
    } else {
      it++;
    }
  }
  return std::move(v0_seg.begin()->second);
};

std::vector<std::list<int>> prism::recover_chains_from_meta_edges(
    const std::map<std::pair<int, int>, std::pair<int, std::vector<int>>>
        &meta) {
  std::vector<std::vector<std::pair<int, int>>> collect;
  for (auto [m, cid_chain] : meta) {
    auto [v0, v1] = m;
    auto [cid, _c] = cid_chain;
    if (collect.size() <= cid) collect.resize(cid + 1);
    collect[cid].emplace_back(v0, v1);
  }

  std::vector<std::list<int>> chains;
  for (int i = 0; i < collect.size(); i++) {
    auto v_list = glue_together(collect[i]);
    std::list<int> chain;
    for (auto c = v_list.begin(); std::next(c) != v_list.end(); c++) {
      auto n = std::next(c);
      auto it = meta.find({*c, *n});
      auto &[cid, segs] = it->second;
      if (segs.empty()) continue;
      assert(cid == i);
      if ((!chain.empty()) && segs.front() == chain.back()) chain.pop_back();
      chain.insert(chain.end(), segs.begin(), segs.end());
    }
    if (v_list.front() == v_list.back() &&
        chain.back() != chain.front())  // circular feature
      chain.push_back(chain.front());
    chains.emplace_back(chain);
  }
  return chains;
}

std::tuple<RowMatd, RowMati, RowMati> prism::subdivide_feature_triangles(
    const RowMatd &mV, const RowMati &mF, const RowMati &feature_edges,
    const std::vector<std::pair<int, int>> &slice_vv,
    std::vector<int> &face_parent) {
  assert(face_parent.size() == mF.rows());
  // splits
  RowMati FF, FFi;
  igl::triangle_triangle_adjacency(mF, FF, FFi);

  std::vector<std::vector<int>> VF, VFi;
  igl::vertex_triangle_adjacency(mV.rows(), mF, VF, VFi);
  // initialize red (2)
  Eigen::VectorXi colors = Eigen::VectorXi::Zero(mF.rows());
  for (auto [v0, v1] : slice_vv) {
    auto [f, e] = vv2fe(v0, v1, mF, VF);
    colors[f] = 2;
    colors[FF(f, e)] = 2;
  }

  auto feature_fe = RowMati(mF.rows(), 3);
  feature_fe.setConstant(-1);
  auto sub_features = std::vector<Eigen::RowVector2i>(feature_edges.rows());
  for (auto e = 0; e < feature_edges.rows(); e++) {
    auto v0 = feature_edges(e, 0), v1 = feature_edges(e, 1);
    auto [f, j] = vv2fe(v0, v1, mF, VF);
    if (f == -1) {
      spdlog::warn("fe[e] {}", feature_edges.row(e));
      spdlog::warn("VF {}/{}", VF[v0], VF[v1]);
      spdlog::error("Subdivide Feature Triangles Error");
      exit(1);
    }
    feature_fe(f, j) = e;
    sub_features[e] = feature_edges.row(e);
  }
  RowMati edge_vert = -RowMati::Ones(mF.rows(), 3);
  prism::local::red_green_coloring(mF, FF, colors);

  auto feature_splitter = [&](int f, int e, int va, int vb, int vc) {
    if (feature_fe(f, e) != -1) {
      // clear and split.
      auto ei = feature_fe(f, e);
      sub_features[ei] << -1, -1;
      sub_features.emplace_back(va, vb);
      sub_features.emplace_back(vb, vc);
    }
  };
  std::vector<Vec3d> V;
  std::vector<Vec3i> F;
  eigen2vec(mV, V);
  eigen2vec(mF, F);
  for (int i = 0; i < mF.rows(); i++) {  // only deal with green here
    if (colors[i] != 1) continue;
    int e = [&colors, &FF](int i) {
      for (int j = 0; j < 3; j++) {
        if (FF(i, j) != -1 && colors[FF(i, j)] == 2) {
          return j;
        }  // find red neighbor
      }
      return -1;
    }(i);
    assert(e != -1);

    int ux = V.size();
    V.push_back((V[F[i][e]] + V[F[i][(e + 1) % 3]]) / 2);
    edge_vert(i, e) = ux;
    edge_vert(FF(i, e), FFi(i, e)) = ux;
    auto v0 = mF(i, e), v1 = mF(i, (e + 1) % 3), v2 = mF(i, (e + 2) % 3);
    F.emplace_back(Vec3i{v0, ux, v2});
    F.emplace_back(Vec3i{v1, v2, ux});
    face_parent.emplace_back(face_parent[i]);
    face_parent.emplace_back(face_parent[i]);
    feature_splitter(i, e, v0, ux, v1);
  }

  for (int f = 0; f < colors.size(); f++) {
    if (colors[f] != 2) continue;
    for (auto e = 0; e < 3; e++) {
      if (edge_vert(f, e) == -1) {  // un-assigned
        edge_vert(f, e) = edge_vert(FF(f, e), FFi(f, e)) = V.size();
        V.push_back((V[mF(f, e)] + V[mF(f, (e + 1) % 3)]) / 2);
      }
      feature_splitter(f, e, mF(f, e), edge_vert(f, e), mF(f, (e + 1) % 3));
    }
    auto v0 = mF(f, 0), v1 = mF(f, 1), v2 = mF(f, 2);
    auto e0 = edge_vert(f, 0), e1 = edge_vert(f, 1), e2 = edge_vert(f, 2);
    F.emplace_back(Vec3i{v0, e0, e2});
    F.emplace_back(Vec3i{v1, e1, e0});
    F.emplace_back(Vec3i{v2, e2, e1});
    F.emplace_back(Vec3i{e0, e1, e2});
    for (auto ii = 0; ii < 4; ii++) face_parent.emplace_back(face_parent[f]);
  }  // insert red faces

  for (int f = 0, freal = 0; f < colors.size();
       f++)               // organize and fill to the front.
    if (colors[f] > 0) {  // removed
      F[f] = F.back();
      F.pop_back();
      face_parent[f] = face_parent.back();
      face_parent.pop_back();
    }

  sub_features.erase(std::remove_if(sub_features.begin(), sub_features.end(),
                                    [](auto &v) -> bool { return v[0] < 0; }),
                     sub_features.end());
  std::for_each(sub_features.begin(), sub_features.end(), [](auto &v) {
    if (v[0] > v[1]) std::swap(v[0], v[1]);
  });
  std::sort(sub_features.begin(), sub_features.end(), [](auto &a, auto &b) {
    return std::lexicographical_compare(a.data(), a.data() + a.size(), b.data(),
                                        b.data() + b.size());
  });
  sub_features.erase(std::unique(sub_features.begin(), sub_features.end()),
                     sub_features.end());

  return std::tuple(RowMatd(Eigen::Map<RowMatd>(V[0].data(), V.size(), 3)),
                    RowMati(Eigen::Map<RowMati>(F[0].data(), F.size(), 3)),
                    RowMati(Eigen::Map<RowMati>(sub_features[0].data(),
                                                sub_features.size(), 2)));
}

#include <prism/local_operations/local_mesh_edit.hpp>
#include <prism/local_operations/retain_triangle_adjacency.hpp>
void prism::split_feature_ears(RowMatd &mV, RowMati &mF,
                               const RowMati &feature_edges,
                               std::vector<int> &face_parent) {
  std::vector<Vec3i> F;
  std::vector<Vec3d> V;
  eigen2vec(mF, F);
  eigen2vec(mV, V);
  auto [FF, FFi] = prism::local::triangle_triangle_adjacency(F);

  std::set<int> verts_on_feat;
  std::set<std::pair<int, int>> edges_on_feat;
  for (auto e = 0; e < feature_edges.rows(); e++) {
    auto v0 = feature_edges(e, 0), v1 = feature_edges(e, 1);
    if (v0 > v1) std::swap(v0, v1);
    edges_on_feat.emplace(v0, v1);
    verts_on_feat.insert(v0);
    verts_on_feat.insert(v1);
  }
  for (auto f = 0; f < F.size(); f++) {
    for (auto e = 0; e < 3; e++) {
      auto v0 = F[f][e], v1 = F[f][(e + 1) % 3];
      if (v0 > v1) std::swap(v0, v1);
      if (edges_on_feat.find({v0, v1}) != edges_on_feat.end()) continue;
      if (verts_on_feat.find(v0) != verts_on_feat.end() &&
          verts_on_feat.find(v1) != verts_on_feat.end()) {
        prism::edge_split(V.size(), F, FF, FFi, f, e);
        face_parent.emplace_back(face_parent[f]);
        face_parent.emplace_back(face_parent[FF[f][e]]);
        V.push_back((V[v0] + V[v1]) / 2);
      }
    }
  }
  vec2eigen(F, mF);
  vec2eigen(V, mV);
}

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/squared_distance_3.h>
#include <geogram/mesh/mesh_AABB.h>
bool prism::feature_sanity(const RowMatd &mV, RowMati &mE) {
  double angle = 0.98;
  double length = 1e-4;
  using K = CGAL::Exact_predicates_inexact_constructions_kernel;
  auto too_close = [&](int i, int j) {
    auto i0 = mE(i, 0), i1 = mE(i, 1);
    auto j0 = mE(j, 0), j1 = mE(j, 1);
    if (i0 == -1 || j0 == -1) return false;  // skip
    if (i0 == j0 || i1 == j1 || i0 == j1 ||
        i1 == j0) {                                 // share vertex. test angle
      if (i0 == j0 || i0 == j1) std::swap(i0, i1);  // i0, shared
      if (j0 == i0) std::swap(j0, j1);              // j0, shared

      auto e0 = (mV.row(i0) - mV.row(i1)).normalized();
      auto e1 = (mV.row(j0) - mV.row(j1)).normalized();
      if (e0.dot(e1) > angle)
        return true;
      else
        return false;
    }
    K::Point_3 p0(mV(i0, 0), mV(i0, 1), mV(i0, 2));
    K::Point_3 p1(mV(i1, 0), mV(i1, 1), mV(i1, 2));
    K::Point_3 q0(mV(j0, 0), mV(j0, 1), mV(j0, 2));
    K::Point_3 q1(mV(j1, 0), mV(j1, 1), mV(j1, 2));
    K::Segment_3 seg0(p0, p1), seg1(q0, q1);
    auto d = CGAL::squared_distance(seg0, seg1);
    if (d < length) return true;
    return false;
  };
  std::vector<GEO::Box> boxes;
  auto num_e = mE.rows();
  for (auto i = 0; i < num_e; i++) {
    if (mE(i, 0) == -1) continue;
    for (auto j = i + 1; j < num_e; j++) {
      if (too_close(i, j)) {
        return false;
        mE.row(i) << -1, -1;
        mE.row(j) << -1, -1;
      }
    }
  }
  return true;
}

#include <igl/avg_edge_length.h>

bool prism::feature_pre_split(RowMatd &V, RowMati &F, RowMati &feature_edges,
                              double threshold, std::vector<int> &face_parent) {
  // ear split will not change features, by definition.
  assert(face_parent.size() == F.rows());

  prism::split_feature_ears(V, F, feature_edges, face_parent);

  if (threshold > 1.) return false;
  double avg_len = igl::avg_edge_length(V, F);
  spdlog::info("ael {}", avg_len);
  while (true) {
    auto slicer = std::vector<std::pair<int, int>>();
    auto slice_cnt = 0;
    auto total_leng = 0.;  // verify that total length should not change much.
    // edges (as v-pairs) exceeding threshold length
    for (int i = 0; i < feature_edges.rows(); i++) {
      auto v0 = feature_edges(i, 0), v1 = feature_edges(i, 1);
      auto edgelen = (V.row(v0) - V.row(v1)).norm();
      if (edgelen > threshold) {
        slicer.emplace_back(v0, v1);
        slice_cnt++;
      }
      total_leng += edgelen;
    }
    for (auto i = 0; i < F.rows(); i++) {
      for (auto e = 0; e < 3; e++) {
        auto v0 = F(i, e);
        auto v1 = F(i, (e + 1) % 3);
        if ((V.row(v0) - V.row(v1)).norm() > threshold) {
          slicer.emplace_back(v0, v1);
          slice_cnt++;
        }
      }
    }
    spdlog::info("V {} E {} len {} slice {}", V.rows(), feature_edges.rows(),
                 total_leng, slice_cnt);
    if (slice_cnt == 0) break;

    std::tie(V, F, feature_edges) = prism::subdivide_feature_triangles(
        V, F, feature_edges, slicer, face_parent);
  }
  spdlog::info("ael {}", igl::avg_edge_length(V, F));
  return true;
}
