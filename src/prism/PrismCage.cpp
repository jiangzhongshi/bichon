#include "PrismCage.hpp"

#include <igl/adjacency_list.h>
#include <igl/boundary_loop.h>
#include <igl/cat.h>
#include <igl/doublearea.h>
#include <igl/remove_unreferenced.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/volume.h>
#include <igl/writeOBJ.h>
#include <igl/write_triangle_mesh.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <highfive/H5Easy.hpp>
#include <prism/geogram/geogram_utils.hpp>
#include <prism/local_operations/retain_triangle_adjacency.hpp>
#include <prism/local_operations/validity_checks.hpp>
#include <stdexcept>

#include "bevel_utils.hpp"
#include "cage_utils.hpp"
#include "feature_utils.hpp"
#include "geogram/AABB.hpp"
#include "prism/cage_check.hpp"
#include "prism/common.hpp"
#include "spatial-hash/AABB_hash.hpp"

auto counterclockwise_reorder = [](const auto &F, const auto &VF,
                                   const auto &VFi) {
  std::vector<std::vector<int>> VFcc, VFicc;
  for (auto v = 0; v < VF.size(); v++) {
    auto &nb = VF[v];
    auto &nbi = VFi[v];
    std::map<int, std::pair<int, int>> finder;
    for (auto i = 0; i < nb.size(); i++) {
      auto f0 = nb[i];
      auto e0 = nbi[i];
      auto v1 = F(f0, (e0 + 1) % 3);
      auto v2 = F(f0, (e0 + 2) % 3);
      finder.emplace(v1, std::pair(v2, f0));
    }
    std::vector<int> collect;
    collect.reserve(nb.size());
    auto v1 = finder.begin()->first;
    while (!finder.empty()) {
      auto it = finder.find(v1);
      if (it == finder.end()) std::runtime_error("not a loop");
      auto [v2, f0] = it->second;
      collect.emplace_back(f0);
      v1 = v2;
      finder.erase(it);
      if (collect.size() > nb.size()) std::runtime_error("not finishing loop");
    }
    std::vector<int> collect_i(collect.size());
    for (auto i = 0; i < collect.size(); i++) {
      collect_i[i] = [&]() {
        for (auto j = 0; j < 3; j++)
          if (F(collect[i], j) == v) return j;
        return -1;
      }();
    }
    VFcc.emplace_back(collect);
    VFicc.emplace_back(collect_i);
  }
  return std::tuple(VFcc, VFicc);
};

PrismCage::PrismCage(const RowMatd &vert, const RowMati &face, double dooseps,
                     double initial_step, SeparateType st)
    : PrismCage::PrismCage(vert, face, RowMati(), Eigen::VectorXi(),
                           Eigen::VectorXi(), RowMatd(), initial_step,
                           st){};

PrismCage::PrismCage(const RowMatd &vert, const RowMati &face,
                     RowMati &&feature_edges, Eigen::VectorXi &&feature_corners,
                     Eigen::VectorXi &&cons_points_fid,
                     RowMatd &&cons_points_bc, double initial_step,
                     SeparateType st) {
  prism::geo::init_geogram();
  RowMatd VN;
  ref.F = face;
  ref.V = vert;
  std::set<int> omni_singu;

  // Constraint Points Setting.
  constraints_per_face.resize(ref.F.rows());
  for (auto i = 0; i < cons_points_fid.size(); i++) {
    constraints_per_face[cons_points_fid[i]].push_back(i);
  }
  constraints_points_bc = cons_points_bc;
  if (constraints_points_bc.size() == 0) {
    assert(cons_points_fid.size() == 0);
    spdlog::info("Default distance sample on vertex.");
    std::vector<std::vector<int>> VF, VFi;
    igl::vertex_triangle_adjacency(ref.V.rows(), ref.F, VF, VFi);
    constraints_points_bc = RowMatd::Zero(ref.V.rows(),3);
    for (auto i=0;i<VF.size(); i++) {
      if (VF[i].empty()) continue; // input unreferenced vertex.
      auto f0 = VF[i][0];
      auto e0 = VFi[i][0];
      constraints_points_bc(i, e0) = 1;
      constraints_per_face[f0].push_back(i);
    }
  }

  bool good_normals =
      prism::cage_utils::most_visible_normals(ref.V, ref.F, VN, omni_singu);
  int pure_singularity = omni_singu.size();
  prism::cage_utils::mark_singular_on_border(ref.V, ref.F, VN, omni_singu);

  if (omni_singu.empty()) {
    assert(good_normals);
    spdlog::info("Succeessful Normals");
  } else {
    spdlog::info("Omni Singularity {} (Border {})", omni_singu.size(),
                 omni_singu.size() - pure_singularity);
    spdlog::trace("<Freeze> Omni \n {}", (omni_singu));
    spdlog::debug("omni {}", omni_singu);
    // reorder vert to make omni in the front
    auto spl =
        prism::cage_utils::split_singular_edges(ref.V, ref.F, VN, omni_singu);
    if (spl.size() > 0) {
      spdlog::debug("Split {} Singular Edges", spl);
      std::vector<Eigen::Vector2i> E;
      eigen2vec(feature_edges, E);
      auto extra = 0;
      for (auto i = 0; i < E.size();
           i++) {  // loop over all feature edges, and split if found in spl.
        auto v0 = E[i][0], v1 = E[i][1];
        if (v0 > v1) std::swap(v0, v1);
        auto it = spl.find({v0, v1});
        if (it == spl.end()) continue;
        auto ux = it->second;
        E[i] << -1, -1;
        E.emplace_back(v0, ux);
        E.emplace_back(v1, ux);
        extra++;
      }
      feature_edges.resize(feature_edges.rows() + extra, 2);
      for (auto cnt = 0, i = 0; i < E.size(); i++) {
        if (E[i][0] == -1) continue;
        feature_edges.row(cnt++) = E[i];
      }
    }

    Eigen::VectorXi vertex_reorder;
    prism::cage_utils::reorder_singularity_to_front(ref.V, ref.F, VN,
                                                    omni_singu, vertex_reorder);

    auto v_order = [&reorder = vertex_reorder](auto &i) { i = reorder(i); };
    std::for_each(feature_corners.data(),
                  feature_corners.data() + feature_corners.size(), v_order);
    std::for_each(feature_edges.data(),
                  feature_edges.data() + feature_edges.size(), v_order);
  }

  int num_cons = omni_singu.size();
  for (auto i = num_cons; i < VN.rows(); i++) {
    if (VN.row(i).norm() < 0.1) {
      spdlog::error("Something wrong with reorder");
    }
  }
  constraints_per_face.resize(ref.F.rows()); // singular split may increase. TODO: re-assign some of the constraints. 
  for (int i = 0; i < ref.F.rows(); i++) {
    auto [s, mt, shift] =
        tetra_split_AorB({ref.F(i, 0), ref.F(i, 1), ref.F(i, 2)});
    for (int j = 0; j < 3; j++) ref.F(i, j) = mt[j];
    for (auto c:  constraints_per_face[i]) {
      Vec3d m = constraints_points_bc.row(c);
      // roll_shift_left
      for (int j = 0; j < 3; j++)
        constraints_points_bc(c,j) = m[(j+s)%3];
      }
  }

  RowMatd dsV = ref.V, dsVN = VN;
  RowMati dsF = ref.F;
  Eigen::VectorXd M;
  igl::doublearea(ref.V, ref.F, M);
  spdlog::info("Area {}", M.minCoeff() / 2);
  std::vector<int> face_parent;
  prism::bevel_utils::edge_based_bevel(ref.V, ref.F, VN, feature_edges, dsV,
                                       dsF, dsVN, face_parent);

  spdlog::info("V/F {}/{}-bevel-> {}/{} ", ref.V.rows(), ref.F.rows(),
               dsV.rows(), dsF.rows());

  if (dsVN.hasNaN()) exit(1);

  for (int i = 0; i < dsF.rows(); i++) {
    auto [s, mt, shift] = tetra_split_AorB({dsF(i, 0), dsF(i, 1), dsF(i, 2)});
    for (int j = 0; j < 3; j++) dsF(i, j) = mt[j];
    if (mt[1] < num_cons || mt[2] < num_cons) throw std::runtime_error("only one vertex in each triangle can be singular.");
  }

  ref.aabb.reset(
      new prism::geogram::AABB(ref.V, ref.F, st == SeparateType::kSurface));
  ref.aabb->num_freeze = num_cons;

  RowMatd inner, outer;
  prism::cage_utils::extrude_for_base_and_top(
      dsV, dsF, *ref.aabb, dsVN, num_cons, inner, outer, initial_step);
  eigen2vec(outer, top);
  eigen2vec(inner, base);

  eigen2vec(dsV, mid);
  eigen2vec(dsF, F);
  {
    std::vector<std::vector<int>> VF, VFi;
    igl::vertex_triangle_adjacency(ref.V.rows(), ref.F, VF, VFi);
    std::tie(ref.VF, ref.VFi) = counterclockwise_reorder(ref.F, VF, VFi);
  }

  std::vector<std::vector<int>> VF, VFi;
  igl::vertex_triangle_adjacency(dsV, dsF, VF, VFi);
  prism::cage_utils::recover_positive_volumes(mid, top, F, dsVN, VF, num_cons,
                                              true);
  prism::cage_utils::recover_positive_volumes(base, mid, F, dsVN, VF, num_cons,
                                              false);

  if (st == SeparateType::kShell) {
    spdlog::info(
        "Using hashgrid collision detection. Improvement in progress.");
    prism::cage_utils::safe_shrink(mid, top, F, VF);
    prism::cage_utils::safe_shrink(mid, base, F, VF);
    prism::cage_utils::recover_positive_volumes(mid, top, F, dsVN, VF, num_cons,
                                                true);
    prism::cage_utils::recover_positive_volumes(base, mid, F, dsVN, VF,
                                                num_cons, false);
    top_grid.reset(new prism::HashGrid(top, F));
    base_grid.reset(new prism::HashGrid(base, F));
  }

  auto chains = std::vector<std::list<int>>();
  prism::feature_chains_from_edges(feature_corners, feature_edges, mid.size(),
                                   chains);
  spdlog::info("chains num {}", chains.size());
  [](const auto &chains, auto &meta_edges) {
    for (auto cid = 0; cid < chains.size(); cid++) {
      auto &c = chains[cid];
      for (auto it = std::next(c.begin()); it != c.end(); it++) {
        auto pit = std::prev(it);
        meta_edges[{*pit, *it}] = {cid, {*pit, *it}};
      }
    }
  }(chains, meta_edges);
  // initial track with itself
  std::set<std::pair<int, int>> feature_edge_set;
  for (auto [m, id_chain] : meta_edges) {
    auto [id, chain] = id_chain;
    for (int i = 0; i < chain.size() - 1; i++) {
      auto v0 = chain[i], v1 = chain[(i + 1)];
      if (v0 > v1) std::swap(v0, v1);
      feature_edge_set.insert({v0, v1});
    }
  }
  track_ref.resize(F.size());

  RowMati TT, TTi;
  igl::triangle_triangle_adjacency(ref.F, TT, TTi);
  for (auto i = 0; i < F.size(); i++) {
    auto pf = face_parent[i];
    track_ref[i].insert(pf);
    for (auto j : {0, 1, 2}) {
      auto v0 = ref.F(pf, j), v1 = ref.F(pf, (j + 1) % 3);
      if (v0 > v1) std::swap(v0, v1);
      if (feature_edge_set.find({v0, v1}) != feature_edge_set.end()) continue;
      track_ref[i].insert(TT(pf, j));
    }
  }
}

auto serialize_meta_edges = [](const auto &meta_edges) {
  std::vector<int> flat, ind;
  for (auto [m, data] : meta_edges) {
    ind.push_back(flat.size());
    flat.push_back(m.first);
    flat.push_back(m.second);
    auto [cid, seg] = data;
    flat.push_back(cid);
    flat.insert(flat.end(), seg.begin(), seg.end());
  }
  ind.push_back(flat.size());
  Eigen::VectorXi mFlat = Eigen::Map<Eigen::VectorXi>(&flat[0], flat.size());
  Eigen::VectorXi mInd = Eigen::Map<Eigen::VectorXi>(&ind[0], ind.size());
  return std::tuple(mFlat, mInd);
};

auto deserialize_meta_edges = [](auto &flat, auto &ind) {
  std::map<std::pair<int, int>, std::pair<int, std::vector<int>>> meta;
  auto prev = 0;
  auto begin = flat.begin();
  for (auto cur : ind) {
    if (cur == 0) continue;  // compatible with 0 leading or not.
    // [prev, cur)
    auto itv0 = begin + prev;
    auto itv1 = std::next(itv0);
    std::vector<int> vec;
    std::copy(itv0 + 2 + 1, begin + cur, std::back_inserter(vec));
    meta[{*itv0, *itv1}] = {*(itv0 + 2), vec};
    prev = cur;
  }
  return meta;
};

void PrismCage::serialize(std::string filename, std::any additionals) const {
  RowMatd mbase, mtop, mV;
  RowMati mF;
  vec2eigen(base, mbase);
  vec2eigen(top, mtop);
  vec2eigen(mid, mV);
  vec2eigen(F, mF);

  Eigen::RowVectorXi track_sizes(track_ref.size());
  std::vector<int> track_flat;
  for (int i = 0; i < track_ref.size(); i++) {
    track_sizes[i] = track_ref[i].size();
    for (auto t : track_ref[i]) {
      track_flat.push_back(t);
    }
  }

  H5Easy::File file(filename, H5Easy::File::Overwrite);

  H5Easy::dump(file, "ref.V", ref.V);
  H5Easy::dump(file, "inpV", ref.inpV);
  H5Easy::dump(file, "ref.F", ref.F);
  H5Easy::dump(file, "mbase", mbase);
  H5Easy::dump(file, "zbase", zig_base);
  H5Easy::dump(file, "ztop", zig_top);
  H5Easy::dump(file, "mtop", mtop);
  H5Easy::dump(file, "mV", mV);
  H5Easy::dump(file, "mF", mF);
  H5Easy::dump(file, "track_flat", track_flat);
  H5Easy::dump(file, "track_size", track_sizes);
  auto [meta_edges_flat, meta_edges_ind] = serialize_meta_edges(meta_edges);
  H5Easy::dump(file, "meta_edges_flat", meta_edges_flat); // somehow, vector<int> triggers UB related to const.
  H5Easy::dump(file, "meta_edges_ind", meta_edges_ind);

  SeparateType st =
      ref.aabb->enabled ? SeparateType::kSurface : SeparateType::kNone;
  if (top_grid != nullptr) st = SeparateType::kShell;
  H5Easy::dump(file, "metadata",
               std::vector<int>{1, static_cast<int>(st), ref.aabb->num_freeze});
  spdlog::info("Save with SeparateType {}",
               st == SeparateType::kSurface
                   ? "kSurface"
                   : (st == SeparateType::kShell ? "kShell" : "kNone"));

  if (additionals.has_value())
    std::any_cast<std::function<void(decltype(file) &)>>(additionals)(file);
}

void PrismCage::load_from_hdf5(std::string filename) {
  H5Easy::File file(filename, H5Easy::File::ReadOnly);
  RowMatd mbase, mtop, mV;
  RowMati mF;
  Eigen::RowVectorXi mTracks;
  Eigen::RowVectorXi track_size;
  std::vector<int> metadata;
  ref.V = H5Easy::load<decltype(ref.V)>(file, "ref.V");
  ref.F = H5Easy::load<decltype(ref.F)>(file, "ref.F");
  ref.inpV = H5Easy::load<decltype(ref.inpV)>(file, "inpV");
  mbase = H5Easy::load<decltype(mbase)>(file, "mbase");
  mtop = H5Easy::load<decltype(mtop)>(file, "mtop");
  if (file.exist("zbase")) {
    zig_base = H5Easy::load<decltype(zig_base)>(file, "zbase");
    zig_top = H5Easy::load<decltype(zig_top)>(file, "ztop");
  }
  mV = H5Easy::load<decltype(mV)>(file, "mV");
  mF = H5Easy::load<decltype(mF)>(file, "mF");
  mTracks = H5Easy::load<decltype(mTracks)>(file, "track_flat");
  track_size = H5Easy::load<decltype(track_size)>(file, "track_size");
  SeparateType st = SeparateType::kSurface;
  if (file.exist("metadata")) {
    metadata = H5Easy::load<decltype(metadata)>(file, "metadata");
    st = static_cast<SeparateType>(metadata[1]);
  }
  if (file.exist("meta_edges_flat")) {
    std::vector<int> flat, ind;
    flat = H5Easy::load<decltype(flat)>(file, "meta_edges_flat");
    ind = H5Easy::load<decltype(ind)>(file, "meta_edges_ind");
    spdlog::debug("loading meta edge, with info {} {}", flat.size(),
                  ind.size());
    meta_edges = deserialize_meta_edges(flat, ind);
  }

  eigen2vec(mbase, base);
  eigen2vec(mtop, top);
  eigen2vec(mV, mid);
  eigen2vec(mF, F);

  // after
  ref.aabb = std::make_unique<prism::geogram::AABB>(
      ref.V, ref.F, st == SeparateType::kSurface);
  for (int i = 0; i < mid.size(); i++) {
    if (base[i] != top[i]) {
      ref.aabb->num_freeze = i;
      break;
    }
  }
  spdlog::info("AABB {}, SeparateType {}",
               ref.aabb->enabled ? "enabled" : "disabled",
               st == SeparateType::kSurface
                   ? "kSurface"
                   : (st == SeparateType::kShell ? "kShell" : "kNone"));
  if (st == SeparateType::kShell) {
    spdlog::info("Loading HashGrid.");
    top_grid.reset(new prism::HashGrid(top, F));
    base_grid.reset(new prism::HashGrid(base, F));
  }
  spdlog::info("Loaded singularity {}", ref.aabb->num_freeze);
  track_ref.clear();
  for (int i = 0, cur = 0; i < track_size.size(); i++) {
    auto ts = track_size[i];
    std::set<int> cur_track(mTracks.data() + cur, mTracks.data() + cur + ts);
    track_ref.emplace_back(cur_track);
    cur += ts;
  }

  {
    std::vector<std::vector<int>> VF, VFi;
    igl::vertex_triangle_adjacency(ref.V.rows(), ref.F, VF, VFi);
    std::tie(ref.VF, ref.VFi) = counterclockwise_reorder(ref.F, VF, VFi);
  }
}

PrismCage::PrismCage(std::string filename) {
  prism::geo::init_geogram();
  namespace fs = std::filesystem;
  assert(fs::path(filename).extension() == ".h5" ||
         fs::path(filename).extension() == ".init");
  spdlog::info("Loading From .H5");
  load_from_hdf5(filename);
}


void PrismCage::cleanup_empty_faces(Eigen::VectorXi& NI, Eigen::VectorXi& NJ)
{
    std::vector<int> face_map;
    cleanup_empty_faces(NI, NJ, face_map);
}

void PrismCage::cleanup_empty_faces(
    Eigen::VectorXi& NI,
    Eigen::VectorXi& NJ,
    std::vector<int>& face_map)
{
    // this is called after collapse pass.
    constexpr auto mark_zero_rows = [](const auto& vecF, RowMati& mat) {
        std::vector<Vec3i> newF;
        newF.reserve(vecF.size());
        for (int i = 0; i < vecF.size(); i++) {
            auto& f = vecF[i];
            if (f[0] != f[1])
                newF.push_back(f);
            else
                newF.push_back({-1, -1, -1});
        }
        mat = Eigen::Map<RowMati>(newF[0].data(), newF.size(), 3);
    };

    RowMati mF;
    mark_zero_rows(F, mF);
    igl::remove_unreferenced(mid.size(), mF, NI, NJ);

    // assuming NJ is sorted ascending
    for (int i = 0; i < NJ.size() - 1; i++) assert(NJ[i] < NJ[i + 1]);

    for (int i = 0; i < NJ.size(); i++) {
        mid[i] = mid[NJ[i]];
        base[i] = base[NJ[i]];
        top[i] = top[NJ[i]];
    }
    mid.resize(NJ.size());
    base.resize(NJ.size());
    top.resize(NJ.size());

    int cur = 0;
    face_map.clear();
    face_map.resize(F.size(), -1);
    for (int i = 0; i < F.size(); i++) {
        if (F[i][0] == F[i][1]) continue;
        if (track_ref[i].size() == 0) spdlog::error("Zero Tracer");
        if (i != cur) track_ref[cur] = std::move(track_ref[i]);
        for (int j = 0; j < 3; j++) F[cur][j] = NI[F[i][j]];
        if (F[cur][0] > F[cur][1] || F[cur][0] > F[cur][2])
            spdlog::error("v0 v1 v2 order wrong at {}", cur);
        face_map[i] = cur;
        cur++;
    }
    track_ref.resize(cur);
    F.resize(cur);

    auto& vid_map = NI;
    // feature meta edges
    std::map<std::pair<int, int>, std::pair<int, std::vector<int>>> new_metas;
    for (auto m : meta_edges) {
        auto [u0, u1] = m.first;
        new_metas[{vid_map[u0], vid_map[u1]}] = m.second;
    }
    meta_edges = std::move(new_metas);

    // hash grid update.
    if (top_grid != nullptr) {
        top_grid->update_after_collapse();
        base_grid->update_after_collapse();
        assert(top_grid->face_stores.size() == F.size());
        assert(base_grid->face_stores.size() == F.size());
    }
}