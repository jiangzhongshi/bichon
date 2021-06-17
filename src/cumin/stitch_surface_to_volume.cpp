#include "stitch_surface_to_volume.hpp"

#include <fmt/ostream.h>
#include <spdlog/spdlog.h>

#include "curve_common.hpp"
#include "curve_utils.hpp"
#include "inversion_check.hpp"
bool prism::curve::stitch_surface_to_volume(
    const RowMatd &base, const RowMatd &top, const RowMati &F_sh,
    const std::vector<RowMatd> &complete_cp,
    const std::tuple<const RowMatd &, const RowMati &, const RowMati &,
                     const std::vector<RowMatd> &> &mat_helpers,
    const Eigen::MatrixXd &Vmsh, const Eigen::MatrixXi &Tmsh,
    RowMatd &final_nodes, RowMati &p4T) {
  for (auto i = 0; i < base.rows(); i++) {
    if (base.row(i) != Vmsh.row(i)) {
      spdlog::critical("Stitch Precondition.");
      throw 1;
    }
  }
  spdlog::info("Setting up stitch: base {}, Fsh {}, Vmsh {} Tmsh {}",
               base.rows(), F_sh.rows(), Vmsh.rows(), Tmsh.rows());
  auto &helper = prism::curve::magic_matrices();
  auto &tri15lag_from_tri10bern = helper.elev_lag_from_bern;
  auto &dxyz = helper.volume_data.vec_dxyz;
  auto tri4_cod = TRI_CODEC.at(helper.tri_order + 1);
  auto tet4_cod = TET_CODEC.at(helper.tri_order + 1);
  assert(tri15lag_from_tri10bern.rows() !=
         tri15lag_from_tri10bern.cols());  // switch to single test
  std::vector<Eigen::VectorXi> all_tuples;
  std::vector<Vec3d> all_nodes;
  int num_tets = Tmsh.rows();

  for (int i = 0; i < F_sh.rows(); i++) {
    auto &cp = complete_cp[i];
    auto mf = F_sh.row(i);
    RowMatd f_base(3, 3), f_top(3, 3);
    for (int j = 0; j < 3; j++) {
      f_base.row(j) = base.row(mf[j]);
      f_top.row(j) = top.row(mf[j]);
    }
    auto type = mf[1] > mf[2];

    // 6T (35N 3D), or singularity case 4T.
    auto singular = (f_base.row(0) == f_top.row(0));
    auto tens = prism::curve::surface_to_decomposed_tetra(
        f_base, tri15lag_from_tri10bern * cp, f_top, singular, type, tri4_cod,
        tet4_cod);
    if (singular) {
      spdlog::trace("singularity handling here f={}", i);
    }
    num_tets += tens.size() / 2;
    // This is in fact in-consistent across opposing triangles, due to floating
    // point evaluations of

    auto &tet = type ? TETRA_SPLIT_A : TETRA_SPLIT_B;
    assert(tet4_cod(0, 0) == 0 && "Contribution Codec (0000,0001,1234 etc).");
    for (auto tt = 0; tt < tens.size() / 2; tt++) {
      auto cur_tet = tet[tt];
      if (singular)
        cur_tet = tet[tt + 1];  // for singularity, move after the first.
      for (auto c = 0; c < tet4_cod.rows(); c++) {
        Eigen::VectorXi codec(tet4_cod.cols());
        for (auto x = 0; x < tet4_cod.cols(); x++) {
          auto p_cod = cur_tet[tet4_cod(c, x)];
          codec[x] =
              mf[p_cod % 3] +
              (p_cod / 3) *
                  base.rows();  // 012 base, 345 top index to be pushed back.
        }
        all_nodes.emplace_back(tens[tt].row(c));
        std::sort(codec.data(), codec.data() + codec.size());
        all_tuples.emplace_back(codec);
      }
    }
  };
  spdlog::trace("tuples {} == nodes {}", all_tuples.size(), all_nodes.size());
  // adding linear elevated trivial nodes
  int real_id = base.rows();
  int layer_ids = real_id * 2;
  for (auto t = 0; t < Tmsh.rows(); t++) {
    // note that arithmetic is done on large indices.
    Eigen::Vector4i cur_tet = Tmsh.row(t);
    for (auto c = 0; c < 4; c++) {
      auto &v = cur_tet[c];
      if (v >= real_id) v += layer_ids;
    }

    for (auto c = 0; c < tet4_cod.rows(); c++) {
      Eigen::VectorXi codec(tet4_cod.cols());
      std::vector<Vec3d> stable_sum;
      for (auto x = 0; x < tet4_cod.cols(); x++) {
        codec[x] = cur_tet[tet4_cod(c, x)];
        stable_sum.emplace_back(Vmsh.row(Tmsh(t, tet4_cod(c, x))));
      }

      std::sort(codec.data(), codec.data() + codec.size());

      // order-invariant sum
      std::sort(stable_sum.begin(), stable_sum.end(), MatLexComp<Vec3d>());
      Vec3d new_nodes = std::accumulate(stable_sum.begin(), stable_sum.end(),
                                        Vec3d::Zero().eval()) /
                        codec.size();
      all_nodes.emplace_back(new_nodes);
      all_tuples.emplace_back(codec);
    }
  }

  std::vector<Vec3d> uniq_nodes;

  // resolve duplicates.
  CodecMap resolve;
  int cnt = 0, ind = 0;  // total and unique.
  int n_nodes = tet4_cod.rows();
  p4T.setZero(num_tets, n_nodes);
  for (auto t : all_tuples) {
    auto it = resolve.lower_bound(t);
    if (it == resolve.end() || it->first != t) {
      it = resolve.emplace_hint(it, t, ind++);
      uniq_nodes.emplace_back(all_nodes[cnt]);
      assert(uniq_nodes.size() == ind);
    }
    // if (all_nodes[cnt] != uniq_nodes[it->second]) {
    //   auto rec = it->second;
    //   spdlog::error("mismatch cnt{}-{}:{}-{} tuple {}vs uniq {}", cnt/35,
    //   cnt%35,  rec/35, rec%35, t.transpose(), all_nodes[cnt] -
    //   uniq_nodes[it->second]); throw 1;
    // }
    p4T(cnt / n_nodes, cnt % n_nodes) = it->second;
    cnt++;
  }

  final_nodes = Eigen::Map<RowMatd>(uniq_nodes[0].data(), uniq_nodes.size(), 3);
  {  // debug
    for (auto i = 0; i < p4T.rows(); i++) {
      RowMatd nodes(n_nodes, 3);
      for (auto j = 0; j < n_nodes; j++) nodes.row(j) = final_nodes.row(p4T(i, j));
      if (!prism::curve::tetrahedron_inversion_check(
              nodes,helper.volume_data.vol_codec, helper.volume_data.vol_jac_codec, helper.volume_data.vol_bern_from_lagr,
                helper.volume_data.vol_jac_bern_from_lagr)) {
        spdlog::critical("blocked by recursive {}A", i);
        spdlog::info(nodes);
        exit(1);
        return false;
      }
    }
  }
  return true;
}