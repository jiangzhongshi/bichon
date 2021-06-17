#ifndef PRISM_CURVE_CURVE_UTILS_HPP
#define PRISM_CURVE_CURVE_UTILS_HPP

#include <igl/list_to_matrix.h>
#include <igl/matrix_to_list.h>

#include <any>
#include <highfive/H5Easy.hpp>
#include <prism/common.hpp>
#include <vector>

#include "curve_common.hpp"
namespace prism::geogram {
struct AABB;
};
namespace prism {
struct Hit;
};

namespace prism::curve {
struct HelperTensors {
  RowMatd bern_val_samples, elev_lag_from_bern;
  std::array<RowMatd, 2> duv_samples;
  Eigen::Matrix<int, -1, 3> tri_codec, tri_elev_codec;
  struct VolumeData {
    std::vector<RowMatd> vec_dxyz;
    RowMatd vol_bern_from_lagr, vol_jac_bern_from_lagr;
    Eigen::Matrix<int, -1, 4> vol_codec, vol_jac_codec;
  } volume_data;
  int tri_order = 3;
  int level = 3;
  struct UpsampleData {
    RowMatd unitV;
    RowMati unitF;
    Vec3i vert_id;
    std::array<std::vector<int>, 3> edge_id;
    std::vector<int> face_id;
  } upsample_data;
  static std::shared_ptr<HelperTensors> tensors_;
  static void init(int tri_order, int level);
};

inline HelperTensors &magic_matrices(int tri_order = -1, int level = -1) {
  if (!HelperTensors::tensors_) {
    if (tri_order == -1) {
      throw 1;
    }
    HelperTensors::init(tri_order, level);
  }
  if (tri_order != -1) {
    if (HelperTensors::tensors_->tri_order != tri_order ||
        HelperTensors::tensors_->level != level) {
      // unmatched.
      throw 1;
    }
  }
  return *HelperTensors::tensors_;
}
}  // namespace prism::curve
namespace prism::curve {

// linear interpolation for trivial HO nodes.
// index storing codec, starts with 0,0,0,0
// lin[codec].mean(axis=1)
inline RowMatd linear_elevate(const RowMatd &lin, const RowMati &codec) {
  auto n = codec.rows(), e = codec.cols();
  assert(lin.cols() == 3 && "coordinate in 3d");
  assert((lin.rows() == 3 || lin.rows() == 4) &&
         "linear base, can be tri or tet");
  RowMatd result = RowMatd::Zero(n, lin.cols());
  for (int i = 0; i < n; i++) {
    std::vector<Vec3d> stable_sum(e);
    for (int j = 0; j < e; j++) stable_sum[j] = lin.row(codec(i, j));
    std::sort(stable_sum.begin(), stable_sum.end(), MatLexComp<Vec3d>());
    result.row(i) = std::accumulate(stable_sum.begin(), stable_sum.end(),
                                    Vec3d::Zero().eval());
  }
  result /= e;
  return result;
};

std::vector<RowMatd> initialize_cp(const std::vector<Vec3d> &mid,
                                   const std::vector<Vec3i> &F,
                                   const RowMati &codec);

// Fit high order for one ring neighborhood
// Input:
//    tri10_lv5: #hodes (e.g. 10 for cubic) by #sample (e.g. for upsample lv 5)
//    b: #nb*#sample by 3 raycast result to fit.
//    duplicates resolution are encapsulated in additional.
std::vector<RowMatd> ring_fit(
    const std::vector<Vec3i> &F, std::vector<RowMatd> &complete_cp,
    const MatLexMap<Eigen::VectorXi, Vec3d> &nb_cp_map,
    const RowMatd &tri10_lv5, const RowMatd &b, const RowMati &tri10_codec,
    const std::any &additional, RowMatd &residual_in, RowMatd &residual_out);

std::tuple<RowMati, std::vector<std::pair<int, int>>, RowMati> upsampled_uv(
    const std::vector<Vec3i> &F, std::vector<int> &res_fid, std::vector<Vec3d> &res_uv);

// RowMatd sample_hit(const std::vector<Vec3d> &base,
//                    const std::vector<Vec3d> &top, const std::vector<Vec3i>
//                    &F, const std::vector<int> &sp_fid, const
//                    std::vector<Vec3d> &sp_uv, const prism::geogram::AABB
//                    &reftree);

// void sample_hit(const std::vector<Vec3d> &base, const std::vector<Vec3d>
// &top,
//                 const std::vector<Vec3i> &F, const std::vector<int> &sp_fid,
//                 const std::vector<Vec3d> &sp_uv,
//                 const prism::geogram::AABB &reftree, std::vector<prism::Hit>
//                 &);

void sample_hit_discrete(
    const std::vector<Vec3d> &base, const std::vector<Vec3d> &mid,
    const std::vector<Vec3d> &top, const std::vector<Vec3i> &F,
    const std::vector<int> &sp_fid, const std::vector<Vec3d> &sp_uv,
    const RowMatd &refV, const RowMati &refF,
    const prism::geogram::AABB &reftree, const std::set<int> &tri_list,
    std::vector<prism::Hit> &hits);

// supported disabled reftree.
void sample_hit(const std::vector<Vec3d> &base, const std::vector<Vec3d> &mid,
                const std::vector<Vec3d> &top, const std::vector<Vec3i> &F,
                const std::vector<int> &sp_fid, const std::vector<Vec3d> &sp_uv,
                const RowMatd &refV, const RowMati &refF,
                const prism::geogram::AABB &reftree,
                const std::set<int> &tri_list, std::vector<prism::Hit> &hits);

// surface control points to deduce the prismatic nodes.
// Prisms are represented as decomposed tetrahedra (6 if non-singularity and 4
// otherwise)
std::vector<RowMatd> surface_to_decomposed_tetra(const RowMatd &base,
                                                 const RowMatd &mid_ho,
                                                 const RowMatd &top,
                                                 bool degenerate, bool type,
                                                 const RowMati &source_cod,
                                                 const RowMati &target_cod);

bool elevated_positive(
    const std::vector<Vec3d> &base, const std::vector<Vec3d> &top,
    const std::vector<Vec3i> &F,
    bool recurse_check, const std::vector<RowMatd> &local_cp);

constexpr auto load_cp = [](std::string filename) {
  std::vector<RowMatd> complete_cp;
  H5Easy::File file(filename, H5Easy::File::ReadOnly);
  auto vec_cp = H5Easy::load<std::vector<std::vector<std::vector<double>>>>(
      file, "complete_cp");
  complete_cp.resize(vec_cp.size());
  for (int i = 0; i < complete_cp.size(); i++)
    igl::list_to_matrix(vec_cp[i], complete_cp[i]);
  return complete_cp;
};

constexpr auto save_cp_inp =
    [](auto &complete_cp, auto &inpV) -> std::function<void(HighFive::File &)> {
  return [complete_cp, inpV](HighFive::File &file) {
    if (complete_cp.size() == 0) return;
    std::vector<std::vector<std::vector<double>>> vec_cp(complete_cp.size());
    for (int i = 0; i < complete_cp.size(); i++)
      igl::matrix_to_list(complete_cp[i], vec_cp[i]);

    H5Easy::dump(file, "complete_cp", vec_cp);
    H5Easy::dump(file, "inpV", inpV);
  };
};

constexpr auto save_cp =
    [](auto &complete_cp) -> std::function<void(HighFive::File &)> {
  return [complete_cp](HighFive::File &file) {
    if (complete_cp.size() == 0) return;
    std::vector<std::vector<std::vector<double>>> vec_cp(complete_cp.size());
    for (int i = 0; i < complete_cp.size(); i++)
      igl::matrix_to_list(complete_cp[i], vec_cp[i]);

    H5Easy::dump(file, "complete_cp", vec_cp);
  };
};
}  // namespace prism::curve
#endif
