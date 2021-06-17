#include "cage_utils.hpp"

#include <igl/adjacency_list.h>
#include <igl/boundary_facets.h>
#include <igl/boundary_loop.h>
#include <igl/grad.h>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>
#include <igl/remove_unreferenced.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/vertex_triangle_adjacency.h>
#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <prism/local_operations/validity_checks.hpp>
#include <set>

#ifdef CGAL_QP
#include "cgal/QP.hpp"
#else
#include "osqp/osqp_normal.hpp"
#endif

#include "local_operations/retain_triangle_adjacency.hpp"
#include "predicates/positive_prism_volume_12.hpp"
#include "predicates/triangle_triangle_intersection.hpp"

#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/ostr.h>

bool prism::cage_utils::all_volumes_are_positive(const std::vector<Vec3d> &base,
                                                 const std::vector<Vec3d> &mid,
                                                 const std::vector<Vec3d> &top,
                                                 const std::vector<Vec3i> &F,
                                                 int num_cons) {
  std::vector<Vec3i> failure;
  for (auto face : F) {
    auto [v0, v1, v2] = face;
    if (!prism::predicates::positive_prism_volume(
            {base[v0], base[v1], base[v2], mid[v0], mid[v1], mid[v2]},
            {v0 < num_cons, v1 < num_cons, v2 < num_cons})) {
      spdlog::trace("BM failure {} {} {}", v0, v1, v2);
      failure.push_back(face);
    }
    if (!prism::predicates::positive_prism_volume(
            {mid[v0], mid[v1], mid[v2], top[v0], top[v1], top[v2]},
            {v0 < num_cons, v1 < num_cons, v2 < num_cons})) {
      spdlog::trace("MT failure {} {} {}", v0, v1, v2);
      failure.push_back(face);
    }
    if (failure.size() > 0) {
      spdlog::dump_backtrace();
      for (auto f : failure) {
        spdlog::error("{} {} {}", f[0], f[1], f[2]);
        spdlog::error("B\n{}\n{}\n{}",
                      base[f[0]].format(Eigen::IOFormat(Eigen::FullPrecision)),
                      base[f[1]].format(Eigen::IOFormat(Eigen::FullPrecision)),
                      base[f[2]].format(Eigen::IOFormat(Eigen::FullPrecision)));
        spdlog::error("M\n{}\n{}\n{}",
                      mid[f[0]].format(Eigen::IOFormat(Eigen::FullPrecision)),
                      mid[f[1]].format(Eigen::IOFormat(Eigen::FullPrecision)),
                      mid[f[2]].format(Eigen::IOFormat(Eigen::FullPrecision)));
        spdlog::error("T\n{}\n{}\n{}",
                      top[f[0]].format(Eigen::IOFormat(Eigen::FullPrecision)),
                      top[f[1]].format(Eigen::IOFormat(Eigen::FullPrecision)),
                      top[f[2]].format(Eigen::IOFormat(Eigen::FullPrecision)));
        spdlog::error("T-M\n{}\n{}\n{}",
                      (top[f[0]] - mid[f[0]])
                          .format(Eigen::IOFormat(Eigen::FullPrecision)),
                      (top[f[1]] - mid[f[1]])
                          .format(Eigen::IOFormat(Eigen::FullPrecision)),
                      (top[f[2]] - mid[f[2]])
                          .format(Eigen::IOFormat(Eigen::FullPrecision)));
        spdlog::error("M-B\n{}\n{}\n{}",
                      (mid[f[0]] - base[f[0]])
                          .format(Eigen::IOFormat(Eigen::FullPrecision)),
                      (mid[f[1]] - base[f[1]])
                          .format(Eigen::IOFormat(Eigen::FullPrecision)),
                      (mid[f[2]] - base[f[2]])
                          .format(Eigen::IOFormat(Eigen::FullPrecision)));
      }
      spdlog::error("Error in main: volume negative");
      exit(1);
    }
  }
  return true;
}

bool prism::cage_utils::most_visible_normals(const RowMatd &V, const RowMati &F,
                                             RowMatd &VN,
                                             std::set<int> &omni_saddle,
                                             double tolerance_singularity) {
  RowMatd FN;
  igl::per_vertex_normals(V, F, VN);
  igl::per_face_normals_stable(V, F, FN);
  if (FN.hasNaN()) {
    spdlog::error("libigl nan");
    exit(1);
  } 
 {
  VN.setZero();
 }
  std::set<int> bad_verts;
  for (int i = 0; i < F.rows(); i++) {
    for (int j = 0; j < 3; j++) {
      if (VN.row(F(i, j)).dot(FN.row(i)) < 1e-1) bad_verts.insert(F(i, j));
    }
  }
  // spdlog::info("Bad Vertices {}", bad_verts.size());
  if (bad_verts.empty()) return true;

  std::vector<std::vector<int>> VF, VFi;
  igl::vertex_triangle_adjacency(V, F, VF, VFi);
  Eigen::VectorXd M;
  igl::doublearea(V, F, M);
  for (auto v : bad_verts) {
    auto normal =
#ifdef CGAL_QP
        cgal::qp_normal
#else
        prism::osqp_normal
#endif
        (FN, VF[v]);
#ifdef CGAL_QP
    // std::cout << "CGAL_QP" << std::endl;
#endif
    if (normal.hasNaN()) {
      spdlog::error("CGAL VN{} nan", v);
      exit(1);
    }
    if (normal.norm() < 1e-1) {
      omni_saddle.insert(v);
      VN.row(v) = normal;

      continue;
    }
    // verify normal
    for (auto f : VF[v]) {
      auto dot = normal.dot(FN.row(f));
      spdlog::trace("dot {}", dot);
      if (dot < tolerance_singularity) {
        spdlog::debug("Numerical Singularity: dot={},f={}", dot, f);
        normal.setZero();
        break;
      }
    }
    if (normal.norm() < 1e-1) {
      omni_saddle.insert(v);
      VN.row(v) = normal;

      continue;
    }

    VN.row(v) = normal;
  }
  return omni_saddle.empty();
}

constexpr auto log_minmax = [](auto &&iterable, std::string name) {
  auto minmax = std::minmax_element(iterable.begin(), iterable.end());
  spdlog::info("{} min {} max {}", name, *minmax.first, *minmax.second);
};

std::vector<double> prism::cage_utils::volume_extrude_steps(
    const RowMatd &V, const RowMati &F, const RowMatd &N, bool outward,
    int num_cons, const std::vector<double> &ray_step) {
  log_minmax(ray_step, "ray");

  // min pool
  std::vector<double> face_step(F.rows(), 1.);
  for (int i = 0; i < F.rows(); i++) {
    for (int j = 0; j < 3; j++)
      face_step[i] = std::min(face_step[i], ray_step[F(i, j)]);
  }
  auto quit_if_too_small = [&outward](double alpha, double eps) {
    if (alpha < eps) {
      spdlog::dump_backtrace();
      spdlog::error("pos-check during {}: alpha too small {}",
                   outward ? "Out" : "In", alpha);
      exit(1);
    }
  };

  for (int i = 0; i < F.rows(); i++) {
    double alpha = face_step[i];
    auto [v0, v1, v2] = std::forward_as_tuple(F(i, 0), F(i, 1), F(i, 2));
    spdlog::trace("V \n{}\n{}\n{}",
                  V.row(v0).format(Eigen::IOFormat(Eigen::FullPrecision)),
                  V.row(v1).format(Eigen::IOFormat(Eigen::FullPrecision)),
                  V.row(v2).format(Eigen::IOFormat(Eigen::FullPrecision)));
    spdlog::trace("N \n{}\n{}\n{}",
                  N.row(v0).format(Eigen::IOFormat(Eigen::FullPrecision)),
                  N.row(v1).format(Eigen::IOFormat(Eigen::FullPrecision)),
                  N.row(v2).format(Eigen::IOFormat(Eigen::FullPrecision)));
    spdlog::trace("{}-{}-{}", v0, v1, v2);
    // volume step
    if (outward) {
      while (!prism::predicates::positive_prism_volume(
          {V.row(v0), V.row(v1), V.row(v2), V.row(v0) + alpha * N.row(v0),
           V.row(v1) + alpha * N.row(v1), V.row(v2) + alpha * N.row(v2)},
          {v0 < num_cons, v1 < num_cons, v2 < num_cons})) {
        alpha = alpha * 0.8;
        quit_if_too_small(alpha, 1e-16);
      }
    } else {
      while (!prism::predicates::positive_prism_volume(
          {V.row(v0) + alpha * N.row(v0), V.row(v1) + alpha * N.row(v1),
           V.row(v2) + alpha * N.row(v2), V.row(v0), V.row(v1), V.row(v2)},
          {v0 < num_cons, v1 < num_cons, v2 < num_cons})) {
        alpha = alpha * 0.8;
        quit_if_too_small(alpha, 1e-16);
      }
    }

    face_step[i] = alpha;
  }
  return std::move(face_step);
}

std::vector<double> prism::cage_utils::extrude_along_normals(
    const RowMatd &V, const RowMati &F, const prism::geogram::AABB &tree,
    const RowMatd &N_in, bool outward, int num_cons,
    double initial_step = 1e-4) {
  RowMatd N = N_in;
  if (!outward) N = -N;

  // Ray Cast
  std::vector<double> ray_step(V.rows(), initial_step);
  for (int i = num_cons; i < tree.geo_vertex_ind.size(); i++) {
    // the beveled vertex is not needed.
    ray_step[i] =
        tree.ray_length(V.row(i), N.row(i), initial_step, i /*ignore*/);
    if (ray_step[i] < 1e-12) {
      spdlog::dump_backtrace();
      spdlog::error(
          "should not happen with preconditions at place. Likely to be "
          "intersection computation problem if this is triggered.");
      ray_step[i] = 1e-9;
      continue;
    }
  }
  return volume_extrude_steps(V, F, N, outward, num_cons, ray_step);
}

void prism::cage_utils::iterative_retract_normal(
    const RowMatd &V, const RowMati &F, const prism::geogram::AABB &tree,
    const RowMatd &N_in, bool outward, int num_cons,
    std::vector<double> &alpha) {
  std::vector<std::vector<int>> VF, VFi;
  igl::vertex_triangle_adjacency(V, F, VF, VFi);
  RowMatd N = N_in;
  if (!outward) N = -N;

  std::set<int> collide;
  for (int i = 0; i < F.rows(); i++) collide.insert(i);

  while (!collide.empty()) {
    auto i = *collide.begin();
    auto [v0, v1, v2] = std::forward_as_tuple(F(i, 0), F(i, 1), F(i, 2));
    auto all_clean = true;
    assert(v0 < v1);                           // well ordered
    assert(v1 >= num_cons && v2 >= num_cons);  // maximum one constraint
    while (tree.intersects_triangle(
        {V.row(v0) + alpha[v0] * N.row(v0), V.row(v1) + alpha[v1] * N.row(v1),
         V.row(v2) + alpha[v2] * N.row(v2)},
        v0 < num_cons)) {
      double tol = std::pow(2, -46);
      all_clean = false;
      alpha[v0] = std::max(alpha[v0] / 2, tol);
      alpha[v1] = std::max(alpha[v1] / 2, tol);
      alpha[v2] = std::max(alpha[v2] / 2, tol);
      spdlog::trace("Backtracing F{}, alpha={}, {}, {}", i, alpha[v0],
                    alpha[v1], alpha[v2]);
      if (alpha[v0] <= tol && alpha[v1] <= tol && alpha[v2] <= tol) {
        spdlog::dump_backtrace();
        spdlog::error("N \n{} \n{} \n{}",
                      N.row(v0).format(Eigen::IOFormat(Eigen::FullPrecision)),
                      N.row(v1).format(Eigen::IOFormat(Eigen::FullPrecision)),
                      N.row(v2).format(Eigen::IOFormat(Eigen::FullPrecision)));
        spdlog::error("F{} with {} {} {} outer {}", i, v0, v1, v2, outward);
        spdlog::error("\n{}\n{}\n{}",
                      V.row(v0).format(Eigen::IOFormat(Eigen::FullPrecision)),
                      V.row(v1).format(Eigen::IOFormat(Eigen::FullPrecision)),
                      V.row(v2).format(Eigen::IOFormat(Eigen::FullPrecision)));
        exit(1);
      }
    }
    if (!all_clean) {
      for (int j = 0; j < 3; j++)
        // for f
        collide.insert(VF[F(i, j)].begin(), VF[F(i, j)].end());
    }
    collide.erase(i);
  }
};

void cap_smooth(const RowMati &F, std::vector<double> &steps) {
  for (int i = 0; i < F.rows(); i++) {
    for (int j = 0; j < 3; j++) {
      auto sum2 = (steps[F(i, (j + 1) % 3)] + steps[F(i, (j + 2) % 3)]);
      while (steps[F(i, j)] > 2 * sum2) steps[F(i, j)] = sum2;
    }
  }
}

void prism::cage_utils::recover_positive_volumes(
    std::vector<Vec3d> &base, std::vector<Vec3d> &top,
    const std::vector<Vec3i> &F, const RowMatd &VN,
    const std::vector<std::vector<int>> &VF, int num_cons, bool move_top) {
  std::set<int> bad_prisms;
  auto checker = [&](auto s) {
    auto [v0, v1, v2] = F[s];
    return prism::predicates::positive_prism_volume(
        {base[v0], base[v1], base[v2], top[v0], top[v1], top[v2]},
        {v0 < num_cons, false, false}, false);
  };
  for (auto s = 0; s < F.size(); s++) {
    if (!checker(s)) bad_prisms.insert(s);
  }
  while (!bad_prisms.empty()) {
    spdlog::debug("bad {}", bad_prisms.size());
    auto it = bad_prisms.begin();
    auto s = *it;
    if (checker(s)) {
      bad_prisms.erase(s);
      continue;
    }
    Eigen::Vector3d heights;
    auto minheight = 1.;
    auto min_id = -1;
    for (auto i = 0; i < 3; i++) {
      auto v = F[s][i];
      heights[i] = (top[v] - base[v]).norm();
      if (heights[i] < minheight) {
        minheight = heights[i];
        min_id = i;
      }
    }
    auto cnt = 0;
    while (!checker(s)) {
      for (auto i = 0; i < 3; i++) {
        if ( i!= min_id) {
          heights[i] = (heights[i] + minheight) / 2;
          auto v = F[s][i];
          if (move_top)
            top[v] = base[v] + heights[i] * VN.row(v);
          else
            base[v] = top[v] - heights[i] * VN.row(v);
        }
      }
      if (cnt++ > 300) {
        spdlog::error("Recover Positive: Too many attempts {}", s);
        spdlog::error("Minimum Height {} heights {}", minheight, heights);
        throw std::runtime_error("Must be wrong");
      }
    }
    for (auto i = 0; i < 3; i++) {
       if ( i!= min_id) {
        auto v = F[s][i];
        set_add_to(VF[v], bad_prisms);
      }
    }
  }
}

void cap_percentile(std::vector<double> &step_in) {
  return;
  std::vector<double> step = step_in;
  std::sort(step.begin(), step.end());
  int num = step.size();
  double sum = 0, mean = 0, var = 0;
  for (auto &s : step) sum += s;
  mean = sum / num;
  for (auto &s : step) var += std::pow(s - mean, 2);
  var = std::sqrt(var / num);

  if (num < 10) return;
  double percentile = step[0.9 * num];
  spdlog::info("percentile 90:{}, 75:{}, 50:{}, mean {} std {}",
               step[0.9 * num], step[0.75 * num], step[0.5 * num], mean, var);
  for (auto &s : step_in) s = std::min(s, percentile);
}

void prism::cage_utils::extrude_for_base_and_top(
    const RowMatd &V, const RowMati &F, const prism::geogram::AABB &tree,
    const RowMatd &N, int num_cons, RowMatd &inner, RowMatd &outer,
    double initial_step) {
  auto out_steps =
      extrude_along_normals(V, F, tree, N, true, num_cons, initial_step);
  auto in_steps =
      extrude_along_normals(V, F, tree, N, false, num_cons, initial_step);

  // pool back to vertices
  std::vector<double> v_out(V.rows(), initial_step),
      v_in(V.rows(), initial_step);
  for (int i = 0; i < F.rows(); i++) {
    for (int j = 0; j < 3; j++) {
      auto vid = F(i, j);
      v_out[vid] = std::min(v_out[vid], out_steps[i]);
      v_in[vid] = std::min(v_in[vid], in_steps[i]);
    }
  }
  log_minmax(v_out, "v_out");
  log_minmax(v_in, "v_in");
  iterative_retract_normal(V, F, tree, N, true, num_cons, v_out);
  iterative_retract_normal(V, F, tree, N, false, num_cons, v_in);
  log_minmax(v_out, "v_out_retract");
  log_minmax(v_in, "v_in_retract");
  auto v_in_copy = v_in;
  auto v_out_copy = v_out;
  for (int i = 0; i < v_out.size(); i++) {
    v_out[i] = std::min(v_out[i], v_in[i]);
    v_in[i] = v_out[i];
  }
  for (int i = 0; i < 3; i++) {
    cap_smooth(F, v_out);
    cap_smooth(F, v_in);
  }
  log_minmax(v_out, "v_out_cap");
  log_minmax(v_in, "v_in_cap");
  inner =
      V -
      Eigen::Map<Eigen::VectorXd>(v_in.data(), v_in.size()).asDiagonal() * N;
  outer =
      V +
      Eigen::Map<Eigen::VectorXd>(v_out.data(), v_out.size()).asDiagonal() * N;

  // This is a simple solution for a rare awkward boundary condition,
  // which leads to numerical self intersection.
  auto verify = [&F = std::as_const(F), &inner, &outer,
                 &tree = std::as_const(tree)]() {
    for (int i = 0; i < F.rows(); i++) {
      auto v0 = F(i, 0), v1 = F(i, 1), v2 = F(i, 2);
      if (tree.intersects_triangle(
              {inner.row(v0), inner.row(v1), inner.row(v2)},
              v0 < tree.num_freeze))
        return false;
      if (tree.intersects_triangle(
              {outer.row(v0), outer.row(v1), outer.row(v2)},
              v0 < tree.num_freeze))
        return false;
    }
    return true;
  };
  if (!verify()) {
    v_out = v_out_copy;
    v_in = v_in_copy;
    inner =
        V -
        Eigen::Map<Eigen::VectorXd>(v_in.data(), v_in.size()).asDiagonal() * N;
    outer =
        V +
        Eigen::Map<Eigen::VectorXd>(v_out.data(), v_out.size()).asDiagonal() *
            N;
  }
}

void prism::cage_utils::tetmesh_from_prismcage(const std::vector<Vec3d> &base,
                                               const std::vector<Vec3d> &top,
                                               const std::vector<Vec3i> &F,
                                               std::vector<Vec3d> &V,
                                               std::vector<Vec4i> &T) {
  int vnum = base.size();
  T.resize(3 * F.size());
  V.resize(2 * vnum);
  for (int i = 0; i < F.size(); i++) {
    auto [v0, v1, v2] = F[i];
    assert(v0 < v1 && v0 < v2);
    auto tet_config = v1 > v2 ? TETRA_SPLIT_A : TETRA_SPLIT_B;
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 4; k++) {
        auto tc = tet_config[j][k];
        if (tc > 2)
          tc = vnum + F[i][tc - 3];
        else
          tc = F[i][tc];
        assert(tc < 2 * vnum);
        T[3 * i + j][k] = tc;
      }
    }
  }
  for (int i = 0; i < vnum; i++) {
    V[i] = base[i];
    V[i + vnum] = top[i];
  }
}

void prism::cage_utils::tetmesh_from_prismcage(
    const std::vector<Vec3d> &base, const std::vector<Vec3d> &mid,
    const std::vector<Vec3d> &top, const std::vector<Vec3i> &F,
    int num_singularity, std::vector<Vec3d> &V, std::vector<Vec4i> &T) {
  int vnum = base.size();
  T.resize(6 * F.size());
  V.resize(3 * vnum);
  for (int i = 0; i < vnum; i++) {
    V[i] = base[i];
    V[i + vnum] = mid[i];
    V[i + 2 * vnum] = top[i];
  }
  for (int i = 0; i < F.size(); i++) {  // top
    auto [v0, v1, v2] = F[i];
    assert(v0 < v1 && v0 < v2);
    auto tet_config = v1 > v2 ? TETRA_SPLIT_A : TETRA_SPLIT_B;
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 4; k++) {
        auto tc = tet_config[j][k];
        if (tc > 2)
          tc = vnum + F[i][tc - 3];
        else
          tc = F[i][tc];
        assert(tc < 2 * vnum);
        T[6 * i + j][k] = tc;             // base-mid
        T[6 * i + j + 3][k] = tc + vnum;  // mid-top
      }
    }
  }
}

void prism::cage_utils::reorder_singularity_to_front(
    RowMatd &V, RowMati &F, RowMatd &VN, const std::set<int> &omni_singu,
    Eigen::VectorXi &old_to_new) {
  if (omni_singu.size() == 0) return;
  int num_verts = V.rows();

  old_to_new = -Eigen::VectorXi::Ones(num_verts);
  std::vector<bool> occupy(num_verts, false);
  int cnt = 0;
  for (auto o : omni_singu) {
    occupy[o] = true;
    old_to_new[o] = cnt++;
  }

  for (int i = 0; i < num_verts; i++)
    if (occupy[i] == false) old_to_new[i] = cnt++;
  RowMatd vert = V;
  RowMatd VN_ = VN;
  for (int i = 0; i < num_verts; i++) {  // not using new to old is a trick due
                                         // to the specific forward shuffle.
    V.row(old_to_new[i]) = vert.row(i);
    VN.row(old_to_new[i]) = VN_.row(i);
  }
  std::for_each(F.data(), F.data() + F.size(),
                [&old_to_new](auto &a) { a = old_to_new[a]; });
}

void prism::cage_utils::mark_singular_on_border(const RowMatd &mV,
                                                const RowMati &mF, RowMatd &VN,
                                                std::set<int> &omni_sing) {
  constexpr auto reduced_triangle_triangle_intersection =
      [](const Vec3d &O, const Vec3d &A, const Vec3d &B, const Vec3d &C,
         const Vec3d &D) {
        if (prism::predicates::segment_triangle_overlap({C, D}, {O, A, B}))
          return true;
        if (prism::predicates::segment_triangle_overlap({A, B}, {O, C, D}))
          return true;
        return false;
      };
  RowMati bv;
  Eigen::VectorXi bf, bfi;
  igl::boundary_facets(mF, bv, bf, bfi);
  spdlog::trace("bv\n{}", bv);
  bfi = bfi.unaryExpr([](const int a) {
    return (a + 1) % 3;
  });  // bfi was across edge. advance by 1.

  for (int i = 0; i < bf.size(); i++) {
    spdlog::trace("{}, {} -> {}", bf[i], bfi[i], mF(bf[i], (bfi[i])));
  }

  RowMati TT, TTi;
  igl::triangle_triangle_adjacency(mF, TT, TTi);

  for (int i = 0; i < bf.size(); i++) {
    int f0 = bf[i], e0 = bfi[i];

    // collect half ring.
    bool along = true;
    std::vector<int> half_ring;
    int center = (igl::triangle_tuple_get_vert(f0, e0, along, mF, TT, TTi));

    if (VN.row(center).norm() < 0.1) continue;
    half_ring.push_back(
        igl::triangle_tuple_get_vert(f0, e0, !along, mF, TT, TTi));
    do {
      igl::triangle_tuple_switch_edge(f0, e0, along, mF, TT, TTi);
      half_ring.push_back(
          igl::triangle_tuple_get_vert(f0, e0, !along, mF, TT, TTi));
      igl::triangle_tuple_switch_face(f0, e0, along, mF, TT, TTi);
    } while (!igl::triangle_tuple_is_on_boundary(f0, e0, along, mF, TT, TTi));

    Vec3d tip = mV.row(center) + VN.row(center) / std::pow(2., 2);
    auto will_collide_one_ring = [&half_ring, &tip, &center,
                                  &reduced_triangle_triangle_intersection](
                                     const RowMatd &mV) {
      // test reduced intersection.
      // (c, hr[t], hr[t+1]) vs. (c, hr[0], D), t=1,2,...n-2
      for (int t = 1; t < half_ring.size() - 1; t++)
        if (reduced_triangle_triangle_intersection(
                mV.row(center), mV.row(half_ring[t]), mV.row(half_ring[t + 1]),
                mV.row(half_ring[0]), tip)) {
          spdlog::trace("center {} t {}, hr[t] {} hr[t+] {}, hr[0] {}", center,
                        t, half_ring[t], half_ring[t + 1], half_ring[0]);
          return true;
        }
      // (c, hr[t], hr[t+1]) vs. (c, hr[d-1], D), t=0, 1,2,...deg-3
      for (int t = 0; t < half_ring.size() - 2; t++)
        if (reduced_triangle_triangle_intersection(
                mV.row(center), mV.row(half_ring[t]), mV.row(half_ring[t + 1]),
                mV.row(half_ring.back()), tip)) {
          spdlog::trace("center {} t {}, hr[t] {} hr[t+] {}, hr[-1] {}", center,
                        t, half_ring[t], half_ring[t + 1], half_ring.back());
          return true;
        }
      return false;
    };

    if (will_collide_one_ring(mV)) {
      omni_sing.insert(center);
      VN.row(center) << 0, 0, 0;
    }
  }
}


#include "spatial-hash/AABB_hash.hpp"
#include "spatial-hash/self_intersection.hpp"
void prism::cage_utils::hashgrid_shrink(const std::vector<Vec3d> &mid, std::vector<Vec3d> &top,
                     const std::vector<Vec3i> &vecF,
                     const std::vector<std::vector<int>> &VF) {
  prism::HashGrid hg(top, vecF, false);
  int vnum = mid.size();
  std::vector<Vec3d> tetV;
  std::vector<Vec4i> tetT;
  prism::cage_utils::tetmesh_from_prismcage(mid, top, vecF, tetV, tetT);
  std::vector<int> affected_faces(vecF.size());
  std::iota(affected_faces.begin(), affected_faces.end(), 0);
  while (!affected_faces.empty()) {
    hg.clear();
    for (auto f : affected_faces)
      for (int j = 0; j < 3; j++) {
        auto i = 3 * f + j;
        Eigen::Matrix<double, 4, 3> local;
        for (auto k = 0; k < local.rows(); k++) local.row(k) = tetV[tetT[i][k]];
        auto aabb_min = local.colwise().minCoeff();
        auto aabb_max = local.colwise().maxCoeff();
        hg.add_element(aabb_min, aabb_max, i);
      }
    auto cand = hg.self_candidates();
    spdlog::debug("cand {}", cand.size());
    std::set<std::pair<int, int>> offend_pairs;
    auto offend_handle = prism::spatial_hash::find_offending_pairs(
        vecF, tetV, tetT, offend_pairs);
    std::for_each(cand.begin(), cand.end(), offend_handle);
    spdlog::debug("offending pairs {}", offend_pairs.size());

    std::set<int> masked_verts;
    for (auto [f0, f1] : offend_pairs) {
      for (auto j = 0; j < 3; j++) {
        masked_verts.insert(vecF[f0][j]);
        masked_verts.insert(vecF[f1][j]);
      }
    }
    spdlog::debug("masked_verts {}", masked_verts.size());

    int vnum = mid.size();
    affected_faces.clear();
    for (auto v : masked_verts) {
      tetV[vnum + v] = (tetV[vnum + v] + 3 * tetV[v]) / 4;
      affected_faces.insert(affected_faces.end(), VF[v].begin(), VF[v].end());
    }
    std::sort(affected_faces.begin(), affected_faces.end());
    affected_faces.erase(
        std::unique(affected_faces.begin(), affected_faces.end()),
        affected_faces.end());
  }
  for (int i = 0; i < vnum; i++) {
    top[i] = tetV[i + vnum];
  }
}

#include <igl/boundary_facets.h>
#include <igl/is_edge_manifold.h>

bool prism::cage_utils::safe_shrink(const std::vector<Vec3d> &mid,
                                    std::vector<Vec3d> &top,
                                    const std::vector<Vec3i> &vecF,
                                    const std::vector<std::vector<int>> &VF) {
  RowMatd mT;
  RowMati mF;
  vec2eigen(top, mT);
  vec2eigen(vecF, mF);
  std::vector<std::pair<int, int>> pairs;
  while (true) {
    prism::geogram::AABB tree(mT, mF);
    pairs.clear();
    tree.self_intersections(pairs);
    spdlog::info("pairs {}", pairs.size());
    if (pairs.empty()) break;
    std::set<int> masked_verts;
    for (auto [f0, f1] : pairs) {
      for (auto j = 0; j < 3; j++) {
        masked_verts.insert(vecF[f0][j]);
        masked_verts.insert(vecF[f1][j]);
      }
    }
    for (auto v : masked_verts) {
      if ((mT.row(v) - mid[v]).norm() < 1e-10) continue; // taken care by precondition
      mT.row(v) = (mT.row(v) + mid[v]) / 2;
    }
  }
  eigen2vec(mT,top);

  return pairs.empty();
}

std::map<std::pair<int, int>, int> prism::cage_utils::split_singular_edges(
    RowMatd &mV, RowMati &mF, RowMatd &mVN, const std::set<int> &omni_singu) {
  std::map<std::pair<int, int>, int> edge_map;
  std::vector<Vec3d> V, VN;
  std::vector<Vec3i> F;
  eigen2vec(mV, V);
  eigen2vec(mF, F);
  eigen2vec(mVN, VN);
  auto is_singular = [&omni_singu](auto v) {
    return omni_singu.find(v) != omni_singu.end();
  };
  auto [TT, TTi] = prism::local::triangle_triangle_adjacency(F);
  for (auto fi = 0; fi < F.size(); fi++) {
    if (is_singular(F[fi][0]) && is_singular(F[fi][1]) && is_singular(F[fi][2])) {
      spdlog::error("Fully Singular Face");
      throw 1;
    }
    for (auto ei = 0; ei < 3; ei++) {
      auto v0 = F[fi][(ei)], v1 = F[fi][(ei + 1) % 3];
      if (is_singular(v0) && is_singular(v1)) {
        spdlog::debug("Splitting Singular Edge {} {}", fi, ei);
        auto fi1 = TT[fi][ei];
        assert(fi1 > fi);
        auto ux = V.size();
        prism::edge_split(V.size(), F, TT, TTi, fi, ei);
        V.push_back((V[v0] + V[v1]) / 2);

        RowMati localF(2, 3);
        localF.row(0) << F[fi][0], F[fi][1], F[fi][2];
        localF.row(1) << F[fi1][0], F[fi1][1], F[fi1][2];
        RowMatd localN;
        igl::per_face_normals_stable(
            Eigen::Map<RowMatd>(V[0].data(), V.size(), 3), localF, localN);
        VN.push_back((localN.row(0) + localN.row(1)).normalized());
        if (v0 > v1) std::swap(v0, v1);
        edge_map.emplace(std::pair(v0, v1), ux);
      }
    }
  }
  auto extra = V.size() - mV.rows();
  vec2eigen(V, mV);
  vec2eigen(VN, mVN);
  vec2eigen(F, mF);
  return edge_map;
}