#include "prism/common.hpp"
#include "prism/phong/projection.hpp"
#include <igl/Timer.h>
#include <prism/energy/map_distortion.hpp>
#include <prism/PrismCage.hpp>
#include <spdlog/spdlog.h>
#include <prism/geogram/AABB.hpp>
#include <prism/predicates/inside_octahedron.hpp>
#include <igl/quadric_binary_plus_operator.h>
#include <igl/circulation.h>

using Quadric = std::tuple<Eigen::MatrixXd, Eigen::RowVectorXd, double>;

bool intersect_check(const std::vector<std::array<Vec3d, 3>> &tris,
                     const std::vector<int> &mask,
                     const prism::geogram::AABB &tree_base,
                     const prism::geogram::AABB &tree_top)
{
  for (int i = 0; i < tris.size(); i++)
  {
    auto tri = tris[i];
    auto m = mask[i];
    if (m > 0)
    {
      for (int j = 0; j < 3; j++)
      {
        tri[j] = tris[i][(j + m) % 3];
      }
    }
    if (tree_base.intersects_triangle(tri, m >= 0))
      return false;
    if (tree_top.intersects_triangle(tri, m >= 0))
      return false;
  }
  return true;
}

bool distort_check(
    const std::vector<std::array<Vec3d, 3>> &tri_verts,
    const std::vector<int> &mask,
    // const std::vector<Vec3d> &V, const std::vector<Vec3i> &tris,
    const std::set<int> &combined_trackee, // indices to prism pcF tracked
    const std::vector<Vec3d> &base, const std::vector<Vec3d> &mid,
    const std::vector<Vec3d> &top, const std::vector<Vec3i> &pcF,
    double distortion_bound, int num_freeze,
    std::vector<std::set<int>> &distributed_refs)
{
  // spdlog::trace("In DC ct#{}, tris{}", combined_trackee.size(), tris.size());
  igl::Timer timer;
  timer.start();
  distributed_refs.resize(tri_verts.size());
  for (int i = 0; i < tri_verts.size(); i++)
  {
    auto &cur_tri = tri_verts[i];

    for (auto t : combined_trackee)
    { // for every traked original triangle.
      auto [v0, v1, v2] = pcF[t];
      std::array<Vec3d, 3> base_vert{base[v0], base[v1], base[v2]};
      std::array<Vec3d, 3> mid_vert{mid[v0], mid[v1], mid[v2]};
      std::array<Vec3d, 3> top_vert{top[v0], top[v1], top[v2]};
      std::array<bool, 3> oct_type;
      prism::determine_convex_octahedron(base_vert, top_vert, oct_type,
                                         num_freeze > v0);
      bool intersected_prism = false;
      if (mask[i] < 0)
      {
        intersected_prism =
            prism::triangle_intersect_octahedron(base_vert, mid_vert, oct_type,
                                                 cur_tri, num_freeze > v0) ||
            prism::triangle_intersect_octahedron(mid_vert, top_vert, oct_type,
                                                 cur_tri, num_freeze > v0);
      }
      else
      {
        auto tri = cur_tri;
        if (auto m = mask[i]; m > 0)
        { // roll
          for (int j = 0; j < 3; j++)
          {
            tri[j] = cur_tri[(j + m) % 3];
          }
        }
        intersected_prism = prism::singularless_triangle_intersect_octahedron(
                                base_vert, mid_vert, oct_type, tri) ||
                            prism::singularless_triangle_intersect_octahedron(
                                mid_vert, top_vert, oct_type, tri);
      }

      if (!intersected_prism)
        continue;
      for (int tc = v0 < num_freeze ? 1 : 0; tc < 3; tc++)
      {
        auto pillar = top_vert[tc] - base_vert[tc];
        auto distortion = prism::energy::map_max_cos_angle(pillar, cur_tri);
        if (distortion < distortion_bound)
        {
          return false;
        }
      }
      distributed_refs[i].insert(t);
    }
  }

  auto elapsed = timer.getElapsedTimeInMicroSec();
  // spdlog::trace("DC true {}", elapsed);

  //
  return true;
}

auto qslim_in_shell_handle(const PrismCage &pc, const prism::geogram::AABB &tree_B, const prism::geogram::AABB &tree_T,
                           std::vector<std::set<int>> &map_track,
                           int &v1, int &v2, std::vector<Quadric> &quadrics)
{
  std::function<bool(
      const Eigen::MatrixXd &,                                         /*V*/
      const Eigen::MatrixXi &,                                         /*F*/
      const Eigen::MatrixXi &,                                         /*E*/
      const Eigen::VectorXi &,                                         /*EMAP*/
      const Eigen::MatrixXi &,                                         /*EF*/
      const Eigen::MatrixXi &,                                         /*EI*/
      const std::set<std::pair<double, int>> &,                        /*Q*/
      const std::vector<std::set<std::pair<double, int>>::iterator> &, /*Qit*/
      const Eigen::MatrixXd &,                                         /*C*/
      const int                                                        /*e*/
      )>
      pre_collapse = [&pc = std::as_const(pc), &tree_B = std::as_const(tree_B),
                      &tree_T = std::as_const(tree_T), &map_track, &v1, &v2](
                         const Eigen::MatrixXd &V, /*V*/
                         const Eigen::MatrixXi &F, /*F*/
                         const Eigen::MatrixXi &E,
                         const Eigen::VectorXi &EMAP,                                     /*EMAP*/
                         const Eigen::MatrixXi &EF,                                       /*EF*/
                         const Eigen::MatrixXi &EI,                                       /*EI*/
                         const std::set<std::pair<double, int>> &,                        /*Q*/
                         const std::vector<std::set<std::pair<double, int>>::iterator> &, /*Qit*/
                         const Eigen::MatrixXd &C,                                        /*C*/
                         const int e) -> bool {
    // need to know the placement C.row(e).
    Vec3d target_placement = C.row(e);
    v1 = E(e, 0);
    v2 = E(e, 1);
    std::vector<int> V2Fe1 = igl::circulation(e, true, EMAP, EF, EI);
    std::vector<int> V2Fe2 = igl::circulation(e, false, EMAP, EF, EI);
    std::sort(V2Fe1.begin(), V2Fe1.end());
    std::sort(V2Fe2.begin(), V2Fe2.end());
    std::vector<int> xor_faces(V2Fe1.size() + V2Fe2.size());
    auto it = std::set_symmetric_difference(V2Fe1.begin(), V2Fe1.end(), V2Fe2.begin(), V2Fe2.end(), xor_faces.begin());
    xor_faces.resize(it - xor_faces.begin());

    std::vector<int> singularity_mask(xor_faces.size(), -1);
    std::vector<std::array<Vec3d, 3>> tri_verts(xor_faces.size());
    for (int i = 0; i < tri_verts.size(); i++)
    {
      for (int j = 0; j < 3; j++)
      {
        int vid = F(xor_faces[i], j);
        if (vid < tree_B.num_freeze)
          singularity_mask[i] = j;
        tri_verts[i][j] = (vid == v1 || vid == v2) ? target_placement : V.row(vid);
      }
    }

    if (!intersect_check(tri_verts, singularity_mask, tree_B, tree_T))
    {
      spdlog::trace("Intersect reject");
      return false;
    }
    std::set<int> combined_trackee;
    for (auto f : xor_faces)
    {
      std::merge(map_track[f].begin(), map_track[f].end(), combined_trackee.begin(),
                 combined_trackee.end(), std::inserter(combined_trackee, combined_trackee.begin()));
    }
    std::vector<std::set<int>> distributed_refs;
    if (!distort_check(tri_verts,
                       singularity_mask,
                       combined_trackee,
                       pc.base, pc.mid, pc.top, pc.F, 1e-4, 0, distributed_refs))
    {
      spdlog::trace("Distort reject");
      return false;
    }
    for (int i = 0; i < xor_faces.size(); i++)
    {
      map_track[xor_faces[i]] = distributed_refs[i];
    }
    spdlog::trace("Succeed Pass");
    return true;
  };

  return pre_collapse;
}