#include "smoother_pillar.hpp"

#include <igl/volume.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <prism/cgal/QP.hpp>

#include "prism_quality.hpp"

RowMatd prism::one_ring_volumes(
    const std::vector<Vec3d>& base, const std::vector<Vec3d>& mid,
    const std::vector<Vec3d>& top, const std::vector<Vec3i>& F,
    const std::vector<int>& nb, const std::vector<int>& nbi,
    const std::array<Vec3d, 3>& modify) {  // #nb * (12*2)
  RowMatd all_vol(nb.size(), 12 * 2);
  // auto [on_base, dir] = modify;
  for (auto index = 0; index < nb.size(); index++) {
    auto v_id = nbi[index];
    auto face = F[nb[index]];
    RowMatd verts(9, 3);
    for (int i = 0; i < 3; i++) {
      verts.row(i) = base[face[i]];
      verts.row(i + 3) = mid[face[i]];
      verts.row(i + 6) = top[face[i]];
    }
    verts.row(v_id) += modify[0];
    verts.row(v_id + 3) += modify[1];
    verts.row(v_id + 6) += modify[2];
    Eigen::VectorXd vol_bot;
    Eigen::VectorXd vol_top;
    igl::volume(verts.topRows(6),
                Eigen::Map<const RowMati>(TWELVE_TETRAS[0].data(), 12, 4),
                vol_bot);
    igl::volume(verts.bottomRows(6),
                Eigen::Map<const RowMati>(TWELVE_TETRAS[0].data(), 12, 4),
                vol_top);
    for (int i = 0; i < 12; i++) {
      all_vol(index, 2 * i) = vol_bot(i);
      all_vol(index, 2 * i + 1) = vol_top(i);
    }
  }
  return all_vol;
}

double prism::get_min_step_to_singularity(
    const std::vector<Vec3d>& base, const std::vector<Vec3d>& mid,
    const std::vector<Vec3d>& top, const std::vector<Vec3i>& F,
    const std::vector<int>& nb, const std::vector<int>& nbi,
    std::array<bool, 3> /*base,mid,top*/ change, const Vec3d& direction,
    int num_freeze) {
  // as long as the same direction, the overall change is linear
  auto vols = prism::one_ring_volumes(base, mid, top, F, nb, nbi);

  std::array<Vec3d, 3> modify{Vec3d(0, 0, 0), Vec3d(0, 0, 0), Vec3d(0, 0, 0)};
  for (int i = 0; i < 3; i++)
    if (change[i]) modify[i] = direction;
  auto vol_step = prism::one_ring_volumes(base, mid, top, F, nb, nbi, modify);

  auto step = 1.;
  for (int i = 0; i < vols.rows(); i++) {
    auto skip = F[nb[i]][0] < num_freeze ? 2 * 4 : 0;
    for (int j = skip; j < vols.cols(); j++) {
      auto v1 = vol_step(i, j), v0 = vols(i, j);
      if (v0 < 0) {
        spdlog::debug("VC result volume-check {} {}", nb[i], v0);
        spdlog::debug("checking negative vol {}, {}", i, j);
        continue;
      }
      if (v1 <= 0) {
        step = std::min(step, v0 / (v0 - v1));
      }
    }
  }
  return step;
};

std::optional<Vec3d> prism::smoother_direction(
    const std::vector<Vec3d>& base, const std::vector<Vec3d>& mid,
    const std::vector<Vec3d>& top, const std::vector<Vec3i>& F, int num_freeze,
    const std::vector<int>& nb,
    const std::vector<int>& nbi, int vid) {
  auto [old_quality, grad] =
      prism::energy::triangle_one_ring_quality(mid, F, nb, nbi, true);
  grad *= -1;  //  descent

  auto step = prism::get_min_step_to_singularity(
      base, mid, top, F, nb, nbi, {true, true, true}, grad, num_freeze);
  spdlog::trace("quality {}, maxstep {}", old_quality, step);

  for (auto i = 0; i < 200; i++) {
    step *= 0.8;
    Vec3d dir = step * grad;
    if (dir.norm() < 1e-14) break;
    auto [new_quality, ignore] = prism::energy::triangle_one_ring_quality(
        mid, F, nb, nbi, /*grad=*/false, /*modify=*/dir);
    spdlog::trace("qualitiy {}, step {} {}", new_quality, i, step);
    if (new_quality < old_quality) {
      spdlog::trace("Good Tri qualitiy {}, step {} {}", new_quality, i, step);
      return dir;
    }
  }
  spdlog::debug("line search failed v={}", vid);
  return {};
}

std::optional<std::pair<Vec3d, Vec3d>> prism::zoom_and_rotate(
    const std::vector<Vec3d>& base, const std::vector<Vec3d>& mid,
    const std::vector<Vec3d>& top, const std::vector<Vec3i>& F, int num_freeze,
    const std::vector<int>& nb,
    const std::vector<int>& nbi, int vid, double target_height) {
  // MPGA: Zoom and Rotate, to optimize shell quality.
  // a (M-T) = B-T
  // B = a M + (1-a) T, a>1
  // dB = da (M - T) + (1-a) dT
  // d(Qt + Qb) = (dQt/dT) dT + (dQb/dB) ((M - T) da + (1-a) dT)
  // dQ/da = (dQb/dB)'(M - T)
  // dQ/dT = (dQt/dT) + (1-a)(dQb/dB)
  std::vector<double> mid_areas(nb.size(), 0.1);
  auto [base_quality, base_grad] = prism::energy::prism_one_ring_quality(
      base, mid, F, nb, nbi, target_height, mid_areas, true, vid);
  auto [top_quality, top_grad] = prism::energy::prism_one_ring_quality(
      mid, top, F, nb, nbi, target_height, mid_areas, false, vid);
  double old_quality = base_quality + top_quality;
  if (std::isnan(old_quality) || old_quality > 1e12) {
    spdlog::trace("Old Quality NaN");
    return {};
  }
  spdlog::trace("Old qualitiy {} = {} + {}", old_quality, base_quality,
                top_quality);
  Vec3d M = mid[vid], T = top[vid], B = base[vid];
  double alpha = (B - T).norm() / (M - T).norm();
  assert(alpha > 1);

  double dQda = base_grad.dot(M - T);
  Vec3d dQdT = top_grad + (1 - alpha) * base_grad;

  double step = 1e-2;
  if (dQda > 0) step = std::min(step, (alpha - 1) / dQda);
  step = std::min(step, get_min_step_to_singularity(base, mid, top, F, nb, nbi,
                                                    {false, false, true}, -dQdT,
                                                    num_freeze));
  for (int i = 0; i < 100; i++) {
    step *= 0.8;
    Vec3d Tdir = -step * dQdT;
    if (Tdir.norm() < 1e-12 && abs(step * dQda) < 1e-12) return {};
    Vec3d curT = T + Tdir;
    double cur_alpha = alpha - step * dQda;
    Vec3d curB = curT * (1 - cur_alpha) + cur_alpha * M;
    Vec3d Bdir = curB - B;
    auto new_Qb = prism::energy::prism_one_ring_quality(
        base, mid, F, nb, nbi, target_height, mid_areas, std::pair(true, Bdir));
    auto new_Qt = prism::energy::prism_one_ring_quality(
        mid, top, F, nb, nbi, target_height, mid_areas, std::pair(false, Tdir));
    spdlog::trace("curent qualitiy {} = {}+{}, step {} {}, alpha={}",
                  new_Qb + new_Qt, new_Qb, new_Qt, i, step, cur_alpha);
    if (new_Qb + new_Qt < old_quality) {
      spdlog::trace("Good qualitiy {}, step {} {}", new_Qb + new_Qt, i, step);
      // exit(0);
      return std::pair(curB, curT);
    }
  }
  spdlog::trace("line search failed");
  return {};
}

std::optional<std::pair<Vec3d, Vec3d>> prism::rotate(
    const std::vector<Vec3d>& base, const std::vector<Vec3d>& mid,
    const std::vector<Vec3d>& top, const std::vector<Vec3i>& F,
    const std::vector<int>& nb,
    const std::vector<int>& nbi, int vid, double _) {
  Vec3d qp_normal(0, 0, 0);
  
  {
    RowMatd FN(nb.size(), 3);
    auto cnt = 0;
    std::vector<int> range;
    for (auto f : nb) {
      FN.row(cnt) =
          (mid[F[f][1]] - mid[F[f][0]]).cross(mid[F[f][2]] - mid[F[f][0]]);
      range.push_back(cnt);
      cnt++;
    }
    qp_normal = prism::cgal::qp_normal(FN, range);
  }
  if (qp_normal.norm() < 0.5) {
    qp_normal = Vec3d(0, 0, 0);
  }
  double target_top = (top[vid] - mid[vid]).norm();
  double target_base = (base[vid] - mid[vid]).norm();
  for (int i = 0; i < nb.size(); i++) {
    auto v1 = F[nb[i]][(nbi[i] + 1) % 3];
    qp_normal += (top[v1] - base[v1]);
  }
  qp_normal.normalize();
  return std::pair(mid[vid] - target_base * qp_normal,
                   mid[vid] + target_top * qp_normal);
}

std::optional<std::pair<Vec3d, Vec3d>> prism::zoom(
    const std::vector<Vec3d>& base, const std::vector<Vec3d>& mid,
    const std::vector<Vec3d>& top, const std::vector<Vec3i>& F,
    const std::vector<int>& nb,
    const std::vector<int>& nbi, int vid,
    double target_thickness) {
  auto target_base = 0., target_top = 0., nb_len = 0.;
  auto nb_num = nb.size();
  for (int i = 0; i < nb.size(); i++) {
    auto v1 = F[nb[i]][(nbi[i] + 1) % 3];
    target_base += (base[v1] - mid[v1]).norm();
    target_top += (top[v1] - mid[v1]).norm();
    nb_len += (mid[v1] - mid[vid]).norm();
  }
  Vec3d cur_normal = (top[vid] - base[vid]).stableNormalized();
  return std::pair(mid[vid] - std::min(nb_len/nb.size(), 1.5*target_base/nb.size()) * cur_normal,
                   mid[vid] + std::min(nb_len/nb.size(), 1.5*target_top/nb.size()) * cur_normal);
}

std::optional<Vec3d> prism::smoother_location_legacy(
    const std::vector<Vec3d>& base, const std::vector<Vec3d>& mid,
    const std::vector<Vec3d>& top, const std::vector<Vec3i>& F, int num_freeze,
    const std::vector<std::vector<int>>& VF,
    const std::vector<std::vector<int>> VFi, int vid, bool on_base) {
  // Set M = (B+T)/2
  // Then d (Q(M,T) + Q(B,M))/dT
  // = 1/2 Q_1(M,T) + Q_2(M,T) + 1/2 Q_2(B,M)
  // Notation: Qb := Q(B,M), Qt := Q(M,T)
  auto nb = VF[vid], nbi = VFi[vid];
  double old_quality = -1.;
  Vec3d grad;
  double dimscale = 0.1;
  std::vector<double> mid_areas(nb.size(), 0.1);
  for (int i=0; i<nb.size(); i++) {
    auto [v0,v1,v2] = F[nb[i]];
    mid_areas[i] = (mid[v1] -mid[v0]).cross(mid[v2] - mid[v0]).stableNorm()/2;
  }
  if (!on_base) {  // dT
    auto [Qt, Qt_d2] = prism::energy::prism_one_ring_quality(
        mid, top, F, nb, nbi, dimscale, mid_areas, false, vid);
    auto [_, Qt_d1] = prism::energy::prism_one_ring_quality(
        mid, top, F, nb, nbi, dimscale, mid_areas, true, vid);
    auto [Qb, Qb_d2] = prism::energy::prism_one_ring_quality(
        base, mid, F, nb, nbi, dimscale, mid_areas, false, vid);
    old_quality = Qb + Qt;
    grad = 0.5 * Qt_d1 + Qt_d2 + 0.5 * Qb_d2;
  } else {  // dB
    auto [Qb, Qb_d1] = prism::energy::prism_one_ring_quality(
        base, mid, F, nb, nbi, dimscale, mid_areas, true, vid);
    auto [_, Qb_d2] = prism::energy::prism_one_ring_quality(
        base, mid, F, nb, nbi, dimscale, mid_areas, false, vid);
    auto [Qt, Qt_d1] = prism::energy::prism_one_ring_quality(
        mid, top, F, nb, nbi, dimscale, mid_areas, true, vid);
    old_quality = Qb + Qt;
    grad = 0.5 * Qb_d2 + Qb_d1 + 0.5 * Qt_d1;
  }
  grad *= -1;  //  descent

  // check the volume is predicate-ok
  std::array<bool, 3> change{false, false, false};
  change[on_base ? 0 : 2] = true;
  auto step = prism::get_min_step_to_singularity(base, mid, top, F, nb, nbi,
                                                 change, grad, num_freeze);

  spdlog::debug("quality {}, maxstep {}", old_quality, step);

  for (auto i = 0; i < 200; i++) {
    step *= 0.8;
    Vec3d dir = step * grad;
    double new_quality = 0;
    if (on_base) {
      new_quality =
          prism::energy::prism_one_ring_quality(base, mid, F, nb, nbi, dimscale, mid_areas,
                                                std::pair(dir, dir / 2)) +
          prism::energy::prism_one_ring_quality(
              mid, top, F, nb, nbi, dimscale,mid_areas, 
              std::pair(dir / 2, Vec3d(0, 0, 0)));
    } else {
      new_quality =
          prism::energy::prism_one_ring_quality(
              base, mid, F, nb, nbi, dimscale,mid_areas,
              std::pair(Vec3d(0, 0, 0), dir / 2)) +
          prism::energy::prism_one_ring_quality(mid, top, F, nb, nbi, dimscale, mid_areas,
                                                std::pair(dir / 2, dir));
    }
    if (new_quality < old_quality) {
      spdlog::debug("current quality {}, step {} {}", new_quality, i, step);
      return (on_base ? base[vid] : top[vid]) + dir;
    }
  }
  spdlog::debug("line search failed");
  return {};
}
