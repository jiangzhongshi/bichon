#include "map_distortion.hpp"

#include <Eigen/Dense>
double prism::energy::map_max_distortion(const Vec3d& ez,  // pillar
                                         const std::array<Vec3d, 3>& source_tri,
                                         const std::array<Vec3d, 3>& target_tri,
                                         DistortionType qt) {
  Vec3d r01 = source_tri[1] - source_tri[0];
  Vec3d source_N = (r01.cross(source_tri[2] - source_tri[0])).stableNormalized();
  Vec3d r02 = source_N.cross(r01);
  r01.normalize();
  r02.normalize();

  Vec3d tN =
      (target_tri[1] - target_tri[0]).cross(target_tri[2] - target_tri[0]);
  tN.normalize();

  if (ez.dot(tN) <= 0)
    return std::numeric_limits<double>::infinity(); 
  Vec3d d01 = r01 - r01.dot(tN) / (ez.dot(tN)) * ez;
  Vec3d d02 = r02 - r02.dot(tN) / (ez.dot(tN)) * ez;

  double det = tN.dot(d01.cross(d02));
  if (det <= 0) return std::numeric_limits<double>::infinity();
  assert(det > 0);
  double frobsq = d01.cwiseAbs2().sum() + d02.cwiseAbs2().sum();
  if (qt == DistortionType::SYMMETRIC_DIRICHLET)
    return frobsq * (1 + 1 / (det * det));
  else
    return -1;
}

double prism::energy::map_max_cos_angle(const Vec3d& ez,  // pillar
                                    const std::array<Vec3d, 3>& target_tri) {
  Vec3d tN =
      (target_tri[1] - target_tri[0]).cross(target_tri[2] - target_tri[0]);
  return ez.dot(tN) / (ez.norm()*tN.norm());
}