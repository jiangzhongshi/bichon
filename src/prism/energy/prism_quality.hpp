#ifndef PRISM_ENERGY_PRISM_QUALIY_HPP
#define PRISM_ENERGY_PRISM_QUALIY_HPP

#include "../common.hpp"
#include <autodiff_mitsuba.h>

namespace prism::energy {
enum class QualityType { SYMMETRIC_DIRICHLET, MIPS_3D };
using DScalar = DScalar2<double, Eigen::Vector3d, Eigen::Matrix3d>;

// the quality measure for the well shapeness of prisms, the larger, the worse!
double prism_full_quality(const std::array<Vec3d, 6>& corners,
                          const Eigen::RowVector3d& dimscale = Vec3d(1, 1, 1),
                          QualityType qt = QualityType::MIPS_3D);

DScalar prism_full_quality(const std::array<Vec3d, 6>& corners,
                           const Eigen::RowVector3d& dimscale, QualityType qt,
                           int id_with_grad);

double prism_one_ring_quality(const std::vector<Vec3d>& base,
                              const std::vector<Vec3d>& top,
                              const std::vector<Vec3i>& F,
                              const std::vector<int>& nb,
                              const std::vector<int>& nbi,
                              double target_height,
                              const std::vector<double>& areas,
                              const std::pair<Vec3d, Vec3d>& modification);

double prism_one_ring_quality(const std::vector<Vec3d>& base,
                              const std::vector<Vec3d>& top,
                              const std::vector<Vec3i>& F,
                              const std::vector<int>& nb,
                              const std::vector<int>& nbi,
                              double target_height,
                              const std::vector<double>& areas,
                              const std::pair<bool, Vec3d>& modification);

// default last parameter means not taking gradient
std::tuple<double /*value*/, Vec3d /*grad*/> prism_one_ring_quality(
    const std::vector<Vec3d>& base, const std::vector<Vec3d>& top,
    const std::vector<Vec3i>& F, const std::vector<int>& nb,
    const std::vector<int>& nbi, double target_height,
    const std::vector<double>& areas, bool on_base, int v_with_grad = -1);

DScalar triangle_quality(const std::array<Vec3d, 3>& vertices, int v_with_grad);
double triangle_quality(const std::array<Vec3d, 3>& vertices);

std::tuple<double, Vec3d> triangle_one_ring_quality(
    const std::vector<Vec3d>& mid, const std::vector<Vec3i>& F,
    const std::vector<int>& nb, const std::vector<int>& nbi,
    bool with_grad, Vec3d modification = Vec3d(0., 0, 0));
}  // namespace prism::energy

#endif