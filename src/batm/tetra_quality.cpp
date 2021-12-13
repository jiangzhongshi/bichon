#include <algorithm>
#include "tetra_utils.hpp"

#include "AMIPS.h"

#include <geogram/basic/geometry.h>
#include <geogram/numerics/predicates.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>
#include <Eigen/Core>
#include <Eigen/Dense>

namespace prism::tet {
double tetra_quality(const Vec3d& p0, const Vec3d& p1, const Vec3d& p2, const Vec3d& p3)
{
    std::array<double, 12> T;
    for (auto i = 0; i < 3; i++) {
        T[i] = p0[i];
        T[i + 3] = p1[i];
        T[i + 6] = p2[i];
        T[i + 9] = p3[i];
    }
    return wmtk::AMIPS_energy(T);
}

double circumradi2(const Vec3d& p0, const Vec3d& p1, const Vec3d& p2, const Vec3d& p3)
{
    std::array<GEO::vec3, 4> geo_v;
    geo_v[0] = GEO::vec3(p0[0], p0[1], p0[2]);
    geo_v[1] = GEO::vec3(p1[0], p1[1], p1[2]);
    geo_v[2] = GEO::vec3(p2[0], p2[1], p2[2]);
    geo_v[3] = GEO::vec3(p3[0], p3[1], p3[2]);
    GEO::vec3 center = GEO::Geom::tetra_circum_center(geo_v[0], geo_v[1], geo_v[2], geo_v[3]);
    return GEO::distance2(center, geo_v[0]);
}

bool tetra_validity(const std::vector<VertAttr>& vert_attrs, const Vec4i& t)
{
    auto flag = GEO::PCK::orient_3d(
        vert_attrs[t[0]].pos.data(),
        vert_attrs[t[1]].pos.data(),
        vert_attrs[t[2]].pos.data(),
        vert_attrs[t[3]].pos.data());
    return flag == 1;
}


Vec3d get_newton_position_from_assemble(
    std::vector<std::array<double, 12>>& assembles,
    const Vec3d& old_pos)
{
    auto newton_direction = [&assembles](auto& pos) -> Vec3d {
        auto total_energy = 0.;
        Eigen::Vector3d total_jac = Eigen::Vector3d::Zero();
        Eigen::Matrix3d total_hess = Eigen::Matrix3d::Zero();

        // E = \sum_i E_i(x)
        // J = \sum_i J_i(x)
        // H = \sum_i H_i(x)
        auto local_id = 0;
        for (auto& T : assembles) {
            for (auto j = 0; j < 3; j++) {
                T[j] = pos[j]; // only filling the front point.
            }
            auto jac = decltype(total_jac)();
            auto hess = decltype(total_hess)();
            total_energy += wmtk::AMIPS_energy(T);
            wmtk::AMIPS_jacobian(T, jac);
            wmtk::AMIPS_hessian(T, hess);
            total_jac += jac;
            total_hess += hess;
            assert(!std::isnan(total_energy));
        }
        Eigen::Vector3d x = total_hess.ldlt().solve(total_jac);
        spdlog::trace("energy {}", total_energy);
        if (total_jac.isApprox(total_hess * x)) // a hacky PSD trick. TODO: change this.
            return -x;
        else {
            spdlog::trace("gradient descent instead.");
            return -total_jac;
        }
    };
    auto compute_energy = [&assembles](const Vec3d& pos) -> double {
        auto total_energy = 0.;
        for (auto& T : assembles) {
            for (auto j = 0; j < 3; j++) {
                T[j] = pos[j]; // only filling the front point x,y,z.
            }
            total_energy += wmtk::AMIPS_energy(T);
        }
        return total_energy;
    };
    auto linesearch = [&compute_energy](const Vec3d& pos, const Vec3d& dir, const int& max_iter) {
        auto lr = 0.8;
        auto old_energy = compute_energy(pos);
        spdlog::trace("dir {}", dir);
        for (auto iter = 1; iter <= max_iter; iter++) {
            Vec3d newpos = pos + std::pow(lr, iter) * dir;
            spdlog::trace("pos {}, dir {}, [{}]", pos, dir, std::pow(lr, iter));
            auto new_energy = compute_energy(newpos);
            spdlog::trace("iter {}, {}, [{}]", iter, new_energy, newpos);
            if (new_energy < old_energy) return newpos; // TODO: armijo conditions.
        }
        return pos;
    };
    auto compute_new_valid_pos = [&linesearch, &newton_direction](const Vec3d& old_pos) {
        auto current_pos = old_pos;
        auto line_search_iters = 12;
        auto newton_iters = 10;
        for (auto iter = 0; iter < newton_iters; iter++) {
            auto dir = newton_direction(current_pos);
            auto newpos = linesearch(current_pos, dir, line_search_iters);
            if ((newpos - current_pos).norm() < 1e-9) // barely moves
            {
                break;
            }
            current_pos = newpos;
        }
        return current_pos;
    };
    return compute_new_valid_pos(old_pos);
};


} // namespace prism::tet