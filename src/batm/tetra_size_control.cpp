#include "tetra_utils.hpp"

#include "prism/geogram/AABB_tet.hpp"
#include "prism/local_operations/remesh_pass.hpp"

#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>
#include <limits>
#include <memory>

namespace prism::tet {

SizeController::SizeController(
    const RowMatd& tetV,
    const RowMati& tetT,
    const Eigen::VectorXd& tetSize)
{
    bg_tree = std::make_shared<prism::geogram::AABB_tet>(tetV, tetT);
    sizes = tetSize;
    assert(tetT.rows() == sizes.size());
}

auto process_result(const Eigen::VectorXd& sizes, const std::vector<size_t>& result)
{
    auto min_size = 1.0;
    if (result.empty()) {
        spdlog::critical("Not found, check");
        return min_size;
    }

    for (auto i : result) min_size = std::min(sizes[i], min_size);
    return min_size;
};

double SizeController::find_size_bound(const std::array<Vec3d, 4>& P) const
{
    auto result = bg_tree->overlap_tetra(P);
    return process_result(sizes, result);
}

double SizeController::find_size_bound(const std::array<Vec3d, 3>& P) const
{
    auto result = bg_tree->overlap_tri(P);
    return process_result(sizes, result);
}

} // namespace prism::tet