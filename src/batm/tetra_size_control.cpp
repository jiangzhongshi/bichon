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

double SizeController::find_size_bound(const std::array<Vec3d, 4>& P) const
{
    auto result = bg_tree->overlap_tetra(P);
    auto min_size = 1.0;
    if (result.empty()) {
        spdlog::critical("Not found, check {} {} {} {}", P[0], P[1], P[2], P[3]);
        return min_size;
    }

    for (auto i : result) min_size = std::min(sizes[i], min_size);
    return min_size;
}

} // namespace prism::tet