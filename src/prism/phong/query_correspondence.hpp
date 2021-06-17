#pragma once

#include "../common.hpp"

struct PrismCage;
namespace prism::geogram {
struct AABB;
};
namespace prism {
struct Hit;
};
namespace prism {
// project query positions onto proxy (pxV, pxF) for fid and bc.
void correspond_bc(const PrismCage &pc, const RowMatd &pxV, const RowMati &pxF,
                   const RowMatd &queryPos, Eigen::VectorXi &resF,
                   RowMatd &resUV);

// internal
bool project_to_proxy_mesh(const std::array<Vec3d, 9> &stack,
                           const prism::geogram::AABB &pxtree, 
                           bool prism_type, const Vec3d &spatial, prism::Hit &hit);

// used in sectional remeshing, with track enabled
bool project_to_ref_mesh(const PrismCage &pc,
                         const std::vector<std::set<int>> &track_to_prism,
                         const std::vector<int> &tris, const Vec3d &point_value,
                         Vec3d &point_on_ref);
} // namespace prism