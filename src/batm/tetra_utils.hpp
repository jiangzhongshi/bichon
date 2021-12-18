#pragma once

#include <memory>
#include <prism/common.hpp>


struct PrismCage;
namespace prism::geogram {
    struct AABB_tet;
}

namespace prism::local {
struct RemeshOptions;
}
/**
 * @brief loosely coupled with PrismCage
 *
 */
namespace prism::tet {

struct SizeController
{
    std::shared_ptr<prism::geogram::AABB_tet> bg_tree;
    Eigen::VectorXd sizes;
    SizeController(const RowMatd & tetV, const RowMati & tetT, const Eigen::VectorXd& tetSize);
    double find_size_bound(const std::array<Vec3d,4>&) const;
};

struct TetAttr
{
    Vec4i conn = {{-1,-1,-1,-1}};
    Vec4i prism_id = {{-1, -1, -1, -1}}; /**The prism cell id for each face.*/
    bool is_removed = false;
};

struct VertAttr
{
    Vec3d pos = Vec3d::Zero();
    int mid_id = -1; /**Points to the vertex id in the shell*/
};

/**
 *
 * @param pc
 * @param tet_v
 * @param tet_t
 * @return std::tuple<
 * std::vector<prism::tet::VertAttr>,
 * std::vector<prism::tet::TetAttr>,
 * std::vector<std::vector<int>>>
 */
std::tuple<
    std::vector<prism::tet::VertAttr>,
    std::vector<prism::tet::TetAttr>,
    std::vector<std::vector<int>>>
prepare_tet_info(
    const PrismCage& pc,
    const RowMatd& tet_v,
    const RowMati& tet_t,
    const Eigen::VectorXi& tet_v_pid);

bool split_edge(
    PrismCage& pc,
    prism::local::RemeshOptions& option,
    std::vector<VertAttr>& vert_attrs,
    std::vector<TetAttr>& tet_attrs,
    std::vector<std::vector<int>>& vert_conn,
    int v0,
    int v1);

bool smooth_vertex(
    PrismCage& pc,
    const prism::local::RemeshOptions& option,
    std::vector<VertAttr>& vert_attrs,
    const std::vector<TetAttr>& tet_attrs,
    const std::vector<std::vector<int>>& vert_conn,
    int v0,
    double size_control);

bool swap_face(
    const PrismCage& pc,
    const prism::local::RemeshOptions& option,
    std::vector<VertAttr>& vert_attrs,
    std::vector<TetAttr>& tet_attrs,
    std::vector<std::vector<int>>& vert_conn,
    int v0_id,
    int v1_id,
    int v2_id,
    double size_control);

bool swap_edge(
    const PrismCage& pc,
    const prism::local::RemeshOptions& option,
    std::vector<VertAttr>& vert_attrs,
    std::vector<TetAttr>& tet_attrs,
    std::vector<std::vector<int>>& vert_conn,
    int v1_id,
    int v2_id,
    double size_control);

bool collapse_edge(
    PrismCage& pc,
    const prism::local::RemeshOptions& option,
    std::vector<VertAttr>& vert_attrs,
    std::vector<TetAttr>& tet_attrs,
    std::vector<std::vector<int>>& vert_conn,
    int v1_id,
    int v2_id,
    double size_control);

void compact_tetmesh(
    std::vector<prism::tet::VertAttr>& vert_info,
    std::vector<prism::tet::TetAttr>& tet_info,
    std::vector<std::vector<int>>& vert_tet_conn,
    PrismCage *pc = nullptr);

bool tetmesh_sanity(
    const std::vector<TetAttr>& tet_attrs,
    const std::vector<VertAttr>& vert_attrs,
    const std::vector<std::vector<int>>& vert_tet_conn,
    const PrismCage& pc);
} // namespace prism::tet

namespace prism::tet {
double circumradi2(const Vec3d& p0, const Vec3d& p1, const Vec3d& p2, const Vec3d& p3);
bool tetra_validity(const std::vector<VertAttr>& vert_attrs, const Vec4i& t);
double tetra_quality(const Vec3d& p0, const Vec3d& p1, const Vec3d& p2, const Vec3d& p3);
Vec3d newton_position_from_stack(
    std::vector<std::array<double, 12>>& assembles);
} // namespace prism::tet
