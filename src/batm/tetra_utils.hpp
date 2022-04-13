#pragma once

#include <array>
#include <memory>
#include <prism/common.hpp>

#include <Rational.h>


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

enum class SmoothType {
    kSurfaceSnap = 0,
    kInteriorNewton,
    kShellZoom,
    kShellPan,
    kShellRotate,
    kMax
};

struct SizeController
{
    std::shared_ptr<prism::geogram::AABB_tet> bg_tree;
    Eigen::VectorXd sizes;
    bool surface_only = false;
    SizeController(const RowMatd& tetV, const RowMati& tetT, const Eigen::VectorXd& tetSize);
    double find_size_bound(const std::array<Vec3d, 4>&) const;
    double find_size_bound(const std::array<Vec3d, 3>&) const;
};

struct TetAttr
{
    Vec4i conn = {{-1, -1, -1, -1}};
    Vec4i prism_id = {{-1, -1, -1, -1}};
    std::array<std::set<int>, 4> track_prisms;
    bool is_removed = false;
};

using Vector3r = Eigen::Matrix<triwild::Rational, 1, 3>;
using Vector2r = Eigen::Matrix<triwild::Rational, 1, 2>;
struct VertAttr
{
    Vec3d pos = Vec3d::Zero();
    Vector3r pos_r = Vector3r::Zero();
    bool rounded = true;
    int mid_id = -1; /**Points to the vertex id in the shell*/
};

using vert_info_t = std::vector<prism::tet::VertAttr>;
using tet_info_t = std::vector<prism::tet::TetAttr>;

using tetmesh_t =
    std::tuple<prism::tet::vert_info_t, prism::tet::tet_info_t, std::vector<std::vector<int>>>;


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
std::tuple<vert_info_t, tet_info_t, std::vector<std::vector<int>>> prepare_tet_info(
    const PrismCage* pc,
    const RowMatd& tet_v,
    const RowMati& tet_t,
    const Eigen::VectorXi& tet_v_pid);


std::tuple<vert_info_t, tet_info_t, std::vector<std::vector<int>>> reload(
    std::string filename,
    const PrismCage* pc);

std::vector<std::pair<int, int>> edge_adjacent_boundary_face(
    const tet_info_t& tet_attrs,
    const std::vector<std::vector<int>>& vert_conn,
    int v0,
    int v1);


bool split_edge(
    PrismCage* pc,
    prism::local::RemeshOptions& option,
    prism::tet::vert_info_t& vert_attrs,
    prism::tet::tet_info_t& tet_attrs,
    std::vector<std::vector<int>>& vert_conn,
    int v0,
    int v1);


bool smooth_vertex(
    PrismCage* pc,
    const prism::local::RemeshOptions& option,
    prism::tet::vert_info_t& vert_attrs,
    const prism::tet::tet_info_t& tet_attrs,
    const std::vector<std::vector<int>>& vert_conn,
    SmoothType smooth_type,
    int v0,
    double size_control);

bool swap_face(
    const PrismCage* pc,
    const prism::local::RemeshOptions& option,
    prism::tet::vert_info_t& vert_attrs,
    prism::tet::tet_info_t& tet_attrs,
    std::vector<std::vector<int>>& vert_conn,
    int v0_id,
    int v1_id,
    int v2_id,
    double size_control);

bool swap_edge(
    const PrismCage* pc,
    const prism::local::RemeshOptions& option,
    prism::tet::vert_info_t& vert_attrs,
    prism::tet::tet_info_t& tet_attrs,
    std::vector<std::vector<int>>& vert_conn,
    int v1_id,
    int v2_id,
    double size_control);

bool collapse_edge(
    PrismCage* pc,
    const prism::local::RemeshOptions& option,
    prism::tet::vert_info_t& vert_attrs,
    prism::tet::tet_info_t& tet_attrs,
    std::vector<std::vector<int>>& vert_conn,
    int v1_id,
    int v2_id,
    double size_control);

bool flip_edge_sf(
    PrismCage* pc,
    const prism::local::RemeshOptions& option,
    std::vector<VertAttr>& vert_attrs,
    std::vector<TetAttr>& tet_attrs,
    std::vector<std::vector<int>>& vert_conn,
    int v0,
    int v1,
    double size_control);

void compact_tetmesh(prism::tet::tetmesh_t& tetmesh, PrismCage* pc = nullptr);

std::optional<Vec3d>
get_snap_position(const PrismCage& pc, const std::vector<int>& neighbor_pris, int v0);

std::tuple<Eigen::VectorXd, Eigen::VectorXd> snap_progress(
    const prism::tet::tetmesh_t& tetmesh,
    PrismCage* pc);

bool common_tet_checkers(
    double quality_threshold,
    const prism::tet::vert_info_t& vert_attrs,
    const prism::tet::tet_info_t& tet_attrs,
    const std::vector<Vec4i>& old_tets,
    const std::vector<Vec4i>& new_tets,
    double size_control);

bool tetmesh_sanity(const prism::tet::tetmesh_t& tetmesh, const PrismCage* pc);
} // namespace prism::tet

namespace prism::tet {
double diameter2(const Vec3d& p0, const Vec3d& p1, const Vec3d& p2, const Vec3d& p3);
bool tetra_validity(const prism::tet::vert_info_t& vert_attrs, const Vec4i& t);
double tetra_quality(const Vec3d& p0, const Vec3d& p1, const Vec3d& p2, const Vec3d& p3);
Vec3d newton_position_from_stack(std::vector<std::array<double, 12>>& assembles);
} // namespace prism::tet

namespace prism::tet::sec {
// section tetmesh, surface insertion related utils

bool divide_tetra(
    std::vector<VertAttr>& vert_attrs,
    std::vector<TetAttr>& tet_attrs,
    std::vector<std::vector<int>>& vert_conn,
    int t_id,
    const Vec3d& p);

bool split_edge(
    prism::tet::vert_info_t& vert_attrs,
    prism::tet::tet_info_t& tet_attrs,
    std::vector<std::vector<int>>& vert_conn,
    int v0,
    int v1);

bool split_face(
    std::vector<VertAttr>& vert_attrs,
    std::vector<TetAttr>& tet_attrs,
    std::vector<std::vector<int>>& vert_conn,
    int v0,
    int v1,
    int v2);

bool round(
    std::vector<VertAttr>& vert_attrs,
    std::vector<TetAttr>& tet_attrs,
    std::vector<std::vector<int>>& vert_conn,
    int v0);
} // namespace prism::tet::sec
