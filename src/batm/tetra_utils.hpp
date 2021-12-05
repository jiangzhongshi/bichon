#include <prism/common.hpp>
#include "prism/local_operations/section_remesh.hpp"

struct PrismCage;

namespace prism::local {
struct RemeshOptions;
}
/**
 * @brief loosely coupled with PrismCage
 *
 */
namespace prism::tet {

struct TetAttr
{
    std::array<int, 4> prism_id = {{-1, -1, -1, -1}}; /**The prism cell id for each face.*/
    bool is_removed = false;
    Vec4i conn;
};

struct VertAttr
{
    Vec3d pos;
    int mid_id = -1; /**Points to the vertex id in the shell*/
};

/**
 * @brief Assuming an initial conforming mesh: tet_v[:len(mV)] = mV
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
prepare_tet_info(PrismCage& pc, RowMatd& tet_v, RowMati& tet_t);

double circumradi2(const Vec3d& p0, const Vec3d& p1, const Vec3d& p2, const Vec3d& p3);

int tetra_validity(const std::vector<VertAttr>& vert_attrs, const Vec4i& t);

bool split_edge(
    PrismCage& pc,
    const prism::local::RemeshOptions& option,
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
    int v0);

} // namespace prism::tet
