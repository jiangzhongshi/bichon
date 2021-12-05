#include <prism/common.hpp>

struct PrismCage;

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

double circumradi2(const Vec3d& p0, const Vec3d& p1, const Vec3d& p2, const Vec3d& p3);

int tetra_validity(const std::vector<VertAttr>& vert_attrs, const Vec4i& t);

bool split_edge(
    PrismCage& pc,
    std::vector<VertAttr>& vert_attrs,
    std::vector<TetAttr>& tet_attrs,
    std::vector<std::vector<int>> vert_conn,
    int v0,
    int v1);

} // namespace prism::tet
