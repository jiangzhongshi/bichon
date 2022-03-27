#include <doctest.h>

#include <batm/tetra_logger.hpp>
#include <batm/tetra_utils.hpp>
#include <nlohmann/json_fwd.hpp>
#include <prism/common.hpp>
#include <prism/geogram/AABB.hpp>
#include <prism/geogram/geogram_utils.hpp>
#include <prism/local_operations/remesh_pass.hpp>
#include <prism/spatial-hash/AABB_hash.hpp>
#include <prism/spatial-hash/self_intersection.hpp>
#include "batm/tetra_remesh_pass.hpp"
#include "prism/PrismCage.hpp"
#include "prism/cage_check.hpp"

#include <geogram/basic/geometry.h>
#include <geogram/numerics/predicates.h>
#include <igl/barycenter.h>
#include <igl/predicates/predicates.h>
#include <igl/read_triangle_mesh.h>
#include <igl/volume.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <condition_variable>
#include <highfive/H5Easy.hpp>
#include <iterator>
#include <memory>
#include <nlohmann/json.hpp>
#include <numeric>
#include <prism/geogram/AABB_tet.hpp>
#include <queue>
#include <tuple>


auto degenerate_config = [](auto& tetv, auto tet, auto& pt) -> std::array<int, 3> {
    using GEO::PCK::orient_3d;
    using GEO::PCK::points_are_colinear_3d;
    std::array<bool, 4> colinear{false, false, false, false};
    for (auto i = 0; i < 4; i++) {
        if (pt == tetv[tet[i]]) return {tet[i], -1, -1}; // vertex
    }
    for (auto i = 0; i < 4; i++) {
        if (orient_3d(
                pt.data(),
                tetv[tet[(i + 1) % 4]].data(),
                tetv[tet[(i + 2) % 4]].data(),
                tetv[tet[(i + 3) % 4]].data()) == 0) {
            for (auto j = 0; j < 3; j++) {
                if (points_are_colinear_3d(
                        pt.data(),
                        tetv[tet[(i + 1 + j) % 4]].data(),
                        tetv[tet[(i + 1 + (j + 1) % 3) % 4]].data()))
                    return {tet[(i + 1 + j) % 4], tet[(i + 1 + (j + 1) % 3) % 4], -1}; // edge
            }
            return {tet[(i + 1) % 4], tet[(i + 2) % 4], tet[(i + 3) % 4]}; // face
        }
    }
    return {-1, -1, -1}; // general
};
auto insert_all_points = [](auto& tetmesh, auto& tets, auto& vt, auto& points) {
    // handle general position first
    std::vector<std::tuple<std::array<int,3>, Vec3d>> degen;
    for (auto i = 0; i < tetmesh.tets.size(); i++) {
        auto tid = tetmesh.tets[i];
        auto pt = points[i];
        auto config = degenerate_config(tetv, tett[tid], pt);
        if (config[0] != -1) { // degenerate, save for later
            if (config[1] == -1) { // point degenerate
                // vertex
                continue;
            } else if (config[2] == -1) {
                // edge
                degen.emplace(config, points);
            } else { // face
                degen.emplace(config, points);
            }
        } else { // insert the single point
                single_point_insert(tetv, tett, vt);
        }
    }
};

TEST_CASE("insert-points")
{
    auto file = H5Easy::File("../buildr/bunny.off.h5.tet.h5");

    auto bgV = H5Easy::load<RowMatd>(file, "tet_v");
    auto bgT = H5Easy::load<RowMati>(file, "tet_t");

    auto tree = prism::geogram::AABB_tet(bgV, bgT);

    auto [tid, bc] = tree.point_query(Vec3d(.5, .5, .5));
    spdlog::info("tid {}", tid);

    
    std::vector<std::vector<int>> VT;
    insert_all_points(bgV, bgT, VT, tids, points);
}
