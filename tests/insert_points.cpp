#include <doctest.h>

#include <batm/tetra_logger.hpp>
#include <batm/tetra_utils.hpp>
#include <functional>
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
#include "prism/predicates/inside_prism_tetra.hpp"
#include "spdlog/common.h"

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


auto degenerate_config(
    const std::vector<prism::tet::VertAttr>& tetv,
    const Vec4i& tet,
    const Vec3d& pt) -> std::array<int, 3>
{
    using GEO::PCK::orient_3d;
    using GEO::PCK::points_are_colinear_3d;
    std::array<bool, 4> colinear{false, false, false, false};
    for (auto i = 0; i < 4; i++) {
        if (pt == tetv[tet[i]].pos) return {tet[i], -1, -1}; // vertex
    }
    for (auto i = 0; i < 4; i++) {
        if (orient_3d(
                pt.data(),
                tetv[tet[(i + 1) % 4]].pos.data(),
                tetv[tet[(i + 2) % 4]].pos.data(),
                tetv[tet[(i + 3) % 4]].pos.data()) == 0) {
            for (auto j = 0; j < 3; j++) {
                if (points_are_colinear_3d(
                        pt.data(),
                        tetv[tet[(i + 1 + j) % 4]].pos.data(),
                        tetv[tet[(i + 1 + (j + 1) % 3) % 4]].pos.data()))
                    return {tet[(i + 1 + j) % 4], tet[(i + 1 + (j + 1) % 3) % 4], -1}; // edge
            }
            return {tet[(i + 1) % 4], tet[(i + 2) % 4], tet[(i + 3) % 4]}; // face
        }
    }
    return {-1, -1, -1}; // general
};

auto insert_all_points(
    prism::tet::tetmesh_t& tetmesh,
    const std::vector<int>& hint_tid,
    const std::vector<Vec3d>& points)
{
    std::map<int, std::set<int>> split_maps;
    auto& [tetv, tett, vt] = tetmesh;

    std::function<int(size_t, const Vec3d&)> find_containing_tet;
    find_containing_tet = [&split_maps, &tetv = tetv, &tett = tett, &find_containing_tet](
                              size_t tid,
                              const Vec3d& pt) -> int {
        auto it = split_maps.find(tid);
        if (it == split_maps.end()) { // leaf
            auto tet = tett[tid].conn;
            if (prism::predicates::point_in_tetrahedron(
                    pt,
                    tetv[tet[0]].pos,
                    tetv[tet[1]].pos,
                    tetv[tet[2]].pos,
                    tetv[tet[3]].pos))
                return tid;
        } else {
            for (auto v : it->second) {
                auto res = find_containing_tet(v, pt);
                if (res != -1) return res;
            }
        }
        return -1;
    };

    std::vector<int> new_vid(points.size());
    prism::local::RemeshOptions option;
    for (auto i = 0; i < points.size(); i++) {
        auto pt = points[i];
        auto tid = find_containing_tet(hint_tid[i], pt); // the final tid

        auto config = degenerate_config(tetv, tett[tid].conn, pt);
        auto ux = tetv.size();
        prism::tet::logger().trace("insert {} with config {}", i, config);
        if (config[0] != -1) {
            if (config[1] == -1) { // point degenerate
                // vertex
                new_vid[i] = config[0];
                continue;
            } else if (config[2] == -1) {
                // edge
                prism::tet::split_edge(nullptr, option, tetv, tett, vt, config[0], config[1]);
                tetv.back().pos = pt;
                new_vid[i] = ux;
            } else {
                // face
                prism::tet::split_face(tetv, tett, vt, config[0], config[1], config[2]);
                tetv.back().pos = pt;
                new_vid[i] = ux;
            }
        } else {
            // general position, insert the single point
            auto num_tet = tett.size();
            prism::tet::divide_tetra(tetv, tett, vt, tid, pt);
            auto& s = split_maps[tid];
            for (auto n = num_tet; n < tett.size(); n++) {
                s.insert(n);
            }
            new_vid[i] = ux;
        }
    }
};

TEST_CASE("insert-bc-points")
{
    auto file = H5Easy::File("../buildr/bunny.off.h5.tet.h5");

    auto bgV = H5Easy::load<RowMatd>(file, "tet_v");
    auto bgT = H5Easy::load<RowMati>(file, "tet_t");

    auto tree = prism::geogram::AABB_tet(bgV, bgT);


    auto [tid, bc] = tree.point_query(Vec3d(.5, .5, .5));
    spdlog::info("tid {}", tid);

    auto tetmesh =
        prism::tet::prepare_tet_info(nullptr, bgV, bgT, Eigen::VectorXi::Constant(bgV.rows(), -1));

    auto& [tetv, tett, vt] = tetmesh;
    SUBCASE("insert-nodes")
    {
        std::vector<int> tids;
        std::vector<Vec3d> points;
        for (auto i = 0; i < tetv.size(); i++) {
            tids.push_back(vt[i].front());
            assert(!vt[i].empty());
            points.push_back(tetv[i].pos);
        }
        auto num_pts = tetv.size();
        insert_all_points(tetmesh, tids, points);
        REQUIRE_EQ(num_pts, tetv.size());
    };

    SUBCASE("insert-bc")
    {
        RowMatd bc;
        igl::barycenter(bgV, bgT, bc);
        std::vector<int> tids;
        std::vector<Vec3d> points;
        for (auto i = 0; i < bgT.rows(); i++) {
            tids.push_back(i);
            points.push_back(bc.row(i));
        }
        auto num_pts = tetv.size();
        insert_all_points(tetmesh, tids, points);
        REQUIRE_EQ(num_pts + points.size(), tetv.size());
    }
}
