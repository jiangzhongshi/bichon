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
#include "batm/surface_insert.hpp"
#include "batm/tetra_remesh_pass.hpp"
#include "prism/PrismCage.hpp"
#include "prism/cage_check.hpp"
#include "prism/predicates/inside_prism_tetra.hpp"
#include "spdlog/common.h"

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

#include <wmtk/utils/InsertTriangleUtils.hpp>

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
        prism::tet::insert_all_points(tetmesh, tids, points);
        REQUIRE_EQ(num_pts + points.size(), tetv.size());
    }
}

TEST_CASE("insert-surface")
{
    auto file = H5Easy::File("../buildr/bunny.off.h5.tet.h5");

    auto bgV = H5Easy::load<RowMatd>(file, "tet_v");
    auto bgT = H5Easy::load<RowMati>(file, "tet_t");

    auto tree = prism::geogram::AABB_tet(bgV, bgT);
}