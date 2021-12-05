#include <doctest.h>
#include <geogram/basic/geometry.h>
#include <igl/Timer.h>
#include <igl/avg_edge_length.h>
#include <igl/combine.h>
#include <igl/read_triangle_mesh.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <batm/tetra_utils.hpp>
#include <highfive/H5Easy.hpp>
#include <iterator>
#include <numeric>
#include <prism/common.hpp>
#include <prism/geogram/AABB.hpp>
#include <prism/geogram/geogram_utils.hpp>
#include <prism/spatial-hash/AABB_hash.hpp>
#include <prism/spatial-hash/self_intersection.hpp>
#include "prism/PrismCage.hpp"
#include "spdlog/common.h"
#include <prism/local_operations/remesh_pass.hpp>


TEST_CASE("amr-sphere-prepare")
{
    using namespace prism::tet;
    std::string filename = "../tests/data/sphere_40.obj.h5";
    PrismCage pc(filename);
    spdlog::set_level(spdlog::level::trace);

    H5Easy::File file(filename, H5Easy::File::ReadOnly);
    auto tet_v = H5Easy::load<RowMatd>(file, "tet_v");
    auto tet_t = H5Easy::load<RowMati>(file, "tet_t");
    spdlog::info("Loading v {},t {} ", tet_v.rows(), tet_t.rows());
    spdlog::info("Shell size v{}, f{}", pc.base.size(), pc.F.size());
    for (auto i = 0; i < tet_t.rows(); i++) {
        auto r = circumradi2(
            tet_v.row(tet_t(i, 0)),
            tet_v.row(tet_t(i, 1)),
            tet_v.row(tet_t(i, 2)),
            tet_v.row(tet_t(i, 3)));
    }

    prism::local::RemeshOptions option;
    auto [vert_info, tet_info, vert_tet_conn] = prism::tet::prepare_tet_info(pc, tet_v, tet_t);
    split_edge(pc, option, vert_info, tet_info, vert_tet_conn, 0, 1);
}