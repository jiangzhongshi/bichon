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
#include <batm/tetra_utils.hpp>


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

    auto vert_info = [&tet_v, &pc]() {
        std::vector<VertAttr> vert_attrs(tet_v.rows());
        auto nshell = pc.mid.size();
        for (auto i = 0; i < tet_v.rows(); i++) {
            vert_attrs[i].pos = tet_v.row(i);
        }
        for (auto i = 0; i < pc.mid.size(); i++) {
            assert(vert_attrs[i].pos == pc.mid[i]);
            vert_attrs[i].mid_id = i;
        }
        return vert_attrs;
    }();

    // Tet-Face (tet_t, k) -> Shell Prisms (pc.mF)
    auto tet_info = [&tet_t, &pc]() {
        std::vector<TetAttr> tet_attrs(tet_t.rows());

        std::map<std::array<int, 3>, int> cell_finder;
        for (auto i = 0; i < pc.F.size(); i++) {
            auto pri = pc.F[i];
            std::sort(pri.begin(), pri.end());
            cell_finder.emplace(pri, i);
        }
        // RowMati marker = RowMati::Constant(tet_t.rows(), 4, -1);
        for (auto i = 0; i < tet_t.rows(); i++) {
            auto& t_a = tet_attrs[i];
            for (auto j = 0; j < 4; j++) {
                t_a.conn[j] = tet_t(i, j);

                auto face = std::array<int, 3>();
                for (auto k = 0; k < 3; k++) face[k] = tet_t(i, (j + k + 1) % 4);
                std::sort(face.begin(), face.end());
                auto it = cell_finder.find(face);
                if (it != cell_finder.end()) { // found
                    t_a.prism_id[j] = it->second;
                }
            }
        }
        return tet_attrs;
    }();

    auto vert_tet_conn = [&tet_info, n_verts = vert_info.size()]() {
        auto vt_conn = std::vector<std::vector<int>>(n_verts);
        for (auto i = 0; i < tet_info.size(); i++) {
            for (auto j = 0; j < 4; j++) vt_conn[tet_info[i].conn[j]].emplace_back(i);
        }
        std::for_each(vt_conn.begin(), vt_conn.end(), [](auto& vec) {
            std::sort(vec.begin(), vec.end());
        });
        return vt_conn;
    }();

    // count how many marks
    [&tet_info, &tet_t, &pc]() {
        auto cnt = 0;
        for (auto i = 0; i < tet_t.rows(); i++) {
            for (auto j = 0; j < 4; j++) {
                if (tet_info[i].prism_id[j] != -1) cnt++;
            }
        }
        CHECK_EQ(cnt, pc.F.size());
    }();

    split_edge(pc, vert_info, tet_info, vert_tet_conn, 0, 1);
}