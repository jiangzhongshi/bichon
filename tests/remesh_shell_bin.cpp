#include "test_common.hpp"
#include <doctest.h>
#include <prism/local_operations/local_mesh_edit.hpp>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include "cumin/curve_validity.hpp"
#include "prism/PrismCage.hpp"
#include "prism/geogram/AABB.hpp"
#include "prism/geogram/geogram_utils.hpp"
#include "prism/local_operations/remesh_pass.hpp"
#include "prism/predicates/positive_prism_volume_12.hpp"
#include "prism/spatial-hash/AABB_hash.hpp"
#include <highfive/H5Easy.hpp>
#include <igl/read_triangle_mesh.h>
#include <prism/cage_check.hpp>
#define DATA_PATH "../build_clang/"

TEST_CASE("cumin collapse") {
  prism::geo::init_geogram();
  spdlog::set_level(spdlog::level::debug);
#ifdef NDEBUG
  spdlog::set_level(spdlog::level::info);
  if (true) {
    RowMatd V;
    RowMati F;
    igl::read_triangle_mesh(
      // "../tests/data/sphere.obj",V,F);
      DATA_PATH "spot.obj", V, F);
    put_in_unit_box(V);
    PrismCage pc(V, F, 0.2, 0.1);
    pc.serialize(DATA_PATH "sphere_temp.h5");
  }
#endif
  PrismCage pc( DATA_PATH "sphere_temp.h5");
  prism::local::RemeshOptions option(pc.mid.size(), 0.5);
  option.distortion_bound = 1e-5;
  option.target_adjustment.resize(pc.mid.size(), 0.03);
  // option.target_thickness = 0.01;

  option.collapse_quality_threshold = 30;
  option.parallel = false;

  int order = 3;
  double normal_th = 0.99;
  double dist_th = 1e-2;
  spdlog::info("order {}", order);
  auto complete_cp =
      prism::curve::initialize_cp(pc.mid, pc.F, TRI_CODEC.at(order));

  for (int i = 0; i < 10; i++) {
    prism::local::wildcollapse_pass(pc, option);
    complete_cp.erase(std::remove_if(complete_cp.begin(), complete_cp.end(),
                                     [](auto &c) { return c(0, 0) == -1; }),
                      complete_cp.end());
    prism::local::localsmooth_pass(pc, option);
    prism::local::wildflip_pass(pc, option);
    REQUIRE(prism::cage_check::cage_is_away_from_ref(pc));
    pc.serialize(fmt::format("spot_noelev_n{}_o{}.h5",normal_th, order),
                 std::function<void(HighFive::File &)>(
                     [complete_cp](HighFive::File &file) {
                       std::vector<std::vector<std::vector<double>>> vec_cp(
                           complete_cp.size());
                       for (int i = 0; i < complete_cp.size(); i++)
                         igl::matrix_to_list(complete_cp[i], vec_cp[i]);

                       H5Easy::dump(file, "complete_cp", vec_cp);
                     }));
    // if (pc.F.size() < 1000)
    // return;
  }
}
