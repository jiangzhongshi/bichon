#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <type_traits>
////////////////////////////////////////////////////////////////////////////////
#include <igl/AABB.h>
#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>

#include <cumin/curve_common.hpp>
#include <cumin/curve_utils.hpp>
#include <cumin/inversion_check.hpp>
#include <prism/PrismCage.hpp>
#include <prism/common.hpp>
#include <prism/geogram/AABB.hpp>
namespace py = pybind11;

using namespace pybind11::literals;
auto find_order_tet = [](int rows) {
  int order = 0;
  while ((order + 1) * (order + 2) * (order + 3) < 6 * rows) {
    order++;
  }
  if ((order + 1) * (order + 2) * (order + 3) != 6 * rows) {
    throw std::runtime_error(
        fmt::format("cp is not a tetrahedron control points, order={}, cp={}",
                    order, rows));
  }
  return order;
};
auto find_order_tri = [](int rows) {
  int order = 0;
  while ((order + 1) * (order + 2) < 2 * rows) {
    order++;
  }
  if ((order + 1) * (order + 2) != 2 * rows) {
    throw std::runtime_error(
        fmt::format("cp is not a tetrahedron control points, order={}, cp={}",
                    order, rows));
  }
  return order;
};
void python_export_curve(py::module &m) {
  m.def(
      "tetrahedron_inversion_check",
      [](const RowMatd &cp) {
        prism::curve::magic_matrices(find_order_tet(cp.rows()) - 1);
        return prism::curve::tetrahedron_inversion_check(cp);
      },
      "", "cp"_a);

  m.def(
      "elevated_positive_check",
      [](const RowMatd &f_base, const RowMatd &f_top, const RowMatd &lagcp,
         bool recurse_check) -> bool {
        auto helper =
            prism::curve::magic_matrices(find_order_tri(lagcp.rows()));
        auto &tri15lag_from_tri10bern = helper.elev_lag_from_bern;
        auto &dxyz = helper.volume_data.vec_dxyz;
        auto tri4_cod = codecs_gen_id(helper.tri_order + 1, 2);
        auto tet4_cod = codecs_gen_id(helper.tri_order + 1, 3);
        // 6T {35 x 3D}
        auto tens = prism::curve::surface_to_decomposed_tetra(
            f_base, lagcp, f_top, f_base.row(0) == f_top.row(0), true, tri4_cod,
            tet4_cod);
        for (auto &d : dxyz) {
          for (auto &t : tens) {
            Eigen::Matrix3d j = d * t;
            auto det = j.determinant();
            if (det <= 0) {
              spdlog::debug("negative {}", det);
              return false;
            }
          }
        }
        if (recurse_check) {
          for (auto &t : tens) {
            // this is codec_bc.
            if (!prism::curve::tetrahedron_inversion_check(t)) {
              spdlog::debug("blocked by recursive");
              return false;
            }
          }
        }
        return true;
      },
      "f_base"_a, "f_top"_a, "lagrcp"_a, "recurse_check"_a = false);
}
