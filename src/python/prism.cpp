#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <type_traits>
////////////////////////////////////////////////////////////////////////////////
#include <igl/AABB.h>
#include <prism/geogram/AABB.hpp>
#include <prism/PrismCage.hpp>

#include <prism/common.hpp>
#include <prism/phong/query_correspondence.hpp>

namespace py = pybind11;

using namespace pybind11::literals;


void python_export_prism(py::module &m)
{
  py::class_<PrismCage> cage(m, "PrismCage");
  cage.def(py::init<const std::string &>())
      .def_property_readonly("base", [](const PrismCage &cage) {
        RowMatd mb;
        vec2eigen(cage.base, mb);
        return mb;
      })
      .def_property_readonly("mid", [](const PrismCage &cage) {
        RowMatd mb;
        vec2eigen(cage.mid, mb);
        return mb;
      })
      .def_property_readonly("top", [](const PrismCage &cage) {
        RowMatd mb;
        vec2eigen(cage.top, mb);
        return mb;
      })
      .def_property_readonly("F", [](const PrismCage &cage) {
        RowMati mb;
      vec2eigen(cage.F, mb); 
      return mb; })
      .def_property_readonly("refV", [](const PrismCage &cage) { return cage.ref.V; })
      .def_property_readonly("refF", [](const PrismCage &cage) { return cage.ref.F; })
      .def("transfer", [](const PrismCage &cage, const RowMatd &pxV, const RowMati &pxF, const RowMatd &queryP) {
        Eigen::VectorXi queryF;
        RowMatd queryUV;
        prism::correspond_bc(cage, pxV, pxF, queryP, queryF, queryUV);
        return std::tuple(queryF, queryUV);
      });
}
