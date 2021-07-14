////////////////////////////////////////////////////////////////////////////////
#include <pybind11/cast.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <type_traits>
#include <spdlog/spdlog.h>
////////////////////////////////////////////////////////////////////////////////

namespace py = pybind11;

using namespace pybind11::literals;
// using namespace prism;

extern void python_export_curve(py::module&);
extern void python_export_spatial(py::module&);
extern void python_export_prism(py::module&);

PYBIND11_MODULE(prism, m) {
  m.doc() = "core prism functions";  // optional module docstring

  m.def("foo",
        [](const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) { return V; },
        "foo", "vertices"_a, "facets"_a);
  m.def("loglevel", [](int level) {
      spdlog::set_level(static_cast<spdlog::level::level_enum>(level));
  }, "set log level");
  python_export_spatial(m);
  python_export_prism(m);
  python_export_curve(m);
}
