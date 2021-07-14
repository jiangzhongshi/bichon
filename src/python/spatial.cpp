#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <type_traits>
////////////////////////////////////////////////////////////////////////////////
#include <igl/AABB.h>
#include <prism/geogram/AABB.hpp>
#include <prism/intersections.hpp>
#include <prism/common.hpp>
#include <prism/spatial-hash/self_intersection.hpp>

namespace py = pybind11;

using namespace pybind11::literals;

void python_export_spatial(py::module& m) {
  // AABB
  py::class_<igl::AABB<Eigen::MatrixXd, 2> > AABB2(m, "AABB2", "using igl::AABB for 2D case, not very robust.");
  AABB2.def(py::init<>())
      .def(py::init<const igl::AABB<Eigen::MatrixXd, 2>&>())
      .def("init",
           [](igl::AABB<Eigen::MatrixXd, 2>& tree, const Eigen::MatrixXd& V,
              const Eigen::MatrixXi& Ele) {
             return tree.init(V, Ele,
                              Eigen::Matrix<double, Eigen::Dynamic, 2>(),
                              Eigen::Matrix<double, Eigen::Dynamic, 2>(),
                              Eigen::VectorXi(), 0);
           })
      .def("squared_distance",
           [](const igl::AABB<Eigen::MatrixXd, 2>& tree,
              const Eigen::MatrixXd& V, const Eigen::MatrixXi& Ele,
              const Eigen::MatrixXd& P, Eigen::MatrixXd& sqrD,
              Eigen::MatrixXi& I, Eigen::MatrixXd& C) {
             return tree.squared_distance(V, Ele, P, sqrD, I, C);
           })
      .def("find",
           [](const igl::AABB<Eigen::MatrixXd, 2>& tree,
              const Eigen::MatrixXd& V, const Eigen::MatrixXi& Ele,
              const Eigen::RowVectorXd& q, bool first = true) {
             auto f = tree.find(V, Ele, q, first);
             if (f.size() == 0)
               return -1;
             else
               return f[0];
           })
      .def("find_all", [](const igl::AABB<Eigen::MatrixXd, 2>& tree,
                          const Eigen::MatrixXd& V, const Eigen::MatrixXi& Ele,
                          const Eigen::MatrixXd& q) {
        Eigen::VectorXi Fid(q.rows());
        Fid.setConstant(-1);
        Eigen::MatrixXd BC = Eigen::MatrixXd::Zero(q.rows(), 3);
        for (int i = 0; i < q.rows(); i++) {
          auto f = tree.find(V, Ele, q.row(i), true);
          if (f.size() != 0) {
            Fid(i) = f[0];
          }
        }
        int n = q.rows();
        Eigen::MatrixXd A(n, 2), B(n, 2), C(n, 2);
        auto F = Ele;
        for (int i = 0; i < n; i++) {
          auto f = Fid(i);
          if (f != -1) {
            A.row(i) = V.row(F(f, 0));
            B.row(i) = V.row(F(f, 1));
            C.row(i) = V.row(F(f, 2));
          } else {
            A.row(i) = V.row(0);
            B.row(i) = V.row(0);
            C.row(i) = V.row(0);
          }
        }
        igl::barycentric_coordinates(q, A, B, C, BC);
        return std::make_tuple(Fid, BC);
      });

  py::class_<prism::geogram::AABB> AABB(m, "AABB");
  AABB.def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXi&>())
      .def("intersects_triangle",
           [](const prism::geogram::AABB& self, const Vec3d& P0, const Vec3d& P1,
              const Vec3d& P2) {
             return self.intersects_triangle({P0, P1, P2});
           })
      .def("segment_query",
           [](const prism::geogram::AABB& self, const Vec3d& P0, const Vec3d& P1) {
             return self.segment_query(P0, P1);
           })
      .def("segment_hit", [](const prism::geogram::AABB& self, const Vec3d& P0,
                             const Vec3d& P1, bool ray = false) {
        prism::Hit hit;
        bool result = self.segment_hit(P0, P1, hit);
        if (result)
          return std::tuple(hit.id, hit.u, hit.v, hit.t);
        else
          return std::tuple(-1, 0., 0., -1.);
      });

     m.def("self_intersect", []
    ( const Eigen::MatrixXd & V,
    const Eigen::MatrixXi& F)->bool {
      std::vector<Vec3d> vecV;
      std::vector<Vec3i> vecF;
      eigen2vec(V, vecV);
      eigen2vec(V, vecV);
      auto pairs = prism::spatial_hash::self_intersections(vecV,vecF);
      return pairs.size() > 0;
    }, "use spatial hash for self intersection check", "V"_a, "F"_a);
}
