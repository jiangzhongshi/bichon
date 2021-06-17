#include "boolean.hpp"

#include <igl/combine.h>
#include <igl/copyleft/cgal/mesh_boolean.h>
#include <igl/copyleft/cgal/remesh_self_intersections.h>

// void prism::boolean_resolve(const RowMatd& V1, const RowMati& F1,
//                             const RowMatd& V2, const RowMati& F2, RowMatd&
//                             Vc, RowMati& Fc, Eigen::VectorXi& J) {
//   igl::copyleft::cgal::mesh_boolean(V1, F1, V2, F2, "resolve", Vc, Fc, J);
// }
namespace prism {
void mesh_boolean(const RowMatd& VA, const RowMati& FA, const RowMatd& VB,
                  const RowMati& FB, igl::MeshBooleanType type, RowMatd& V_ABi,
                  RowMati& F_ABi, Eigen::VectorXi& Birth) {
  igl::copyleft::cgal::mesh_boolean(VA, FA, VB, FB, type, V_ABi, F_ABi, Birth);
}
void remesh_intersect(const RowMatd& V1, const RowMati& F1, const RowMatd& V2,
                      const RowMati& F2, RowMatd& Vr, RowMati& Fr,
                      Eigen::VectorXi& CJ, Eigen::VectorXi& IM) {
  Eigen::MatrixXd VV;
  Eigen::MatrixXi FF;
  igl::combine(std::vector<RowMatd>{V1, V2}, std::vector<RowMati>{F1, F2}, VV,
               FF);
  igl::copyleft::cgal::RemeshSelfIntersectionsParam params;
  params.stitch_all = true;
  Eigen::MatrixXi IF, I;
  Eigen::MatrixXd matVr;
  Eigen::MatrixXi matFr;
  igl::copyleft::cgal::remesh_self_intersections(VV, FF, params, matVr, matFr,
                                                 IF, CJ, IM);
  Vr = matVr;
  Fr = matFr;
}
}  // namespace prism