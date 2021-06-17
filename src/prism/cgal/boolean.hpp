#ifndef PRISM_CGAL_BOOLEAN_HPP
#define PRISM_CGAL_BOOLEAN_HPP

#include <prism/common.hpp>
#include <igl/MeshBooleanType.h>
namespace prism {
void boolean_resolve(const RowMatd& V1, const RowMati& F1, const RowMatd& V2,
                     const RowMati& F2, RowMatd& Vc, RowMati& Fc, Eigen::VectorXi& J);
void mesh_boolean(const RowMatd& VA, const RowMati& FA, const RowMatd& VB,
                  const RowMati& FB, igl::MeshBooleanType type, RowMatd& V_ABi,
                  RowMati& F_ABi, Eigen::VectorXi& Birth);
}
#endif