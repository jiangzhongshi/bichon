#include <igl/per_face_normals.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/vertex_triangle_adjacency.h>

#include <algorithm>
#include <highfive/H5Easy.hpp>
#include <prism/PrismCage.hpp>
#include <prism/bevel_utils.hpp>
#include <prism/cage_check.hpp>
#include <prism/cage_utils.hpp>
#include <prism/feature_utils.hpp>
#include <prism/geogram/AABB.hpp>
#include <prism/geogram/geogram_utils.hpp>
#include <prism/local_operations/validity_checks.hpp>
#include <queue>

#include "cumin/curve_validity.hpp"
#include "prism/PrismCage.hpp"
#include "prism/common.hpp"
#include "prism/local_operations/remesh_with_feature.hpp"
#include "test_common.hpp"

TEST_CASE("edge based bevel") {
  RowMatd V;
  RowMati F;
  igl::read_triangle_mesh(
      "/home/zhongshi/Workspace/libigl/tutorial/data/cube.off", V, F);
  prism::geo::init_geogram();
  RowMatd VN;
  std::set<int> omni_singu;

  bool good_normals =
      prism::cage_utils::most_visible_normals(V, F, VN, omni_singu);

  std::vector<int> face_parent;
  RowMatd dsV, dsVN;
  RowMati dsF;
  prism::bevel_utils::edge_based_bevel(V, F, VN, RowMati(), dsV, dsF, dsVN,
                                       face_parent);
  prism::bevel_utils::verify_edge_bevel(dsV, dsF, dsVN, RowMati());
  {
    auto file = H5Easy::File("temp.h5", H5Easy::File::Overwrite);
    H5Easy::dump(file, "V", dsV);
    H5Easy::dump(file, "F", dsF);
    H5Easy::dump(file, "VN", dsVN);
    H5Easy::dump(file, "fp", face_parent);
  }
}