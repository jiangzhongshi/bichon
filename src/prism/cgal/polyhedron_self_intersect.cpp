#include "polyhedron_self_intersect.hpp"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/box_intersection_d.h>
#include <igl/vertex_triangle_adjacency.h>
#include <prism/common.hpp>
#include <iostream>
#include <vector>
namespace prism::cgal {
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Surface_mesh<K::Point_3> Mesh;
typedef boost::graph_traits<Mesh>::face_descriptor face_descriptor;

typedef K::Point_3 Point_3;
typedef K::Triangle_3 Triangle_3;

namespace PMP = CGAL::Polygon_mesh_processing;
void Eigen_to_Mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                   Mesh& mesh) {
  for (int i = 0; i < V.rows(); i++)
    mesh.add_vertex(K::Point_3(V(i, 0), V(i, 1), V(i, 2)));
  for (int i = 0; i < F.rows(); i++)
    mesh.add_face(Mesh::Vertex_index(F(i, 0)), Mesh::Vertex_index(F(i, 1)),
                  Mesh::Vertex_index(F(i, 2)));
}

bool polyhedron_self_intersect_legacy(const Eigen::MatrixXd& V,
                                      const Eigen::MatrixXi& F) {
  Mesh mesh;
  Eigen_to_Mesh(V, F, mesh);
  bool intersecting = PMP::does_self_intersect(
      mesh, PMP::parameters::vertex_point_map(get(CGAL::vertex_point, mesh)));
  return intersecting;
}

void polyhedron_all_self_intersections_legacy(const Eigen::MatrixXd& V,
                                              const Eigen::MatrixXi& F) {
  Mesh mesh;
  Eigen_to_Mesh(V, F, mesh);
  std::vector<std::pair<face_descriptor, face_descriptor>> intersected_tris;
  PMP::self_intersections(mesh, std::back_inserter(intersected_tris));
  std::cout << intersected_tris.size() << " pairs of triangles intersect."
            << std::endl;
}

bool polyhedron_self_intersect(const Eigen::MatrixXd& V,
                               const Eigen::MatrixXi& F,
                               std::vector<std::pair<int, int>>& pairs) {
  typedef std::vector<Triangle_3> Triangles;
  typedef Triangles::iterator Iterator;
  typedef CGAL::Box_intersection_d::Box_with_handle_d<double, 3, Iterator> Box;
  std::vector<std::set<int>> vert_adj_triangles(F.rows());  // include itself
  std::vector<std::vector<int>> VF, VFi;
  igl::vertex_triangle_adjacency(V, F, VF, VFi);
  for (int i = 0; i < F.rows(); i++) {
    for (int j = 0; j < 3; j++)
      vert_adj_triangles[i].insert(VF[F(i, j)].begin(), VF[F(i, j)].end());
  }

  std::vector<Point_3> vertices;
  vertices.reserve(V.rows());
  std::vector<Triangle_3> triangles;
  triangles.reserve(F.rows());
  for (int i = 0; i < V.rows(); i++)
    vertices.emplace_back(V(i, 0), V(i, 1), V(i, 2));
  for (int i = 0; i < F.rows(); i++)
    triangles.emplace_back(vertices[F(i, 0)], vertices[F(i, 1)],
                           vertices[F(i, 2)]);

  // Create the corresponding vector of bounding boxes
  std::vector<Box> boxes;
  for (auto i = triangles.begin(); i != triangles.end(); ++i)
    boxes.push_back(Box(i->bbox(), i));

  auto reporter = [&triangles, &vert_adj_triangles, &pairs](const Box& a,
                                                            const Box& b) {
    int f0 = (a.handle() - triangles.begin());
    int f1 = (b.handle() - triangles.begin());
    std::set<int>& adj_f0 = vert_adj_triangles[f0];
    if (std::find(adj_f0.begin(), adj_f0.end(), f1) != adj_f0.end()) return;
    if (!a.handle()->is_degenerate() && !b.handle()->is_degenerate() &&
        CGAL::do_intersect(*(a.handle()), *(b.handle()))) {
      pairs.emplace_back(f0, f1);
    }
  };
  CGAL::box_self_intersection_d(boxes.begin(), boxes.end(), reporter);
  return pairs.size() > 0;
}

// declaration
bool tetrahedron_tetrahedron_intersection(
    const CGAL::Epick::Tetrahedron_3& Atet,
    const CGAL::Epick::Tetrahedron_3& Btet);

bool tetrashell_self_intersect(const Eigen::MatrixXd& base,
                               const Eigen::MatrixXd& top,
                               const Eigen::MatrixXi& F,
                               const std::vector<bool>& mask,
                               std::vector<std::pair<int, int>>& pairs) {
  typedef std::vector<K::Tetrahedron_3>::iterator Iterator;
  typedef CGAL::Box_intersection_d::Box_with_handle_d<double, 3, Iterator> Box;
  std::vector<std::set<int>> vert_adj_triangles(F.rows());  // include itself
  std::vector<std::vector<int>> VF, VFi;
  igl::vertex_triangle_adjacency(base, F, VF, VFi);
  for (int i = 0; i < F.rows(); i++) {
    for (int j = 0; j < 3; j++)
      vert_adj_triangles[i].insert(VF[F(i, j)].begin(), VF[F(i, j)].end());
  }

  std::vector<Point_3> vertices;
  vertices.reserve(2 * base.rows());
  std::vector<Triangle_3> triangles;
  std::vector<K::Tetrahedron_3> tetrahedra;
  triangles.reserve(F.rows());
  tetrahedra.reserve(3 * F.rows());
  int v_num = base.rows();
  for (int i = 0; i < base.rows(); i++)
    vertices.emplace_back(base(i, 0), base(i, 1), base(i, 2));
  for (int i = 0; i < top.rows(); i++)
    vertices.emplace_back(top(i, 0), top(i, 1), top(i, 2));
  for (int i = 0; i < F.rows(); i++) {
    int v0 = F(i, 0), v1 = F(i, 1), v2 = F(i, 2);
    triangles.emplace_back(vertices[F(i, 0)], vertices[F(i, 1)],
                           vertices[F(i, 2)]);
    auto tetconfig = v1 > v2 ? TETRA_SPLIT_A : TETRA_SPLIT_B;
    std::array<Point_3, 6> stackV{
        vertices[F(i, 0)],         vertices[F(i, 1)],
        vertices[F(i, 2)],         vertices[F(i, 0) + v_num],
        vertices[F(i, 1) + v_num], vertices[F(i, 2) + v_num],
    };
    for (int j = 0; j < 3; j++) {
      tetrahedra.emplace_back(stackV[tetconfig[j][0]], stackV[tetconfig[j][1]],
                              stackV[tetconfig[j][2]], stackV[tetconfig[j][3]]);
    }
  }
  // Create the corresponding vector of bounding boxes
  std::vector<Box> boxes;
  // for (int i = 0; i < tetrahedra.size(); i++)
  for (auto i = tetrahedra.begin(); i != tetrahedra.end(); ++i) {
    int fid = std::distance(tetrahedra.begin(), i) / 3;
    if (mask[fid]) boxes.push_back(Box(i->bbox(), i));
  }

  auto reporter = [&tetrahedra, &vert_adj_triangles, &pairs](const Box& a,
                                                             const Box& b) {
    int t0 = std::distance(tetrahedra.begin(), a.handle());
    int t1 = std::distance(tetrahedra.begin(), b.handle());
    auto& tetA = tetrahedra[t0];
    auto& tetB = tetrahedra[t1];

    std::set<int>& adj_f0 = vert_adj_triangles[t0 / 3];
    if (t0 / 3 == t1 / 3) return;
    if (std::find(adj_f0.begin(), adj_f0.end(), t1 / 3) != adj_f0.end()) return;
    if (!tetA.is_degenerate() && !tetB.is_degenerate() &&
        prism::cgal::tetrahedron_tetrahedron_intersection(tetA, tetB)) {
      pairs.emplace_back(t0 / 3, t1 / 3);
    }
  };

  CGAL::box_self_intersection_d(boxes.begin(), boxes.end(), reporter);
  return pairs.size() > 0;
}

bool polyhedron_self_intersect_edges(
    const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
    std::vector<std::pair<int, int>>& fe_pairs) {
  typedef std::vector<Triangle_3> Triangles;
  typedef Triangles::iterator Iterator;
  typedef CGAL::Box_intersection_d::Box_with_handle_d<double, 3, Iterator> Box;
  std::vector<std::set<int>> vert_adj_triangles(F.rows());  // include itself
  std::vector<std::vector<int>> VF, VFi;
  igl::vertex_triangle_adjacency(V, F, VF, VFi);
  for (int i = 0; i < F.rows(); i++) {
    for (int j = 0; j < 3; j++)
      vert_adj_triangles[i].insert(VF[F(i, j)].begin(), VF[F(i, j)].end());
  }

  std::vector<Point_3> vertices;
  vertices.reserve(V.rows());
  std::vector<Triangle_3> triangles;
  triangles.reserve(F.rows());
  for (int i = 0; i < V.rows(); i++)
    vertices.emplace_back(V(i, 0), V(i, 1), V(i, 2));
  for (int i = 0; i < F.rows(); i++)
    triangles.emplace_back(vertices[F(i, 0)], vertices[F(i, 1)],
                           vertices[F(i, 2)]);

  // Create the corresponding vector of bounding boxes
  std::vector<Box> boxes;
  for (auto i = triangles.begin(); i != triangles.end(); ++i)
    boxes.push_back(Box(i->bbox(), i));

  auto reporter = [&triangles, &vert_adj_triangles, &F, &vertices, &fe_pairs](
                      const Box& a, const Box& b) {
    int f0 = (a.handle() - triangles.begin());
    int f1 = (b.handle() - triangles.begin());
    std::set<int>& adj_f0 = vert_adj_triangles[f0];
    if (std::find(adj_f0.begin(), adj_f0.end(), f1) != adj_f0.end()) return;
    if (a.handle()->is_degenerate() || b.handle()->is_degenerate()) return;
    for (int i = 0; i < 3; i++) {
      if (CGAL::do_intersect(
              *(b.handle()),
              K::Segment_3(vertices[F(f0, i)], vertices[F(f0, (i + 1) % 3)]))) {
        ;
        fe_pairs.emplace_back(f0, i);
      }
      if (CGAL::do_intersect(
              *(a.handle()),
              K::Segment_3(vertices[F(f1, i)], vertices[F(f1, (i + 1) % 3)]))) {
        ;
        fe_pairs.emplace_back(f1, i);
      }
    }
  };
  CGAL::box_self_intersection_d(boxes.begin(), boxes.end(), reporter);
  return fe_pairs.size() > 0;
}

bool polyhedron_self_intersect(const Eigen::MatrixXd& V,
                               const Eigen::MatrixXi& F) {
  std::vector<std::pair<int, int>> pairs;
  return polyhedron_self_intersect(V, F, pairs);
}

}  // namespace prism::cgal