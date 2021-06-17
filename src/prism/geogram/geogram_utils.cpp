#include "geogram_utils.hpp"
void prism::geo::init_geogram() {
  static bool first_time = true;

  if (first_time) {
    first_time = false;
  } else {
    return;
  }

  // Do not install custom signal handlers
  setenv("GEO_NO_SIGNAL_HANDLER", "1", 1);

  // Init logger first so we can hide geogram output from init
  GEO::Logger::initialize();

  // Do not show geogram output
  GEO::Logger::instance()->unregister_all_clients();
  GEO::Logger::instance()->set_quiet(true);

#if 0
    // Use the following code to disable multi-threading in geogram (for debugging purposes).
    GEO::Process::enable_multithreading(false);
    GEO::Process::set_max_threads(1);
#endif

  // Initialize global instances (e.g., logger), register MeshIOHandler, etc.
  GEO::initialize(GEO::GEOGRAM_NO_HANDLER);

  // Import standard command line arguments, and custom ones
  GEO::CmdLine::import_arg_group("standard");
  GEO::CmdLine::import_arg_group("pre");
  GEO::CmdLine::import_arg_group("algo");
  // GEO::CmdLine::import_arg_group("sys");
  GEO::CmdLine::set_arg("sys:assert", "throw");
}

void prism::geo::to_geogram_mesh(const RowMatd &V, const RowMati &F,
                                 GEO::Mesh &M) {
  init_geogram();
  M.clear();
  // Setup vertices
  M.vertices.create_vertices((int)V.rows());
  for (int i = 0; i < (int)M.vertices.nb(); ++i) {
    GEO::vec3 &p = M.vertices.point(i);
    p[0] = V(i, 0);
    p[1] = V(i, 1);
    p[2] = V(i, 2);
  }
  {
    GEO::Attribute<int> indices(M.vertices.attributes(), "vertex_id");
    for (int i = 0; i < V.rows(); i++)
      indices[i] = i;
  }

  if (F.cols() == 2) {
    M.edges.create_edges((int)F.rows());
    for (int i = 0; i < F.rows(); i++)
      for (int j = 0; j < F.cols(); j++) {
        M.edges.set_vertex(i,j,F(i,j));
      }
    GEO::Attribute<int> indices(M.edges.attributes(), "edge_id");
    for (int i = 0; i < F.rows(); i++)
      indices[i] = i;
  } else if (F.cols() == 3) {
    M.facets.create_triangles((int)F.rows());
    // Setup faces
    for (int i = 0; i < F.rows(); i++)
      for (int j = 0; j < F.cols(); j++)
        M.facets.set_vertex(i, j, F(i, j));
    M.facets.connect();
    GEO::Attribute<int> indices(M.facets.attributes(), "facet_id");
    for (int i = 0; i < F.rows(); i++)
      indices[i] = i;

  } else {
    M.cells.create_tets((int)F.rows());
    // Setup faces
    for (int i = 0; i < F.rows(); i++)
      for (int j = 0; j < 4; j++)
        M.cells.set_vertex(i, j, F(i, j));
    M.cells.connect();
    GEO::Attribute<int> indices(M.cells.attributes(), "cell_id");
    for (int i = 0; i < F.rows(); i++)
      indices[i] = i;
  }
}

void prism::geo::from_geogram_mesh(const GEO::Mesh &M, RowMatd &V, RowMati &T) {
  init_geogram();
  V.resize(M.vertices.nb(), 3);
  for (int i = 0; i < (int)M.vertices.nb(); ++i) {
    GEO::vec3 p = M.vertices.point(i);
    V.row(i) << p[0], p[1], p[2];
  }
  if (M.cells.nb() > 0) {
    assert(M.cells.are_simplices());
    T.resize(M.cells.nb(), 4);
    for (int c = 0; c < (int)M.cells.nb(); ++c) {
      for (int lv = 0; lv < 4; ++lv) {
        T(c, lv) = M.cells.vertex(c, lv);
      }
    }
  } else if (M.facets.nb() > 0) {
    assert(M.facets.are_simplices());
    T.resize(M.facets.nb(), 3);
    for (int c = 0; c < (int)M.facets.nb(); ++c) {
      for (int lv = 0; lv < 4; ++lv) {
        T(c, lv) = M.facets.vertex(c, lv);
      }
    }
  }
}
