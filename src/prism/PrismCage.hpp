#ifndef PRISM_PRISMCAGE_HPP
#define PRISM_PRISMCAGE_HPP

#include <array>
#include <list>
#include <memory>
#include <set>
#include <tuple>
#include <any>
#include <mutex>

#include "common.hpp"

namespace prism::geogram {
struct AABB;
};
namespace prism {
struct HashGrid;
};

struct PrismCage {
  enum class SeparateType { kShell, kSurface, kNone };
  ///////////////////////////////////////
  // The input reference surface, with fixed structure throughout the
  // optimization. contains V, F and acceleration AABB.
  ///////////////////////////////////////
  struct RefSurf {
    RowMatd inpV;
    RowMatd V;
    RowMati F;
    std::vector<std::vector<int>> VF, VFi;
    std::unique_ptr<prism::geogram::AABB> aabb;
  };
  RefSurf ref;

  void serialize(std::string filename, std::any additional = {}) const;
  // data for zig
  RowMatd zig_top;
  RowMatd zig_base;
  ///////////////////////////////////////
  // Data for the Cage
  // Base Vertex, Top Vertex, F, TT, TTi
  // All with std::vector to enable dynamic change.
  ///////////////////////////////////////
  std::vector<Vec3d> base;
  std::vector<Vec3d> top;
  std::vector<Vec3d> mid;
  std::vector<Vec3i> F;

  std::vector<std::set<int>> track_ref;
  std::shared_ptr<prism::HashGrid> base_grid = nullptr;
  std::shared_ptr<prism::HashGrid> top_grid = nullptr;
  std::mutex grid_mutex;


  // marked feature edges.
  // a map from endpoints to chain id and list of vertices. As a feature representation.
  prism::meta_type_t meta_edges;

  // specified constraint points (for distance bound) on each face.
  std::vector<std::vector<int>> constraints_per_face;
  RowMatd constraints_points_bc;
  ///////////////////////////////////////
  // Constructor and Initialize cage
  ///////////////////////////////////////
  PrismCage() = default;
  PrismCage(std::string);
  PrismCage(const RowMatd &vert, const RowMati &face, double dooseps = 0.2,
            double initial_step = 1e-4, SeparateType st=SeparateType::kSurface);
  PrismCage(const RowMatd &vert, const RowMati &face,
            RowMati&& feature_edges, 
            Eigen::VectorXi&& feature_corners, 
            Eigen::VectorXi && cons_points_fid, RowMatd&& cons_points_bc,
            double initial_step = 1e-4, SeparateType st=SeparateType::kSurface);
  void load_from_hdf5(std::string);
  void construct_cage(const RowMatd &);
  void init_track();
  void cleanup_empty_faces(Eigen::VectorXi &NI, Eigen::VectorXi &NJ);
};

#endif