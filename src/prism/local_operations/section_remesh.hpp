#ifndef PRISM_LOCAL_SECTION_REMESH_HPP
#define PRISM_LOCAL_SECTION_REMESH_HPP

#include "../common.hpp"

struct PrismCage;
namespace prism::geogram {
struct AABB;
}
namespace prism::section {

struct RemeshOptions {
  std::function<double(const Vec3d&)> sizing_field;
  int iteration_number = 3;
  int smooth_per_iteration = 5;
  double distortion_bound = 0.1;
  bool parallel = true;
  double collapse_quality_threshold = 30;
  bool collapse_improve_quality = true;
  bool split_improve_quality = true;
  RemeshOptions() = default;
  RemeshOptions(int v_num, double edge_len) {
    sizing_field = [edge_len](const Vec3d&) { return edge_len; };
  }
};
int wildcollapse_pass(const PrismCage& pc, const prism::geogram::AABB& base_tree,
                      const prism::geogram::AABB& top_tree, RemeshOptions& option,
                      std::vector<Vec3d>& V, std::vector<Vec3i>& F,
                      std::vector<std::set<int>>& track_to_prism,
                      std::vector<double>& target_adjustment);

bool project_to_ref_mesh(const PrismCage& pc,
                         const std::vector<std::set<int>>& track_ref,
                         const std::vector<int>& tris,
                         const Vec3d& point_value, Vec3d& point_on_ref);
void wildsplit_pass(const PrismCage& pc, const prism::geogram::AABB& base_tree,
                    const prism::geogram::AABB& top_tree, RemeshOptions& option,
                    std::vector<Vec3d>& V, std::vector<Vec3i>& F,
                    std::vector<std::set<int>>& track_ref,
                    std::vector<double>& target_adjustment);

void wildflip_pass(const PrismCage& pc, const prism::geogram::AABB& base_tree,
                   const prism::geogram::AABB& top_tree, RemeshOptions& option,
                   std::vector<Vec3d>& V, std::vector<Vec3i>& F,
                   std::vector<std::set<int>>& track_ref);
void localsmooth_pass(const PrismCage& pc, const prism::geogram::AABB& base_tree,
                      const prism::geogram::AABB& top_tree, RemeshOptions& option,
                      std::vector<Vec3d>& V, std::vector<Vec3i>& F,
                      std::vector<std::set<int>>& track_ref);
}  // namespace prism::section

#endif