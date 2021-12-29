#ifndef PRISM_SPATIAL_HASH_AABB_HASH
#define PRISM_SPATIAL_HASH_AABB_HASH
#include <list>
#include <memory>
#include <set>
/// @brief An entry into the hash grid as a (key, value) pair.

#include "../common.hpp"

namespace prism {

using HashMap =
    std::map<std::tuple<long, long, long>, std::shared_ptr<std::list<int>>>;
using HashPtr =
    std::pair<std::shared_ptr<std::list<int>>, std::list<int>::iterator>;

    
// The following does not store AABB or any geometry at all, only indices.
struct HashGrid {
  HashGrid(const Vec3d &lower, const Vec3d &upper, double cell)
      : m_domain_min(lower), m_domain_max(upper), m_cell_size(cell) {
    m_grid_size = int(std::ceil((upper - lower).maxCoeff() / m_cell_size));
  };
  HashGrid(const std::vector<Vec3d> &V, const std::vector<Vec3i> &F,
           bool filled = true);
  HashGrid(const RowMatd &V, const RowMati &F, bool filled = true);
  void insert_triangles(const RowMatd &V, const RowMati &F,
                        const std::vector<int> &fid);
  void insert_triangles(const std::vector<Vec3d> &V,
                        const std::vector<Vec3i> &F,
                        const std::vector<int> &fid);
  std::vector<std::pair<int, int>> self_candidates() const;
  void query(const Vec3d &lower, const Vec3d &upper, std::set<int> &) const;
  void add_element(const Vec3d &lower, const Vec3d &upper, const int index);
  void remove_element(const int index);
  void bound_convert(const Vec3d &, Eigen::Array<long, 3, 1> &) const;
  bool clear() {m_face_items.clear(); face_stores.clear();return true;}

  // reorder and update after edge collapse
  void update_after_collapse();
  // spatial partition parameters
  Vec3d m_domain_min;
  Vec3d m_domain_max;
  double m_cell_size;
  size_t m_grid_size;

  // key is (integral) spatial coordinate
  // val points to list of faces (abstract, w/o geometry).
  HashMap m_face_items;
  std::vector<std::vector<HashPtr>> face_stores;  // facilitate element removal
};
}  // namespace prism
#endif