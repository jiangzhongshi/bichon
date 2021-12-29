#include "AABB_hash.hpp"

#include <igl/avg_edge_length.h>
#include <spdlog/spdlog.h>
#include <numeric>


prism::HashGrid::HashGrid(const std::vector<Vec3d> &V,
                          const std::vector<Vec3i> &F, bool filled)
    : HashGrid(Eigen::Map<const RowMatd>(V[0].data(), V.size(), 3),
               Eigen::Map<const RowMati>(F[0].data(), F.size(), 3), filled) {}

prism::HashGrid::HashGrid(const RowMatd &matV, const RowMati &matF,
                          bool filled) {
  m_domain_min = matV.colwise().minCoeff();
  m_domain_max = matV.colwise().maxCoeff();
  double avg_len = igl::avg_edge_length(matV, matF);
  m_cell_size = 2 * avg_len;
  m_grid_size =
      long(std::ceil((m_domain_max - m_domain_min).maxCoeff() / m_cell_size));
  if (filled) {
    std::vector<int> fid(matF.rows());
    std::iota(fid.begin(), fid.end(), 0);
    insert_triangles(matV, matF, fid);
  }
}

void prism::HashGrid::bound_convert(const Vec3d &aabb_min,
                                    Eigen::Array<long, 3, 1> &int_min) const {
  int_min = ((aabb_min - m_domain_min) / m_cell_size).cast<long>().array();
  // out of bound might happen, particularly when trying to extend the shell large.
  // assert((int_min.minCoeff() >= -1)); 
  // assert((int_min.maxCoeff() <= m_grid_size));
  int_min.max(0).min(m_grid_size - 1);
}

void prism::HashGrid::add_element(const Vec3d &aabb_min, const Vec3d &aabb_max,
                                  const int index) {
  if (index >= face_stores.size()) face_stores.resize(index + 1);
  assert(face_stores[index].size() == 0);
  Eigen::Array<long, 3, 1> int_min, int_max;
  bound_convert(aabb_min, int_min);
  bound_convert(aabb_max, int_max);
  assert(int_min[0] <= int_max[0]);

  auto &hg = m_face_items;
  for (auto x = int_min.x(); x <= int_max.x(); ++x)
    for (auto y = int_min.y(); y <= int_max.y(); ++y)
      for (auto z = int_min.z(); z <= int_max.z(); ++z) {
        auto key = std::tuple(x, y, z);
        auto it = hg.lower_bound(key);
        if (it != hg.end() && it->first == key) {
          it->second->push_back(index);
        } else {
          it = hg.emplace_hint(
              it, key, std::make_shared<std::list<int>>(std::list<int>{index}));
        }
        auto ptr = std::prev(it->second->end());
        face_stores[index].emplace_back(it->second, ptr);
      }
}

void prism::HashGrid::remove_element(const int index) {
  for (auto [ptr_list, iter_list] : face_stores[index]) {
    ptr_list->erase(iter_list);
  }
  face_stores[index].clear();
}

void prism::HashGrid::query(const Vec3d &aabb_min, const Vec3d &aabb_max,
                            std::set<int> &result) const {
  Eigen::Array<long, 3, 1> int_min, int_max;
  bound_convert(aabb_min, int_min);
  bound_convert(aabb_max, int_max);
  assert(int_min[0] <= int_max[0]);

  auto &hg = m_face_items;
  std::vector<int> vec_result;
  for (auto x = int_min.x(); x <= int_max.x(); ++x)
    for (auto y = int_min.y(); y <= int_max.y(); ++y)
      for (auto z = int_min.z(); z <= int_max.z(); ++z) {
        auto it = hg.find(std::forward_as_tuple(x, y, z));
        if (it != hg.end()) {
          vec_result.insert(vec_result.end(), it->second->begin(),
                            it->second->end());
        }
      }
  result = std::set<int>(vec_result.begin(), vec_result.end());
}

void prism::HashGrid::insert_triangles(const std::vector<Vec3d> &V,
                                       const std::vector<Vec3i> &F,
                                       const std::vector<int> &fid) {
  face_stores.resize(F.size());
  for (auto i : fid) {
    Eigen::Matrix3d local;
    for (auto k : {0, 1, 2}) local.row(k) = V[F[i][k]];
    auto aabb_min = local.colwise().minCoeff();
    auto aabb_max = local.colwise().maxCoeff();
    add_element(aabb_min, aabb_max, i);
  }

}

void prism::HashGrid::insert_triangles(const RowMatd &V, const RowMati &F,
                                       const std::vector<int> &fid) {
  face_stores.resize(F.rows());
  for (auto i : fid) {
    Eigen::Matrix3d local;
    for (auto k : {0, 1, 2}) local.row(k) = V.row(F(i, k));
    auto aabb_min = local.colwise().minCoeff();
    auto aabb_max = local.colwise().maxCoeff();
    add_element(aabb_min, aabb_max, i);
  }

}

std::vector<std::pair<int, int>> prism::HashGrid::self_candidates() const {
  std::vector<std::pair<int, int>> candidates;
  for (auto [_, l_ptr] : m_face_items) {
    auto n = l_ptr->size();
    for (auto it = l_ptr->begin(); it != l_ptr->end(); it++) {
      for (auto jt = std::next(it); jt != l_ptr->end(); jt++) {
        assert(*it < *jt && "just observation so there is no order flipping");
        candidates.emplace_back(*it, *jt);
      }
    }
  }
  std::sort(candidates.begin(), candidates.end());
  candidates.erase(std::unique(candidates.begin(), candidates.end()),
                   candidates.end());
  return candidates;
}

void prism::HashGrid::update_after_collapse() {
  std::vector<int> old2new(face_stores.size(), -1);
  std::vector<std::vector<HashPtr>> new_face_stores;
  new_face_stores.reserve(face_stores.size());
  for (int i = 0, cnt = 0; i < face_stores.size(); i++) {
    if (!face_stores[i].empty()) {
      new_face_stores.push_back(face_stores[i]);
      old2new[i] = cnt++;
    }
  }
  face_stores = std::move(new_face_stores);

  std::for_each(m_face_items.begin(), m_face_items.end(),
                [&old2new](auto &item) {
                  auto ptr_list = item.second;
                  for (auto it = ptr_list->begin(); it != ptr_list->end();) {
                    *it = old2new[*it];
                    if (*it == -1) {
                      ptr_list->erase(it++); // this does not happen since it is already taken care of. However, leave here for future extension.
                      spdlog::warn("Should not happen");
                    }
                    else
                      it++;
                  }
                });
}