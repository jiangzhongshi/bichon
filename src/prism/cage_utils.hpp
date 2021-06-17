#ifndef PRIISM_CAGE_UTILS_HPP
#define PRIISM_CAGE_UTILS_HPP
#include <prism/common.hpp>
// #include <prism/geogram/AABB.hpp>
namespace prism::geogram{struct AABB;};
namespace prism {struct HashGrid;};
namespace prism::cage_utils {

// terminology from On the ‘most normal’ normal—Part 2, Aubry et al.
bool most_visible_normals(const RowMatd& V, const RowMati& F, RowMatd& VN,
                          std::set<int>& omni_saddle, double tolerance_singularity = 1e-2);

// hint with ray cast. Volume shrinking step.
std::vector<double> extrude_along_normals(const RowMatd& V, const RowMati& F,
                                          const prism::geogram::AABB& tree,
                                          const RowMatd& N, bool outward,
                                          int num_cons, double initial_step);
// no hint version
std::vector<double> volume_extrude_steps(const RowMatd& V, const RowMati& F,
                                          const RowMatd& N, bool outward,
                                          int num_cons,
                                          const std::vector<double>& ray_step);

// iteratively retract with respect to AABB collision.
void iterative_retract_normal(const RowMatd& V, const RowMati& F,
                              const prism::geogram::AABB& tree,
                              const RowMatd& N_in, bool outward, int num_cons,
                              std::vector<double>& alpha);

// call above twice and retract for intersections.
void extrude_for_base_and_top(const RowMatd& V, const RowMati& F,
                              const prism::geogram::AABB& tree,
                              const RowMatd& N, int num_cons, RowMatd& inner,
                              RowMatd& outer, double initial_step);

// modifies top in place.
void hashgrid_shrink(const std::vector<Vec3d> &mid, std::vector<Vec3d> &top,
                     const std::vector<Vec3i> &vecF, const std::vector<std::vector<int>>& VF);

bool safe_shrink(const std::vector<Vec3d> &mid, std::vector<Vec3d> &top,
                     const std::vector<Vec3i> &vecF, const std::vector<std::vector<int>>& VF);
                     
void reorder_singularity_to_front(RowMatd& V, RowMati& F, RowMatd& VN,
                                  const std::set<int>& omni_singu,
                                  Eigen::VectorXi& idx_map);

// mark out the "singularity-like" vertices on the border
void mark_singular_on_border(const RowMatd& V, const RowMati& F, RowMatd& VN,
                             std::set<int>& omni_sing);

// prism_id = tet_id / 3, intra: base-top 0,1,2
// V = stack([base, top])
void tetmesh_from_prismcage(const std::vector<Vec3d>& base,
                            const std::vector<Vec3d>& top,
                            const std::vector<Vec3i>& F, std::vector<Vec3d>& V,
                            std::vector<Vec4i>& T);

// prism_id = tet_id / 6, intra: base-mid 0,1,2 then mid-top 3,4,5
// V = stack([base, mid, top])
// auto prism_id = tet_id / 6;
// auto pillar_id = tet_id % 3;
// bool bottom = (tet_id%6) < 3;
void tetmesh_from_prismcage(const std::vector<Vec3d>& base,
                            const std::vector<Vec3d>& mid,
                            const std::vector<Vec3d>& top,
                            const std::vector<Vec3i>& F, int num_singularity,
                            std::vector<Vec3d>& V, std::vector<Vec4i>& T);

// verification function
bool all_volumes_are_positive(const std::vector<Vec3d>& base,
                              const std::vector<Vec3d>& mid,
                              const std::vector<Vec3d>& top,
                              const std::vector<Vec3i>& F, int num_cons);

std::map<std::pair<int,int>, int> split_singular_edges(RowMatd& V, RowMati& F, RowMatd& VN,
                                  const std::set<int>& omni_singu);

void recover_positive_volumes(std::vector<Vec3d>& mid, std::vector<Vec3d>& top,
                              const std::vector<Vec3i>& F, const RowMatd& VN,
                              const std::vector<std::vector<int>>& VF,
                              int num_cons,
                              bool outward);
}  // namespace prism::cage_utils

#endif