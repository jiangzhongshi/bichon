#ifndef PRISM_PREDICATE_TETRAHEDRON_TETRAHEDRON_INTERSECTION_HPP
#define PRISM_PREDICATE_TETRAHEDRON_TETRAHEDRON_INTERSECTION_HPP

#include <array>
#include "../common.hpp"

namespace prism::predicates {
bool tetrahedron_tetrahedron_overlap(const std::array<Vec3d, 4>&,
                                    const std::array<Vec3d, 4>&);
}  // namespace prism

#endif