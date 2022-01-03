#include "tetrahedron_overlap.hpp"

#include <geogram/numerics/predicates.h>

#include "inside_octahedron.hpp"
#include "inside_prism_tetra.hpp"
#include "triangle_triangle_intersection.hpp"

bool prism::predicates::tetrahedron_tetrahedron_overlap(
    const std::array<Vec3d, 4>& Atet,
    const std::array<Vec3d, 4>& Btet)
{
#ifndef NDEBUG
    if (GEO::PCK::orient_3d(Atet[0].data(), Atet[1].data(), Atet[2].data(), Atet[3].data()) == 0)
        return false;
    if (GEO::PCK::orient_3d(Btet[0].data(), Btet[1].data(), Btet[2].data(), Btet[3].data()) == 0)
        return false;
#endif
    for (int i = 0; i < 4; i++) {
        if (prism::predicates::point_in_tetrahedron(Atet[i], Btet[0], Btet[1], Btet[2], Btet[3]))
            return true;

        if (prism::predicates::point_in_tetrahedron(Btet[i], Atet[0], Atet[1], Atet[2], Atet[3]))
            return true;
    }

    constexpr std::array<std::array<int, 2>, 6> edges{
        {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}};
    constexpr std::array<std::array<int, 3>, 4> faces{{{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}}};

    for (auto e : edges) {
        for (auto f : faces) {
            if (prism::predicates::segment_triangle_overlap(
                    {Atet[e[0]], Atet[e[1]]},
                    {Btet[f[0]], Btet[f[1]], Btet[f[2]]}))
                return true;
            if (prism::predicates::segment_triangle_overlap(
                    {Btet[e[0]], Btet[e[1]]},
                    {Atet[f[0]], Atet[f[1]], Atet[f[2]]}))
                return true;
        }
    }

    return false;
}

bool prism::predicates::triangle_tetrahedron_overlap(
    const std::array<Vec3d, 3>& tri,
    const std::array<Vec3d, 4>& tet)
{
#ifndef NDEBUG
    if (GEO::PCK::orient_3d(tet[0].data(), tet[1].data(), tet[2].data(), tet[3].data()) == 0)
        return false;
#endif
    for (int i = 0; i < 3; i++) {
        if (prism::predicates::point_in_tetrahedron(tri[i], tet[0], tet[1], tet[2], tet[3]))
            return true;
    }

    constexpr std::array<std::array<int, 2>, 6> edges{
        {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}};
    constexpr std::array<std::array<int, 3>, 4> faces{{{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}}};

    for (auto e : edges) {
        if (prism::predicates::segment_triangle_overlap({tet[e[0]], tet[e[1]]}, tri)) return true;
    }
    for (auto f : faces) {
        for (auto k = 0; k < 3; k++)
            if (prism::predicates::segment_triangle_overlap(
                    {tri[k], tri[(k + 1) % 3]},
                    {tet[f[0]], tet[f[1]], tet[f[2]]}))
                return true;
    }

    return false;
}