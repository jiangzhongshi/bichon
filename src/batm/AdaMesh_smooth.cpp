#include "AdaMesh.hpp"
#include "Rational.h"
#include "wmtk/utils/AMIPS.h"

namespace wmtk::prism {
bool AdaMesh::round(size_t i)
{
    auto v = tuple_from_vertex(i);
    if (!v.is_valid(*this)) return true;

    auto& m_vertex_attribute = vertex_attrs;
    if (m_vertex_attribute[i].rounded) return true;

    auto old_pos = m_vertex_attribute[i].pos_r;
    m_vertex_attribute[i].pos_r << m_vertex_attribute[i].pos[0], m_vertex_attribute[i].pos[1],
        m_vertex_attribute[i].pos[2];

    auto conn_tets = get_one_ring_tets_for_vertex(v);
    m_vertex_attribute[i].rounded = true;
    for (auto& tet : conn_tets) {
        if (is_invert(tet)) {
            m_vertex_attribute[i].rounded = false;
            m_vertex_attribute[i].pos_r = old_pos;
            return false;
        }
    }

    return true;
}

bool AdaMesh::is_invert(const Tuple& t)
{
    auto& m = *this;
    if (!t.is_valid(m)) return false;
    // TODO:
    return true;
}

double AdaMesh::quality(const Tuple& t)
{
    auto vs = oriented_tet_vids(t);
    auto rational_energy_compute = [&vertex_attrs = vertex_attrs](auto& vs) {
        std::array<apps::Rational, 12> T;
        for (auto j = 0; j < 4; j++) {
            auto& va = vertex_attrs[vs[j]];
            for (auto k = 0; k < 3; k++) {
                T[j * 3 + k] = va.pos_r[k];
            }
        }
        return wmtk::AMIPS_energy_rational_p3<apps::Rational, apps::Rational>(T);
    };

    std::array<double, 12> T;
    double energy = -1;
    for (int j = 0; j < 4; j++) {
        auto& va = vertex_attrs[vs[j]];
        if (!va.rounded) {
            energy = rational_energy_compute(vs);
			break;
        }
        for (auto k = 0; k < 3; k++) {
            T[j * 3 + k] = va.pos[k];
        }
    }

    if (energy == -1) energy = wmtk::AMIPS_energy_stable_p3<apps::Rational>(T);
    if (std::isinf(energy) || std::isnan(energy) || energy < 27 - 1e-3) return 1e50;
    return energy;
}
} // namespace wmtk::prism
