#include "AdaMesh.hpp"
#include "Rational.h"
#include "wmtk/TetMesh.h"
#include "wmtk/utils/AMIPS.h"

#include "wmtk/TetMeshOperations.hpp"

bool face_attribute_tracker(
    const wmtk::TetMesh& m,
    const std::vector<wmtk::TetMesh::Tuple>& incident_tets,
    const wmtk::prism::AdaMesh::FaceAttCol& m_face_attribute,
    std::map<std::array<size_t, 3>, wmtk::prism::AdaMesh::FaceAttributes>& changed_faces)
{
    changed_faces.clear();
    auto middle_face = std::set<int>();
    for (auto t : incident_tets) {
        for (auto j = 0; j < 4; j++) {
            auto f_t = m.tuple_from_face(t.tid(m), j);
            auto global_fid = f_t.fid(m);
            auto vs = m.get_face_vertices(f_t);
            auto vids = std::array<size_t, 3>{{vs[0].vid(m), vs[1].vid(m), vs[2].vid(m)}};
            std::sort(vids.begin(), vids.end());
            auto [it, suc] = changed_faces.emplace(vids, m_face_attribute[global_fid]);
            if (!suc) {
                changed_faces.erase(it); // erase if already there.
                middle_face.insert(global_fid);
            }
        }
    }

    // for (auto f : middle_face) {
    //     if (m_face_attribute[f].m_is_surface_fs || m_face_attribute[f].m_is_bbox_fs >= 0) {
    //         wmtk::logger().debug("Attempting to Swap a boundary/bbox face, reject.");
    //         return false;
    //     }
    // }
    return true;
}


void tracker_assign_after(
    const wmtk::TetMesh& m,
    const std::vector<wmtk::TetMesh::Tuple>& incident_tets,
    const std::map<std::array<size_t, 3>, wmtk::prism::AdaMesh::FaceAttributes>& changed_faces,
    wmtk::prism::AdaMesh::FaceAttCol& m_face_attribute)
{
    auto middle_face = std::vector<size_t>();
    auto new_faces = std::set<std::array<size_t, 3>>();

    for (auto t : incident_tets) {
        for (auto j = 0; j < 4; j++) {
            auto f_t = m.tuple_from_face(t.tid(m), j);
            auto global_fid = f_t.fid(m);
            auto vs = m.get_face_vertices(f_t);
            auto vids = std::array<size_t, 3>{{vs[0].vid(m), vs[1].vid(m), vs[2].vid(m)}};
            std::sort(vids.begin(), vids.end());
            auto it = (changed_faces.find(vids));
            if (it == changed_faces.end()) {
                middle_face.push_back(global_fid);
                continue;
            }

            m_face_attribute[global_fid] = it->second; // m_face_attribute[it->second];
        }
    }
    for (auto f : middle_face) {
        m_face_attribute[f].reset();
    }
}

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
    double energy = -1.;
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

    if (energy == -1.) energy = wmtk::AMIPS_energy_stable_p3<apps::Rational>(T);
    if (std::isinf(energy) || std::isnan(energy) || energy < 27 - 1e-3) return 1e50;
    return energy;
}

struct Split : public wmtk::SplitEdge
{
    std::map<std::array<size_t, 3>, AdaMesh::FaceAttributes> cache_changed_faces;
    AdaMesh::FaceAttCol& m_face_attribute;

    Split(const TetMesh& m_, AdaMesh::FaceAttCol& attr_)
        : wmtk::SplitEdge(m_)
        , m_face_attribute(attr_){};

    bool before(const wmtk::TetMesh::Tuple& tup)
    {
        auto& m = static_cast<const AdaMesh&>(this->m);
        auto incident_tets = m.get_incident_tets_for_edge(tup);
        cache_changed_faces.clear();
        for (auto t : incident_tets) {
            for (auto j = 0; j < 4; j++) {
                auto f_t = m.tuple_from_face(t.tid(m), j);
                auto global_fid = f_t.fid(m);
                auto vs = m.get_face_vertices(f_t);
                auto vids = std::array<size_t, 3>{{vs[0].vid(m), vs[1].vid(m), vs[2].vid(m)}};
                std::sort(vids.begin(), vids.end());
                auto [it, suc] = cache_changed_faces.emplace(vids, m.m_face_attribute[global_fid]);
            }
        }

        return true;
    };

    bool after(const std::vector<wmtk::TetMesh::Tuple>& new_tets)
    {
        auto& m = static_cast<const AdaMesh&>(this->m);
        auto v1 = edge_verts[0];
        auto v2 = edge_verts[1];
        // if same face, then inherit,
        for (auto t : new_tets) {
            for (auto j = 0; j < 4; j++) {
                auto f_t = m.tuple_from_face(t.tid(m), j);
                auto global_fid = f_t.fid(m);
                auto vs = m.get_face_vertices(f_t);
                auto vids = std::array<size_t, 3>{{vs[0].vid(m), vs[1].vid(m), vs[2].vid(m)}};

                // for the middle faces special: if (v1, ux, -) or (v2, ux, -) == vids, query for
                // each middle face becomes two child.
                // (v1,v2,-)
                constexpr auto id_in_arr = [](auto& arr, auto& x) -> int {
                    for (auto i = 0; i < arr.size(); i++)
                        if (arr[i] == x) return i;
                    return -1;
                };
                auto ux_id = id_in_arr(vids, ux);
                if (ux_id != -1) {
                    for (auto b = 0; b < 2; b++) {
                        if (id_in_arr(vids, edge_verts[b]) != -1) {
                            vids[ux_id] = edge_verts[(b + 1) % 2];
                        }
                    }
                }

                std::sort(vids.begin(), vids.end());
                auto it = (cache_changed_faces.find(vids));
                if (it != cache_changed_faces.end()) {
                    m_face_attribute[global_fid] = it->second;
                }
            }
        }
        // TODO, some need clear?
        return true;
    }
};

} // namespace wmtk::prism
