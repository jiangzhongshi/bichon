#pragma once

#include <prism/common.hpp>
#include "tetra_utils.hpp"

#include <memory>
#include <queue>

namespace prism::tet {
void serializer(std::string filename, const PrismCage* pc, const prism::tet::tetmesh_t& tetmesh);

int edge_split_pass_with_sizer(
    PrismCage* pc,
    prism::local::RemeshOptions& option,
    prism::tet::tetmesh_t& tetmesh,
     const prism::tet::SizeController* sizer,
    double scale);

int collapse_pass(
    PrismCage* pc,
    prism::local::RemeshOptions& option,
    prism::tet::tetmesh_t& tetmesh,
     const prism::tet::SizeController* sizer,
    bool strict = false);

int edgeswap_pass(
    PrismCage* pc,
    prism::local::RemeshOptions& option,
    prism::tet::tetmesh_t& tetmesh,
     const prism::tet::SizeController* sizer);

int faceswap_pass(
    PrismCage* pc,
    prism::local::RemeshOptions& option,
    prism::tet::tetmesh_t& tetmesh,
     const prism::tet::SizeController* sizer);

std::priority_queue<std::tuple<double, int, int, int>> construct_face_queue(
    const prism::tet::vert_info_t& vert_info,
    const prism::tet::tet_info_t& tet_info);

std::priority_queue<std::tuple<double, int, int>> construct_edge_queue(
    const prism::tet::vert_info_t& vert_info,
    const prism::tet::tet_info_t& tet_info);

std::priority_queue<std::tuple<double, int, int>> construct_collapse_queue(
    const prism::tet::vert_info_t& vert_info,
    const prism::tet::tet_info_t& tet_info);

int vertexsmooth_pass(
    PrismCage* pc,
    prism::local::RemeshOptions& option,
    prism::tet::tetmesh_t& tetmesh,
    const prism::tet::SizeController* sizer,
    double thick);

int edge_split_pass_for_dof(
    PrismCage* pc,
    prism::local::RemeshOptions& option,
    prism::tet::tetmesh_t& tetmesh);

std::vector<double> size_progress(
    const prism::tet::SizeController* sizer,
    const prism::tet::tetmesh_t& tetmesh);
} // namespace prism::tet