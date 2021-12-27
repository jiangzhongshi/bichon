#pragma once

#include <prism/common.hpp>
#include "tetra_utils.hpp"

#include <memory>
#include <queue>

namespace prism::tet {
void serializer(
    std::string filename,
    const PrismCage* pc,
    const prism::tet::vert_info_t& vert_info,
    const prism::tet::tet_info_t& tet_info);
void edge_split_pass_with_sizer(
    PrismCage* pc,
    prism::local::RemeshOptions& option,
    prism::tet::vert_info_t& vert_info,
    prism::tet::tet_info_t& tet_info,
    std::vector<std::vector<int>>& vert_tet_conn,
    const std::unique_ptr<prism::tet::SizeController>& sizer);

int collapse_pass(
    PrismCage* pc,
    prism::local::RemeshOptions& option,
    prism::tet::vert_info_t& vert_info,
    prism::tet::tet_info_t& tet_info,
    std::vector<std::vector<int>>& vert_tet_conn,
    const std::unique_ptr<prism::tet::SizeController>& sizer);

int edgeswap_pass(
    PrismCage* pc,
    prism::local::RemeshOptions& option,
    prism::tet::vert_info_t& vert_info,
    prism::tet::tet_info_t& tet_info,
    std::vector<std::vector<int>>& vert_tet_conn,
    double sizing);

int faceswap_pass(
    PrismCage* pc,
    prism::local::RemeshOptions& option,
    prism::tet::vert_info_t& vert_info,
    prism::tet::tet_info_t& tet_info,
    std::vector<std::vector<int>>& vert_tet_conn,
    double sizing);

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
    prism::tet::vert_info_t& vert_info,
    prism::tet::tet_info_t& tet_info,
    std::vector<std::vector<int>>& vert_tet_conn,
    double thick);


int edge_split_pass_for_dof(
    PrismCage* pc,
    prism::local::RemeshOptions& option,
    prism::tet::vert_info_t& vert_attrs,
    prism::tet::tet_info_t& tet_attrs,
    std::vector<std::vector<int>>& vert_conn);
} // namespace prism::tet