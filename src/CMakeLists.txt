add_library(prism_library)
target_sources(prism_library
PRIVATE
    prism/PrismCage.cpp
    prism/extraction.cpp
    prism/feature_utils.cpp
    prism/cage_utils.cpp
    prism/polyshell_utils.cpp
    prism/bevel_utils.cpp
    prism/geogram/geogram_utils.cpp
    prism/cgal/polyhedron_self_intersect.cpp
    prism/cgal/triangle_triangle_intersection.cpp
    # prism/cgal/tetrahedron_tetrahedron_intersection.cpp
    prism/cgal/QP.cpp
    prism/geogram/AABB.cpp
    prism/geogram/AABB_tet.cpp
    prism/phong/projection.cpp
    prism/phong/trilinear_projection.cpp
    prism/phong/tangent_orientation.cpp
    prism/phong/query_correspondence.cpp
    prism/predicates/inside_prism_tetra.cpp
    prism/predicates/inside_octahedron.cpp
    prism/predicates/positive_prism_volume_12.cpp
    prism/predicates/triangle_triangle_intersection.cpp
    prism/predicates/tetrahedron_overlap.cpp
    prism/energy/prism_quality.cpp
    prism/energy/map_distortion.cpp
    prism/energy/smoother_pillar.cpp
    prism/local_operations/remesh_pass.cpp
    prism/local_operations/smooth_pass.cpp
    prism/local_operations/remesh_pass_collapse.cpp
    prism/local_operations/remesh_pass_polyshell.cpp
    prism/local_operations/remesh_pass_feature.cpp
    prism/local_operations/section_remesh.cpp
    prism/local_operations/mesh_coloring.cpp
    prism/local_operations/local_mesh_edit.cpp
    prism/local_operations/validity_checks.cpp
    prism/local_operations/retain_triangle_adjacency.cpp
    prism/spatial-hash/AABB_hash.cpp
    prism/spatial-hash/self_intersection.cpp
    prism/osqp/osqp_normal.cpp
    prism/cage_check.cpp
    prism/intersections.cpp
  )

target_compile_definitions(prism_library PUBLIC CGAL_QP)
target_compile_features(prism_library PUBLIC cxx_std_17)
target_link_libraries(prism_library PUBLIC spdlog::spdlog igl::core osqpstatic highfive geogram mitsuba_autodiff igl::cgal igl::predicates)
target_include_directories(prism_library PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/)
set_target_properties(prism_library PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(prism::prism ALIAS prism_library)


if (NOT PRISM_LIB_ONLY)
add_library(cumin_library
  cumin/tetshell_utils.cpp
  cumin/curve_utils.cpp
  cumin/curve_pass.cpp
  cumin/inversion_check.cpp
  cumin/bernstein_eval.cpp
  cumin/high_order_optimization.cpp
  cumin/stitch_surface_to_volume.cpp
  cumin/curve_validity.cpp)
target_compile_features(cumin_library PUBLIC cxx_std_17)
target_link_libraries(cumin_library PUBLIC prism_library libTetShell)
target_include_directories(cumin_library PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/)
target_compile_definitions(cumin_library PUBLIC CUMIN_MAGIC_DATA_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../python/curve/data/")

add_executable(cumin_bin)

target_sources(cumin_bin PRIVATE pipeline_schedules.cpp curve_in_shell.cpp)
target_link_libraries(cumin_bin prism_library cumin_library CLI11::CLI11 json libTetShell)

if (PRISM_ENABLE_ASAN)
  target_compile_options(cumin_bin PUBLIC "-fsanitize=address")
  target_link_options(cumin_bin PUBLIC "-fsanitize=address")
endif()

add_library(batm_library
          batm/tetra_utils.cpp
          batm/tetra_logger.cpp
          batm/tetra_quality.cpp
          batm/tetra_size_control.cpp
          batm/tetra_remesh_pass.cpp
          batm/AMIPS.cpp
          )
target_compile_features(batm_library PUBLIC cxx_std_17)
target_link_libraries(batm_library PUBLIC prism_library)
target_include_directories(batm_library PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/)


option(PYBICHON "Enable Python Binding" OFF)
if (PYBICHON)
  find_package(PythonInterp QUIET)
  prism_download_pybind11()
  add_subdirectory(${PRISM_EXTERNAL}/pybind11/ pybind11)
  add_subdirectory(python)
endif()
endif()