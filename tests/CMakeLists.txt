cmake_minimum_required(VERSION 3.1)
project(prism_tests)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../cmake)


### Download unit test framework
if (NOT TARGET doctest)
  prism_download_doctest()
  add_library(doctest INTERFACE)
  target_include_directories(doctest SYSTEM INTERFACE ${PRISM_EXTERNAL}/doctest/doctest/)
endif()

set(CMAKE_MODULE_PATH "${PRISM_EXTERNAL}/sanitizers-cmake/cmake" ${CMAKE_MODULE_PATH})



find_package(MPFR)
IF(NOT ${MPFR_FOUND})
	MESSAGE(FATAL_ERROR "Cannot find MPFR")
ENDIF()

MESSAGE(STATUS "Found MPFR ${MPFR_INCLUDES}")

# Create test executable
add_executable(prism_tests test_main.cpp)
target_sources(prism_tests PRIVATE 
                cgal_AABB.cpp
                cgal_qp.cpp
                spatial_hash.cpp
                predicates.cpp
                phong.cpp
                numerical_self_intersection.cpp)
target_sources(prism_tests PRIVATE 
                bevel_init.cpp
                curved_tetra_mips.cpp
                surface_stitch.cpp
                zig_shell_collapse.cpp
                curve_fitting.cpp
                tangential_smooth_bin.cpp
                remesh_shell_bin.cpp)

target_sources(prism_tests PRIVATE
                tetra_opt.cpp
                sizing_background_mesh.cpp
                sphere_refine.cpp)
target_link_libraries(prism_tests PUBLIC doctest cumin_library prism::prism batm_library json libTetShell)
 #spdlog::spdlog igl::core highfive geogram mitsuba_autodiff igl::cgal)

target_compile_features(prism_tests PUBLIC cxx_std_17)
target_include_directories(prism_tests PUBLIC ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/../src/)
target_include_directories(prism_tests PUBLIC ${PRISM_EXTERNAL}/rational/)
# target_include_directories(prism_tests PUBLIC ${MPFR_INCLUDES})
# target_include_directories(prism_tests PUBLIC ${PRISM_EXTERNAL}/libigl/external/eigen/unsupported/test/mpreal)
# target_link_libraries(prism_tests PUBLIC ${MPFR_LIBRARIES})

target_compile_options(prism_tests PUBLIC "-Wno-deprecated")

if (PRISM_ENABLE_ASAN)
  message(STATUS "[prism] enabled asan ubsan for test")
  target_compile_options(prism_tests PUBLIC "-fsanitize=undefined")
  target_compile_options(prism_tests PUBLIC "-fsanitize=address")
  target_link_options(prism_tests PUBLIC "-fsanitize=address")
endif()
