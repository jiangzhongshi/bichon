cmake_minimum_required(VERSION 3.9)
project(prism)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(PRISM_EXTERNAL ${CMAKE_CURRENT_SOURCE_DIR}/external/)

include(PrismDependencies)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
cmake_policy(SET CMP0069 NEW)
# set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})


if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
add_compile_options (-fdiagnostics-color=always)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
add_compile_options (-fcolor-diagnostics)
endif ()

option(PRISM_LIB_ONLY "only prism lib" OFF)

add_subdirectory(src)

option(PRISM_TESTS "Unit Tests" ON)
option(PRISM_ENABLE_ASAN "enable ASAN for prism" OFF)
if(PRISM_TESTS)
  include(CTest)
  add_subdirectory(tests)
endif()