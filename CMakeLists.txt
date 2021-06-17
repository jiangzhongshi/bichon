cmake_minimum_required(VERSION 3.9)
project(prism)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_ARGS "-v" "-w" "dupbuild=warn") 

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(PRISM_EXTERNAL ${CMAKE_CURRENT_SOURCE_DIR}/external/)

include(PrismDependencies)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
cmake_policy(SET CMP0069 NEW)
# set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory(src)

option(PRISM_TESTS "Unit Tests" ON)
if(PRISM_TESTS)
  include(CTest)
  add_subdirectory(tests)
endif()