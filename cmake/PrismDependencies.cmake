
include(PrismDownloadExternal)

if (NOT TARGET igl::core)
  prism_download_libigl()
  set(LIBIGL_EIGEN_VERSION 3.3.7)
  set(CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE TRUE)
  option(LIBIGL_USE_STATIC_LIBRARY "" OFF)
  option(LIBIGL_WITH_CGAL              "Use CGAL"           ON)
  option(LIBIGL_WITH_EMBREE              "Use embree"           OFF)
  option(LIBIGL_WITH_PREDICATES        "Use exact predicates"         ON)
  set(LIBIGL_INCLUDE_DIR ${PRISM_EXTERNAL}/libigl/include)
  find_package(LIBIGL REQUIRED)
endif()

# fmt
if(NOT TARGET fmt::fmt)
	prism_download_fmt()
	add_subdirectory(${PRISM_EXTERNAL}/fmt)
endif()

if(NOT TARGET spdlog::spdlog)
  prism_download_spdlog()
	add_library(spdlog INTERFACE)
	add_library(spdlog::spdlog ALIAS spdlog)
	target_include_directories(spdlog INTERFACE ${PRISM_EXTERNAL}/spdlog/include)
	target_compile_definitions(spdlog INTERFACE -DSPDLOG_FMT_EXTERNAL)
	target_link_libraries(spdlog INTERFACE fmt::fmt)
endif()

if (NOT TARGET json)
  prism_download_json()
  add_library(json INTERFACE)
  target_include_directories(json SYSTEM INTERFACE ${PRISM_EXTERNAL}/json/include)
endif()

if(NOT TARGET CLI11::CLI11)
    prism_download_cli11()
    add_subdirectory(${PRISM_EXTERNAL}/cli11)
endif()

if(NOT TARGET highfive)
  prism_download_HighFive()
  option(HIGHFIVE_USE_EIGEN ON)

  find_package(HDF5 REQUIRED)
  add_library(highfive INTERFACE)
  target_include_directories(highfive SYSTEM INTERFACE ${PRISM_EXTERNAL}/HighFive/include/ ${HDF5_INCLUDE_DIRS})
  target_link_libraries(highfive INTERFACE ${HDF5_LIBRARIES})
  target_compile_definitions(highfive INTERFACE H5_BUILT_AS_DYNAMIC_LIB)
endif()

if(NOT TARGET geogram::geogram)
  prism_download_geogram()
  include(geogram)
endif()

function(prism_download_file url name)
  if (NOT EXISTS ${name})
    file(DOWNLOAD ${url} ${name})
  endif()
endfunction()


if (NOT TARGET cvc3_rational)
  prism_download_file(https://raw.githubusercontent.com/wildmeshing/wildmeshing-toolkit/main/app/tetwild/Rational.hpp
  ${PRISM_EXTERNAL}/rational/Rational.h)
  add_library(cvc3_rational INTERFACE)
  target_include_directories(cvc3_rational INTERFACE ${PRISM_EXTERNAL}/rational/)
  target_link_libraries(cvc3_rational INTERFACE ${GMP_LIBRARIES})
endif()

if (NOT TARGET mitsuba_autodiff)
  prism_download_file(https://raw.githubusercontent.com/polyfem/polyfem/master/src/utils/autodiff.h
  ${PRISM_EXTERNAL}/autodiff/autodiff_mitsuba.h)
  add_library(mitsuba_autodiff INTERFACE)
  target_include_directories(mitsuba_autodiff INTERFACE ${PRISM_EXTERNAL}/autodiff/)
endif()

if (NOT TARGET osqp)
  prism_download_project(osqp
  GIT_REPOSITORY https://github.com/oxfordcontrol/osqp.git
  GIT_TAG da403d4b41e86b7dc00237047ea4f00354d902ed
  # GIT_SHALLOW true
  # GIT_SUBMODULES qdldl
  )
  add_subdirectory(${PRISM_EXTERNAL}/osqp)
endif()

if (NOT TARGET libTetShell)
  prism_download_tetshell()
  option(TETSHELL_LIBONLY ON)
  add_subdirectory(${PRISM_EXTERNAL}/tetshell)
endif()

if (NOT TARGET wildmeshing_toolkit)
  prism_download_wmtk()
  add_subdirectory(${PRISM_EXTERNAL}/wildmeshing_toolkit)
endif()