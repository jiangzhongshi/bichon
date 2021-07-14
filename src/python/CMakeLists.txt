################################################################################
# Find Python
################################################################################

if(NOT ${PYTHONINTERP_FOUND})
	execute_process(
		COMMAND
		python -c "import sys; sys.stdout.write(sys.version)"
		OUTPUT_VARIABLE PYTHON_VERSION)

	if(NOT PYTHON_VERSION)
		message(FATAL_ERROR "Unable to run python")
	endif()

	string(REGEX MATCH "^3\.*" IS_PYTHON3 ${PYTHON_VERSION})

	if(NOT IS_PYTHON3)
		message(FATAL_ERROR "Unable to find python 3")
	else()
		set(PYTHON_EXECUTABLE "python")
	endif()
endif()

################################################################################
# Pybind11
################################################################################

if(UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
  if(NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -flto")
  endif()
endif()


pybind11_add_module(prism_python
  module.cpp
  prism.cpp
  spatial.cpp
  curve.cpp
)

add_library(prism::python ALIAS prism_python)

target_link_libraries(prism_python
	PUBLIC igl::core prism_library
)
target_link_libraries(prism_python
	PUBLIC cumin_library
)

# Generate position independent code
set_target_properties(prism_python PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Output location
string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)
if(${U_CMAKE_BUILD_TYPE} MATCHES RELEASE)
	  set_target_properties(prism_python PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python)
    set_target_properties(prism_python PROPERTIES OUTPUT_NAME "prism")
else()
    set_target_properties(prism_python PROPERTIES OUTPUT_NAME "prism")
	  set_target_properties(prism_python PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python/debug)
endif()
set_target_properties(prism_python PROPERTIES CXX_VISIBILITY_PRESET hidden)
