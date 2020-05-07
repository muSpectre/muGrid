# =============================================================================
# file   CMakeLists.txt
#
# @author Till Junge <till.junge@epfl.ch>
#
# @date   08 Jan 2018
#
# @brief  Main configuration file
#
# @section LICENSE
#
# Copyright © 2018 Till Junge
#
# µSpectre is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3, or (at
# your option) any later version.
#
# µSpectre is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with µSpectre; see the file COPYING. If not, write to the
# Free Software Foundation, Inc., 59 Temple Place - Suite 330,
# Boston, MA 02111-1307, USA.
#
# Additional permission under GNU GPL version 3 section 7
#
# If you modify this Program, or any covered work, by linking or combining it
# with proprietary FFT implementations or numerical libraries, containing parts
# covered by the terms of those libraries' licenses, the licensors of this
# Program grant you additional permission to convey the resulting work.
# =============================================================================

cmake_minimum_required(VERSION 3.5.0)

project(muSpectre)

# Enable overriding variables/options in directories above (necessary from CMake 3.15)
if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

# ------------------------------------------------------------------------------
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT DEFINED BUILD_SHARED_LIBS)
  set(BUILD_SHARED_LIBS ON)
endif()
set(MUSPECTRE_PYTHON_MAJOR_VERSION 3)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# ------------------------------------------------------------------------------
option(MUSPECTRE_MAKE_DOC_TARGET "If on, a target dev_doc (which builds the documentation) is added" OFF)
option(MUSPECTRE_MAKE_TESTS "If on, several ctest targets will be built automatically" ON)
option(MUSPECTRE_MAKE_EXAMPLES "If on, the executables in the bin folder will be compiled" ON)
option(MUSPECTRE_MAKE_BENCHMARKS "If on, the benchmarks will be compiled" ON)
option(MUSPECTRE_MPI_PARALLEL "If on, MPI-parallel solvers become available" OFF)
option(SPLIT_CELL "If on, split cell, cork and gmp are accessible" OFF)
set(MUCHOICE "muSpectre" CACHE STRING "Which µLevel is required")
set_property(CACHE MUCHOICE PROPERTY STRINGS
  "muGrid"
  "muFFT"
  "muSpectre"
  )

set(MUSPECTRE_RUNNING_IN_CI "OFF" CACHE INTERNAL "changes output format for tests")

set(MUSPECTRE_NAMESPACE "${MUCHOICE}::")
set(MUSPECTRE_TARGETS_EXPORT ${MUCHOICE}Targets)
set(MUGRID_NAMESPACE ${MUSPECTRE_NAMESPACE})
set(MUGRID_TARGETS_EXPORT ${MUSPECTRE_TARGETS_EXPORT})
set(MUFFT_NAMESPACE ${MUGRID_NAMESPACE})
set(MUFFT_TARGETS_EXPORT ${MUGRID_TARGETS_EXPORT})
set(EIGEN_EXPORT_NAME ${MUSPECTRE_TARGETS_EXPORT})
set(PYBIND11_EXPORT_NAME ${MUSPECTRE_TARGETS_EXPORT})

if(MUSPECTRE_MUFFT_ONLY OR MUSPECTRE_MUGRID_ONLY)
  set(MUSPECTRE_MAKE_TESTS OFF CACHE BOOL "" FORCE)
  set(MUSPECTRE_MAKE_EXAMPLES OFF CACHE BOOL "" FORCE)
endif()

# Set options suitable for using muSpectre as a subproject
if(NOT CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set(MUSPECTRE_MAKE_DOC_TARGET OFF)
  set(MUSPECTRE_MAKE_TESTS OFF)
  set(MUSPECTRE_MAKE_EXAMPLES OFF)
  set(MUSPECTRE_MAKE_BENCHMARKS OFF)
endif()

# ------------------------------------------------------------------------------
include(muspectreTools)
include(muTools)
include(cpplint)
mark_as_advanced_prefix(CPPLINT)



# ------------------------------------------------------------------------------
add_external_package(Eigen3 VERSION 3.3.0 CONFIG)
add_external_package(pybind11 VERSION 2.2.4 CONFIG)
find_package(PythonLibsNew ${MUSPECTRE_PYTHON_MAJOR_VERSION} MODULE REQUIRED)


if(${SPLIT_CELL})
  add_external_package(gmp)
  add_external_package(corkpp)
endif()

add_subdirectory(external)

muSpectre_set_global_compile_options()
# ------------------------------------------------------------------------------
if(APPLE)
  include_directories(
    ${CMAKE_INSTALL_PREFIX}/include
    ${Boost_INCLUDE_DIR})
endif()


if (${MUSPECTRE_MAKE_TESTS})
  enable_testing()
  add_subdirectory(tests)
endif()



################################################################################
# compile the library
add_compile_options(-Werror)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)
if(MUCHOICE MATCHES "muSpectre")
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/language_bindings)
endif()

if(MUCHOICE MATCHES "muFFT")
  add_subdirectory(${CMAKE_SOURCE_DIR}/language_bindings/libmufft)
  add_subdirectory(${CMAKE_SOURCE_DIR}/language_bindings/libmugrid)
endif()
if(MUCHOICE MATCHES "muGrid")
  add_subdirectory(${CMAKE_SOURCE_DIR}/language_bindings/libmugrid)
endif()

if (${MUSPECTRE_MAKE_DOC_TARGET})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/doc)
endif()

################################################################################
if(${MUSPECTRE_MAKE_EXAMPLES} AND ${MUCHOICE} MATCHES "muSpectre")
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples)
endif()

################################################################################
# compile benchmarks
if(${MUSPECTRE_MAKE_BENCHMARKS})
  file(GLOB benchmarks "${CMAKE_SOURCE_DIR}/benchmarks/benchmark*cc")
  foreach(benchmark ${benchmarks})
    get_filename_component(benchmark_name ${benchmark} NAME_WE)
    add_executable(${benchmark_name} ${benchmark})
    target_link_libraries(${benchmark_name} ${BOOST_LIBRARIES} muSpectre)
  endforeach(benchmark ${benchmark})
endif()
