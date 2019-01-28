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

# ------------------------------------------------------------------------------
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(BUILD_SHARED_LIBS ON)
set(MUSPECTRE_PYTHON_MAJOR_VERSION 3)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

# ------------------------------------------------------------------------------
option(MUSPECTRE_MAKE_DOC_TARGET "If on, a target dev_doc (which builds the documentation) is added" OFF)
option(MUSPECTRE_MAKE_TESTS "If on, several ctest targets will be built automatically" ON)
option(MUSPECTRE_MAKE_EXAMPLES "If on, the executables in the bin folder will be compiled" ON)
option(MUSPECTRE_MAKE_BENCHMARKS "If on, the benchmarks will be compiled" ON)
option(MUSPECTRE_MPI_PARALLEL "If on, MPI-parallel solvers become available" OFF)
option(MUSPECTRE_MUGRID_ONLY "Only setup muGrid" OFF)
option(MUSPECTRE_MUFFT_ONLY "Compile only muFFT" OFF)
set(MUSPECTRE_RUNNING_IN_CI "OFF" CACHE INTERNAL "changes output format for tests")

if(MUSPECTRE_MUFFT_ONLY OR MUSPECTRE_MUGRID_ONLY)
  set(MUSPECTRE_MAKE_TESTS OFF CACHE BOOL "" FORCE)
  set(MUSPECTRE_MAKE_EXAMPLES OFF CACHE BOOL "" FORCE)
endif()

# ------------------------------------------------------------------------------
include(muspectreTools)
include(cpplint)
mark_as_advanced_prefix(CPPLINT)

muSpectre_set_global_compile_options()

# ------------------------------------------------------------------------------
add_external_package(Eigen3 VERSION 3.3.0 CONFIG)
add_external_package(pybind11 VERSION 2.2 CONFIG)
find_package(PythonLibsNew ${MUSPECTRE_PYTHON_MAJOR_VERSION} MODULE REQUIRED)

add_subdirectory(external)

# ------------------------------------------------------------------------------
if(APPLE)
  include_directories(
    ${CMAKE_INSTALL_PREFIX}/include
    ${Boost_INCLUDE_DIRS})
endif()

if (${MUSPECTRE_MAKE_TESTS})
  enable_testing()
  add_subdirectory(tests)
endif()

################################################################################
# compile the library
add_compile_options(-Werror)
add_subdirectory(${CMAKE_SOURCE_DIR}/src)
if(NOT MUSPECTRE_MUFFT_ONLY AND NOT MUSPECTRE_MUGRID_ONLY)
  add_subdirectory(${CMAKE_SOURCE_DIR}/language_bindings)
endif()

if (${MUSPECTRE_MAKE_DOC_TARGET})
  add_subdirectory(${CMAKE_SOURCE_DIR}/doc)
endif()

################################################################################
if(${MUSPECTRE_MAKE_EXAMPLES})
  add_subdirectory(${CMAKE_SOURCE_DIR}/bin)
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

if(MUSPECTRE_MUGRID_ONLY AND NOT MUSPECTRE_MUFFT_ONLY)
  cpplint_add_subdirectory("${CMAKE_SOURCE_DIR}/src/libmugrid" "")
elseif(MUSPECTRE_MUFFT_ONLY)
  cpplint_add_subdirectory("${CMAKE_SOURCE_DIR}/src/libmugrid" "")
  cpplint_add_subdirectory("${CMAKE_SOURCE_DIR}/src/libmufft" "")
else()
  cpplint_add_subdirectory("${CMAKE_SOURCE_DIR}/src" "")
  cpplint_add_subdirectory("${CMAKE_SOURCE_DIR}/tests" "")
  cpplint_add_subdirectory("${CMAKE_SOURCE_DIR}/language_bindings" "")
  cpplint_add_subdirectory("${CMAKE_SOURCE_DIR}/bin" "--filter=-build/namespaces")
endif()
