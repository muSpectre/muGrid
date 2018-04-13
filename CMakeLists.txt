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
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3, or (at
# your option) any later version.
#
# µSpectre is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GNU Emacs; see the file COPYING. If not, write to the
# Free Software Foundation, Inc., 59 Temple Place - Suite 330,
# Boston, MA 02111-1307, USA.
# =============================================================================

cmake_minimum_required(VERSION 3.5.0)

project(µSpectre)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(BUILD_SHARED_LIBS ON)
set(MUSPECTRE_PYTHON_MAJOR_VERSION 3)

add_compile_options(-Wall -Wextra -Weffc++)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)
set(MAKE_DOC_TARGET "OFF" CACHE BOOL "If on, a target dev_doc (which builds the documentation) is added")
set(MAKE_TESTS "ON" CACHE BOOL "If on, several ctest targets will be built automatically")
set(MAKE_EXAMPLES "ON" CACHE BOOL "If on, the executables in the bin folder will be compiled")
set(MAKE_BENCHMARKS "ON" CACHE BOOL "If on, the benchmarks will be compiled")
set(MPI_PARALLEL "OFF" CACHE BOOL "If on, MPI-parallel solvers become available")
set(RUNNING_IN_CI "OFF" CACHE INTERNAL "changes output format for tests")

if(${MAKE_TESTS})
  enable_testing()
  find_package(Boost COMPONENTS unit_test_framework REQUIRED)
endif(${MAKE_TESTS})

if(${MPI_PARALLEL})
  add_definitions(-DWITH_MPI)
  find_package(MPI)
  if (NOT ${MPI_FOUND})
    message(SEND_ERROR "You chose MPI but CMake cannot find the MPI package")
  endif(NOT ${MPI_FOUND})
endif(${MPI_PARALLEL})

include(muspectreTools)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
  # using Clang
  add_compile_options(-Wno-missing-braces)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  # using GCC
  add_compile_options(-Wno-non-virtual-dtor)
  string( TOLOWER "${CMAKE_BUILD_TYPE}" build_type )
  if ("release" STREQUAL "${build_type}" )
    add_compile_options(-march=native)
  endif()
  if ("debug" STREQUAL "${build_type}" )
    add_compile_options(-O0)
  endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  # using Intel C++
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  # using Visual Studio C++
endif()

# Do not trust old gcc. the std::optional has memory bugs
if(${CMAKE_COMPILER_IS_GNUCC})
  if(${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 6.0.0)
    add_definitions(-DNO_EXPERIMENTAL)
  endif()
endif()


add_external_package(Eigen3 VERSION 3.3.0 CONFIG)
add_external_package(pybind11 VERSION 2.2 CONFIG)

include_directories(
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}
  )

if(APPLE)
  include_directories(${CMAKE_INSTALL_PREFIX}/include ${Boost_INCLUDE_DIRS})
endif()


#build tests (these are before we add -Werror to the compile options)
if (${MAKE_TESTS})
  ##############################################################################

  # build tests
  file( GLOB TEST_SRCS "${CMAKE_SOURCE_DIR}/tests/test_*.cc")

  add_executable(main_test_suite tests/main_test_suite.cc ${TEST_SRCS})
  target_link_libraries(main_test_suite ${Boost_LIBRARIES} muSpectre)

  muSpectre_add_test(main_test_suite TYPE BOOST main_test_suite --report_level=detailed)

  ##############################################################################
  # copy python test
  file( GLOB PY_TEST_SRCS "${CMAKE_SOURCE_DIR}/tests/python_*.py")

  foreach(pytest ${PY_TEST_SRCS})
    get_filename_component(pytest_name ${pytest} NAME)
    configure_file(
      ${pytest}
      "${CMAKE_BINARY_DIR}/${pytest_name}"
      COPYONLY)
  endforeach(pytest ${PY_TEST_SRCS})
  find_package(PythonInterp ${MUSPECTRE_PYTHON_MAJOR_VERSION} REQUIRED)
  muSpectre_add_test(python_binding_test  TYPE PYTHON python_binding_tests.py)

  if(${MPI_PARALLEL})
    ############################################################################
    # add MPI tests
    file( GLOB TEST_SRCS "${CMAKE_SOURCE_DIR}/tests/mpi_test_*.cc")

    add_executable(mpi_main_test_suite tests/mpi_main_test_suite.cc ${TEST_SRCS})
    target_link_libraries(mpi_main_test_suite ${Boost_LIBRARIES} muSpectre)

    muSpectre_add_test(mpi_main_test_suite1 TYPE BOOST MPI_NB_PROCS 1
      mpi_main_test_suite --report_level=detailed)
    muSpectre_add_test(mpi_main_test_suite2 TYPE BOOST MPI_NB_PROCS 2
      mpi_main_test_suite --report_level=detailed)

    muSpectre_add_test(python_mpi_binding_test1 TYPE PYTHON MPI_NB_PROCS 1
      python_mpi_binding_tests.py)
    muSpectre_add_test(python_mpi_binding_test2 TYPE PYTHON MPI_NB_PROCS 2
      python_mpi_binding_tests.py)
  endif(${MPI_PARALLEL})
endif(${MAKE_TESTS})

################################################################################
# compile the library
add_compile_options( -Werror)
add_subdirectory(
  ${CMAKE_SOURCE_DIR}/src/
  )
add_subdirectory(
  ${CMAKE_SOURCE_DIR}/language_bindings/
  )

if (${MAKE_DOC_TARGET})
  add_subdirectory(
    ${CMAKE_SOURCE_DIR}/doc/
    )
endif()

################################################################################
if (${MAKE_EXAMPLES})
  #compile executables

  set(binaries
    ${CMAKE_SOURCE_DIR}/bin/demonstrator1.cc
    ${CMAKE_SOURCE_DIR}/bin/demonstrator2.cc
    ${CMAKE_SOURCE_DIR}/bin/hyper-elasticity.cc
    ${CMAKE_SOURCE_DIR}/bin/small_case.cc)

  if (${MPI_PARALLEL})
    set (binaries
      ${binaries}
      ${CMAKE_SOURCE_DIR}/bin/demonstrator_mpi.cc
      )
  endif (${MPI_PARALLEL})

  foreach(binaryfile ${binaries})
    get_filename_component(binaryname  ${binaryfile} NAME_WE)
    add_executable(${binaryname} ${binaryfile})
    target_link_libraries(${binaryname} ${Boost_LIBRARIES} muSpectre)
  endforeach(binaryfile ${binaries})

  #or copy them
  file (GLOB pybins "${CMAKE_SOURCE_DIR}/bin/*.py")
  foreach(pybin ${pybins})

    get_filename_component(binaryname  ${pybin} NAME_WE)
    configure_file(
      ${pybin}
      "${CMAKE_BINARY_DIR}/${binaryname}.py"
      COPYONLY)
  endforeach(pybin ${pybins})
endif (${MAKE_EXAMPLES})

################################################################################
# compile benchmarks
if(${MAKE_BENCHMARKS})
  file(GLOB benchmarks "${CMAKE_SOURCE_DIR}/benchmarks/benchmark*cc")
  foreach(benchmark ${benchmarks})
    get_filename_component(benchmark_name ${benchmark} NAME_WE)
    add_executable(${benchmark_name} ${benchmark})
    target_link_libraries(${benchmark_name} ${BOOST_LIBRARIES} muSpectre)
  endforeach(benchmark ${benchmark})
endif(${MAKE_BENCHMARKS})
