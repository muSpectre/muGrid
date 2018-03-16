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

add_compile_options(-Wall -Wextra -Weffc++)

enable_testing()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)
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

find_package(Boost COMPONENTS unit_test_framework REQUIRED)

add_external_package(Eigen3 VERSION 3.3 CONFIG)
add_external_package(pybind11 VERSION 2.2 CONFIG)

include_directories(
  ${CMAKE_SOURCE_DIR}/tests
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}
  )

if(APPLE)
  include_directories(${CMAKE_INSTALL_PREFIX}/include ${Boost_INCLUDE_DIRS})
endif()

find_package(MPI)
find_package(FFTW REQUIRED)
find_package(FFTWMPI)
find_package(PFFT)

if(MPI_FOUND)
  add_definitions(-DWITH_MPI)
  include_directories(SYSTEM ${MPI_C_INCLUDE_PATH})
endif()


#build tests
file( GLOB TEST_SRCS "${CMAKE_SOURCE_DIR}/tests/test_*.cc")

add_executable(main_test_suite tests/main_test_suite.cc ${TEST_SRCS})
target_link_libraries(main_test_suite ${Boost_LIBRARIES} muSpectre)
add_test(main_test_suite main_test_suite --report_level=detailed --build_info=TRUE)

if(MPI_FOUND)
  file( GLOB TEST_SRCS "${CMAKE_SOURCE_DIR}/tests/mpi_test_*.cc")

  add_executable(mpi_main_test_suite tests/mpi_main_test_suite.cc ${TEST_SRCS})
  target_link_libraries(mpi_main_test_suite ${Boost_LIBRARIES} muSpectre)
  add_test(mpi_main_test_suite main_test_suite --report_level=detailed --build_info=TRUE) 
endif()


# compile the library
add_compile_options( -Werror)
add_subdirectory(
  ${CMAKE_SOURCE_DIR}/src/
  )
add_subdirectory(
  ${CMAKE_SOURCE_DIR}/language_bindings/
  )

add_subdirectory(
  ${CMAKE_SOURCE_DIR}/doc/
  )


#compile executables
file( GLOB binaries "${CMAKE_SOURCE_DIR}/bin/*.cc")
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


# compile benchmarks

file(GLOB benchmarks "${CMAKE_SOURCE_DIR}/benchmarks/benchmark*cc")
foreach(benchmark ${benchmarks})
  get_filename_component(benchmark_name ${benchmark} NAME_WE)
  add_executable(${benchmark_name} ${benchmark})
  target_link_libraries(${benchmark_name} ${BOOST_LIBRARIES} muSpectre)
endforeach(benchmark ${benchmark})


# copy python test
#build tests
file( GLOB PY_TEST_SRCS "${CMAKE_SOURCE_DIR}/tests/python_*.py")

foreach(pytest ${PY_TEST_SRCS})
  get_filename_component(pytest_name ${pytest} NAME)
  configure_file(
    ${pytest}
    "${CMAKE_BINARY_DIR}/${pytest_name}"
    COPYONLY)
endforeach(pytest ${PY_TEST_SRCS})
add_test(python_binding_test python_binding_tests.py)
