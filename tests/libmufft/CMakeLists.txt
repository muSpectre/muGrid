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
set(mufft_tests)

macro(muFFT_add_test)
  muTools_add_test(${ARGN} LINK_LIBRARIES ${MUFFT_NAMESPACE}muFFT TEST_LIST mufft_tests)
endmacro()

# build library tests
file(GLOB TEST_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/test_*.cc")
muFFT_add_test(main_mufft_test_suite
  SOURCES main_test_suite.cc ${TEST_SRCS}
  TYPE BOOST --report_level=detailed)

file(GLOB PY_TEST_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/python_*.py")
foreach(pytest ${PY_TEST_SRCS})
  get_filename_component(pytest_name ${pytest} NAME)
  configure_file(
    ${pytest}
    "${CMAKE_CURRENT_BINARY_DIR}/${pytest_name}"
    COPYONLY)
endforeach(pytest ${PY_TEST_SRCS})
muFFT_add_test(mufft_python_binding_test TYPE
  PYTHON "${CMAKE_CURRENT_BINARY_DIR}/python_binding_tests.py")

if(${MUSPECTRE_MPI_PARALLEL})
  # add MPI tests
  file(GLOB TEST_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/mpi_test_*.cc")

  muFFT_add_test(mufft_mpi_main_test_suite1
    SOURCES mpi_main_test_suite.cc ${TEST_SRCS}
    TARGET mpi_main_test_suite
    TYPE BOOST MPI_NB_PROCS 1 --report_level=detailed)
  muFFT_add_test(mufft_mpi_main_test_suite2
    TARGET mpi_main_test_suite
    TYPE BOOST MPI_NB_PROCS 2 --report_level=detailed)

  muFFT_add_test(mufft_python_mpi_binding_test1 TYPE PYTHON MPI_NB_PROCS 1
    python_mpi_binding_tests.py)
  muFFT_add_test(mufft_python_mpi_binding_test2 TYPE PYTHON MPI_NB_PROCS 2
    python_mpi_binding_tests.py)
endif()

##############################################################################
# license test


add_custom_target(test_mufft DEPENDS ${mufft_tests})

add_subdirectory(license)
