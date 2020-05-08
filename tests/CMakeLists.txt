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

find_package(Boost COMPONENTS unit_test_framework REQUIRED)

function(get_test_name test_name test_fname prefix)
  get_filename_component(test_fname_we ${test_fname} NAME_WE)
  string(REGEX REPLACE "(${prefix}_?)(.*)" "\\2" _test_name "${test_fname_we}")
  set(${test_name} ${_test_name} PARENT_SCOPE)
endfunction()

add_subdirectory(libmugrid)
if(MUCHOICE MATCHES "muGrid")
  return()
endif()

add_subdirectory(libmufft)
if(MUCHOICE MATCHES "muFFT")
  return()
endif()

set(muspectre_tests)

macro(muSpectre_add_test)
  muTools_add_test(${ARGN} LINK_LIBRARIES ${MUSPECTRE_NAMESPACE}muSpectre TEST_LIST muspectre_tests)
endmacro()


# build library tests
file(GLOB TEST_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/test_*.cc")
if(${SPLIT_CELL})
  file( GLOB SPLIT_TEST_SRCS "${CMAKE_SOURCE_DIR}/tests/split_test_*.cc")
  list (APPEND TEST_SRCS ${SPLIT_TEST_SRCS})
endif(${SPLIT_CELL})
muSpectre_add_test(main_test_suite
  SOURCES main_test_suite.cc ${TEST_SRCS}
  TYPE BOOST --report_level=detailed)

muSpectre_add_test(mattb_test
  SOURCES main_test_suite.cc test_materials_toolbox.cc
  TYPE BOOST --report_level=detailed)


##############################################################################
# build py_comparison tests
file(GLOB PY_COMP_TEST_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/py_comparison_*.cc")
find_package(PythonInterp ${MUSPECTRE_PYTHON_MAJOR_VERSION} REQUIRED)
foreach(py_comp_test ${PY_COMP_TEST_SRCS})
  get_test_name(py_comp_test_name ${py_comp_test} "py_comparison_test")
  set(py_comp_test_fname "py_comparison_test_${py_comp_test_name}")

  pybind11_add_module(${py_comp_test_name} ${py_comp_test})
  target_link_libraries(${py_comp_test_name} PRIVATE muSpectre::muSpectre)

  configure_file(
    "${py_comp_test_fname}.py"
    "${CMAKE_CURRENT_BINARY_DIR}/${py_comp_test_fname}.py"
    COPYONLY)
  muSpectre_add_test(${py_comp_test_fname}
    TYPE PYTHON "${py_comp_test_fname}.py")
endforeach(py_comp_test ${PY_COMP_TEST_SRCS})

##############################################################################
# copy python test

file(GLOB PY_TEST_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/python_*.py")
foreach(pytest ${PY_TEST_SRCS})
  get_filename_component(pytest_name ${pytest} NAME)
  configure_file(
    ${pytest}
    "${CMAKE_CURRENT_BINARY_DIR}/${pytest_name}"
    COPYONLY)
endforeach(pytest ${PY_TEST_SRCS})
muSpectre_add_test(python_binding_tests TYPE PYTHON python_binding_tests.py)

file(GLOB REFERENCE_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/reference_computations/*.ref.*")
foreach(ref ${REFERENCE_SRCS})
  get_filename_component(ref_name ${ref} NAME)
  configure_file(
    ${ref}
    "${CMAKE_CURRENT_BINARY_DIR}/reference_computations/${ref_name}"
    COPYONLY)
endforeach(ref ${REFERENCE_SRCS})

if(${MUSPECTRE_MPI_PARALLEL})
  # add MPI tests
  file(GLOB TEST_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/mpi_test_*.cc")

  muSpectre_add_test(mpi_main_test_suite1
    SOURCES mpi_main_test_suite.cc ${TEST_SRCS}
    TARGET mpi_main_test_suite
    TYPE BOOST MPI_NB_PROCS 1 --report_level=detailed)

  muSpectre_add_test(mpi_main_test_suite2
    TARGET mpi_main_test_suite
    TYPE BOOST MPI_NB_PROCS 2 --report_level=detailed)

  muSpectre_add_test(python_mpi_binding_test1 TYPE PYTHON MPI_NB_PROCS 1
    python_mpi_binding_tests.py)

  muSpectre_add_test(python_mpi_binding_test2 TYPE PYTHON MPI_NB_PROCS 2
    python_mpi_binding_tests.py)
endif()

add_custom_target(test_muspectre DEPENDS ${muspectre_tests})

add_subdirectory(lint)

add_subdirectory(license)
