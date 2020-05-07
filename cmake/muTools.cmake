#==============================================================================
# file   muspectreTools.cmake
#
# @author Nicolas Richart <nicolas.,richart@epfl.ch>
#
# @date   11 Jan 2018
#
# @brief  some tool to help to do stuff with cmake in µSpectre
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
# along with µSpectre; see the file COPYING. If not, write to the
# Free Software Foundation, Inc., 59 Temple Place - Suite 330,
# Boston, MA 02111-1307, USA.
#

#[=[.rst:
µTools
------

Common cmake function to the µ* projects

::

muTools_move_to_project(target)

Moves the output of a given target to the root PROJECT_BINARY_DIR

::

muTools_add_test(test_name
  [HEADER_ONLY]
  [TYPE BOOST|PYTHON]
  [MPI_NB_PROCS]
  [TARGET]
  [SOURCES]
  [ARG_LIST]
  [LINK_LIBRARIES libraries|targets]
  )


]=]
#


function(muTools_move_to_project target)
  get_property(_output_name TARGET ${target} PROPERTY OUTPUT_NAME)
  get_filename_component(_output_name_exe "${_output_name}" NAME)
  file(RELATIVE_PATH _output_name
    ${CMAKE_CURRENT_BINARY_DIR} "${PROJECT_BINARY_DIR}/${target}")
  set_property(TARGET ${target}
    PROPERTY OUTPUT_NAME "${_output_name}")
endfunction()

# ------------------------------------------------------------------------------
function(muTools_add_test test_name)
  include(CMakeParseArguments)

  set(_mat_flags
    HEADER_ONLY
    )

  set(_mat_one_variables
    TYPE
    MPI_NB_PROCS
    TARGET
    TEST_LIST
    )

  set(_mat_multi_variables
    SOURCES
    ARG_LIST
    LINK_LIBRARIES
    )

  cmake_parse_arguments(_mat_args
    "${_mat_flags}"
    "${_mat_one_variables}"
    "${_mat_multi_variables}"
    ${ARGN}
    )

  if ("${_mat_args_TYPE}" STREQUAL "BOOST")
  elseif("${_mat_args_TYPE}" STREQUAL "PYTHON")
  else ()
    message (SEND_ERROR "Can only handle types 'BOOST' and 'PYTHON'")
  endif ("${_mat_args_TYPE}" STREQUAL "BOOST")

  if ("${_mat_args_TYPE}" STREQUAL "BOOST")
    if(DEFINED _mat_args_TARGET)
      set(target_test_name ${_mat_args_TARGET})
    else()
      set(target_test_name ${test_name})
    endif()

    if(NOT TARGET ${target_test_name})
      add_executable(${target_test_name} ${_mat_args_SOURCES})
      if(_mat_args_TEST_LIST)
        set(_tmp ${${_mat_args_TEST_LIST}})
        list(APPEND _tmp ${target_test_name})
        set(${_mat_args_TEST_LIST} ${_tmp} PARENT_SCOPE)
      endif()

      muTools_move_to_project(${target_test_name})
      target_link_libraries(${target_test_name}
        PRIVATE Boost::boost Boost::unit_test_framework cxxopts ${_mat_args_LINK_LIBRARIES})

      if(_mat_HEADER_ONLY)
        foreach(_target ${_mat_args_LINK_LIBRARIES})
          if(TARGET ${_target})
            get_target_property(_features ${_target} INTERFACE_COMPILE_FEATURES)
            target_compile_features(${target_test_name}
              PRIVATE ${_features})
          endif()
        endforeach()
      endif()
    endif()
  endif()

  if ("${_mat_args_TYPE}" STREQUAL "BOOST")
    set(_exe $<TARGET_FILE:${target_test_name}> ${_mat_args_UNPARSED_ARGUMENTS})
  else()
    set(_exe ${_mat_args_UNPARSED_ARGUMENTS} ${_mat_args_ARG_LIST})
  endif()


  if (${MUSPECTRE_RUNNING_IN_CI} AND NOT ${_mat_args_MPI_NB_PROCS})
    if ("${_mat_args_TYPE}" STREQUAL "BOOST")
      list(APPEND _exe "--logger=JUNIT,all,test_results_${test_name}.xml")
    elseif("${_mat_args_TYPE}" STREQUAL "PYTHON")
      set(_exe ${PYTHON_EXECUTABLE} -m pytest --junitxml test_results_${test_name}.xml ${_exe})
    endif ("${_mat_args_TYPE}" STREQUAL "BOOST")
  else ()
    if("${_mat_args_TYPE}" STREQUAL "PYTHON")
      set(_exe ${PYTHON_EXECUTABLE} ${_exe})
    endif ("${_mat_args_TYPE}" STREQUAL "PYTHON")
  endif (${MUSPECTRE_RUNNING_IN_CI} AND NOT ${_mat_args_MPI_NB_PROCS})

  if(${_mat_args_MPI_NB_PROCS})
    find_package(MPI REQUIRED)
    set(_exe ${MPIEXEC_EXECUTABLE} ${MPIEXEC_PREFLAGS} ${MPIEXEC_NUMPROC_FLAG} ${_mat_args_MPI_NB_PROCS} ${MPIEXEC_POSTFLAGS} ${_exe})
  endif(${_mat_args_MPI_NB_PROCS})

  add_test(
    NAME ${test_name}
    COMMAND ${_exe}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endfunction()

##############################################################################

# license test
set(LICENSETEST_TARGET license CACHE STRING "license test")
# project root directory
add_custom_target(${LICENSETEST_TARGET})

function(license_add_subdirectory DIR LICENSETEST)
  if(LICENSETEST)
  else()
    message(FATAL_ERROR "license script: NOT FOUND!")
  endif()

  set(LICENSETEST_EXC ${PYTHON_EXECUTABLE} ${LICENSETEST})
  # create relative path to the directory
  file(RELATIVE_PATH TEST_NAME ${CMAKE_SOURCE_DIR} ${DIR})
  string(REGEX REPLACE "/" "." TEST_NAME ${TEST_NAME})

  # perform license check
  set(TARGET_NAME ${LICENSETEST_TARGET}.${TEST_NAME})
  string(LENGTH ${TARGET_NAME} NAME_LENGTH)
  if (${NAME_LENGTH} GREATER 39)
    string(SUBSTRING ${TARGET_NAME} 0 15 START)
    math(EXPR START_ID "${NAME_LENGTH} - 21")
    string(SUBSTRING ${TARGET_NAME} ${START_ID} ${NAME_LENGTH} END)
    string(CONCAT TARGET_NAME ${START} "..." ${END})
  endif(${NAME_LENGTH} GREATER  39)

  add_custom_target(${TARGET_NAME}
    COMMAND
    ${LICENSETEST_EXC}
    ${DIR}
    DEPENDS ${DIR}
    COMMENT "license test"
    )
  # run this target when root cpplint.py test is triggered
  add_dependencies(${LICENSETEST_TARGET} ${TARGET_NAME})

  # add this test to CTest
  add_test(${TARGET_NAME} ${CMAKE_MAKE_PROGRAM} ${TARGET_NAME})
  unset(LICENSETEST)
endfunction()
