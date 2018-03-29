#==============================================================================
# file   CMakeLists.txt
#
# @author Till Junge <till.junge@epfl.ch>
#
# @date   08 Jan 2018
#
# @brief  configuration for python binding using pybind11
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
#

# FIXME! The user should have a choice to configure this path.
execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-m" "site" "--user-site"
  RESULT_VARIABLE _PYTHON_SUCCESS
  OUTPUT_VARIABLE PYTHON_USER_SITE
  ERROR_VARIABLE _PYTHON_ERROR_VALUE)
if(NOT _PYTHON_SUCCESS MATCHES 0)
  message(FATAL_ERROR
          "Python config failure:\n${_PYTHON_ERROR_VALUE}")
endif()
string(REGEX REPLACE "\n" "" PYTHON_USER_SITE ${PYTHON_USER_SITE})

set (PY_BINDING_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/bind_py_module.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/bind_py_common.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/bind_py_cell.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/bind_py_material.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/bind_py_solvers.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/bind_py_fftengine.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/bind_py_projections.cc
  )


pybind11_add_module(pyMuSpectreLib ${PY_BINDING_SRCS})
target_link_libraries(pyMuSpectreLib PRIVATE muSpectre)
# Want to rename the output, so that the python module is called muSpectre
set_target_properties(pyMuSpectreLib PROPERTIES OUTPUT_NAME _muSpectre)

add_custom_target(pyMuSpectre ALL SOURCES muSpectre/__init__.py)
add_custom_command(TARGET pyMuSpectre POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                       ${CMAKE_SOURCE_DIR}/language_bindings/python/muSpectre $<TARGET_FILE_DIR:pyMuSpectreLib>/muSpectre)

install(TARGETS pyMuSpectreLib LIBRARY DESTINATION ${PYTHON_USER_SITE})
install(FILES muSpectre/__init__.py DESTINATION ${PYTHON_USER_SITE}/muSpectre)