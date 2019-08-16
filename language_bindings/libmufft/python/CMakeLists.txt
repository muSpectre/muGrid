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

# FIXME! The user should have a choice to configure this path.
execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-m" "site" "--user-site"
  RESULT_VARIABLE _PYTHON_SUCCESS
  OUTPUT_VARIABLE PYTHON_USER_SITE
  ERROR_VARIABLE _PYTHON_ERROR_VALUE)
if(_PYTHON_SUCCESS GREATER 2)
  message(FATAL_ERROR
          "Python config failure:\n${_PYTHON_ERROR_VALUE}")
endif()
string(REGEX REPLACE "\n" "" PYTHON_USER_SITE ${PYTHON_USER_SITE})

set (PY_BINDING_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/bind_py_module.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/bind_py_communicator.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/bind_py_common.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/bind_py_derivatives.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/bind_py_fftengine.cc
  )

if (${USE_FFTWMPI})
  add_definitions(-DWITH_FFTWMPI)
endif(${USE_FFTWMPI})

if (${USE_PFFT})
  add_definitions(-DWITH_PFFT)
endif(${USE_PFFT})

find_package(PythonLibsNew ${MUFFT_PYTHON_MAJOR_VERSION} MODULE REQUIRED)

# On OS X, add -Wno-deprecated-declarations
IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  add_compile_options(-Wno-deprecated-declarations)
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

pybind11_add_module(pyMuFFTLib ${PY_BINDING_SRCS})
target_link_libraries(pyMuFFTLib PRIVATE muFFT)
# Want to rename the output, so that the python module is called muFFT
set_target_properties(pyMuFFTLib PROPERTIES OUTPUT_NAME _muFFT)

target_include_directories(pyMuFFTLib PUBLIC ${PYTHON_INCLUDE_DIRS})

add_custom_target(pyMuFFT ALL SOURCES
  muFFT/__init__.py
  muFFT/Communicator.py
  muFFT/NetCDF.py)
add_custom_command(TARGET pyMuFFT POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                   ${CMAKE_CURRENT_SOURCE_DIR}/muFFT
                   $<TARGET_FILE_DIR:pyMuFFTLib>/muFFT)

install(TARGETS pyMuFFTLib LIBRARY DESTINATION ${PYTHON_USER_SITE})
install(FILES muFFT/__init__.py muFFT/Communicator.py muFFT/NetCDF.py
  DESTINATION ${PYTHON_USER_SITE}/muFFT)
