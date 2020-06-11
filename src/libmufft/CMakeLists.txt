# =============================================================================
# file   CMakeLists.txt
#
# @author Nicolas Richart <nicolas.richart@epfl.ch>
#
# @date   24 Jan 2019
#
# @brief  Configuration for libmuFFT
#
# @section LICENSE
#
# Copyright © 2018 Till Junge
#
# µGrid is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3, or (at your option) any later version.
#
# µGrid is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with µGrid; see the file COPYING. If not, write to the Free Software
# Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
#
# Additional permission under GNU GPL version 3 section 7
#
# If you modify this Program, or any covered work, by linking or combining it
# with proprietary FFT implementations or numerical libraries, containing parts
# covered by the terms of those libraries' licenses, the licensors of this
# Program grant you additional permission to convey the resulting work.
# =============================================================================
project(muFFT)
cmake_minimum_required(VERSION 3.5)

add_library(muFFT "")

set(PRE_CONFIGURE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/version.cc.skeleton")
set(POST_CONFIGURE_FILE "${CMAKE_CURRENT_BINARY_DIR}/version.cc")
include(git_watcher.cmake)

# Create a library out of the compiled post-configure file.
add_library(version STATIC ${POST_CONFIGURE_FILE})
set_target_properties(version PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
  POSITION_INDEPENDENT_CODE ON)
target_include_directories(version PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
add_dependencies(version check_git)
target_link_libraries(muFFT PUBLIC version)

set(mufft_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/derivative.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/fft_engine_base.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/fft_utils.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/fftw_engine.cc
  )

set(fft_engine_FFTWMPI_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/fftwmpi_engine.cc
  )
set(fft_engine_PFFT_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/pfft_engine.cc
  )

find_package(FFTW3 COMPONENTS double REQUIRED)
target_link_libraries(muFFT PUBLIC fftw3::double)

if(BUILD_SHARED_LIBS STREQUAL "OFF")
  set_target_properties(muFFT PROPERTIES
    POSITION_INDEPENDENT_CODE ON
  )
endif()

if(${MUSPECTRE_MPI_PARALLEL})
  find_package(MPI REQUIRED)
  set(mufft_SRCS ${mufft_SRCS} communicator.cc)

  target_link_libraries(muFFT PUBLIC ${MPI_LIBRARIES})
  target_include_directories(muFFT SYSTEM PUBLIC ${MPI_C_INCLUDE_PATH})
  target_compile_definitions(muFFT PUBLIC -DWITH_MPI)

  option(MUFFT_USE_FFTWMPI "If on, the mpi-parallel FFTW engine is built" ON)
  option(MUFFT_USE_PFFT "If on, the mpi-parallel PFFT engine is built" OFF)

  set(_using_mpi FALSE)
  foreach (_fft FFTWMPI PFFT)
    if (${MUFFT_USE_${_fft}})
      if("${_fft}" STREQUAL "FFTWMPI")
        find_package(FFTW3 COMPONENTS double mpi REQUIRED)
        target_link_libraries(muFFT PUBLIC fftw3::double::mpi)
      else()
        find_package(${_fft} REQUIRED)
        # TODO this private should perhaps become private
        target_link_libraries(muFFT PUBLIC ${${_fft}_LIBRARIES})
        target_include_directories(muFFT PUBLIC ${${_fft}_INCLUDES})
      endif()

      target_compile_definitions(muFFT PUBLIC -DWITH_${_fft})

      list(APPEND mufft_SRCS ${fft_engine_${_fft}_SRCS})
      set(_using_mpi TRUE)
    endif()
  endforeach()
  ##############################################################################
  if(NOT ${_using_mpi})
    message(SEND_ERROR "You activated MPI but turned on none of the MPI-parallel FFT engines")
  endif()
endif()

target_sources(muFFT PRIVATE ${mufft_SRCS})
target_link_libraries(muFFT PUBLIC ${MUFFT_NAMESPACE}muGrid)

# defining exported include directories
target_include_directories(muFFT
  INTERFACE $<INSTALL_INTERFACE:include/>
  )

# small trick for build includes in public
set_property(TARGET muFFT APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>)


if(MUFFT_NAMESPACE)
  add_library(${MUFFT_NAMESPACE}muFFT ALIAS muFFT)
endif()


# ------------------------------------------------------------------------------
if(NOT MUFFT_TARGETS_EXPORT)
  if ("${MUCHOICE}" STREQUAL muFFT) 
    set(MUFFT_TARGETS_EXPORT muFFTTargets)
endif()


install(TARGETS muFFT
  EXPORT ${MUFFT_TARGETS_EXPORT}
  ARCHIVE
    DESTINATION lib
  LIBRARY
    DESTINATION lib
  PUBLIC_HEADER
    DESTINATION include/libmufft)

if("${MUFFT_TARGETS_EXPORT}" STREQUAL "muFFTTargets")
  if(MUFFT_NAMESPACE)
    set(_namespace NAMESPACE ${MUFFT_NAMESPACE})
  else()
    unset(_namespace)
  endif()

  install(EXPORT ${MUFFT_TARGETS_EXPORT}
    ${_namespace}
    DESTINATION share/cmake/${PROJECT_NAME}
    COMPONENT dev)

  #Export for build tree
  export(EXPORT ${MUFFT_TARGETS_EXPORT}
    ${_namespace}
    FILE "${CMAKE_BINARY_DIR}/${MUFFT_TARGETS_EXPORT}.cmake")
  endif()
endif()
