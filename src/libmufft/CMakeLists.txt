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

add_library(muFFT)

set(mufft_SRCS
  fft_utils.cc
  fft_engine_base.cc
  fftw_engine.cc
  )

set(fft_engine_FFTWMPI_SRCS
  fftwmpi_engine.cc
  )
set(fft_engine_PFFT_SRCS
  pfft_engine.cc
  )

find_package(FFTW REQUIRED)
target_link_libraries(muFFT PUBLIC ${FFTW_LIBRARIES})
target_include_directories(muFFT PUBLIC ${FFTW_INCLUDES})
      
if(${MUSPECTRE_MPI_PARALLEL})
  find_package(MPI REQUIRED)

  target_link_libraries(muFFT PUBLIC ${MPI_LIBRARIES})
  target_include_directories(muFFT SYSTEM PUBLIC ${MPI_C_INCLUDE_PATH})
  target_compile_definitions(muFFT PUBLIC -DWITH_MPI)

  set(MUFFT_USE_FFTWMPI "ON" CACHE BOOL "If on, the mpi-parallel FFTW engine is built")
  set(MUFFT_USE_PFFT "OFF" CACHE BOOL "If on, the mpi-parallel PFFT engine is built")

  set(_using_mpi FALSE)
  foreach (_fft FFTWMPI PFFT)
    if (${MUFFT_USE_${_fft}})
      find_package(${_fft} REQUIRED)

      if(NOT ${${_fft}_FOUND})
	message(SEND_ERROR "You chose ${_fft} but CMake cannot find it")
      endif()

      # TODO this private should perhaps become private
      target_link_libraries(muFFT PUBLIC ${${_fft}_LIBRARIES})
      target_include_directories(muFFT PUBLIC ${${_fft}_INCLUDES})
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
target_link_libraries(muFFT PUBLIC muSpectre::muGrid)

# defining exported include directories
target_include_directories(muFFT
  PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/libmufft>
  INTERFACE $<INSTALL_INTERFACE:include/>
  )

# small trick for build includes in public
set_property(TARGET muFFT APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>)

add_library(muSpectre::muFFT ALIAS muFFT)

# set_target_properties(muFFT PROPERTIES
#   CXX_STANDARD 14
#   CXX_STANDARD_REQUIRED TRUE)
