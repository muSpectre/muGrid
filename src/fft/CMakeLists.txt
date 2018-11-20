# =============================================================================
# file   CMakeLists.txt
#
# @author Till Junge <till.junge@epfl.ch>
#
# @date   08 Jan 2018
#
# @brief  configuration for fft-related files
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

################################################################################
if (${MPI_PARALLEL})
  set(USE_FFTWMPI "ON" CACHE BOOL "If on, the mpi-parallel FFTW engine is built")
  set(USE_PFFT "OFF" CACHE BOOL "If on, the mpi-parallel PFFT engine is built")

  ##############################################################################
  if (${USE_FFTWMPI})
    find_package(FFTWMPI)

    if (NOT ${FFTWMPI_FOUND})
      message (SEND_ERROR "You chose FFTWMPI but CMake cannot find it")
    endif (NOT ${FFTWMPI_FOUND})

    list(APPEND PRIVATE_MUSPECTRE_LIBS
      ${FFTWMPI_LIBRARIES})

    target_include_directories(muSpectre PRIVATE ${FFTWMPI_INCLUDES})
  endif (${USE_FFTWMPI})

  ##############################################################################
  if (${USE_PFFT})
    find_package(PFFT)

    if (NOT ${PFFT_FOUND})
      message (SEND_ERROR "You chose PFFT but CMake cannot find it")
    endif (NOT ${PFFT_FOUND})

    list(APPEND PRIVATE_MUSPECTRE_LIBS
      ${PFFT_LIBRARIES})

    target_include_directories(muSpectre PUBLIC ${PFFT_INCLUDES})
  endif (${USE_PFFT})

  ##############################################################################
  if (NOT ${USE_FFTWMPI} AND NOT ${USE_PFFT})
    message (SEND_ERROR "You activated MPI but turned on none of the MPI-parallel FFT engines")
  endif (NOT ${USE_FFTWMPI} AND NOT ${USE_PFFT})

endif(${MPI_PARALLEL})

set(PRIVATE_MUSPECTRE_LIBS ${PRIVATE_MUSPECTRE_LIBS} PARENT_SCOPE)

if (${USE_FFTWMPI})
  add_definitions(-DWITH_FFTWMPI)
endif(${USE_FFTWMPI})

if (${USE_PFFT})
  add_definitions(-DWITH_PFFT)
endif(${USE_PFFT})


set (fft_engine_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/fft_utils.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/fft_engine_base.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/fftw_engine.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/projection_base.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/projection_default.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/projection_finite_strain.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/projection_small_strain.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/projection_finite_strain_fast.cc
  )

if(${USE_FFTWMPI})
  set(fft_engine_SRC
    ${fft_engine_SRC}
    ${CMAKE_CURRENT_SOURCE_DIR}/fftwmpi_engine.cc
    )
endif(${USE_FFTWMPI})

if (${USE_PFFT})
  set(fft_engine_SRC
    ${fft_engine_SRC}
    ${CMAKE_CURRENT_SOURCE_DIR}/pfft_engine.cc
    )
endif(${USE_PFFT})

target_sources(muSpectre PRIVATE ${fft_engine_SRC})
