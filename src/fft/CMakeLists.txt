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

find_package(FFTWMPI)
find_package(PFFT)

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

if(FFTWMPI_FOUND)
  set(fft_engine_SRC
    ${fft_engine_SRC}
    ${CMAKE_CURRENT_SOURCE_DIR}/fftwmpi_engine.cc
    )
endif()

if (PFFT_FOUND)
  set(fft_engine_SRC
    ${fft_engine_SRC}
    ${CMAKE_CURRENT_SOURCE_DIR}/pfft_engine.cc
    )
endif()

set (muSpectre_SRC
  ${muSpectre_SRC}
  ${fft_engine_SRC}
  PARENT_SCOPE)
