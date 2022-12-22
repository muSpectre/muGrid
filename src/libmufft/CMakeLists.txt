# =============================================================================
# @file   CMakeLists.txt
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
# A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
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

check_cxx_source_compiles(
        "#include <experimental/optional>
int main() {
std::experimental::optional<double> A{};
}"
        HAS_STD_OPTIONAL)

if(NOT HAS_STD_OPTIONAL)
    target_compile_definitions(muFFT PUBLIC -DNO_EXPERIMENTAL)
    # Do not trust old gcc. the std::optional has memory bugs
    if(${CMAKE_COMPILER_IS_GNUCC})
      if(${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 6.0.0)
        add_definitions(-DNO_EXPERIMENTAL)
    endif()
  endif()
endif()

set(mufft_HDRS
  ${CMAKE_CURRENT_SOURCE_DIR}/mufft_common.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/derivative.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/fft_engine_base.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/fft_utils.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/pocketfft_engine.hh
  )

set(mufft_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/version.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/derivative.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/fft_engine_base.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/fft_utils.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/pocketfft_engine.cc
  )

set(fft_engine_FFTW_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/fftw_engine.cc
  )

set(fft_engine_FFTW_HDRS
  ${CMAKE_CURRENT_SOURCE_DIR}/fftw_engine.hh
  )

set(fft_engine_FFTWMPI_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/fftwmpi_engine.cc
  )

set(fft_engine_FFTWMPI_HDRS
  ${CMAKE_CURRENT_SOURCE_DIR}/fftwmpi_engine.hh
  )

set(fft_engine_PFFT_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/pfft_engine.cc
  )

set(fft_engine_PFFT_HDRS
  ${CMAKE_CURRENT_SOURCE_DIR}/pfft_engine.hh
  )

if (${MUFFT_FFTW})
  find_package(FFTW3 COMPONENTS double REQUIRED)
  target_link_libraries(muFFT PUBLIC fftw3::double)
  list(APPEND mufft_SRCS ${fft_engine_FFTW_SRCS})
  list(APPEND mufft_HDRS ${fft_engine_FFTW_HDRS})
endif(${MUFFT_FFTW})

if(BUILD_SHARED_LIBS STREQUAL "OFF")
  set_target_properties(muFFT PROPERTIES
    POSITION_INDEPENDENT_CODE ON
  )
endif()

if(${MUSPECTRE_MPI_PARALLEL})
  find_package(MPI REQUIRED)

  target_link_libraries(muFFT PUBLIC ${MPI_LIBRARIES})
  target_include_directories(muFFT SYSTEM PUBLIC ${MPI_C_INCLUDE_PATH})

  set(_using_mpi FALSE)
  foreach (_fft FFTWMPI PFFT)
    if (${MUFFT_${_fft}})
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
      list(APPEND mufft_HDRS ${fft_engine_${_fft}_HDRS})
      set(_using_mpi TRUE)
    endif()
  endforeach()
  ##############################################################################
  if(NOT ${_using_mpi})
    message(SEND_ERROR "You activated MPI but turned on none of the MPI-parallel FFT engines")
  endif()
  ##############################################################################
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
  set(MUFFT_TARGETS_EXPORT muFFTTargets eigen3)
endif()


install(TARGETS muFFT
  EXPORT ${MUFFT_TARGETS_EXPORT}
  ARCHIVE
    DESTINATION lib
  LIBRARY
    DESTINATION lib
  PUBLIC_HEADER
    DESTINATION include/libmufft)

install(FILES ${mufft_HDRS} DESTINATION include/libmufft)

if("${MUFFT_TARGETS_EXPORT}" STREQUAL "muFFTTargets")
  if(MUFFT_NAMESPACE)
    set(_namespace NAMESPACE ${MUFFT_NAMESPACE})
  endif()

  # install(EXPORT ${MUFFT_TARGETS_EXPORT}
  #   ${_namespace}
  #   DESTINATION share/cmake/${PROJECT_NAME}
  #   COMPONENT dev)

  # #Export for build tree
  # export(EXPORT ${MUFFT_TARGETS_EXPORT}
  #   ${_namespace}
  #   FILE "${CMAKE_BINARY_DIR}/${MUFFT_TARGETS_EXPORT}.cmake")
endif()
