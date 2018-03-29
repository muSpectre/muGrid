# =============================================================================
# file   CMakeLists.txt
#
# @author Till Junge <till.junge@epfl.ch>
#
# @date   08 Jan 2018
#
# @brief  Configuration for libmuSpectre
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

include_directories(
  common
  materials
  cell
  )

set (muSpectre_SRC
  )
add_subdirectory(common)
add_subdirectory(materials)
add_subdirectory(fft)
add_subdirectory(cell)
add_subdirectory(solver)

find_package(MPI)
find_package(FFTW REQUIRED)
find_package(FFTWMPI)
find_package(PFFT)


# The following checks whether std::optional exists and replaces it by
# boost::optional if necessary
include(CheckCXXSourceCompiles)
check_cxx_source_compiles(
    "#include <experimental/optional>
int main() {
std::experimental::optional<double> A{};
}"
    HAS_STD_OPTIONAL)

add_definitions(-DBAR)
if( NOT HAS_STD_OPTIONAL)
  add_definitions(-DNO_EXPERIMENTAL)
endif()

file(GLOB_RECURSE _headers RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.hh")

list(APPEND muSpectre_SRC ${_headers})
add_library(muSpectre ${muSpectre_SRC})

set(muSpectre_INCLUDES ${FFTW_INCLUDES})
set(muSpectre_LIBRARIES Eigen3::Eigen ${FFTW_LIBRARIES})
if(FFTWMPI_FOUND)
  set(muSpectre_INCLUDES
    ${muSpectre_INCLUDES}
    ${FFTWMPI_INCLUDES})
  set(muSpectre_LIBRARIES
    ${muSpectre_LIBRARIES}
    ${FFTWMPI_LIBRARIES})
endif()
if(PFFT_FOUND)
  set(muSpectre_INCLUDES
    ${muSpectre_INCLUDES}
    ${PFFT_INCLUDES})
  set(muSpectre_LIBRARIES
    ${muSpectre_LIBRARIES}
    ${PFFT_LIBRARIES})
endif()
if(FFTWMPI_FOUND OR PFFT_FOUND)
  set(muSpectre_INCLUDES
    ${muSpectre_INCLUDES}
    ${MPI_INCLUDES})
  set(muSpectre_LIBRARIES
    ${muSpectre_LIBRARIES}
    ${MPI_LIBRARIES})
endif()
target_include_directories(muSpectre INTERFACE ${muSpectre_INCLUDES})
target_link_libraries(muSpectre ${muSpectre_LIBRARIES})

target_link_libraries(muSpectre Eigen3::Eigen ${FFTW_LIBRARIES})
target_compile_definitions(muSpectre PRIVATE  ${-DNO_EXPERIMENTAL})

set_property(TARGET muSpectre PROPERTY PUBLIC_HEADER ${_headers})

install(TARGETS muSpectre
        LIBRARY DESTINATION lib
        PUBLIC_HEADER DESTINATION include)