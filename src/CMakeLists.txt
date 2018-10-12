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
# along with µSpectre; see the file COPYING. If not, write to the
# Free Software Foundation, Inc., 59 Temple Place - Suite 330,
# Boston, MA 02111-1307, USA.
# =============================================================================

add_library(muSpectre "")
set("PRIVATE_MUSPECTRE_LIBS" "")

add_subdirectory(common)
add_subdirectory(materials)
add_subdirectory(fft)
add_subdirectory(cell)
add_subdirectory(solver)


if (${MPI_PARALLEL})
  target_link_libraries(muSpectre PUBLIC ${MPI_LIBRARIES})
  target_include_directories(muSpectre SYSTEM PUBLIC ${MPI_C_INCLUDE_PATH})
endif(${MPI_PARALLEL})


find_package(FFTW REQUIRED)


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

set(muSpectre_INCLUDES ${FFTW_INCLUDES})
target_include_directories(muSpectre INTERFACE ${muSpectre_INCLUDES})

target_link_libraries(muSpectre PRIVATE ${FFTW_LIBRARIES} ${PRIVATE_MUSPECTRE_LIBS})

target_link_libraries(muSpectre PUBLIC Eigen3::Eigen)

set_property(TARGET muSpectre PROPERTY PUBLIC_HEADER ${_headers})

install(TARGETS muSpectre
        LIBRARY DESTINATION lib
        PUBLIC_HEADER DESTINATION include)
