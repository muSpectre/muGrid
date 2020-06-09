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

set(MUSPECTRE_TARGETS_EXPORT "${MUCHOICE}Targets")
set(MUFFT_TARGETS_EXPORT "${MUSPECTRE_TARGETS_EXPORT}")
set(MUGRID_TARGETS_EXPORT "${MUSPECTRE_TARGETS_EXPORT}")

add_subdirectory(libmugrid)

if(MUCHOICE MATCHES "muGrid")
  return()
endif()

add_subdirectory(libmufft)

if(MUCHOICE MATCHES "muFFT")
  return()
endif()

add_library(muSpectre "")

add_subdirectory(common)
add_subdirectory(materials)
add_subdirectory(projection)
add_subdirectory(cell)
add_subdirectory(solver)

# The following checks whether std::optional exists and replaces it by
# boost::optional if necessary
include(CheckCXXSourceCompiles)
check_cxx_source_compiles(
    "#include <experimental/optional>
int main() {
std::experimental::optional<double> A{};
}"
    HAS_STD_OPTIONAL)

if(NOT HAS_STD_OPTIONAL)
    target_compile_definitions(muSpectre PUBLIC -DNO_EXPERIMENTAL)
    # Do not trust old gcc. the std::optional has memory bugs
    if(${CMAKE_COMPILER_IS_GNUCC})
      if(${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 6.0.0)
        add_definitions(-DNO_EXPERIMENTAL)
    endif()
  endif()
endif()

file(GLOB_RECURSE _headers RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.hh")
list(APPEND muSpectre_SRC ${_headers})
set(muSpectre_INCLUDES ${FFTW_INCLUDES})

if(${SPLIT_CELL})
  target_compile_definitions(muSpectre PUBLIC -DWITH_SPLIT)
endif()


if(${SPLIT_CELL})
    target_link_libraries(muSpectre PUBLIC Eigen3::Eigen   corkpp gmp )
else()
  target_link_libraries(muSpectre PUBLIC Eigen3::Eigen)
endif()

target_include_directories(muSpectre INTERFACE ${muSpectre_INCLUDES})

target_link_libraries(muSpectre PUBLIC
  ${MUFFT_NAMESPACE}muFFT ${MUGRID_NAMESPACE}muGrid)

set_property(TARGET muSpectre PROPERTY PUBLIC_HEADER ${_headers})

# defining exported include directories
target_include_directories(muSpectre
  PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
  INTERFACE $<INSTALL_INTERFACE:include/libmuspectre>
  )

# small trick for build includes in public
set_property(TARGET muSpectre APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>)


add_library(${MUSPECTRE_NAMESPACE}muSpectre ALIAS muSpectre)

# ------------------------------------------------------------------------------
# if(NOT MUSPECTRE_TARGETS_EXPORT)
#   set(MUSPECTRE_TARGETS_EXPORT muSpectreTargets)
# endif()

# install(TARGETS muSpectre
#   EXPORT ${MUSPECTRE_TARGETS_EXPORT}
#   LIBRARY DESTINATION lib
#   PUBLIC_HEADER DESTINATION include/libmuspectre)

# if("${MUSPECTRE_TARGETS_EXPORT}" STREQUAL "muSpectreTargets")
#   install(EXPORT ${MUSPECTRE_TARGETS_EXPORT}
#     NAMESPACE "${MUSPECTRE_NAMESPACE}"
#     DESTINATION share/cmake/${PROJECT_NAME}
#     COMPONENT dev)

#   #Export for build tree
#   export(EXPORT ${MUSPECTRE_TARGETS_EXPORT}
#     NAMESPACE "${MUSPECTRE_NAMESPACE}"
#     FILE "${CMAKE_BINARY_DIR}/${MUSPECTRE_TARGETS_EXPORT}.cmake")
# endif()
