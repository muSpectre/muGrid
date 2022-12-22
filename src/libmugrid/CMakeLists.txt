# =============================================================================
# @file   CMakeLists.txt
#
# @author Nicolas Richart <nicolas.richart@epfl.ch>
#
# @date   24 Jan 2019
#
# @brief  Configuration for libmuGrid
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

project(muGrid)
cmake_minimum_required(VERSION 3.5)

if(POLICY CMP0076)
  cmake_policy(SET CMP0076 NEW)
endif()

set(mugrid_HDRS
  ${CMAKE_CURRENT_SOURCE_DIR}/ccoord_operations.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp_compliance.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/eigen_tools.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/exception.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/field.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/field_collection.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/field_collection_global.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/field_collection_local.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/field_map.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/field_map_static.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/field_typed.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/file_io_base.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/gradient_operator_base.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/gradient_operator_default.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/grid_common.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/iterators.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/mapped_field.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/mapped_state_field.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/numpy_tools.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/options_dictionary.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/physics_domain.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/raw_memory_operations.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/ref_array.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/ref_vector.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/state_field.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/state_field_map.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/state_field_map_static.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/tensor_algebra.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/T4_map_proxy.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/units.hh
  )

set(mugrid_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/exception.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/grid_common.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ccoord_operations.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/field_collection.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/field_collection_global.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/field_collection_local.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/file_io_base.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/field.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/field_typed.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/field_map.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/gradient_operator_default.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/physics_domain.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/options_dictionary.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/raw_memory_operations.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/state_field.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/state_field_map.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/units.cc
  )

set(file_io_NETCDF_HDRS
  file_io_netcdf.hh
  )

set(file_io_NETCDF_SRCS
  file_io_netcdf.cc
  )

add_library(muGrid "")

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
target_link_libraries(muGrid PUBLIC version)

check_cxx_source_compiles(
        "#include <experimental/optional>
int main() {
std::experimental::optional<double> A{};
}"
        HAS_STD_OPTIONAL)

if(NOT HAS_STD_OPTIONAL)
    target_compile_definitions(muGrid PUBLIC -DNO_EXPERIMENTAL)
    # Do not trust old gcc. the std::optional has memory bugs
    if(${CMAKE_COMPILER_IS_GNUCC})
      if(${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 6.0.0)
        add_definitions(-DNO_EXPERIMENTAL)
    endif()
  endif()
endif()

# defining external dependencies
target_link_libraries(muGrid PUBLIC ${CMAKE_DL_LIBS})
target_link_libraries(muGrid PUBLIC Eigen3::Eigen)

# defining exported include directories
target_include_directories(muGrid
  INTERFACE $<INSTALL_INTERFACE:include/>
  )
# small trick for build includes in public
set_property(TARGET muGrid APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>)

target_compile_features(muGrid PUBLIC
  cxx_auto_type cxx_constexpr cxx_decltype cxx_decltype_auto)

if(${MUSPECTRE_MPI_PARALLEL})
  find_package(MPI REQUIRED)

  target_link_libraries(muGrid PUBLIC ${MPI_LIBRARIES})
  target_include_directories(muGrid SYSTEM PUBLIC ${MPI_C_INCLUDE_PATH})

  set(mugrid_SRCS ${mugrid_SRCS} ../libmugrid/communicator.cc)
endif()

if(${MUGRID_NETCDF_IO})
  if(${MUSPECTRE_MPI_PARALLEL})
    # find and link PnetCDF
    find_package(PnetCDF REQUIRED)
    target_link_libraries(muGrid PUBLIC pnetcdf)
  else()
    # find and link NetCDF
    find_package(NetCDF REQUIRED)
    target_link_libraries(muGrid PUBLIC netcdf)
  endif()
  list(APPEND mugrid_HDRS ${file_io_NETCDF_HDRS})
  list(APPEND mugrid_SRCS ${file_io_NETCDF_SRCS})
endif()

target_sources(muGrid PRIVATE ${mugrid_SRCS})

if(MUGRID_NAMESPACE)
  add_library(${MUGRID_NAMESPACE}muGrid ALIAS muGrid)
endif()


# ------------------------------------------------------------------------------
if(NOT MUGRID_TARGETS_EXPORT)
  set(MUGRID_TARGETS_EXPORT muGridTargets eigen3)
endif()

install(TARGETS muGrid
  EXPORT ${MUGRID_TARGETS_EXPORT}
  ARCHIVE
    DESTINATION lib
  LIBRARY
    DESTINATION lib
  INCLUDES
    DESTINATION include/libmugrid)

install(FILES ${mugrid_HDRS} DESTINATION include/libmugrid)

if("${MUGRID_TARGETS_EXPORT}" STREQUAL "muGridTargets")
  if(MUGRID_NAMESPACE)
    set(_namespace NAMESPACE ${MUGRID_NAMESPACE})
  endif()

  # install(EXPORT ${MUGRID_TARGETS_EXPORT}
  #   ${_namespace}
  #   DESTINATION share/cmake/${PROJECT_NAME}
  #   COMPONENT dev)

  # #Export for build tree
  # export(EXPORT ${MUGRID_TARGETS_EXPORT}
  #   ${_namespace}
  #   FILE "${CMAKE_BINARY_DIR}/${MUGRID_TARGETS_EXPORT}.cmake")
endif()
