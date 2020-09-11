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
  exception.hh
  numpy_tools.hh
  tensor_algebra.hh
  T4_map_proxy.hh
  mapped_state_field.hh
  cpp_compliance.hh
  eigen_tools.hh
  mapped_field.hh
  ref_array.hh
  field_map_static.hh
  ref_vector.hh
  field_map.hh
  grid_common.hh
  iterators.hh
  state_field.hh
  field_collection_local.hh
  state_field_map_static.hh
  field_collection_global.hh
  field_typed.hh
  field.hh
  field_collection.hh
  file_io_base.hh
  ccoord_operations.hh
  raw_memory_operations.hh
  state_field_map.hh
  units.hh
  )

set(mugrid_SRCS
  exception.cc
  grid_common.cc
  ccoord_operations.cc
  raw_memory_operations.cc
  field_collection.cc
  field_collection_global.cc
  field_collection_local.cc
  field.cc
  field_typed.cc
  field_map.cc
  file_io_base.cc
  state_field.cc
  state_field_map.cc
  units.cc
  )

set(file_io_NETCDF_HDRS
  file_io_netcdf.hh
  )

set(file_io_NETCDF_SRCS
  file_io_netcdf.cc
  )

add_library(muGrid "")

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
  target_compile_definitions(muGrid PUBLIC -DWITH_MPI)

  set(mugrid_SRCS ${mugrid_SRCS} ../libmugrid/communicator.cc)
endif()

if(${NETCDF_IO})
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
