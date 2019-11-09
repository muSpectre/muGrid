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
  ccoord_operations.hh
  field_base.hh
  field_collection_base.hh
  field_collection_global.hh
  field_collection.hh
  field_collection_local.hh
  field_helpers.hh
  field.hh
  field_map_base.hh
  field_map_dynamic.hh
  field_map.hh
  field_map_matrixlike.hh
  field_map_scalar.hh
  field_map_tensor.hh
  field_typed.hh
  grid_common.hh
  iterators.hh
  numpy.hh
  statefield.hh
  eigen_tools.hh
  T4_map_proxy.hh
  tensor_algebra.hh
  )

set(mugrid_SRCS
  ccoord_operations.cc
  nfield_collection.cc
  nfield_collection_global.cc
  nfield_collection_local.cc
  nfield.cc
  nfield_typed.cc
  nfield_map.cc
  state_nfield.cc
  state_nfield_map.cc
  )

add_library(muGrid "")

# defining external dependencies
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
