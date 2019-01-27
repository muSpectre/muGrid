# =============================================================================
# file   CMakeLists.txt
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

project(muGrid)
cmake_minimum_required(VERSION 3.5)

set(mugrid_HDRS
  ${CMAKE_CURRENT_SOURCE_DIR}/ccoord_operations.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/field_base.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/field_collection_base.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/field_collection_global.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/field_collection.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/field_collection_local.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/field_helpers.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/field.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/field_map_base.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/field_map_dynamic.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/field_map.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/field_map_matrixlike.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/field_map_scalar.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/field_map_tensor.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/field_typed.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/grid_common.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/iterators.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/statefield.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/eigen_tools.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/T4_map_proxy.hh
  ${CMAKE_CURRENT_SOURCE_DIR}/tensor_algebra.hh
  )

add_library(muGrid INTERFACE)
target_sources(muGrid INTERFACE ${mugrid_HDRS})

# defining external dependencies
target_link_libraries(muGrid INTERFACE Eigen3::Eigen)

# defining exported include directories
target_include_directories(muGrid
  INTERFACE $<INSTALL_INTERFACE:include/>
  )
# small trick for build includes in public
set_property(TARGET muGrid APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>)

target_compile_features(muGrid INTERFACE
  cxx_auto_type cxx_constexpr cxx_decltype cxx_decltype_auto)

add_library(muSpectre::muGrid ALIAS muGrid)
