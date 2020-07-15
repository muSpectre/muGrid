# =============================================================================
# file   CMakeLists.txt
#
# @author Till Junge <till.junge@epfl.ch>
#
# @date   08 Jan 2018
#
# @brief  configuration for material laws
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

set (materials_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/material_base.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/material_linear_anisotropic.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/material_linear_orthotropic.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/material_linear_elastic1.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/material_linear_elastic2.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/material_linear_elastic3.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/material_linear_elastic4.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/material_linear_elastic_damage1.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/material_linear_elastic_damage2.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/material_linear_elastic_generic1.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/material_linear_elastic_generic2.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/material_visco_elastic_ss.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/material_visco_elastic_damage_ss1.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/material_visco_elastic_damage_ss2.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/material_neo_hookean_elastic.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/material_hyper_elasto_plastic1.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/material_hyper_elasto_plastic2.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/material_stochastic_plasticity.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/s_t_material_linear_elastic_generic1.cc
  )
if(${SPLIT_CELL})
  set (materials_SRC
    ${materials_SRC}
    ${CMAKE_CURRENT_SOURCE_DIR}/laminate_homogenisation.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/material_laminate.cc
    )
endif()
target_sources(muSpectre PRIVATE ${materials_SRC})
