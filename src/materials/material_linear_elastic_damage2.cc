/**
 * @file   material_linear_elastic_damage2.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   04 May 2020
 *
 * @brief  implementation of MaterialLinearElasticDamage2
 *
 * Copyright © 2020 Ali Falsafi
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 *
 */

#include "materials/material_linear_elastic_damage2.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  MaterialLinearElasticDamage2<DimM>::MaterialLinearElasticDamage2(
      const std::string & name, const Index_t & spatial_dimension,
      const Index_t & nb_quad_pts, const Real & young, const Real & poisson,
      const Real & kappa_init, const Real & alpha, const Real & beta)
      : Parent{name, spatial_dimension, nb_quad_pts},
        material_child(name + "_child", spatial_dimension, nb_quad_pts, young,
                       poisson, kappa_init, alpha, beta,
                       this->internal_fields) {}

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void
  MaterialLinearElasticDamage2<DimM>::add_pixel(const size_t & pixel_index) {
    this->internal_fields->add_pixel(pixel_index);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void
  MaterialLinearElasticDamage2<DimM>::add_pixel(const size_t & pixel_index,
                                                const Real & kappa_variation) {
    this->internal_fields->add_pixel(pixel_index);
    this->get_kappa_field().get_state_field().current().push_back(
        this->material_child.get_kappa_init() + kappa_variation);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialLinearElasticDamage2<DimM>::save_history_variables() {
    this->get_kappa_field().get_state_field().cycle();
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialLinearElasticDamage2<DimM>::initialise() {
    if (not this->is_initialised_flag) {
      Parent::initialise();
      this->save_history_variables();
    }
  }

  /* ---------------------------------------------------------------------- */

  template class MaterialLinearElasticDamage2<twoD>;
  template class MaterialLinearElasticDamage2<threeD>;

}  // namespace muSpectre
