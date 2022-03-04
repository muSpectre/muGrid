/**
 * @file   material_visco_elastic_damage_ss2.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   29 Apr 2020
 *
 * @brief  implementation of material_visco_damage_ss2
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

#include "materials/material_visco_elastic_damage_ss2.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  MaterialViscoElasticDamageSS2<DimM>::MaterialViscoElasticDamageSS2(
      const std::string & name, const Index_t & spatial_dimension,
      const Index_t & nb_quad_pts, const Real & young_inf, const Real & young_v,
      const Real & eta_v, const Real & poisson, const Real & kappa_init,
      const Real & alpha, const Real & beta, const Real & dt)
      : Parent{name, spatial_dimension, nb_quad_pts},
        material_child(name + "_child", spatial_dimension, nb_quad_pts,
                       young_inf, young_v, eta_v, poisson, kappa_init, alpha,
                       beta, dt, this->internal_fields) {}

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void
  MaterialViscoElasticDamageSS2<DimM>::add_pixel(const size_t & pixel_index) {
    this->internal_fields->add_pixel(pixel_index);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void
  MaterialViscoElasticDamageSS2<DimM>::add_pixel(const size_t & pixel_index,
                                                 const Real & kappa_variation) {
    this->internal_fields->add_pixel(pixel_index);
    this->get_kappa_field().get_state_field().current().push_back(
        this->material_child.get_kappa_init() + kappa_variation);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialViscoElasticDamageSS2<DimM>::save_history_variables() {
    this->get_history_integral().get_state_field().cycle();
    this->get_s_null_prev_field().get_state_field().cycle();
    this->get_kappa_field().get_state_field().cycle();
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialViscoElasticDamageSS2<DimM>::initialise() {
    if (not this->is_initialised_flag) {
      Parent::initialise();
      this->get_history_integral().get_map().get_current() =
          Eigen::Matrix<Real, DimM, DimM>::Identity();
      this->get_s_null_prev_field().get_map().get_current() =
          Eigen::Matrix<Real, DimM, DimM>::Identity();
      this->save_history_variables();
    }
  }

  /* ---------------------------------------------------------------------- */

  template class MaterialViscoElasticDamageSS2<twoD>;
  template class MaterialViscoElasticDamageSS2<threeD>;

}  // namespace muSpectre
