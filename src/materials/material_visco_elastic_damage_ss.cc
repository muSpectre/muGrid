/**
 * @file material_visco_elastic_damage_ss.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   20 Dec 2019
 *
 * @brief  The implementation of the methods of the MaterialViscoElasticDamageSS
 *
 * Copyright © 2019 Ali Falsafi
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

#include "materials/material_visco_elastic_damage_ss.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  MaterialViscoElasticDamageSS<DimM>::MaterialViscoElasticDamageSS(
      const std::string & name, const Index_t & spatial_dimension,
      const Index_t & nb_quad_pts, const Real & young_inf, const Real & young_v,
      const Real & eta_v, const Real & poisson, const Real & kappa_init,
      const Real & alpha, const Real & beta, const Real & dt)
    : Parent{name, spatial_dimension, nb_quad_pts},
        material_child(name + "_child", spatial_dimension, nb_quad_pts,
                       young_inf, young_v, eta_v, poisson, dt,
                       this->internal_fields),
        kappa_prev_field{this->get_prefix() + "strain measure",
                         *this->internal_fields, QuadPtTag},
        kappa_init{kappa_init}, alpha{alpha}, beta{beta} {}

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialViscoElasticDamageSS<DimM>::save_history_variables() {
    this->get_history_integral().get_state_field().cycle();
    this->get_s_null_prev_field().get_state_field().cycle();
    this->get_kappa_prev_field().get_state_field().cycle();
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialViscoElasticDamageSS<DimM>::initialise() {
    Parent::initialise();
    this->get_history_integral().get_map().get_current() =
        Eigen::Matrix<Real, DimM, DimM>::Identity();
    this->get_s_null_prev_field().get_map().get_current() =
        Eigen::Matrix<Real, DimM, DimM>::Identity();
    this->kappa_prev_field.get_map().get_current() = this->kappa_init;
    this->save_history_variables();
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  auto MaterialViscoElasticDamageSS<DimM>::evaluate_stress(
      const Eigen::Ref<const T2_t> & E, T2StRef_t h_prev, T2StRef_t s_null_prev,
      ScalarStRef_t kappa_prev) -> T2_t {
    this->update_damage_measure(E, kappa_prev);
    auto && damage{this->compute_damage_measure(kappa_prev.current())};
    auto && S{damage *
              this->material_child.evaluate_stress(E, h_prev, s_null_prev)};
    return S;
  }

  /* ----------------------------------------------------------------------*/
  template <Index_t DimM>
  auto MaterialViscoElasticDamageSS<DimM>::evaluate_stress_tangent(
      const Eigen::Ref<const T2_t> & E, T2StRef_t h_prev, T2StRef_t s_null_prev,
      ScalarStRef_t kappa_prev) -> std::tuple<T2_t, T4_t> {
    this->update_damage_measure(E, kappa_prev);
    auto && damage{this->compute_damage_measure(kappa_prev.current())};
    auto && SC_pristine{
        this->material_child.evaluate_stress_tangent(E, h_prev, s_null_prev)};
    auto && S{damage * std::get<0>(SC_pristine)};
    auto && C{damage * std::get<1>(SC_pristine)};
    return std::make_tuple(S, C);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialViscoElasticDamageSS<DimM>::update_damage_measure(
      const Eigen::Ref<const T2_t> & E, ScalarStRef_t kappa_prev) {
    auto kappa_current{this->compute_strain_measure(E)};
    auto kappa_old{kappa_prev.old()};

    if (kappa_current > kappa_old) {
      kappa_prev.current() = kappa_current;
    } else {
      kappa_prev.current() = kappa_prev.old();
    }
    kappa_current = kappa_prev.current();
    kappa_old = kappa_prev.old();
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  template <class Derived>
  Real MaterialViscoElasticDamageSS<DimM>::compute_strain_measure(
      const Eigen::MatrixBase<Derived> & E) {
    auto && elastic_stress{this->material_child.evaluate_elastic_stress(E)};
    auto && measure{sqrt(::muGrid::Matrices::ddot<DimM>(
        std::move(elastic_stress), std::move(E)))};
    return measure;
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  Real MaterialViscoElasticDamageSS<DimM>::compute_damage_measure(
      const Real & kappa) {
    return this->beta +
           (1.0 - this->beta) *
               ((1.0 - std::exp(-kappa / this->alpha)) / (kappa / this->alpha));
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  muGrid::MappedT2StateField<Real, Mapping::Mut, DimM, IterUnit::SubPt> &
  MaterialViscoElasticDamageSS<DimM>::get_history_integral() {
    return this->material_child.get_history_integral();
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  muGrid::MappedT2StateField<Real, Mapping::Mut, DimM, IterUnit::SubPt> &
  MaterialViscoElasticDamageSS<DimM>::get_s_null_prev_field() {
    return this->material_child.get_s_null_prev_field();
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  muGrid::MappedScalarStateField<Real, Mapping::Mut, IterUnit::SubPt> &
  MaterialViscoElasticDamageSS<DimM>::get_kappa_prev_field() {
    return this->kappa_prev_field;
  }

  /* ----------------------------------------------------------------------*/
  template class MaterialViscoElasticDamageSS<twoD>;
  template class MaterialViscoElasticDamageSS<threeD>;

}  // namespace muSpectre
