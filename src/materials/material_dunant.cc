/**
 * @file   material_dunant.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   13 Jul 2020
 *
 * @brief  implementation of Material Dunant
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

#include "material_dunant.hh"

namespace muSpectre {

  template <Index_t DimM>
  MaterialDunant<DimM>::MaterialDunant(
      const std::string & name, const Index_t & spatial_dimension,
      const Index_t & nb_quad_pts, const Real & young, const Real & poisson,
      const Real & kappa_init, const Real & alpha,
      const std::shared_ptr<muGrid::LocalFieldCollection> &
          parent_field_collection)
      : Parent{name, spatial_dimension, nb_quad_pts, parent_field_collection},
        material_child(name + "_child", spatial_dimension, nb_quad_pts, young,
                       poisson, this->internal_fields),
        kappa_init_field{this->get_prefix() + "kappa init",
                         *this->internal_fields, QuadPtTag},
        kappa_field{this->get_prefix() + "strain measure",
                    *this->internal_fields, QuadPtTag},
        kappa_init{kappa_init}, kappa_fin{(alpha > 0)
                                              ? kappa_init + kappa_init / alpha
                                              : 1e3 * kappa_init},
        alpha{alpha} {}
  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialDunant<DimM>::save_history_variables() {
    this->get_kappa_field().get_state_field().cycle();
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialDunant<DimM>::initialise() {
    if (not this->is_initialised_flag) {
      Parent::initialise();
      this->save_history_variables();
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  auto MaterialDunant<DimM>::evaluate_stress(const T2_t & E,
                                             const size_t & quad_pt_index)
      -> T2_t {
    auto && kappa{this->get_kappa_field()[quad_pt_index]};
    auto && kappa_init{this->get_kappa_init_field()[quad_pt_index]};
    return this->evaluate_stress(std::move(E), kappa, kappa_init);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  auto MaterialDunant<DimM>::evaluate_stress_tangent(
      const T2_t & E, const size_t & quad_pt_index) -> std ::tuple<T2_t, T4_t> {
    auto && kappa{this->get_kappa_field()[quad_pt_index]};
    auto && kappa_init{this->get_kappa_init_field()[quad_pt_index]};
    return this->evaluate_stress_tangent(std::move(E), kappa, kappa_init);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  auto MaterialDunant<DimM>::evaluate_stress(const T2_t & E,
                                             ScalarStRef_t kappa,
                                             const Real & kappa_init_f)
      -> T2_t {
    this->update_damage_measure(E, kappa);
    auto && reduction{this->compute_reduction(kappa.current(), kappa_init_f)};
    T2_t S{reduction * this->material_child.evaluate_stress(E, 0)};
    return S;
  }

  /* ----------------------------------------------------------------------*/
  template <Index_t DimM>
  auto MaterialDunant<DimM>::evaluate_stress_tangent(const T2_t & E,
                                                     ScalarStRef_t kappa,
                                                     const Real & kappa_init_f)
      -> std::tuple<T2_t, T4_t> {
    auto && step_status_current = this->update_damage_measure(E, kappa);
    auto && reduction{this->compute_reduction(kappa.current(), kappa_init_f)};
    auto && SC_pristine{this->material_child.evaluate_stress_tangent(E, 0)};
    auto && S_pristine{std::get<0>(SC_pristine)};
    auto && C_pristine{std::get<1>(SC_pristine)};
    T2_t S{reduction * S_pristine};
    switch (step_status_current) {
    case StepState::elastic: {
      T4_t C{reduction * C_pristine};
      return std::make_tuple(S, C);
      break;
    }
    case StepState::fully_damaged: {
      T4_t C{T4_t::Zero()};
      return std::make_tuple(S, C);
      break;
    }
    case StepState::damaging: {
      T2_t dk_dE{(1.0 / kappa.current()) * E};
      Real dr_dk{-1.0 * ((1.0 + this->alpha) * this->kappa_init) /
                 std::pow(kappa.current(), 2)};
      T2_t drdE{dr_dk * dk_dE};
      T4_t C{reduction * C_pristine + Matrices::outer(drdE, S_pristine)};
      return std::make_tuple(S, C);
      break;
    }
    default:
      std::stringstream error_message{};
      error_message << "Undefined step status!!!"
                    << "\n";
      throw MaterialError{error_message.str()};
      break;
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  auto MaterialDunant<DimM>::update_damage_measure(const T2_t & E,
                                                   ScalarStRef_t kappa)
      -> StepState {
    StepState state{StepState::elastic};
    auto && kappa_current{this->compute_strain_measure(E)};
    if (kappa_current > kappa.old()) {
      kappa.current() = kappa_current;
      state = StepState::damaging;
      this->last_step_was_nonlinear |= kappa_current <= this->kappa_fin;
    } else {
      kappa.current() = kappa.old();
    }
    if (kappa_current > kappa_fin) {
      state = StepState::fully_damaged;
    }
    return state;
  }

  /* ---------------------------------------------------------------------- */

  template <Index_t DimM>
  void MaterialDunant<DimM>::add_pixel(const size_t & pixel_index) {
    this->internal_fields->add_pixel(pixel_index);
    this->get_kappa_field().get_state_field().current().push_back(
        this->get_kappa_init());
    this->get_kappa_init_field().get_field().push_back(this->get_kappa_init());
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialDunant<DimM>::add_pixel(const size_t & pixel_index,
                                       const Real & kappa_variation) {
    this->internal_fields->add_pixel(pixel_index);
    this->get_kappa_field().get_state_field().current().push_back(
        this->get_kappa_init() + kappa_variation);
    this->get_kappa_init_field().get_field().push_back(this->get_kappa_init() +
                                                       kappa_variation);
  }

  /* --------------------------------------------------------------------- */
  template <Index_t DimM>
  template <class Derived>
  Real MaterialDunant<DimM>::compute_strain_measure(
      const Eigen::MatrixBase<Derived> & E) {
    // Different damage criteria can be considered here.
    // In this material, the considered damage criterion is the strain norm
    // measure = || E || =  √(E : E) →
    auto && k{sqrt(muGrid::Matrices::ddot<DimM>(E, E))};
    return k;
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  Real MaterialDunant<DimM>::compute_reduction(const Real & kappa,
                                               const Real & kappa_init_f) {
    auto && reduction_measure{((1.0 + this->alpha) * (kappa_init_f / kappa)) -
                              this->alpha};
    return reduction_measure * static_cast<int>(reduction_measure > 0.);
  }

  /* ----------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialDunant<DimM>::clear_last_step_nonlinear() {
    this->last_step_was_nonlinear = false;
  }

  /* ----------------------------------------------------------------------*/
  template class MaterialDunant<twoD>;
  template class MaterialDunant<threeD>;

}  // namespace muSpectre
