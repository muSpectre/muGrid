/**
 * @file   material_dunant_max.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   13 Jul 2020
 *
 * @brief  implementation of Material DunantT
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

#include "material_dunant_max.hh"

namespace muSpectre {

  template <Index_t DimM>
  MaterialDunantMax<DimM>::MaterialDunantMax(
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
  void MaterialDunantMax<DimM>::save_history_variables() {
    this->get_kappa_field().get_state_field().cycle();
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialDunantMax<DimM>::initialise() {
    if (not this->is_initialised_flag) {
      Parent::initialise();
      this->save_history_variables();
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  auto MaterialDunantMax<DimM>::evaluate_stress(const T2_t & E,
                                                const size_t & quad_pt_index)
      -> T2_t {
    auto && kappa{this->get_kappa_field()[quad_pt_index]};
    auto && kappa_init{this->get_kappa_init_field()[quad_pt_index]};
    return this->evaluate_stress(std::move(E), kappa, kappa_init);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  auto MaterialDunantMax<DimM>::evaluate_stress_tangent(
      const T2_t & E, const size_t & quad_pt_index) -> std ::tuple<T2_t, T4_t> {
    auto && kappa{this->get_kappa_field()[quad_pt_index]};
    auto && kappa_init{this->get_kappa_init_field()[quad_pt_index]};
    return this->evaluate_stress_tangent(std::move(E), kappa, kappa_init);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  auto MaterialDunantMax<DimM>::evaluate_stress(const T2_t & E,
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
  auto MaterialDunantMax<DimM>::evaluate_stress_tangent(
      const T2_t & E, ScalarStRef_t kappa, const Real & kappa_init_f)
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
      auto && spectral_decomp_strain{muGrid::spectral_decomposition(E)};
      Vec_t eigen_vals{spectral_decomp_strain.eigenvalues()};
      T2_t eigen_vecs{spectral_decomp_strain.eigenvectors()};

      Vec_t q_max{eigen_vecs.col(DimM - 1)};  // last eig. vector
      auto && dk_dE{q_max * q_max.transpose()};  // ∂K/∂E = qₘₐₓ ⊗ qₘₐₓ
      Real dr_dk{-1.0 * ((1.0 + this->alpha) * this->kappa_init) /
                 std::pow(kappa.current(), 2)};
      T2_t drdE{dr_dk * dk_dE};  // ∂r/∂E = ∂r/∂K * ∂K/∂E
      T4_t C{reduction * C_pristine + Matrices::outer(S_pristine, drdE)};
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
  auto MaterialDunantMax<DimM>::update_damage_measure(const T2_t & E,
                                                      ScalarStRef_t kappa)
      -> StepState {
    StepState state{StepState::elastic};
    auto && kappa_current{this->compute_strain_measure(E)};
    // Further damage only happens if the maximum eigen value is positive
    if (kappa_current > kappa.old() and kappa_current > 0.0) {
      kappa.current() = kappa_current;
      state = StepState::damaging;
      this->last_step_was_nonlinear |= kappa_current <= this->kappa_fin;
    } else {
      kappa.current() = kappa.old();
    }
    if (kappa_current > this->kappa_fin) {
      state = StepState::fully_damaged;
    }
    return state;
  }

  /* ---------------------------------------------------------------------- */

  template <Index_t DimM>
  void MaterialDunantMax<DimM>::add_pixel(const size_t & pixel_index) {
    this->internal_fields->add_pixel(pixel_index);
    this->get_kappa_field().get_state_field().current().push_back(
        this->get_kappa_init());
    this->get_kappa_init_field().get_field().push_back(this->get_kappa_init());
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialDunantMax<DimM>::add_pixel(const size_t & pixel_index,
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
  Real MaterialDunantMax<DimM>::compute_strain_measure(
      const Eigen::MatrixBase<Derived> & E) {
    // Different damage criteria can be considered here.
    // In this material, the considered damage criterion is the strain norm
    // measure = max λᵢ  → (λᵢ: eigen values(E))
    auto && spectral_decomp_strain{muGrid::spectral_decomposition(E)};
    Vec_t strain_eig_vals(spectral_decomp_strain.eigenvalues());
    Real lambda_max{strain_eig_vals(DimM - 1)};
    auto && k{lambda_max};
    return k;
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  Real MaterialDunantMax<DimM>::compute_reduction(const Real & kappa,
                                                  const Real & kappa_init_f) {
    auto && reduction_measure{((1.0 + this->alpha) * (kappa_init_f / kappa)) -
                              this->alpha};
    return reduction_measure * static_cast<int>(reduction_measure > 0.);
  }

  /* ----------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialDunantMax<DimM>::clear_last_step_nonlinear() {
    this->last_step_was_nonlinear = false;
  }

  /* ----------------------------------------------------------------------*/
  template class MaterialDunantMax<twoD>;
  template class MaterialDunantMax<threeD>;

}  // namespace muSpectre
