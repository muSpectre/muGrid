/**
 * @file   material_dunant_tc.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   13 Jul 2020
 *
 * @brief  implementation of Material DunantTC
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

#include "material_dunant_tc.hh"

namespace muSpectre {

  template <Index_t DimM>
  MaterialDunantTC<DimM>::MaterialDunantTC(
      const std::string & name, const Index_t & spatial_dimension,
      const Index_t & nb_quad_pts, const Real & young, const Real & poisson,
      const Real & kappa_init, const Real & alpha, const Real & rho_c,
      const Real & rho_t,
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
        alpha{alpha}, rho_c{rho_c}, rho_t{rho_t} {}
  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialDunantTC<DimM>::save_history_variables() {
    this->get_kappa_field().get_state_field().cycle();
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialDunantTC<DimM>::initialise() {
    if (not this->is_initialised_flag) {
      Parent::initialise();
      this->save_history_variables();
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  auto MaterialDunantTC<DimM>::compute_M2(const T2_t & E)
      -> std::tuple<T2_t, T2_t> {
    auto && spectral_decomp_strain{muGrid::spectral_decomposition(E)};
    T2_t M2_t{T2_t::Zero()};
    for (Index_t i{0}; i < DimM; ++i) {
      if (spectral_decomp_strain.eigenvalues()[i] > 0) {
        M2_t += spectral_decomp_strain.eigenvectors().col(i) *
                spectral_decomp_strain.eigenvectors().col(i).transpose();
      }
    }
    T2_t M2_c{T2_t::Identity() - M2_t};
    return std::make_tuple(M2_c, M2_t);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  auto MaterialDunantTC<DimM>::compute_dM2_dE(const T2_t & E)
      -> std::tuple<T4_t, T4_t, bool> {
    bool varying_M{true};
    auto && spectral_decomp_strain{muGrid::spectral_decomposition(E)};
    Vec_t eigen_vals{spectral_decomp_strain.eigenvalues()};
    T2_t eigen_vecs{spectral_decomp_strain.eigenvectors()};
    switch (DimM) {
    case twoD: {
      // considering that we know that eigenvalues are in an **ascending order**
      //  otherwise this algorithm might be invalid.
      if (eigen_vals[0] * eigen_vals[1] >= 0.0) {
        // (*I*) the eigenvalues have same sign => the masking tensor is
        // constant
        varying_M = false;
        return std::make_tuple(T4_t::Zero(), T4_t::Zero(), varying_M);
      } else {
        T4_t dM2_t_dE{T4_t::Zero()};
        // (*II*) ⇒ the first eigen value is (-) and the 2nd one is (+)
        Vec_t q_0{eigen_vecs.col(0)};  //!< eig. vec. of neg. eig. val.
        Vec_t q_1{eigen_vecs.col(1)};  //!< eig. vec. of pos. eig. val.
        for (Index_t k{0}; k < DimM; ++k) {
          for (Index_t l{0}; l < DimM; ++l) {
            for (Index_t n{0}; n < DimM; ++n) {
              for (Index_t p{0}; p < DimM; ++p) {
                get(dM2_t_dE, k, l, n, p) +=
                    ((q_0[n] * q_1[p]) * (q_0[k] * q_1[l] + q_1[k] * q_0[l])) /
                    (eigen_vals[1] - eigen_vals[0]);
              }
            }
          }
        }
        return std::make_tuple(-dM2_t_dE, dM2_t_dE, varying_M);
      }
      break;
    }
    case threeD: {
      // considering that we know that eigenvalues are in an **ascending order**
      if (eigen_vals[0] < 0 and eigen_vals[1] < 0 and eigen_vals[2] > 0) {
        // (*I*). q₁, q₀<0 and q₂>0
        T4_t dM2_t_dE{T4_t::Zero()};
        Vec_t q_0{eigen_vecs.col(0)};  //!< eig. vec. of neg. eig. val.
        Vec_t q_1{eigen_vecs.col(1)};  //!< eig. vec. of neg. eig. val.
        Vec_t q_2{eigen_vecs.col(2)};  //!< eig. vec. of pos. eig. val.
        for (Index_t k{0}; k < DimM; ++k) {
          for (Index_t l{0}; l < DimM; ++l) {
            for (Index_t n{0}; n < DimM; ++n) {
              for (Index_t p{0}; p < DimM; ++p) {
                get(dM2_t_dE, k, l, n, p) +=
                    (((q_0[n] * q_2[p]) * (q_0[k] * q_2[l] + q_2[k] * q_0[l])) /
                     (eigen_vals[2] - eigen_vals[0])) +
                    (((q_1[n] * q_2[p]) * (q_1[k] * q_2[l] + q_2[k] * q_1[l])) /
                     (eigen_vals[2] - eigen_vals[1]));
              }
            }
          }
        }
        return std::make_tuple(-dM2_t_dE, dM2_t_dE, varying_M);
      } else if (eigen_vals[0] < 0 and eigen_vals[1] > 0 and
                 eigen_vals[2] > 0) {
        // (*II*). q₀<0 and q₁, q₂>0
        T4_t dM2_c_dE{T4_t::Zero()};
        Vec_t q_0{eigen_vecs.col(0)};  //!< eig. vec. of neg. eig. val.
        Vec_t q_1{eigen_vecs.col(1)};  //!< eig. vec. of pos. eig. val.
        Vec_t q_2{eigen_vecs.col(2)};  //!< eig. vec. of pos. eig. val.
        for (Index_t k{0}; k < DimM; ++k) {
          for (Index_t l{0}; l < DimM; ++l) {
            for (Index_t n{0}; n < DimM; ++n) {
              for (Index_t p{0}; p < DimM; ++p) {
                get(dM2_c_dE, k, l, n, p) +=
                    (((q_1[n] * q_0[p]) * (q_0[k] * q_1[l] + q_1[k] * q_0[l])) /
                     (eigen_vals[0] - eigen_vals[1])) +
                    (((q_2[n] * q_0[p]) * (q_0[k] * q_2[l] + q_2[k] * q_0[l])) /
                     (eigen_vals[0] - eigen_vals[2]));
              }
            }
          }
        }
        return std::make_tuple(dM2_c_dE, -dM2_c_dE, varying_M);
      } else {
        // (*III*). otherwise the masking tensors are constant
        varying_M = false;
        return std::make_tuple(T4_t::Zero(), T4_t::Zero(), varying_M);
      }
      break;
    }
    default: {
      throw MaterialError("Only 2D or 3D supported");
      break;
    }
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  auto MaterialDunantTC<DimM>::compute_dEct_dE(const T2_t & E, const T2_t & M2,
                                               const T4_t & dM2_dE,
                                               const bool & varying_M) -> T4_t {
    T4_t dEct_dE{Matrices::outer_under(M2, M2.transpose())};
    if (varying_M) {
      T2_t ME{M2 * E};
      T2_t EM{E * M2};
      for (Index_t a{0}; a < DimM; ++a) {
        for (Index_t b{0}; b < DimM; ++b) {
          for (Index_t d{0}; d < DimM; ++d) {
            for (Index_t n{0}; n < DimM; ++n) {
              for (Index_t p{0}; p < DimM; ++p) {
                get(dEct_dE, a, d, n, p) +=
                    (get(dM2_dE, a, b, n, p) * EM(b, d) +
                     ME(a, b) * get(dM2_dE, b, d, n, p));
              }
            }
          }
        }
      }
    }
    return dEct_dE;
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  auto MaterialDunantTC<DimM>::evaluate_stress(const T2_t & E,
                                               const size_t & quad_pt_index)
      -> T2_t {
    auto && kappa{this->get_kappa_field()[quad_pt_index]};
    auto && kappa_init{this->get_kappa_init_field()[quad_pt_index]};
    return this->evaluate_stress(std::move(E), kappa, kappa_init);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  auto MaterialDunantTC<DimM>::evaluate_stress_tangent(
      const T2_t & E, const size_t & quad_pt_index) -> std ::tuple<T2_t, T4_t> {
    auto && kappa{this->get_kappa_field()[quad_pt_index]};
    auto && kappa_init{this->get_kappa_init_field()[quad_pt_index]};
    return this->evaluate_stress_tangent(std::move(E), kappa, kappa_init);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  auto MaterialDunantTC<DimM>::evaluate_stress(const T2_t & E,
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
  auto MaterialDunantTC<DimM>::evaluate_stress_tangent(
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
      Real dr_dk{-1.0 * ((1.0 + this->alpha) * kappa_init_f) /
                 std::pow(kappa.current(), 2)};
      // computing the masking matrices
      T2_t M2_c{T2_t::Zero()};
      T2_t M2_t{T2_t::Zero()};
      std::tie(M2_c, M2_t) = this->compute_M2(E);
      T2_t E_c{M2_c * E * M2_c};
      T2_t E_t{M2_t * E * M2_t};

      // computing ∂M/∂ε
      T4_t dM2c_dE{T4_t::Zero()};
      T4_t dM2t_dE{T4_t::Zero()};
      bool varying_M{false};
      std::tie(dM2c_dE, dM2t_dE, varying_M) = this->compute_dM2_dE(E);

      // computing ∂εₜ/∂ε and ∂ε_c/∂ε
      T4_t dEc_dE{this->compute_dEct_dE(E, M2_c, dM2c_dE, varying_M)};
      T4_t dEt_dE{this->compute_dEct_dE(E, M2_t, dM2t_dE, varying_M)};

      /*
        κ = √[(ρᶜ εᶜ:ε + ρᵗ εᵗ:ε)/(ρᶜ + ρᵗ)] ⇒
                                ρᶜ * (∂εᶜ/∂ε::ε + εᶜ) + ρₜ * (∂εᵗ/∂ε::ε + εᵗ)
                    ∂κ/∂ε = —————————————————————————
                                      2 * κ * (ρᶜ + ρᵗ)
      */
      T2_t dk_dE{
          (this->rho_c * (Matrices::tensmult(dEc_dE.transpose(), E) + E_c) +
           this->rho_t * (Matrices::tensmult(dEt_dE.transpose(), E) + E_t)) /
          (2.0 * kappa.current() * (this->rho_c + this->rho_t))};

      T2_t drdE{dr_dk * dk_dE};  // ∂r/∂E = ∂r/∂K * ∂K/∂E

      /*
         σ = σ⁰ * r ⇒
         ∂σ/∂ε = ( C⁰ * r + σ⁰ ⊗ ∂r/∂ε)
      */
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
  auto MaterialDunantTC<DimM>::update_damage_measure(const T2_t & E,
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
    if (kappa_current > this->kappa_fin) {
      state = StepState::fully_damaged;
    }
    return state;
  }

  /* ---------------------------------------------------------------------- */

  template <Index_t DimM>
  void MaterialDunantTC<DimM>::add_pixel(const size_t & pixel_index) {
    this->internal_fields->add_pixel(pixel_index);
    this->get_kappa_field().get_state_field().current().push_back(
        this->get_kappa_init());
    this->get_kappa_init_field().get_field().push_back(this->get_kappa_init());
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialDunantTC<DimM>::add_pixel(const size_t & pixel_index,
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
  Real MaterialDunantTC<DimM>::compute_strain_measure(
      const Eigen::MatrixBase<Derived> & E) {
    // Different damage criteria can be considered here.
    // In this material, the considered damage criterion is the strain norm
    // measure = κ = ||E_weighted|| =   √((cᶜ * E : Eᶜ + cᵗ * E : Eᵗ)/(cᶜ + cᵗ))
    auto && spectral_decomp_strain{muGrid::spectral_decomposition(E)};
    Vec_t strain_eig_vals(spectral_decomp_strain.eigenvalues());
    Vec_t strain_eig_vals_c(Vec_t::Zero());
    Vec_t strain_eig_vals_t(Vec_t::Zero());

    for (Index_t i{0}; i < DimM; ++i) {
      if (strain_eig_vals(i) > 0.0) {
        strain_eig_vals_t(i) = strain_eig_vals(i);
      } else {
        strain_eig_vals_c(i) = strain_eig_vals(i);
      }
    }

    auto && E2_c{strain_eig_vals_c.dot(strain_eig_vals)};
    auto && E2_t{strain_eig_vals_t.dot(strain_eig_vals)};

    return sqrt((this->rho_c * E2_c + this->rho_t * E2_t) /
                (this->rho_c + this->rho_t));
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  Real MaterialDunantTC<DimM>::compute_reduction(const Real & kappa,
                                                 const Real & kappa_init_f) {
    auto && reduction_measure{((1.0 + this->alpha) * (kappa_init_f / kappa)) -
                              this->alpha};
    return reduction_measure * static_cast<int>(reduction_measure > 0.);
  }

  /* ----------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialDunantTC<DimM>::clear_last_step_nonlinear() {
    this->last_step_was_nonlinear = false;
  }

  /* ----------------------------------------------------------------------*/
  template class MaterialDunantTC<twoD>;
  template class MaterialDunantTC<threeD>;

}  // namespace muSpectre
