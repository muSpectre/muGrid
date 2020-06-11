/**
 * @file   material_hyper_elasto_plastic1.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   21 Feb 2018
 *
 * @brief  implementation for MaterialHyperElastoPlastic1
 *
 * Copyright © 2018 Till Junge
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

#include "common/muSpectre_common.hh"
#include "materials/stress_transformations_Kirchhoff.hh"
#include "materials/material_hyper_elasto_plastic1.hh"

#include <libmugrid/T4_map_proxy.hh>

namespace muSpectre {

  //--------------------------------------------------------------------------//
  template <Index_t DimM>
  MaterialHyperElastoPlastic1<DimM>::MaterialHyperElastoPlastic1(
      const std::string & name, const Index_t & spatial_dimension,
      const Index_t & nb_quad_pts, const Real & young, const Real & poisson,
      const Real & tau_y0, const Real & H,
      const std::shared_ptr<muGrid::LocalFieldCollection> &
          parent_field_collection)
      : Parent{name, spatial_dimension, nb_quad_pts,
      parent_field_collection},
        plast_flow_field{this->get_prefix() + "cumulated plastic flow εₚ",
                         *this->internal_fields, QuadPtTag},
        F_prev_field{this->get_prefix() + "Previous placement gradient Fᵗ",
                     *this->internal_fields, QuadPtTag},
        be_prev_field{this->get_prefix() +
                          "Previous left Cauchy-Green deformation bₑᵗ",
                      *this->internal_fields, QuadPtTag},
        young{young}, poisson{poisson}, lambda{Hooke::compute_lambda(young,
                                                                     poisson)},
        mu{Hooke::compute_mu(young, poisson)},
        K{Hooke::compute_K(young, poisson)}, tau_y0{tau_y0}, H{H},
        // the factor .5 comes from equation (18) in Geers 2003
        // (https://doi.org/10.1016/j.cma.2003.07.014)
        C_holder{std::make_unique<const muGrid::T4Mat<Real, DimM>>(
            0.5 * Hooke::compute_C_T4(lambda, mu))},
        C{*this->C_holder} {}

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialHyperElastoPlastic1<DimM>::save_history_variables() {
    this->plast_flow_field.get_state_field().cycle();
    this->F_prev_field.get_state_field().cycle();
    this->be_prev_field.get_state_field().cycle();
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialHyperElastoPlastic1<DimM>::initialise() {
    Parent::initialise();
    this->F_prev_field.get_map().get_current() =
        Eigen::Matrix<Real, DimM, DimM>::Identity();
    this->be_prev_field.get_map().get_current() =
        Eigen::Matrix<Real, DimM, DimM>::Identity();
    this->save_history_variables();
  }
  //--------------------------------------------------------------------------//
  template <Index_t DimM>
  auto MaterialHyperElastoPlastic1<DimM>::stress_n_internals_worker(
      const T2_t & F, T2StRef_t & F_prev, T2StRef_t & be_prev,
      ScalarStRef_t & eps_p, const Real & lambda, const Real & mu,
      const Real & tau_y0, const Real & H) -> Worker_t {
    // the notation in this function follows Geers 2003
    // (https://doi.org/10.1016/j.cma.2003.07.014).

    // computation of trial state
    using Mat_t = Eigen::Matrix<Real, DimM, DimM>;
    Mat_t f{F * F_prev.old().inverse()};
    // trial elastic left Cauchy–Green deformation tensor
    Mat_t be_star{f * be_prev.old() * f.transpose()};
    const muGrid::SelfAdjointDecomp_t<DimM> spectral_decomp{
        muGrid::spectral_decomposition(be_star)};
    Mat_t ln_be_star{muGrid::logm_alt(spectral_decomp)};
    Mat_t tau_star{.5 * Hooke::evaluate_stress(lambda, mu, ln_be_star)};
    // deviatoric part of Kirchhoff stress
    Mat_t tau_d_star{tau_star - tau_star.trace() / DimM * tau_star.Identity()};
    Real tau_eq_star{std::sqrt(
        3 * .5 * (tau_d_star.array() * tau_d_star.transpose().array()).sum())};
    // tau_eq_star can only be zero if tau_d_star is identically zero,
    // so the following is not an approximation;
    Real division_safe_tau_eq_star{tau_eq_star + Real(tau_eq_star == 0.)};
    Mat_t N_star{3 * .5 * tau_d_star / division_safe_tau_eq_star};
    // this is eq (27), and the std::max enforces the Kuhn-Tucker relation (16)
    Real phi_star{std::max(tau_eq_star - tau_y0 - H * eps_p.old(), 0.)};

    // return mapping
    Real Del_gamma{phi_star / (H + 3 * mu)};
    Mat_t tau{tau_star - 2 * Del_gamma * mu * N_star};

    // update the previous values to the new ones
    F_prev.current() = F;
    be_prev.current() = muGrid::expm(ln_be_star - 2 * Del_gamma * N_star);
    eps_p.current() = eps_p.old() + Del_gamma;

    // transmit info whether this is a plastic step or not
    bool is_plastic{phi_star > 0};
    return Worker_t(std::move(tau), std::move(tau_eq_star),
                    std::move(Del_gamma), std::move(N_star),
                    std::move(is_plastic), spectral_decomp);
  }

  //--------------------------------------------------------------------------//
  template <Index_t DimM>
  auto MaterialHyperElastoPlastic1<DimM>::evaluate_stress(
      const T2_t & F, T2StRef_t F_prev, T2StRef_t be_prev, ScalarStRef_t eps_p,
      const Real & lambda, const Real & mu, const Real & tau_y0, const Real & H)
      -> T2_t {
    Eigen::Matrix<Real, DimM, DimM> tau{};
    std::tie(tau, std::ignore, std::ignore, std::ignore, std::ignore,
             std::ignore) =
        this->stress_n_internals_worker(F, F_prev, be_prev, eps_p, lambda, mu,
                                        tau_y0, H);

    return tau;
  }

  //--------------------------------------------------------------------------//
  template <Index_t DimM>
  auto MaterialHyperElastoPlastic1<DimM>::evaluate_stress_tangent(
      const T2_t & F, T2StRef_t F_prev, T2StRef_t be_prev, ScalarStRef_t eps_p,
      const Real & lambda, const Real & mu, const Real & tau_y0, const Real & H,
      const Real & K, const Eigen::Ref<const muGrid::T4Mat<Real, DimM>> & C)
      -> std::tuple<T2_t, T4_t> {
    //! after the stress computation, all internals are up to date
    auto && vals{this->stress_n_internals_worker(F, F_prev, be_prev, eps_p,
                                                 lambda, mu, tau_y0, H)};
    auto & tau{std::get<0>(vals)};
    auto & tau_eq_star{std::get<1>(vals)};
    auto & Del_gamma{std::get<2>(vals)};
    auto & N_star{std::get<3>(vals)};
    auto & is_plastic{std::get<4>(vals)};
    auto & spec_decomp{std::get<5>(vals)};
    using Mat_t = Eigen::Matrix<Real, DimM, DimM>;
    using Vec_t = Eigen::Matrix<Real, DimM, 1>;
    using T4_t = muGrid::T4Mat<Real, DimM>;

    auto && a0 = Del_gamma * mu / tau_eq_star;
    auto && a1 = mu / (H + 3 * mu);
    T4_t mat_tangent{
        is_plastic ? ((K / 2. - mu / 3 + a0 * mu) * Matrices::Itrac<DimM>() +
                      (1 - 3 * a0) * mu * Matrices::Isymm<DimM>() +
                      2 * mu * (a0 - a1) * Matrices::outer(N_star, N_star))
                   : C};

    // compute derivative ∂ln(be_star)/∂be_star, see (77) through (80)
    T4_t dlnbe_dbe{T4_t::Zero()};
    {
      const Vec_t & eig_vals{spec_decomp.eigenvalues()};
      const Vec_t log_eig_vals{eig_vals.array().log().matrix()};
      const Mat_t & eig_vecs{spec_decomp.eigenvectors()};

      Mat_t g_vals{};
      // see (78), (79)
      for (int i{0}; i < DimM; ++i) {
        g_vals(i, i) = 1 / eig_vals(i);
        for (int j{i + 1}; j < DimM; ++j) {
          if (std::abs((eig_vals(i) - eig_vals(j)) / eig_vals(i)) < 1e-12) {
            g_vals(i, j) = g_vals(j, i) = g_vals(i, i);
          } else {
            g_vals(i, j) = g_vals(j, i) = ((log_eig_vals(j) - log_eig_vals(i)) /
                                           (eig_vals(j) - eig_vals(i)));
          }
        }
      }

      for (int i{0}; i < DimM; ++i) {
        for (int j{0}; j < DimM; ++j) {
          Mat_t dyad = eig_vecs.col(i) * eig_vecs.col(j).transpose();
          T4_t outerDyad = Matrices::outer(dyad, dyad.transpose());
          dlnbe_dbe += g_vals(i, j) * outerDyad;
        }
      }
    }

    // compute variation δbe_star
    T2_t I{Matrices::I2<DimM>()};
    // computation of trial state
    Mat_t f{F * F_prev.old().inverse()};
    // trial elastic left Cauchy–Green deformation tensor
    Mat_t be_star{f * be_prev.old() * f.transpose()};

    T4_t dbe_dF{Matrices::outer_under(I, be_star) +
                Matrices::outer_over(be_star, I)};

    // T4_t dtau_dbe{mat_tangent * dlnbe_dbe * dbe4s};
    T4_t dtau_dF{mat_tangent * dlnbe_dbe * dbe_dF};
    // return std::tuple<Mat_t, T4_t>(tau, dtau_dbe);
    return std::tuple<Mat_t, T4_t>(tau, dtau_dF);
  }

  template class MaterialHyperElastoPlastic1<twoD>;
  template class MaterialHyperElastoPlastic1<threeD>;
}  // namespace muSpectre
