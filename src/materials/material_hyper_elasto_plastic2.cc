/**
 * @file   material_hyper_elasto_plastic2.cc
 *
 * @author Richard Leute <richard.leute@imtek.uni-freiburg.de>
 *
 * @date   08 Jul 2019
 *
 * @brief  copy of material_hyper_elasto_plastic1 with Young, Poisson, yield
 *         criterion and  hardening modulus per pixel. As defined in de Geus
 *         2017 (https://doi.org/10.1016/j.cma.2016.12.032) and further
 *         explained in Geers 2003 (https://doi.org/10.1016/j.cma.2003.07.014).
 *
 * Copyright © 2019 Till Junge
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
#include "materials/material_hyper_elasto_plastic2.hh"

#include <libmugrid/T4_map_proxy.hh>

namespace muSpectre {

  //--------------------------------------------------------------------------//
  template <Dim_t DimS, Dim_t DimM>
  MaterialHyperElastoPlastic2<DimS, DimM>::MaterialHyperElastoPlastic2(
      std::string name)
      : Parent{name}, plast_flow_field{this->internal_fields,
                                       "cumulated plastic flow εₚ"},
        F_prev_field{this->internal_fields, "Previous placement gradient Fᵗ"},
        be_prev_field{this->internal_fields,
                      "Previous left Cauchy-Green deformation bₑᵗ"},
        lambda_field{this->internal_fields, "local first Lame constant"},
        mu_field(this->internal_fields,
                 "local second Lame constant(shear modulus)"),
        tau_y0_field{this->internal_fields, "local initial yield stress"},
        H_field{this->internal_fields, "local hardening modulus"},
        K_field(this->internal_fields, "local Bulk modulus") {}

  /* ----------------------------------------------------------------------
   */
  template <Dim_t DimS, Dim_t DimM>
  void MaterialHyperElastoPlastic2<DimS, DimM>::save_history_variables() {
    this->plast_flow_field.get_field().cycle();
    this->F_prev_field.get_field().cycle();
    this->be_prev_field.get_field().cycle();
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void MaterialHyperElastoPlastic2<DimS, DimM>::initialise() {
    Parent::initialise();
    this->F_prev_field.get_map().current() =
        Eigen::Matrix<Real, DimM, DimM>::Identity();
    this->be_prev_field.get_map().current() =
        Eigen::Matrix<Real, DimM, DimM>::Identity();
    this->save_history_variables();
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void MaterialHyperElastoPlastic2<DimS, DimM>::add_pixel(
      const Ccoord_t<DimS> & /*pixel*/) {
    throw std::runtime_error(
        "This material needs pixels with Young's modulus, Poisson's ratio, "
        "initial yield stress and hardening modulus.");
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void MaterialHyperElastoPlastic2<DimS, DimM>::add_pixel(
      const Ccoord_t<DimS> & pixel, const Real & Youngs_modulus,
      const Real & Poisson_ratio, const Real & tau_y0, const Real & H) {
    this->internal_fields.add_pixel(pixel);
    // store the first(lambda) and second(mu) Lame constant in the field
    Real lambda{Hooke::compute_lambda(Youngs_modulus, Poisson_ratio)};
    Real mu{Hooke::compute_mu(Youngs_modulus, Poisson_ratio)};
    this->lambda_field.get_field().push_back(lambda);
    this->mu_field.get_field().push_back(mu);
    this->tau_y0_field.get_field().push_back(tau_y0);
    this->H_field.get_field().push_back(H);
    Real K{Hooke::compute_K(Youngs_modulus, Poisson_ratio)};
    this->K_field.get_field().push_back(K);
  }

  //--------------------------------------------------------------------------//
  template <Dim_t DimS, Dim_t DimM>
  auto MaterialHyperElastoPlastic2<DimS, DimM>::stress_n_internals_worker(
      const T2_t & F, StrainStRef_t & F_prev, StrainStRef_t & be_prev,
      FlowStRef_t & eps_p, const Real lambda, const Real mu, const Real tau_y0,
      const Real H) -> Worker_t {
    // the notation in this function follows Geers 2003
    // (https://doi.org/10.1016/j.cma.2003.07.014).

    // computation of trial state
    using Mat_t = Eigen::Matrix<Real, DimM, DimM>;
    Mat_t f{F * F_prev.old().inverse()};
    // trial elastic left Cauchy–Green deformation tensor
    Mat_t be_star{f * be_prev.old() * f.transpose()};
    const muGrid::Decomp_t<DimM> spectral_decomp{
        muGrid::spectral_decomposition(be_star)};
    Mat_t ln_be_star{muGrid::logm_alt(spectral_decomp)};
    Mat_t tau_star{.5 *
                   Hooke::evaluate_stress(lambda, mu, ln_be_star)};
    // deviatoric part of Kirchhoff stress
    Mat_t tau_d_star{tau_star - tau_star.trace() / DimM * tau_star.Identity()};
    Real tau_eq_star{std::sqrt(
        3 * .5 * (tau_d_star.array() * tau_d_star.transpose().array()).sum())};
    // tau_eq_star can only be zero if tau_d_star is identically zero,
    // so the following is not an approximation;
    Real division_safe_tau_eq_star{tau_eq_star + Real(tau_eq_star == 0.)};
    Mat_t N_star{3 * .5 * tau_d_star / division_safe_tau_eq_star};
    // this is eq (27), and the std::max enforces the Kuhn-Tucker relation (16)
    Real phi_star{
        std::max(tau_eq_star - tau_y0 - H * eps_p.old(), 0.)};

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
  template <Dim_t DimS, Dim_t DimM>
  auto MaterialHyperElastoPlastic2<DimS, DimM>::evaluate_stress(
      const T2_t & F, StrainStRef_t F_prev, StrainStRef_t be_prev,
      FlowStRef_t eps_p, const Real lambda, const Real mu, const Real tau_y0,
      const Real H) -> T2_t {
    Eigen::Matrix<Real, DimM, DimM> tau;
    std::tie(tau, std::ignore, std::ignore, std::ignore, std::ignore,
             std::ignore) =
        this->stress_n_internals_worker(F, F_prev, be_prev, eps_p, lambda, mu,
                                        tau_y0, H);

    return tau;
  }

  //--------------------------------------------------------------------------//
  template <Dim_t DimS, Dim_t DimM>
  auto MaterialHyperElastoPlastic2<DimS, DimM>::evaluate_stress_tangent(
      const T2_t & F, StrainStRef_t F_prev, StrainStRef_t be_prev,
      FlowStRef_t eps_p, const Real lambda, const Real mu, const Real tau_y0,
      const Real H, const Real K) -> std::tuple<T2_t, T4_t> {
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

    // compute stiffness tensor
    // the factor .5 comes from equation (18) in Geers 2003
    // (https://doi.org/10.1016/j.cma.2003.07.014)
    auto && a0 = is_plastic ? Del_gamma * mu / tau_eq_star : 1./3;
    auto && a1 = is_plastic ? mu / (H + 3 * mu) : 0;
    T4_t mat_tangent{is_plastic ?
                     ((K / 2. - mu / 3 + a0 * mu) *
                         Matrices::Itrac<DimM>() +
                      (1 - 3 * a0) * mu * Matrices::Isymm<DimM>() +
                      2 * mu * (a0 - a1) * Matrices::outer(N_star, N_star))
                     : T4_t{0.5 * Hooke::compute_C_T4(lambda, mu)}};

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

  template class MaterialHyperElastoPlastic2<twoD, twoD>;
  template class MaterialHyperElastoPlastic2<twoD, threeD>;
  template class MaterialHyperElastoPlastic2<threeD, threeD>;
}  // namespace muSpectre