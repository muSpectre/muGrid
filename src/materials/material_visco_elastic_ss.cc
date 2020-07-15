/**
 * @file   material_visco_elastic_ss.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   20 Dec 2019
 *
 * @brief  The implementation of the methods of the
 *  MaterialViscoElasticSS
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

#include "materials/material_visco_elastic_ss.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  MaterialViscoElasticSS<DimM>::MaterialViscoElasticSS(
      const std::string & name, const Index_t & spatial_dimension,
      const Index_t & nb_quad_pts, const Real & young_inf, const Real & young_v,
      const Real & eta_v, const Real & poisson, const Real & dt,
      const std::shared_ptr<muGrid::LocalFieldCollection> &
          parent_field_collection)
      : Parent{name, spatial_dimension, nb_quad_pts, parent_field_collection},
        s_null_prev_field{this->get_prefix() + "Pure elastic stress",
                          *this->internal_fields, QuadPtTag},
        h_prev_field{this->get_prefix() + "history integral",
                     *this->internal_fields, QuadPtTag},
        young_inf{young_inf}, young_v{young_v}, eta_v{eta_v}, poisson{poisson},
        lambda_inf{Hooke::compute_lambda(young_inf, poisson)},
        mu_inf{Hooke::compute_mu(young_inf, poisson)}, K_inf{Hooke::compute_K(
                                                           young_inf, poisson)},
        lambda_v{Hooke::compute_lambda(young_v, poisson)},
        mu_v{Hooke::compute_mu(young_v, poisson)}, K_v{Hooke::compute_K(
                                                       young_v, poisson)},
        tau_v{eta_v / young_v}, young_tot{young_v + young_inf},
        K_tot{Hooke::compute_K(young_tot, poisson)}, mu_tot{Hooke::compute_mu(
                                                         young_tot, poisson)},
        lambda_tot{Hooke::compute_lambda(young_tot, poisson)},
        gamma_inf{young_inf / young_tot}, gamma_v{young_v / young_tot}, dt{dt} {
    if (not(this->dt > 0.0)) {
      throw std::runtime_error(
          "The time step must be set to a strictly positive value.");
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialViscoElasticSS<DimM>::save_history_variables() {
    this->h_prev_field.get_state_field().cycle();
    this->s_null_prev_field.get_state_field().cycle();
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialViscoElasticSS<DimM>::initialise() {
    Parent::initialise();
    this->h_prev_field.get_map().get_current() =
        Eigen::Matrix<Real, DimM, DimM>::Identity();
    this->s_null_prev_field.get_map().get_current() =
        Eigen::Matrix<Real, DimM, DimM>::Identity();
    this->save_history_variables();
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  auto MaterialViscoElasticSS<DimM>::evaluate_stress(
      const Eigen::Ref<const T2_t> & E, T2StRef_t h_prev, T2StRef_t s_null_prev)
      -> T2_t {
    auto && e{MatTB::compute_deviatoric<DimM>(E)};
    auto && s_null{this->evaluate_elastic_deviatoric_stress(e)};
    auto && h{std::exp(-dt / this->tau_v) * h_prev.old() +
              std::exp(-dt / (2 * this->tau_v)) * (s_null - s_null_prev.old())};
    h_prev.current() = h;
    s_null_prev.current() = s_null;
    return this->evaluate_elastic_volumetric_stress(E) + gamma_inf * s_null +
           gamma_v * h;
  }

  /* ----------------------------------------------------------------------*/
  template <Index_t DimM>
  auto MaterialViscoElasticSS<DimM>::evaluate_stress_tangent(
      const Eigen::Ref<const T2_t> & F, T2StRef_t h_prev, T2StRef_t s_null_prev)
      -> std::tuple<T2_t, T4_t> {
    // using auto && gives wrong results (probably memory issue)
    T4_t Iasymm{Matrices::Isymm<DimM>() -
                (1.0 / 3.0) * Matrices::Itrac<DimM>()};
    auto && C_null_bar{2 * this->mu_tot * Iasymm};
    auto && g_star{gamma_inf + gamma_v * std::exp(-dt / (2 * this->tau_v))};
    // using auto && gives wrong results (probably memory issue) for both of
    // these two places that I could not use auto &&, a constexpr variable
    // form muGrid::Matrices namespace is used and that might be the source of
    // the problem
    T4_t C{this->lambda_tot * Matrices::Itrac<DimM>() + g_star * C_null_bar};
    return std::make_tuple(this->evaluate_stress(F, h_prev, s_null_prev), C);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  muGrid::MappedT2StateField<Real, Mapping::Mut, DimM, IterUnit::SubPt> &
  MaterialViscoElasticSS<DimM>::get_history_integral() {
    return this->h_prev_field;
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  muGrid::MappedT2StateField<Real, Mapping::Mut, DimM, IterUnit::SubPt> &
  MaterialViscoElasticSS<DimM>::get_s_null_prev_field() {
    return this->s_null_prev_field;
  }

  /* ----------------------------------------------------------------------*/
  template class MaterialViscoElasticSS<twoD>;
  template class MaterialViscoElasticSS<threeD>;

}  // namespace muSpectre
