/**
 * @file   material_hyper_elasto_plastic2.cc
 *
 * @author Richard Leute <richard.leute@imtek.uni-freiburg.de>
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   08 Apr 2020
 *
 * @brief  copy of material_hyper_elasto_plastic2 with the extension that
 * enables it to use functions and fields of a contained
 * material_hyper_elasto_plastic1 witout name collision
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
  template <Index_t DimM>
  MaterialHyperElastoPlastic2<DimM>::MaterialHyperElastoPlastic2(
      const std::string & name, const Index_t & spatial_dimension,
      const Index_t & nb_quad_pts)
      : Parent{name, spatial_dimension, nb_quad_pts},
        material_child(name + "_child", spatial_dimension, nb_quad_pts, 0.0,
                       0.0, 0.0, 0.0, this->internal_fields),
        lambda_field{this->get_prefix() + "local first Lame constant",
                     *this->internal_fields, QuadPtTag},
        mu_field(this->get_prefix() +
                     "local second Lame constant(shear modulus)",
                 *this->internal_fields, QuadPtTag),
        tau_y0_field{this->get_prefix() + "local initial yield stress",
                     *this->internal_fields, QuadPtTag},
        H_field{this->get_prefix() + "local hardening modulus",
                *this->internal_fields, QuadPtTag},
        K_field(this->get_prefix() + "local Bulk modulus",
                *this->internal_fields, QuadPtTag) {}

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialHyperElastoPlastic2<DimM>::save_history_variables() {
    this->material_child.save_history_variables();
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialHyperElastoPlastic2<DimM>::initialise() {
    this->material_child.initialise();
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void
  MaterialHyperElastoPlastic2<DimM>::add_pixel(const size_t & /*pixel_id*/) {
    throw muGrid::RuntimeError(
        "This material needs pixels with Young's modulus, Poisson's ratio, "
        "initial yield stress and hardening modulus.");
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialHyperElastoPlastic2<DimM>::add_pixel(const size_t & pixel_id,
                                                    const Real & Youngs_modulus,
                                                    const Real & Poisson_ratio,
                                                    const Real & tau_y0,
                                                    const Real & H) {
    // this->material_child.add_pixel(pixel_id);
    this->internal_fields->add_pixel(pixel_id);
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
  template <Index_t DimM>
  auto MaterialHyperElastoPlastic2<DimM>::evaluate_stress(
      const T2_t & F, PrevStrain_ref F_prev, PrevStrain_ref be_prev,
      FlowField_ref eps_p, const Real lambda, const Real mu, const Real tau_y0,
      const Real H) -> T2_t {
    return this->material_child.evaluate_stress(F, F_prev, be_prev, eps_p,
                                                lambda, mu, tau_y0, H);
  }

  //--------------------------------------------------------------------------//
  template <Index_t DimM>
  auto MaterialHyperElastoPlastic2<DimM>::evaluate_stress_tangent(
      const T2_t & F, PrevStrain_ref F_prev, PrevStrain_ref be_prev,
      FlowField_ref eps_p, const Real lambda, const Real mu, const Real tau_y0,
      const Real H, const Real K) -> std::tuple<T2_t, T4_t> {
    auto && C{T4_t{0.5 * Hooke::compute_C_T4(lambda, mu)}};
    return this->material_child.evaluate_stress_tangent(
        F, F_prev, be_prev, eps_p, lambda, mu, tau_y0, H, K, C);
  }

  template class MaterialHyperElastoPlastic2<twoD>;
  template class MaterialHyperElastoPlastic2<threeD>;
}  // namespace muSpectre
