/**
 * @file   material_hyper_elastic2.cc
 *
 * @author Indre Joedicke <indre.joedicke@imtek.uni-freiburg.de>
 *
 * @date   19 Oct 2021
 *
 * @brief Hyper elastic material with distribution of stiffness properties.
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

#include "material_hyper_elastic2.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  MaterialHyperElastic2<DimM>::MaterialHyperElastic2(
      const std::string & name, const Index_t & spatial_dimension,
      const Index_t & nb_quad_pts)
      : Parent{name, spatial_dimension, nb_quad_pts},
        lambda_field{this->get_prefix() + "local first Lame constant",
                     *this->internal_fields, QuadPtTag},
        mu_field(this->get_prefix() +
                     "local second Lame constant(shear modulus)",
                 *this->internal_fields, QuadPtTag) {
    this->last_step_was_nonlinear = false;
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialHyperElastic2<DimM>::add_pixel(const size_t & /*pixel*/) {
    throw muGrid::RuntimeError(
        "This material needs pixels with Youngs modulus and Poisson ratio.");
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialHyperElastic2<DimM>::add_pixel(const size_t & pixel,
                                               const Real & Young_modulus,
                                               const Real & Poisson_ratio) {
    this->internal_fields->add_pixel(pixel);
    // store the first(lambda) and second(mu) Lame constant in the field
    Real lambda = Hooke::compute_lambda(Young_modulus, Poisson_ratio);
    Real mu = Hooke::compute_mu(Young_modulus, Poisson_ratio);
    this->lambda_field.get_field().push_back(lambda);
    this->mu_field.get_field().push_back(mu);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialHyperElastic2<DimM>::set_youngs_modulus(
      const size_t & quad_pt_id, const Real & Youngs_modulus) {
    auto && lambda_map{this->lambda_field.get_map()};
    auto && mu_map{this->mu_field.get_map()};

    // compute poisson from first and second lame constant (lambda and mu)
    const Real & lambda_old = lambda_map[quad_pt_id];
    const Real & mu_old = mu_map[quad_pt_id];
    const Real Poisson_ratio = Hooke::compute_poisson(lambda_old, mu_old);

    // compute updated first and second lame constant (lambda and mu)
    const Real lambda_new =
        Hooke::compute_lambda(Youngs_modulus, Poisson_ratio);
    const Real mu_new = Hooke::compute_mu(Youngs_modulus, Poisson_ratio);

    // assign new values to fields
    lambda_map[quad_pt_id] = lambda_new;
    mu_map[quad_pt_id] = mu_new;
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialHyperElastic2<DimM>::set_poisson_ratio(
      const size_t & quad_pt_id, const Real & Poisson_ratio) {
    auto && lambda_map{this->lambda_field.get_map()};
    auto && mu_map{this->mu_field.get_map()};

    // compute young from first and second lame constant (lambda and mu)
    const Real & lambda_old = lambda_map[quad_pt_id];
    const Real & mu_old = mu_map[quad_pt_id];
    const Real Youngs_modulus = Hooke::compute_young(lambda_old, mu_old);

    // compute updated first and second lame constant (lambda and mu)
    const Real lambda_new =
        Hooke::compute_lambda(Youngs_modulus, Poisson_ratio);
    const Real mu_new = Hooke::compute_mu(Youngs_modulus, Poisson_ratio);

    // assign new values to fields
    lambda_map[quad_pt_id] = lambda_new;
    mu_map[quad_pt_id] = mu_new;
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  Real
  MaterialHyperElastic2<DimM>::get_youngs_modulus(const size_t & quad_pt_id) {
    auto && lambda_map{this->lambda_field.get_map()};
    auto && mu_map{this->mu_field.get_map()};

    // compute poisson from first and second lame constant (lambda and mu)
    const Real & lambda = lambda_map[quad_pt_id];
    const Real & mu = mu_map[quad_pt_id];

    return Hooke::compute_young(lambda, mu);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  Real
  MaterialHyperElastic2<DimM>::get_poisson_ratio(const size_t & quad_pt_id) {
    auto && lambda_map{this->lambda_field.get_map()};
    auto && mu_map{this->mu_field.get_map()};

    // compute poisson from first and second lame constant (lambda and mu)
    const Real & lambda = lambda_map[quad_pt_id];
    const Real & mu = mu_map[quad_pt_id];

    return Hooke::compute_poisson(lambda, mu);
  }

  template class MaterialHyperElastic2<twoD>;
  template class MaterialHyperElastic2<threeD>;

}  // namespace muSpectre
