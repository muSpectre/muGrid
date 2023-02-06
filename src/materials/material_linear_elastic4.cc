/**
 * @file   material_linear_elastic4.cc
 *
 * @author Richard Leute <richard.leute@imtek.uni-freiburg.de
 *
 * @date   15 March 2018
 *
 * @brief linear elastic material with distribution of stiffness properties.
 *        In difference to material_linear_elastic3 two Lame constants are
 *        stored per pixel instead of the whole elastic matrix C.
 *        Uses the MaterialMuSpectre facilities to keep it simple.
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

#include "material_linear_elastic4.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  MaterialLinearElastic4<DimM>::MaterialLinearElastic4(
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
  void MaterialLinearElastic4<DimM>::add_pixel(const size_t & /*pixel*/) {
    throw muGrid::RuntimeError(
        "This material needs pixels with Youngs modulus and Poisson ratio.");
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialLinearElastic4<DimM>::add_pixel(const size_t & pixel,
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
  void MaterialLinearElastic4<DimM>::add_pixel(
      const size_t & pixel_id,
      const Eigen::Ref<const Eigen::Matrix<Real, Eigen::Dynamic, 1>> &
          Youngs_modulus,
      const Eigen::Ref<const Eigen::Matrix<Real, Eigen::Dynamic, 1>> &
          Poisson_ratio) {
    auto & nb_sub_pts(this->lambda_field.get_field().get_nb_sub_pts());
    if (Youngs_modulus.rows() != nb_sub_pts) {
      std::stringstream error{};
      error << "Got a wrong shape " << std::to_string(Youngs_modulus.rows())
            << "×" << std::to_string(Youngs_modulus.cols())
            << " for the Youngs modulus vector.\nI expected the shape: "
            << std::to_string(this->lambda_field.get_field().get_nb_sub_pts())
            << "×"
            << "1";
      throw MaterialError(error.str());
    }
    if (Poisson_ratio.rows() != nb_sub_pts) {
      std::stringstream error{};
      error << "Got a wrong shape " << std::to_string(Poisson_ratio.rows())
            << "×" << std::to_string(Poisson_ratio.cols())
            << " for the Poisson ratio vector.\nI expected the shape: "
            << std::to_string(this->lambda_field.get_field().get_nb_sub_pts())
            << "×"
            << "1";
      throw MaterialError(error.str());
    }

    this->internal_fields->add_pixel(pixel_id);

    for (Index_t i{0}; i < nb_sub_pts; i++) {
      Real lambda{Hooke::compute_lambda(Youngs_modulus(i), Poisson_ratio(i))};
      Real mu{Hooke::compute_mu(Youngs_modulus(i), Poisson_ratio(i))};
      this->lambda_field.get_field().push_back_single(lambda);
      this->mu_field.get_field().push_back_single(mu);
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialLinearElastic4<DimM>::set_youngs_modulus(
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
  void MaterialLinearElastic4<DimM>::set_poisson_ratio(
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
  void MaterialLinearElastic4<DimM>::set_youngs_modulus_and_poisson_ratio(
       const size_t & quad_pt_id, const Real & Youngs_modulus,
       const Real & Poisson_ratio) {
    auto && lambda_map{this->lambda_field.get_map()};
    auto && mu_map{this->mu_field.get_map()};

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
  MaterialLinearElastic4<DimM>::get_youngs_modulus(const size_t & quad_pt_id) {
    auto && lambda_map{this->lambda_field.get_map()};
    auto && mu_map{this->mu_field.get_map()};

    // compute poisson from first and second lame constant (lambda and mu)
    const Real & lambda = lambda_map[quad_pt_id];
    const Real & mu = mu_map[quad_pt_id];

    auto E = Hooke::compute_young(lambda, mu);

    return E;
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  Real
  MaterialLinearElastic4<DimM>::get_poisson_ratio(const size_t & quad_pt_id) {
    auto && lambda_map{this->lambda_field.get_map()};
    auto && mu_map{this->mu_field.get_map()};

    // compute poisson from first and second lame constant (lambda and mu)
    const Real & lambda = lambda_map[quad_pt_id];
    const Real & mu = mu_map[quad_pt_id];

    return Hooke::compute_poisson(lambda, mu);
  }

  template class MaterialLinearElastic4<twoD>;
  template class MaterialLinearElastic4<threeD>;

}  // namespace muSpectre
