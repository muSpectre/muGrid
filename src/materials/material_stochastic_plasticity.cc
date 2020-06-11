/**
 * @file   material_stochastic_plasticity.cc
 *
 * @author Richard Leute <richard.leute@imtek.uni-freiburg.de>
 *
 * @date   24 Jan 2019
 *
 * @brief  material for stochastic plasticity as described in Z. Budrikis et al.
 *         Nature Comm. 8:15928, 2017. It only works together with "python
 *         -script", which performes the avalanche loop. This makes the material
 *         slower but more easy to modify and test.
 *         (copied from material_linear_elastic4.cc)
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser
 * General Public License for more details.
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

#include "materials/material_stochastic_plasticity.hh"

#include <sstream>

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  MaterialStochasticPlasticity<DimM>::MaterialStochasticPlasticity(
      const std::string & name, const Index_t & spatial_dimension,
      const Index_t & nb_quad_pts)
      : Parent{name, spatial_dimension, nb_quad_pts},
        lambda_field{this->get_prefix() + "local first Lame constant",
                     *this->internal_fields, QuadPtTag},
        mu_field{this->get_prefix() +
                     "local second Lame constant(shear modulus)",
                 *this->internal_fields, QuadPtTag},
        plastic_increment_field{this->get_prefix() + "plastic increment",
                                *this->internal_fields, QuadPtTag},
        stress_threshold_field{this->get_prefix() + "threshold",
                               *this->internal_fields, QuadPtTag},
        eigen_strain_field{this->get_prefix() + "eigen strain",
                           *this->internal_fields, QuadPtTag} {}

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialStochasticPlasticity<DimM>::add_pixel(const size_t & /*pixel*/) {
    throw muGrid::RuntimeError(
        "This material needs pixels with Youngs modulus and Poisson ratio.");
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialStochasticPlasticity<DimM>::add_pixel(
      const size_t & pixel, const Real & Young_modulus,
      const Real & Poisson_ratio, const Real & plastic_increment,
      const Real & stress_threshold,
      const Eigen::Ref<const Eigen::Matrix<Real, Eigen::Dynamic,
                                           Eigen::Dynamic>> & eigen_strain) {
    // check if the users input eigen strain has the right dimension
    if (eigen_strain.cols() != DimM || eigen_strain.rows() != DimM) {
      std::stringstream error{};
      error << "Got a wrong shape " << std::to_string(eigen_strain.rows())
            << "×" << std::to_string(eigen_strain.cols())
            << " for the eigen strain matrix.\nI expected the shape: "
            << std::to_string(DimM) << "×" << std::to_string(DimM);
      throw muGrid::RuntimeError(error.str());
    }
    this->internal_fields->add_pixel(pixel);
    // store the first(lambda) and second(mu) Lame constant in the field
    Real lambda = Hooke::compute_lambda(Young_modulus, Poisson_ratio);
    Real mu = Hooke::compute_mu(Young_modulus, Poisson_ratio);
    this->lambda_field.get_field().push_back(lambda);
    this->mu_field.get_field().push_back(mu);
    this->plastic_increment_field.get_field().push_back(plastic_increment);
    this->stress_threshold_field.get_field().push_back(stress_threshold);
    const Eigen::Map<const Eigen::Array<Real, DimM * DimM, 1>> strain_map(
        eigen_strain.data());
    this->eigen_strain_field.get_field().push_back(strain_map);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialStochasticPlasticity<DimM>::set_plastic_increment(
      const size_t & quad_pt_id, const Real & increment) {
    auto && plastic_increment_map{this->plastic_increment_field.get_map()};
    plastic_increment_map[quad_pt_id] = increment;
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialStochasticPlasticity<DimM>::set_stress_threshold(
      const size_t & quad_pt_id, const Real & threshold) {
    auto && stress_threshold_map{this->stress_threshold_field.get_map()};
    stress_threshold_map[quad_pt_id] = threshold;
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialStochasticPlasticity<DimM>::set_eigen_strain(
      const size_t & quad_pt_id,
      Eigen::Ref<Eigen::Matrix<Real, DimM, DimM>> & eigen_strain) {
    auto && eigen_strain_map{this->eigen_strain_field.get_map()};
    eigen_strain_map[quad_pt_id] = eigen_strain;
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  const Real & MaterialStochasticPlasticity<DimM>::get_plastic_increment(
      const size_t & quad_pt_id) {
    auto && plastic_increment_map{this->plastic_increment_field.get_map()};
    return plastic_increment_map[quad_pt_id];
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  const Real & MaterialStochasticPlasticity<DimM>::get_stress_threshold(
      const size_t & quad_pt_id) {
    auto && stress_threshold_map{this->stress_threshold_field.get_map()};
    return stress_threshold_map[quad_pt_id];
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  const Eigen::Ref<Eigen::Matrix<Real, DimM, DimM>>
  MaterialStochasticPlasticity<DimM>::get_eigen_strain(
      const size_t & quad_pt_id) {
    auto && eigen_strain_map{this->eigen_strain_field.get_map()};
    return eigen_strain_map[quad_pt_id];
  }

  template <Index_t DimM>
  void MaterialStochasticPlasticity<DimM>::reset_overloaded_quad_pts() {
    this->overloaded_quad_pts.clear();
  }

  template class MaterialStochasticPlasticity<twoD>;
  template class MaterialStochasticPlasticity<threeD>;

}  // namespace muSpectre
