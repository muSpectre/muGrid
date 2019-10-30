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
  template <Dim_t DimS, Dim_t DimM>
  MaterialStochasticPlasticity<DimS, DimM>::MaterialStochasticPlasticity(
      std::string name)
      : Parent{name}, lambda_field{this->internal_fields,
                                   "local first Lame constant"},
        mu_field{this->internal_fields,
                 "local second Lame constant(shear modulus)"},
        plastic_increment_field{this->internal_fields, "plastic increment"},
        stress_threshold_field{this->internal_fields, "threshold"},
        eigen_strain_field{this->internal_fields, "eigen strain"},
        overloaded_pixels{std::vector<Ccoord_t<DimS>>(0)} {}

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void MaterialStochasticPlasticity<DimS, DimM>::
  add_pixel(const Ccoord_t<DimS> & /*pixel*/) {
    throw std::runtime_error
      ("This material needs pixels with Youngs modulus and Poisson ratio.");
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void MaterialStochasticPlasticity<DimS, DimM>::add_pixel(
      const Ccoord_t<DimS> & pixel, const Real & Young_modulus,
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
      throw std::runtime_error(error.str());
      }
    this->internal_fields.add_pixel(pixel);
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
  template <Dim_t DimS, Dim_t DimM>
  void MaterialStochasticPlasticity<DimS, DimM>::set_plastic_increment(
      const Ccoord_t<DimS> pixel, const Real increment) {
    auto && plastic_increment_map{this->plastic_increment_field.get_map()};
    plastic_increment_map[pixel] = increment;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void MaterialStochasticPlasticity<DimS, DimM>::set_stress_threshold(
      const Ccoord_t<DimS> pixel, const Real threshold) {
    auto && stress_threshold_map{this->stress_threshold_field.get_map()};
    stress_threshold_map[pixel] = threshold;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void MaterialStochasticPlasticity<DimS, DimM>::set_eigen_strain(
      const Ccoord_t<DimS> pixel,
      Eigen::Ref<Eigen::Matrix<Real, DimM, DimM>> & eigen_strain) {
    auto && eigen_strain_map{this->eigen_strain_field.get_map()};
    eigen_strain_map[pixel] = eigen_strain;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  const Real & MaterialStochasticPlasticity<DimS, DimM>::get_plastic_increment(
      const Ccoord_t<DimS> pixel) {
    auto && plastic_increment_map{this->plastic_increment_field.get_map()};
    return plastic_increment_map[pixel];
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  const Real & MaterialStochasticPlasticity<DimS, DimM>::get_stress_threshold(
      const Ccoord_t<DimS> pixel) {
    auto && stress_threshold_map{this->stress_threshold_field.get_map()};
    return stress_threshold_map[pixel];
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  const Eigen::Ref<Eigen::Matrix<Real, DimM, DimM>>
  MaterialStochasticPlasticity<DimS, DimM>::get_eigen_strain(
      const Ccoord_t<DimS> pixel) {
    auto && eigen_strain_map{this->eigen_strain_field.get_map()};
    return eigen_strain_map[pixel];
  }

  template <Dim_t DimS, Dim_t DimM>
  void MaterialStochasticPlasticity<DimS, DimM>::reset_overloaded_pixels() {
    this->overloaded_pixels.clear();
  }

  template class MaterialStochasticPlasticity<twoD, twoD>;
  template class MaterialStochasticPlasticity<twoD, threeD>;
  template class MaterialStochasticPlasticity<threeD, threeD>;

}  // namespace muSpectre
