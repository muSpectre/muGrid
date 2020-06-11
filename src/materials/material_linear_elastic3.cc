/**
 * @file   material_linear_elastic3.cc
 *
 * @author Richard Leute <richard.leute@imtek.uni-freiburg.de
 *
 * @date   20 Feb 2018
 *
 * @brief  implementation for linear elastic material with distribution of
 * stiffness properties. Uses the MaterialMuSpectre facilities to keep it
 * simple.
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
 * * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 *
 */

#include "materials/material_linear_elastic3.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  MaterialLinearElastic3<DimM>::MaterialLinearElastic3(
      const std::string & name, const Index_t & spatial_dimension,
      const Index_t & nb_quad_pts)
      : Parent{name, spatial_dimension, nb_quad_pts},
        C_field{this->get_prefix() + "local stiffness tensor",
                *this->internal_fields, QuadPtTag} {
    this->last_step_was_nonlinear = false;
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialLinearElastic3<DimM>::add_pixel(const size_t & /*pixel*/) {
    throw muGrid::RuntimeError(
        "this material needs pixels with Youngs modulus and Poisson ratio.");
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialLinearElastic3<DimM>::add_pixel(const size_t & pixel,
                                               const Real & Young,
                                               const Real & Poisson) {
    this->internal_fields->add_pixel(pixel);
    Real lambda = Hooke::compute_lambda(Young, Poisson);
    Real mu = Hooke::compute_mu(Young, Poisson);
    auto C_tensor = Hooke::compute_C(lambda, mu);
    Eigen::Map<const Eigen::Array<Real, DimM * DimM * DimM * DimM, 1>> C(
        C_tensor.data());
    this->C_field.get_field().push_back(C);
  }

  template class MaterialLinearElastic3<twoD>;
  template class MaterialLinearElastic3<threeD>;

}  // namespace muSpectre
