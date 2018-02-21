/**
 * @file   material_linear_elastic3.cc
 *
 * @author Richard Leute <richard.leute@imtek.uni-freiburg.de
 *
 * @date   20 Feb 2018
 *
 * @brief  implementation for linear elastic material with distribution of stiffness properties.
 *        Uses the MaterialMuSpectre facilities to keep it simple.
 *
 * Copyright © 2018 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Emacs; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#include "materials/material_linear_elastic3.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  MaterialLinearElastic3<DimS, DimM>::
  MaterialLinearElastic3(std::string name)
    :Parent{name},
     C_field{make_field<Field_t>("local stiffness tensor", this->internal_fields)},
     internal_variables(C_field.get_const_map())
    {}

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void MaterialLinearElastic3<DimS, DimM>::
  add_pixel(const Ccoord_t<DimS> & /*pixel*/) {
    throw std::runtime_error
      ("this material needs pixels with Youngs modulus and Poisson ratio.");
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void MaterialLinearElastic3<DimS, DimM>::
  add_pixel(const Ccoord_t<DimS> & pixel,
            const Real & Young, const Real & Poisson) {
    this->internal_fields.add_pixel(pixel);
    Real lambda = Hooke::compute_lambda(Young, Poisson);
    Real mu     = Hooke::compute_mu(Young, Poisson);
    auto C_tensor = Hooke::compute_C(lambda, mu);
    Eigen::Map<const Eigen::Array<Real, DimM*DimM*DimM*DimM, 1>> C(C_tensor.data());
    this->C_field.push_back(C);
  }

  template class MaterialLinearElastic3<twoD, twoD>;
  template class MaterialLinearElastic3<twoD, threeD>;
  template class MaterialLinearElastic3<threeD, threeD>;

}  // muSpectre
