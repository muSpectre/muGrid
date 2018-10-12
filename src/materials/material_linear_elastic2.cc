/**
 * @file   material_linear_elastic2.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   04 Feb 2018
 *
 * @brief  implementation for linear elastic material with eigenstrain
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
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#include "materials/material_linear_elastic2.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  MaterialLinearElastic2<DimS, DimM>::
  MaterialLinearElastic2(std::string name, Real young, Real poisson)
    :Parent{name}, material{name, young, poisson},
     eigen_field{make_field<Field_t>("Eigenstrain", this->internal_fields)},
     internal_variables(eigen_field.get_const_map())
    {}

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void MaterialLinearElastic2<DimS, DimM>::
  add_pixel(const Ccoord_t<DimS> & /*pixel*/) {
    throw std::runtime_error
      ("this material needs pixels with and eigenstrain");
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void MaterialLinearElastic2<DimS, DimM>::
  add_pixel(const Ccoord_t<DimS> & pixel,
            const StrainTensor & E_eig) {
    this->internal_fields.add_pixel(pixel);
    Eigen::Map<const Eigen::Array<Real, DimM*DimM, 1>> strain_array(E_eig.data());
    this->eigen_field.push_back(strain_array);
  }

  template class MaterialLinearElastic2<twoD, twoD>;
  template class MaterialLinearElastic2<twoD, threeD>;
  template class MaterialLinearElastic2<threeD, threeD>;

}  // muSpectre
