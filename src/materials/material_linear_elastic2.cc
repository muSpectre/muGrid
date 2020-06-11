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

#include "materials/material_linear_elastic2.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  MaterialLinearElastic2<DimM>::MaterialLinearElastic2(
      const std::string & name, const Index_t & spatial_dimension,
      const Index_t & nb_quad_pts, Real young, Real poisson)
      : Parent{name, spatial_dimension, nb_quad_pts},
        material{name, spatial_dimension, nb_quad_pts, young, poisson},
        eigen_strains{this->get_prefix() + "Eigenstrain",
                      *this->internal_fields, QuadPtTag} {
    this->last_step_was_nonlinear = false;
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialLinearElastic2<DimM>::add_pixel(const size_t & /*pixel_index*/) {
    throw std::runtime_error("this material needs pixels with an eigenstrain");
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void MaterialLinearElastic2<DimM>::add_pixel(const size_t & pixel_index,
                                               const StrainTensor & E_eig) {
    this->internal_fields->add_pixel(pixel_index);
    Eigen::Map<const Eigen::Array<Real, DimM * DimM, 1>> strain_array(
        E_eig.data());
    this->eigen_strains.get_field().push_back(strain_array);
  }

  template class MaterialLinearElastic2<twoD>;
  template class MaterialLinearElastic2<threeD>;

}  // namespace muSpectre
