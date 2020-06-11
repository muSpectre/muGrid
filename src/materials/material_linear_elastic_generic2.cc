/**
 * @file   material_linear_elastic_generic2.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   20 Dec 2018
 *
 * @brief  Implementation for generic linear elastic law with eigenstrains
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

#include "material_linear_elastic_generic2.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  MaterialLinearElasticGeneric2<DimM>::MaterialLinearElasticGeneric2(
      const std::string & name, const Index_t & spatial_dimension,
      const Index_t & nb_quad_pts, const CInput_t & C_voigt)
      : Parent{name, spatial_dimension, nb_quad_pts}, worker{name,
                                                             spatial_dimension,
                                                             nb_quad_pts,
                                                             C_voigt},
        eigen_field{this->get_prefix() + "Eigenstrain",
                    *this->internal_fields, QuadPtTag} {
    this->last_step_was_nonlinear = false;
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void
  MaterialLinearElasticGeneric2<DimM>::add_pixel(const size_t & /*pixel*/) {
    throw muGrid::RuntimeError(
        "this material needs pixels with and eigenstrain");
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void
  MaterialLinearElasticGeneric2<DimM>::add_pixel(const size_t & pixel_index,
                                                 const StrainTensor & E_eig) {
    this->internal_fields->add_pixel(pixel_index);
    Eigen::Map<const Eigen::Array<Real, DimM * DimM, 1>> strain_array(
        E_eig.data());
    this->eigen_field.get_field().push_back(strain_array);
  }

  template class MaterialLinearElasticGeneric2<twoD>;
  template class MaterialLinearElasticGeneric2<threeD>;

}  // namespace muSpectre
