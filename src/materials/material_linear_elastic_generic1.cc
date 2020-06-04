/**
 * @file   material_linear_elastic_generic1.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   21 Sep 2018
 *
 * @brief  implementation for MaterialLinearElasticGeneric
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

#include "materials/material_linear_elastic_generic1.hh"
#include "common/voigt_conversion.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  MaterialLinearElasticGeneric1<DimM>::MaterialLinearElasticGeneric1(
      const std::string & name, const Index_t & spatial_dimension,
      const Index_t & nb_quad_pts, const CInput_t & C_voigt)
      : Parent{name, spatial_dimension, nb_quad_pts},
        C_holder{std::make_unique<muGrid::T4Mat<Real, DimM>>()},
        C{*this->C_holder} {
    MatTB::make_C_from_C_voigt<DimM>(C_voigt, *this->C_holder);
    this->last_step_was_nonlinear = false;
  }

  /* ---------------------------------------------------------------------- */
  template class MaterialLinearElasticGeneric1<twoD>;
  template class MaterialLinearElasticGeneric1<threeD>;

}  // namespace muSpectre
