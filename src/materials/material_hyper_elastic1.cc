/**
 * @file   material_hyper_elastic1.cc
 *
 * @author Indre Joedicke <indre.joedicke@imtek.uni-freiburg.de>
 *
 * @date   18 Oct 2021
 *
 * @brief  Implementation for materialhyperelastic1
 *
 * Copyright © 2017 Till Junge
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

#include "materials/material_hyper_elastic1.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  MaterialHyperElastic1<DimM>::MaterialHyperElastic1(
      const std::string & name, const Index_t & spatial_dimension,
      const Index_t & nb_quad_pts, const Real & young, const Real & poisson,
      const std::shared_ptr<muGrid::LocalFieldCollection> &
          parent_field_collection)
      : Parent{name, spatial_dimension, nb_quad_pts, parent_field_collection},
        young{young}, poisson{poisson}, lambda{Hooke::compute_lambda(young,
                                                                     poisson)},
        mu{Hooke::compute_mu(young, poisson)},
        C_holder{
            std::make_unique<Stiffness_t>(Hooke::compute_C_T4(lambda, mu))},
        C{*C_holder} {
    this->last_step_was_nonlinear = false;
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  auto MaterialHyperElastic1<DimM>::get_C() const -> const Stiffness_t & {
    return this->C;
  }
  template class MaterialHyperElastic1<twoD>;
  template class MaterialHyperElastic1<threeD>;

}  // namespace muSpectre
