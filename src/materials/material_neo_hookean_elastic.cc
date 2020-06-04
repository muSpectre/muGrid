/**
 * @file   material_neo_hookean_elastic.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   27 Feb 2020
 *
 * @brief  Implementation of material Neo-Hookean
 *
 * Copyright © 2020 Ali Falsafi
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

#include "materials/material_neo_hookean_elastic.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  MaterialNeoHookeanElastic<DimM>::MaterialNeoHookeanElastic(
      const std::string & name, const Index_t & spatial_dimension,
      const Index_t & nb_quad_pts, const Real & young, const Real & poisson)
      : Parent{name, spatial_dimension, nb_quad_pts}, young{young},
        poisson{poisson}, lambda{Hooke::compute_lambda(young, poisson)},
        mu{Hooke::compute_mu(young, poisson)}, K{Hooke::compute_K(young,
                                                                  poisson)},
        C_linear_holder{
            std::make_unique<Stiffness_t>(Hooke::compute_C_T4(lambda, mu))},
        C_linear{*C_linear_holder} {}

  /* ---------------------------------------------------------------------- */
  template class MaterialNeoHookeanElastic<twoD>;
  template class MaterialNeoHookeanElastic<threeD>;

}  // namespace muSpectre
