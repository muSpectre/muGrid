/**
 * @file   material_linear_elastic1.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   14 Nov 2017
 *
 * @brief  Implementation for materiallinearelastic1
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
 * General Public License for more details.
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
 */

#include "materials/material_linear_elastic1.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  MaterialLinearElastic1<DimS, DimM>::MaterialLinearElastic1(std::string name,
                                                           Real young,
                                                           Real poisson)
    :Parent(name), young{young}, poisson{poisson},
     lambda{Hooke::compute_lambda(young, poisson)},
     mu{Hooke::compute_mu(young, poisson)},
     C{Hooke::compute_C(lambda, mu)}
  {}

  template class MaterialLinearElastic1<twoD, twoD>;
  template class MaterialLinearElastic1<twoD, threeD>;
  template class MaterialLinearElastic1<threeD, threeD>;

}  // muSpectre
