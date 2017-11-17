/**
 * file   material_hyperelastic1.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   14 Nov 2017
 *
 * @brief  Implementation for materialhyperelastic1
 *
 * @section LICENCE
 *
 * Copyright (C) 2017 Till Junge
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

#include "materials/material_hyperelastic1.hh"
#include "common/tensor_algebra.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  Materialhyperelastic1<DimS, DimM>::MaterialHyperElastic1(std::string name,
                                                           Real young,
                                                           Real poisson)
    :young{young}, poisson{poisson},
     lambda{young*poisson/((1+poisson)*(1-2*poisson))},
     mu{young/(2*(1+poisson))},
     C{lambda*Tensors::outer<DimM>(Tensors::I2<DimM>(),Tensors::I2<DimM>()) +
        2*mu*Tensors::I4S<DimM>()}
  {}

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  MaterialHyperElastic1<DimS, DimM>::evaluate_stress(const Strain_t & E,
                                                    Stress_t & S) {
    S = E.trace()*lambda * Strain_t::Identity() + 2*mu*E;
  }

}  // muSpectre
