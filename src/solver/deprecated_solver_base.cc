/**
 * @file   deprecated_solver_base.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   18 Dec 2017
 *
 * @brief  definitions for solvers
 *
 * Copyright © 2017 Till Junge
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

#include "solver/deprecated_solver_base.hh"
#include "solver/deprecated_solver_cg.hh"
#include "common/field.hh"
#include "common/iterators.hh"

#include <iostream>
#include <memory>


namespace muSpectre {

  //----------------------------------------------------------------------------//
  template <Dim_t DimS, Dim_t DimM>
  DeprecatedSolverBase<DimS, DimM>::DeprecatedSolverBase(Cell_t & cell, Real tol, Uint maxiter,
                                     bool verbose )
    : cell{cell}, tol{tol}, maxiter{maxiter}, verbose{verbose}
  {}


  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void DeprecatedSolverBase<DimS, DimM>::reset_counter() {
    this->counter = 0;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  Uint DeprecatedSolverBase<DimS, DimM>::get_counter() const {
    return this->counter;
  }

  template class DeprecatedSolverBase<twoD, twoD>;
  //template class DeprecatedSolverBase<twoD, threeD>;
  template class DeprecatedSolverBase<threeD, threeD>;

}  // muSpectre
