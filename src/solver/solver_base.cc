/**
 * file   solver_base.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   24 Apr 2018
 *
 * @brief  implementation of SolverBase
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

#include "solver/solver_base.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  SolverBase::SolverBase(Cell & cell, Real tol, Uint maxiter, bool verbose):
    cell(cell), tol{tol}, maxiter{maxiter}, verbose{verbose}
  {}

  /* ---------------------------------------------------------------------- */
  bool SolverBase::has_converged() const {
    return this->converged;
  }

  /* ---------------------------------------------------------------------- */
  void SolverBase::reset_counter() {
    this->counter = 0;
    this->converged = false;
  }

  /* ---------------------------------------------------------------------- */
  Uint SolverBase::get_counter() const {
    return this->counter;
  }

  /* ---------------------------------------------------------------------- */
  Real SolverBase::get_tol() const {
    return this->tol;
  }

  /* ---------------------------------------------------------------------- */
  Uint SolverBase::get_maxiter() const {
    return this->maxiter;
  }

}  // muSpectre
