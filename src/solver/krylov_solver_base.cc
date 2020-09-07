/**
 * @file   krylov_solver_base.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   24 Apr 2018
 *
 * @brief  implementation of KrylovSolverBase
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

#include "solver/krylov_solver_base.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  KrylovSolverBase::KrylovSolverBase(Cell & cell, Real tol, Uint maxiter,
                                     Verbosity verbose)
      : cell(cell), tol{tol}, maxiter{maxiter}, verbose{verbose} {}

  /* ---------------------------------------------------------------------- */
  KrylovSolverBase::Convergence KrylovSolverBase::get_convergence() const {
    return this->convergence;
  }

  /* ---------------------------------------------------------------------- */
  void KrylovSolverBase::reset_counter() {
    this->counter = 0;
    this->convergence = Convergence::DidNotConverge;
  }

  /* ---------------------------------------------------------------------- */
  void KrylovSolverBase::set_trust_region(Real new_trust_region) {
    std::stringstream s;
    s << "Setting a trust region is not supported by the " << this->get_name()
      << " solver. (The desired trust region value was " << new_trust_region
      << ".)";
    throw SolverError(s.str());
  }

  /* ---------------------------------------------------------------------- */
  Uint KrylovSolverBase::get_counter() const { return this->counter; }

  /* ---------------------------------------------------------------------- */
  Real KrylovSolverBase::get_tol() const { return this->tol; }

  /* ---------------------------------------------------------------------- */
  Uint KrylovSolverBase::get_maxiter() const { return this->maxiter; }

}  // namespace muSpectre
