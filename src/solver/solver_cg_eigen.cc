/**
 * @file   solver_cg_eigen.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   19 Jan 2018
 *
 * @brief  implementation for binding to Eigen's conjugate gradient solver
 *
 * Copyright (C) 2018 Till Junge
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

#include "solver_cg_eigen.hh"

#include <sstream>
#include <iomanip>

namespace muSpectre {

  //----------------------------------------------------------------------------//
  template <class SolverType, Dim_t DimS, Dim_t DimM>
  SolverEigen<SolverType, DimS, DimM>::SolverEigen(Cell_t& cell, Real tol, Uint maxiter,
                                                   bool verbose)
    :Parent(cell, tol, maxiter, verbose),
     adaptor{cell.get_adaptor()},
     solver{}
  {}

  //----------------------------------------------------------------------------//
  template <class SolverType, Dim_t DimS, Dim_t DimM>
  void
  SolverEigen<SolverType, DimS, DimM>::initialise() {
    this->solver.setTolerance(this->tol);
    this->solver.setMaxIterations(this->maxiter);
    this->solver.compute(this->adaptor);
  }

  //----------------------------------------------------------------------------//
  template <class SolverType, Dim_t DimS, Dim_t DimM>
  typename SolverEigen<SolverType, DimS, DimM>::SolvVectorOut
  SolverEigen<SolverType, DimS, DimM>::solve(const SolvVectorInC rhs, SolvVectorIn x_0) {
    auto & this_solver = static_cast<SolverType&> (*this);
    SolvVectorOut retval = this->solver.solveWithGuess(rhs, x_0);
    this->counter += this->solver.iterations();
    if (this->solver.info() != Eigen::Success) {
      std::stringstream err {};
      err << this_solver.name() << " has not converged,"
          << " After " << this->solver.iterations() << " steps, the solver "
          << " FAILED with  |r|/|b| = "
          << std::setw(15) << this->solver.error()
          << ", cg_tol = " << this->tol << std::endl;
      throw ConvergenceError(err.str());
    }
    if (this->verbose) {
      std::cout << " After " << this->solver.iterations() << " "
                << this_solver.name() << " steps, |r|/|b| = "
                << std::setw(15) << this->solver.error()
                << ", cg_tol = " << this->tol << std::endl;
    }
    return retval;
  }

  /* ---------------------------------------------------------------------- */
  template <class SolverType, Dim_t DimS, Dim_t DimM>
  typename SolverEigen<SolverType, DimS, DimM>::Tg_req_t
  SolverEigen<SolverType, DimS, DimM>::get_tangent_req() const {
    return tangent_requirement;
  }

  template class SolverEigen<SolverCGEigen<twoD, twoD>, twoD, twoD>;
  template class SolverEigen<SolverCGEigen<threeD, threeD>, threeD, threeD>;
  template class SolverCGEigen<twoD, twoD>;
  template class SolverCGEigen<threeD, threeD>;

  template class SolverEigen<SolverGMRESEigen<twoD, twoD>, twoD, twoD>;
  template class SolverEigen<SolverGMRESEigen<threeD, threeD>, threeD, threeD>;
  template class SolverGMRESEigen<twoD, twoD>;
  template class SolverGMRESEigen<threeD, threeD>;

  template class SolverEigen<SolverBiCGSTABEigen<twoD, twoD>, twoD, twoD>;
  template class SolverEigen<SolverBiCGSTABEigen<threeD, threeD>, threeD, threeD>;
  template class SolverBiCGSTABEigen<twoD, twoD>;
  template class SolverBiCGSTABEigen<threeD, threeD>;

  template class SolverEigen<SolverDGMRESEigen<twoD, twoD>, twoD, twoD>;
  template class SolverEigen<SolverDGMRESEigen<threeD, threeD>, threeD, threeD>;
  template class SolverDGMRESEigen<twoD, twoD>;
  template class SolverDGMRESEigen<threeD, threeD>;

  template class SolverEigen<SolverMINRESEigen<twoD, twoD>, twoD, twoD>;
  template class SolverEigen<SolverMINRESEigen<threeD, threeD>, threeD, threeD>;
  template class SolverMINRESEigen<twoD, twoD>;
  template class SolverMINRESEigen<threeD, threeD>;

} // muSpectre
