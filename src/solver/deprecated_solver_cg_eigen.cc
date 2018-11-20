/**
 * @file   deprecated_solver_cg_eigen.cc
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

#include "deprecated_solver_cg_eigen.hh"

#include <sstream>
#include <iomanip>

namespace muSpectre {

  //----------------------------------------------------------------------------//
  template <class DeprecatedSolverType, Dim_t DimS, Dim_t DimM>
  DeprecatedSolverEigen<DeprecatedSolverType, DimS, DimM>::DeprecatedSolverEigen(Cell_t& cell, Real tol, Uint maxiter,
                                                   bool verbose)
    :Parent(cell, tol, maxiter, verbose),
     adaptor{cell.get_adaptor()},
     solver{}
  {}

  //----------------------------------------------------------------------------//
  template <class DeprecatedSolverType, Dim_t DimS, Dim_t DimM>
  void
  DeprecatedSolverEigen<DeprecatedSolverType, DimS, DimM>::initialise() {
    this->solver.setTolerance(this->tol);
    this->solver.setMaxIterations(this->maxiter);
    this->solver.compute(this->adaptor);
  }

  //----------------------------------------------------------------------------//
  template <class DeprecatedSolverType, Dim_t DimS, Dim_t DimM>
  typename DeprecatedSolverEigen<DeprecatedSolverType, DimS, DimM>::SolvVectorOut
  DeprecatedSolverEigen<DeprecatedSolverType, DimS, DimM>::solve(const SolvVectorInC rhs, SolvVectorIn x_0) {
    auto & this_solver = static_cast<DeprecatedSolverType&> (*this);
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
  template <class DeprecatedSolverType, Dim_t DimS, Dim_t DimM>
  typename DeprecatedSolverEigen<DeprecatedSolverType, DimS, DimM>::Tg_req_t
  DeprecatedSolverEigen<DeprecatedSolverType, DimS, DimM>::get_tangent_req() const {
    return tangent_requirement;
  }

  template class DeprecatedSolverEigen<DeprecatedSolverCGEigen<twoD, twoD>, twoD, twoD>;
  template class DeprecatedSolverEigen<DeprecatedSolverCGEigen<threeD, threeD>, threeD, threeD>;
  template class DeprecatedSolverCGEigen<twoD, twoD>;
  template class DeprecatedSolverCGEigen<threeD, threeD>;

  template class DeprecatedSolverEigen<DeprecatedSolverGMRESEigen<twoD, twoD>, twoD, twoD>;
  template class DeprecatedSolverEigen<DeprecatedSolverGMRESEigen<threeD, threeD>, threeD, threeD>;
  template class DeprecatedSolverGMRESEigen<twoD, twoD>;
  template class DeprecatedSolverGMRESEigen<threeD, threeD>;

  template class DeprecatedSolverEigen<DeprecatedSolverBiCGSTABEigen<twoD, twoD>, twoD, twoD>;
  template class DeprecatedSolverEigen<DeprecatedSolverBiCGSTABEigen<threeD, threeD>, threeD, threeD>;
  template class DeprecatedSolverBiCGSTABEigen<twoD, twoD>;
  template class DeprecatedSolverBiCGSTABEigen<threeD, threeD>;

  template class DeprecatedSolverEigen<DeprecatedSolverDGMRESEigen<twoD, twoD>, twoD, twoD>;
  template class DeprecatedSolverEigen<DeprecatedSolverDGMRESEigen<threeD, threeD>, threeD, threeD>;
  template class DeprecatedSolverDGMRESEigen<twoD, twoD>;
  template class DeprecatedSolverDGMRESEigen<threeD, threeD>;

  template class DeprecatedSolverEigen<DeprecatedSolverMINRESEigen<twoD, twoD>, twoD, twoD>;
  template class DeprecatedSolverEigen<DeprecatedSolverMINRESEigen<threeD, threeD>, threeD, threeD>;
  template class DeprecatedSolverMINRESEigen<twoD, twoD>;
  template class DeprecatedSolverMINRESEigen<threeD, threeD>;

} // muSpectre
