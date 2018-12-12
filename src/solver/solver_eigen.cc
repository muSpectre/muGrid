/**
 * file   solver_eigen.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   15 May 2018
 *
 * @brief  Implementations for bindings to Eigen's iterative solvers
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

#include "solver/solver_eigen.hh"

#include <iomanip>
#include <sstream>

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <class SolverType>
  SolverEigen<SolverType>::SolverEigen(Cell &cell, Real tol, Uint maxiter,
                                       bool verbose)
      : Parent(cell, tol, maxiter, verbose), adaptor{cell.get_adaptor()},
        solver{}, result{} {}

  /* ---------------------------------------------------------------------- */
  template <class SolverType> void SolverEigen<SolverType>::initialise() {
    this->solver.setTolerance(this->get_tol());
    this->solver.setMaxIterations(this->get_maxiter());
    this->solver.compute(this->adaptor);
  }

  /* ---------------------------------------------------------------------- */
  template <class SolverType>
  auto SolverEigen<SolverType>::solve(const ConstVector_ref rhs) -> Vector_map {
    // for crtp
    auto &this_solver = static_cast<SolverType &>(*this);
    this->result = this->solver.solve(rhs);
    this->counter += this->solver.iterations();

    if (this->solver.info() != Eigen::Success) {
      std::stringstream err{};
      err << this_solver.get_name() << " has not converged,"
          << " After " << this->solver.iterations() << " steps, the solver "
          << " FAILED with  |r|/|b| = " << std::setw(15) << this->solver.error()
          << ", cg_tol = " << this->tol << std::endl;
      throw ConvergenceError(err.str());
    }

    if (this->verbose) {
      std::cout << " After " << this->solver.iterations() << " "
                << this_solver.get_name()
                << " steps, |r|/|b| = " << std::setw(15) << this->solver.error()
                << ", cg_tol = " << this->tol << std::endl;
    }
    return Vector_map(this->result.data(), this->result.size());
  }

  /* ---------------------------------------------------------------------- */
  template class SolverEigen<SolverCGEigen>;
  template class SolverEigen<SolverGMRESEigen>;
  template class SolverEigen<SolverBiCGSTABEigen>;
  template class SolverEigen<SolverDGMRESEigen>;
  template class SolverEigen<SolverMINRESEigen>;

}  // namespace muSpectre
