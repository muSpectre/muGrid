/**
 * file   solvers.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   24 Apr 2018
 *
 * @brief  Free functions for solving rve problems
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

#ifndef SRC_SOLVER_SOLVERS_HH_
#define SRC_SOLVER_SOLVERS_HH_

#include "solver/solver_base.hh"

#include <Eigen/Dense>

#include <vector>
#include <string>

namespace muSpectre {

  using LoadSteps_t = std::vector<Eigen::MatrixXd>;

  /**
   * Uses the Newton-conjugate Gradient method to find the static
   * equilibrium of a cell given a series of mean applied strains
   */
  std::vector<OptimizeResult> newton_cg(Cell &cell,
                                        const LoadSteps_t &load_steps,
                                        SolverBase &solver, Real newton_tol,
                                        Real equil_tol, Dim_t verbose = 0);

  /**
   * Uses the Newton-conjugate Gradient method to find the static
   * equilibrium of a cell given a mean applied strain
   */
  OptimizeResult newton_cg(Cell &cell,
                           const Eigen::Ref<Eigen::MatrixXd> load_step,
                           SolverBase &solver, Real newton_tol, Real equil_tol,
                           Dim_t verbose = 0) {
    LoadSteps_t load_steps{load_step};
    return newton_cg(cell, load_steps, solver, newton_tol, equil_tol, verbose)
        .front();
  }

  /* ---------------------------------------------------------------------- */
  /**
   * Uses the method proposed by de Geus method to find the static
   * equilibrium of a cell given a series of mean applied strains
   */
  std::vector<OptimizeResult> de_geus(Cell &cell, const LoadSteps_t &load_steps,
                                      SolverBase &solver, Real newton_tol,
                                      Real equil_tol, Dim_t verbose = 0);

  /* ---------------------------------------------------------------------- */
  /**
   * Uses the method proposed by de Geus method to find the static
   * equilibrium of a cell given a mean applied strain
   */
  OptimizeResult de_geus(Cell &cell,
                         const Eigen::Ref<Eigen::MatrixXd> load_step,
                         SolverBase &solver, Real newton_tol, Real equil_tol,
                         Dim_t verbose = 0) {
    return de_geus(cell, LoadSteps_t{load_step}, solver, newton_tol, equil_tol,
                   verbose)[0];
  }

}  // namespace muSpectre

#endif  // SRC_SOLVER_SOLVERS_HH_
