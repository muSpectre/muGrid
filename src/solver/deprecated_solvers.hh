/**
 * @file   solvers.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   20 Dec 2017
 *
 * @brief  Free functions for solving
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

#ifndef SRC_SOLVER_DEPRECATED_SOLVERS_HH_
#define SRC_SOLVER_DEPRECATED_SOLVERS_HH_

#include "solver/solver_common.hh"
#include "solver/deprecated_solver_base.hh"

#include <Eigen/Dense>

#include <vector>
#include <string>

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  /**
   * Uses the Newton-conjugate Gradient method to find the static
   * equilibrium of a cell given a series of mean applied strains
   */
  template <Dim_t DimS, Dim_t DimM = DimS>
  std::vector<OptimizeResult>
  deprecated_newton_cg(CellBase<DimS, DimM> &cell,
                       const GradIncrements<DimM> &delF0,
                       DeprecatedSolverBase<DimS, DimM> &solver,
                       Real newton_tol, Real equil_tol, Dim_t verbose = 0);

  /* ---------------------------------------------------------------------- */
  /**
   * Uses the Newton-conjugate Gradient method to find the static
   * equilibrium of a cell given a mean applied strain
   */
  template <Dim_t DimS, Dim_t DimM = DimS>
  inline OptimizeResult
  deprecated_newton_cg(CellBase<DimS, DimM> &cell, const Grad_t<DimM> &delF0,
                       DeprecatedSolverBase<DimS, DimM> &solver,
                       Real newton_tol, Real equil_tol, Dim_t verbose = 0) {
    return deprecated_newton_cg(cell, GradIncrements<DimM>{delF0}, solver,
                                newton_tol, equil_tol, verbose)[0];
  }

  /* ---------------------------------------------------------------------- */
  /**
   * Uses the method proposed by de Geus method to find the static
   * equilibrium of a cell given a series of mean applied strains
   */
  template <Dim_t DimS, Dim_t DimM = DimS>
  std::vector<OptimizeResult>
  deprecated_de_geus(CellBase<DimS, DimM> &cell,
                     const GradIncrements<DimM> &delF0,
                     DeprecatedSolverBase<DimS, DimM> &solver, Real newton_tol,
                     Real equil_tol, Dim_t verbose = 0);

  /* ---------------------------------------------------------------------- */
  /**
   * Uses the method proposed by de Geus method to find the static
   * equilibrium of a cell given a mean applied strain
   */
  template <Dim_t DimS, Dim_t DimM = DimS>
  OptimizeResult
  deprecated_de_geus(CellBase<DimS, DimM> &cell, const Grad_t<DimM> &delF0,
                     DeprecatedSolverBase<DimS, DimM> &solver, Real newton_tol,
                     Real equil_tol, Dim_t verbose = 0) {
    return deprecated_de_geus(cell, GradIncrements<DimM>{delF0}, solver,
                              newton_tol, equil_tol, verbose)[0];
  }

}  // namespace muSpectre

#endif  // SRC_SOLVER_DEPRECATED_SOLVERS_HH_
