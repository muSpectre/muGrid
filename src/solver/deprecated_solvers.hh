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

#ifndef DEPRECATED_SOLVERS_H
#define DEPRECATED_SOLVERS_H

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
  template <Dim_t DimS, Dim_t DimM=DimS>
  std::vector<OptimizeResult>
  deprecated_newton_cg (CellBase<DimS, DimM> & cell,
                        const GradIncrements<DimM> & delF0,
                        DeprecatedSolverBase<DimS, DimM> & solver,
                        Real newton_tol,
                        Real equil_tol,
                        Dim_t verbose = 0);

  /* ---------------------------------------------------------------------- */
  /**
   * Uses the Newton-conjugate Gradient method to find the static
   * equilibrium of a cell given a mean applied strain
   */
  template <Dim_t DimS, Dim_t DimM=DimS>
  inline OptimizeResult
  deprecated_newton_cg (CellBase<DimS, DimM> & cell, const Grad_t<DimM> & delF0,
                        DeprecatedSolverBase<DimS, DimM> & solver,
                        Real newton_tol,
                        Real equil_tol,
                        Dim_t verbose = 0){
    return deprecated_newton_cg(cell, GradIncrements<DimM>{delF0},
                                solver, newton_tol, equil_tol, verbose)[0];
  }

  /* ---------------------------------------------------------------------- */
  /**
   * Uses the method proposed by de Geus method to find the static
   * equilibrium of a cell given a series of mean applied strains
   */
  template <Dim_t DimS, Dim_t DimM=DimS>
  std::vector<OptimizeResult>
  deprecated_de_geus (CellBase<DimS, DimM> & cell,
                      const GradIncrements<DimM> & delF0,
                      DeprecatedSolverBase<DimS, DimM> & solver,
                      Real newton_tol,
                      Real equil_tol,
                      Dim_t verbose = 0);

  /* ---------------------------------------------------------------------- */
  /**
   * Uses the method proposed by de Geus method to find the static
   * equilibrium of a cell given a mean applied strain
   */
  template <Dim_t DimS, Dim_t DimM=DimS>
  OptimizeResult
  deprecated_de_geus (CellBase<DimS, DimM> & cell, const Grad_t<DimM> & delF0,
                      DeprecatedSolverBase<DimS, DimM> & solver,
                      Real newton_tol,
                      Real equil_tol,
                      Dim_t verbose = 0){
    return deprecated_de_geus(cell, GradIncrements<DimM>{delF0},
                              solver, newton_tol, equil_tol, verbose)[0];
  }


}  // muSpectre

#endif /* DEPRECATED_SOLVERS_H */
