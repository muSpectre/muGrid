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

#ifndef SOLVERS_H
#define SOLVERS_H

#include "solver/solver_base.hh"

#include <Eigen/Dense>

#include <vector>
#include <string>

namespace muSpectre {

  /**
   * emulates scipy.optimize.OptimizeResult
   */
  struct OptimizeResult
  {
    //! Strain ε or Gradient F at solution
    Eigen::ArrayXXd grad;
    //! Cauchy stress σ or first Piola-Kirchhoff stress P at solution
    Eigen::ArrayXXd stress;
    //! whether or not the solver exited successfully
    bool success;
    //! Termination status of the optimizer. Its value depends on the
    //! underlying solver. Refer to message for details.
    Int status;
    //! Description of the cause of the termination.
    std::string message;
    //! number of iterations
    Uint nb_it;
    //! number of cell evaluations
    Uint nb_fev;
  };

  /**
   * Field type that solvers expect gradients to be expressed in
   */
  template <Dim_t Dim>
  using Grad_t = Matrices::Tens2_t<Dim>;
  /**
   * multiple increments can be submitted at once (useful for
   * path-dependent materials)
   */
  template <Dim_t Dim>
  using GradIncrements = std::vector<Grad_t<Dim>,
                                     Eigen::aligned_allocator<Grad_t<Dim>>>;

  /* ---------------------------------------------------------------------- */
  /**
   * Uses the Newton-conjugate Gradient method to find the static
   * equilibrium of a cell given a series of mean applied strains
   */
  template <Dim_t DimS, Dim_t DimM=DimS>
  std::vector<OptimizeResult>
  newton_cg (CellBase<DimS, DimM> & cell,
             const GradIncrements<DimM> & delF0,
             SolverBase<DimS, DimM> & solver, Real newton_tol,
             Real equil_tol,
             Dim_t verbose = 0);

  /* ---------------------------------------------------------------------- */
  /**
   * Uses the Newton-conjugate Gradient method to find the static
   * equilibrium of a cell given a mean applied strain
   */
  template <Dim_t DimS, Dim_t DimM=DimS>
  inline OptimizeResult
  newton_cg (CellBase<DimS, DimM> & cell, const Grad_t<DimM> & delF0,
             SolverBase<DimS, DimM> & solver, Real newton_tol,
             Real equil_tol,
             Dim_t verbose = 0){
    return newton_cg(cell, GradIncrements<DimM>{delF0},
                     solver, newton_tol, equil_tol, verbose)[0];
  }

  /* ---------------------------------------------------------------------- */
  /**
   * Uses the method proposed by de Geus method to find the static
   * equilibrium of a cell given a series of mean applied strains
   */
  template <Dim_t DimS, Dim_t DimM=DimS>
  std::vector<OptimizeResult>
  de_geus (CellBase<DimS, DimM> & cell,
           const GradIncrements<DimM> & delF0,
           SolverBase<DimS, DimM> & solver, Real newton_tol,
           Real equil_tol,
           Dim_t verbose = 0);

  /* ---------------------------------------------------------------------- */
  /**
   * Uses the method proposed by de Geus method to find the static
   * equilibrium of a cell given a mean applied strain
   */
  template <Dim_t DimS, Dim_t DimM=DimS>
  OptimizeResult
  de_geus (CellBase<DimS, DimM> & cell, const Grad_t<DimM> & delF0,
           SolverBase<DimS, DimM> & solver, Real newton_tol,
           Real equil_tol,
           Dim_t verbose = 0){
    return de_geus(cell, GradIncrements<DimM>{delF0},
                   solver, newton_tol, equil_tol, verbose)[0];
  }

  /* ---------------------------------------------------------------------- */
  /**
   * check whether a strain is symmetric, for the purposes of small
   * strain problems
   */
  bool check_symmetry(const Eigen::Ref<const Eigen::ArrayXXd>& eps,
                      Real rel_tol = 1e-8);

}  // muSpectre

#endif /* SOLVERS_H */
