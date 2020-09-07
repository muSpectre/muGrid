/**
 * @file   solvers.hh
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

#ifndef SRC_SOLVER_SOLVERS_HH_
#define SRC_SOLVER_SOLVERS_HH_

#include "solver/krylov_solver_base.hh"

#include <Eigen/Dense>

#include <vector>
#include <string>
#include <functional>

namespace muSpectre {

  /**
   * Input type for specifying a load regime
   */
  using LoadSteps_t = std::vector<Eigen::MatrixXd>;

  using MappedField_t =
      muGrid::MappedField<muGrid::FieldMap<Real, Mapping::Mut>>;

  using EigenStrainFunc_t = typename std::function<void(
      const size_t &, muGrid::TypedFieldBase<Real> &)>;

#ifdef NO_EXPERIMENTAL
  using EigenStrainOptFunc_ref = typename muGrid::optional<EigenStrainFunc_t &>;
#else
  using EigenStrainOptFunc_ref =
      typename muGrid::optional<std::reference_wrapper<EigenStrainFunc_t>>;
#endif
  enum class IsStrainInitialised { True, False };

  /**
   * This class contains bool variables used to store the termination
   * criteria of the newton-cg solver
   */
  class ConvergenceCriterion {
   public:
    // constructor
    ConvergenceCriterion();
    // destructor
    virtual ~ConvergenceCriterion() = default;

    //! getter for was_last_step_linear_test
    bool & get_was_last_step_linear_test();

    //! getter for equil_tol_test
    bool & get_equil_tol_test();

    //! getter for newton_tol_test
    bool & get_newton_tol_test();

    //! reset the bool members of the ConvergenceCriterion to false
    void reset();

   protected:
    bool was_last_step_linear_test;  //!< the linearity termination criterion
    bool equil_tol_test;             //!< the equilibrium termination criterion
    bool newton_tol_test;            //!< the change in strain criterion
  };

  /**
   * Uses the Newton-conjugate Gradient method to find the static
   * equilibrium of a cell given a series of mean applied strain(ε for
   * Formulation::small_strain and H (=F-I) for Formulation::finite_strain).
   * The initial macroscopic strain state is set to zero in cell
   * initialisation.
   */
  std::vector<OptimizeResult> newton_cg(
      Cell & cell, const LoadSteps_t & load_steps, KrylovSolverBase & solver,
      const Real & newton_tol, const Real & equil_tol,
      const Verbosity & verbose = Verbosity::Silent,
      const IsStrainInitialised & strain_init = IsStrainInitialised::False,
      EigenStrainOptFunc_ref eigen_strain_func = muGrid::nullopt);
  /**
   * Uses the Newton-conjugate Gradient method to find the static
   * equilibrium of a cell given a mean applied strain.
   */
  inline OptimizeResult newton_cg(
      Cell & cell, const Eigen::Ref<Eigen::MatrixXd> load_step,
      KrylovSolverBase & solver, const Real & newton_tol,
      const Real & equil_tol, const Verbosity & verbose = Verbosity::Silent,
      const IsStrainInitialised & strain_init = IsStrainInitialised::False,
      EigenStrainOptFunc_ref eigen_strain_func = muGrid::nullopt) {
    LoadSteps_t load_steps{load_step};
    auto ret_val{newton_cg(cell, load_steps, solver, newton_tol, equil_tol,
                           verbose, strain_init, eigen_strain_func)
                     .front()};
    return ret_val;
  }

  /* ---------------------------------------------------------------------- */
  /**
   * Uses the method proposed by de Geus method to find the static
   * given a series of mean applied strain(ε for Formulation::small_strain
   * and H (=F-I) for Formulation::finite_strain). The initial macroscopic
   * strain state is set to zero in cell initialisation.
   */
  std::vector<OptimizeResult>
  de_geus(Cell & cell, const LoadSteps_t & load_steps,
          KrylovSolverBase & solver, Real newton_tol, Real equil_tol,
          Verbosity verbose = Verbosity::Silent,
          IsStrainInitialised strain_init = IsStrainInitialised::False);

  /* ---------------------------------------------------------------------- */
  /**
   * Uses the method proposed by de Geus method to find the static
   * equilibrium of a cell given a mean applied strain.
   */
  inline OptimizeResult
  de_geus(Cell & cell, const Eigen::Ref<Eigen::MatrixXd> load_step,
          KrylovSolverBase & solver, Real newton_tol, Real equil_tol,
          Verbosity verbose = Verbosity::Silent,
          IsStrainInitialised strain_init = IsStrainInitialised::False) {
    return de_geus(cell, LoadSteps_t{load_step}, solver, newton_tol, equil_tol,
                   verbose, strain_init)[0];
  }

  /* ---------------------------------------------------------------------- */
  /**
   * Uses the Newton-conjugate Gradient method to find the static
   * equilibrium of a cell given a series of mean applied strain(ε for
   * Formulation::small_strain and H (=F-I) for Formulation::finite_strain).
   * The initial macroscopic strain state is set to zero in cell
   * initialisation.
   */
  std::vector<OptimizeResult> trust_region_newton_cg(
      Cell & cell, const LoadSteps_t & load_steps, KrylovSolverBase & solver,
      const Real & max_trust_region, const Real & newton_tol,
      const Real & equil_tol, const Real & inc_tr_tol, const Real & dec_tr_tol,
      const Verbosity & verbose, const IsStrainInitialised & strain_init,
      EigenStrainOptFunc_ref eigen_strain_func = muGrid::nullopt);

  /**
   * Uses the Newton-conjugate Gradient method to find the static
   * equilibrium of a cell given a mean applied strain.
   */
  inline OptimizeResult trust_region_newton_cg(
      Cell & cell, const Eigen::Ref<Eigen::MatrixXd> load_step,
      KrylovSolverBase & solver, const Real & max_trust_region,
      const Real & newton_tol, const Real & inc_tr_tol, const Real & dec_tr_tol,
      const Real & reduction_tol, const Verbosity & verbose = Verbosity::Silent,
      const IsStrainInitialised & strain_init = IsStrainInitialised::False,
      EigenStrainOptFunc_ref eigen_strain_func = muGrid::nullopt) {
    LoadSteps_t load_steps{load_step};
    auto ret_val{trust_region_newton_cg(cell, load_steps, solver,
                                        max_trust_region, newton_tol,
                                        inc_tr_tol, dec_tr_tol, reduction_tol,
                                        verbose, strain_init, eigen_strain_func)
                     .front()};
    return ret_val;
  }

}  // namespace muSpectre

#endif  // SRC_SOLVER_SOLVERS_HH_
