/**
 * @file   krylov_solver_trust_region_cg.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *         Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   25 July 2020
 *
 * @brief  Conjugate-gradient solver with a trust region
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

#ifndef SRC_SOLVER_KRYLOV_SOLVER_TRUST_REGION_CG_HH_
#define SRC_SOLVER_KRYLOV_SOLVER_TRUST_REGION_CG_HH_

#include "solver/krylov_solver_trust_region_base.hh"

namespace muSpectre {
  enum class ResetCG {
    no_reset = 0,
    fixed_iter_count = 1,
    user_defined_iter_count = 2,
    gradient_orthogonality = 3,
    valid_direction = 4
  };

  /**
   * implements the `muSpectre::KrylovSolverBase` interface using a
   * conjugate gradient solver with a trust region. This Krylov solver is meant
   * to be used with the nonlinear `trust_region_newton_cg` solver.
   */
  class KrylovSolverTrustRegionCG : public KrylovSolverTrustRegionBase {
   public:
    using Parent =
        KrylovSolverTrustRegionBase;  //!< standard short-hand for base class
    //! for storage of fields
    using Vector_t = Parent::Vector_t;
    //! Input vector for solvers
    using Vector_ref = Parent::Vector_ref;
    //! Input vector for solvers
    using ConstVector_ref = Parent::ConstVector_ref;
    //! Output vector for solvers
    using Vector_map = Parent::Vector_map;

    //! Default constructor
    KrylovSolverTrustRegionCG() = delete;

    //! Copy constructor
    KrylovSolverTrustRegionCG(const KrylovSolverTrustRegionCG & other) = delete;

    /**
     * Constructor takes a Cell, tolerance, max number of iterations
     * , initial trust region radius ,verbosity flag, and reset flag as input. A
     * negative value for the tolerance tells the solver to automatically adjust
     * it.
     */
    KrylovSolverTrustRegionCG(std::shared_ptr<MatrixAdaptable> matrix_holder,
                              const Real & tol = -1.0,
                              const Uint & maxiter = 1000,
                              const Real & trust_region = 1.0,
                              const Verbosity & verbose = Verbosity::Silent,
                              const ResetCG & reset = ResetCG::no_reset,
                              const Uint & reset_iter_count = 0);

    /**
     * Constructor without matrix adaptable. The adaptable has to be supplied
     * using KrylovSolverBase::set_matrix(...) before initialisation for this
     * solver to be usable
     */
    KrylovSolverTrustRegionCG(const Real & tol = -1.0,
                              const Uint & maxiter = 1000,
                              const Real & trust_region = 1.0,
                              const Verbosity & verbose = Verbosity::Silent,
                              const ResetCG & reset = ResetCG::no_reset,
                              const Uint & reset_iter_count = 0);

    //! Move constructor
    KrylovSolverTrustRegionCG(KrylovSolverTrustRegionCG && other) = default;

    //! Destructor
    virtual ~KrylovSolverTrustRegionCG() = default;

    //! Copy assignment operator
    KrylovSolverTrustRegionCG &
    operator=(const KrylovSolverTrustRegionCG & other) = delete;

    //! Move assignment operator
    KrylovSolverTrustRegionCG &
    operator=(KrylovSolverTrustRegionCG && other) = delete;

    //! initialisation does not need to do anything in this case
    void initialise() final{};

    //! returns the solver's name
    std::string get_name() const final { return "TrustRegionCG"; }

    //! set the matrix
    void set_matrix(std::shared_ptr<MatrixAdaptable> matrix_adaptable) final;

    //! set the matrix
    void set_matrix(std::weak_ptr<MatrixAdaptable> matrix_adaptable) final;

    //! the actual solver
    Vector_map solve(const ConstVector_ref rhs) final;

    Real reset_cg();

   protected:
    // to be called in set_matrix
    void set_internal_arrays();
    muGrid::Communicator comm{};
    //! find the minimzer on the trust region bound
    Vector_map bound(const ConstVector_ref rhs);

    ResetCG reset;  //!< Determines whether restart will be carried out in
                    //!< solver steps or  not(if necessary)

    Uint reset_iter_count;  //!< if user wants to give the iteration count after
                            //! which the reset needs to be triggered

    Vector_t r_k;         //!< residual
    Vector_t p_k;         //!< search direction
    Vector_t Ap_k;        //!< directional stiffness
    Vector_t x_k;         //!< current solution
    Vector_t r_k_copy{};  //! used to keep a copy of the residual if needed
  };

}  // namespace muSpectre

#endif  // SRC_SOLVER_KRYLOV_SOLVER_TRUST_REGION_CG_HH_
