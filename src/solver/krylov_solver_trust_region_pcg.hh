/**
 * @file   krylov_solver_trust_region_pcg.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   01 Apr 2021
 *
 * @brief  Preconditioned conjugate gradient solver.
 *
 * Copyright © 2020 Till Junge
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
 * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 *
 */

#ifndef SRC_SOLVER_KRYLOV_SOLVER_TRUST_REGION_PCG_HH_
#define SRC_SOLVER_KRYLOV_SOLVER_TRUST_REGION_PCG_HH_

#include "krylov_solver_preconditioned_features.hh"
#include "krylov_solver_trust_region_cg.hh"

namespace muSpectre {

  class KrylovSolverTrustRegionPCG : public KrylovSolverTrustRegionCG,
                                     public KrylovSolverPreconditionedFeatures {
   public:
    //! standard short-hand for base class
    using Parent = KrylovSolverTrustRegionCG;
    using FeaturesPC = KrylovSolverPreconditionedFeatures;
    //! Input vector for solvers
    using ConstVector_ref = typename Parent::ConstVector_ref;
    //! Output vector for solvers
    using Vector_map = typename Parent::Vector_map;

    //! Default constructor
    KrylovSolverTrustRegionPCG() = delete;

    //! Copy constructor
    KrylovSolverTrustRegionPCG(const KrylovSolverTrustRegionPCG & other) =
        delete;

    //! Move constructor
    KrylovSolverTrustRegionPCG(KrylovSolverTrustRegionPCG && other) = default;

    /**
     * Constructor takes a system matrix, inverse of preconditioner, tolerance,
     * max number of iterations and verbosity flag as input
     *
     * @param matrix system matrix, "A" in algorithm above
     *
     * @param inv_preconditioner inverse of preconditioner, "M⁻¹" in algorithm
     * above
     *
     * @param tol relative convergence criterion, is compared to
     * |residual|/|initial_residual|
     *
     * @param maxiter maximum allowed number of iterations
     *
     * @param verbose level of verbosity
     */
    KrylovSolverTrustRegionPCG(
        std::shared_ptr<MatrixAdaptable> matrix,
        std::shared_ptr<MatrixAdaptable> inv_preconditioner, const Real & tol,
        const Uint & maxiter, const Real & trust_region = 1.0,
        const Verbosity & verbose = Verbosity::Silent,
        const ResetCG & reset = ResetCG::no_reset,
        const Index_t & reset_iter_count = muGrid::Unknown);

    /**
     * Constructor without matrix adaptable. The adaptable has to be supplied
     * usinge KrylovSolverBase::set_matrix(...) before initialisation for this
     * solver to be usable
     */
    KrylovSolverTrustRegionPCG(
        const Real & tol, const Uint & maxiter, const Real & trust_region = 1.0,
        const Verbosity & verbose = Verbosity::Silent,
        const ResetCG & reset = ResetCG::no_reset,
        const Index_t & reset_iter_count = muGrid::Unknown);

    //! Destructor
    virtual ~KrylovSolverTrustRegionPCG() = default;

    //! Copy assignment operator
    KrylovSolverTrustRegionPCG &
    operator=(const KrylovSolverTrustRegionPCG & other) = delete;

    //! Move assignment operator
    KrylovSolverTrustRegionPCG &
    operator=(KrylovSolverTrustRegionPCG && other) = delete;

    //! initialisation does not need to do anything in this case
    void initialise() final{};

    //! set the matrix
    // void set_matrix(std::shared_ptr<MatrixAdaptable> matrix_adaptable) final;

    //! returns the solver's name
    std::string get_name() const final;

    //! set the matrix
    void set_matrix(std::shared_ptr<MatrixAdaptable> matrix_adaptable) final;

    //! set the matrix
    void set_matrix(std::weak_ptr<MatrixAdaptable> matrix_adaptable) final;

    /**
     * override for setting the preconditioner in order to clarify that the
     * inverse preconditioner is needed for this class.
     *
     * @param inv_preconditioner inverse of preconditioner, "M⁻¹" in algorithm
     * above
     */
    void set_preconditioner(
        std::shared_ptr<MatrixAdaptable> inv_preconditioner) final;

    //! the actual solver
    Vector_map solve(const ConstVector_ref rhs) final;

   protected:
    void set_internal_arrays();

    //! find the minimzer on the trust region bound (To be called if necessary
    //! during the solution procedure)
    Vector_map bound(const ConstVector_ref rhs);

    Vector_t r_k;         //!< residual
    Vector_t y_k;         //!< preconditioned current solution
    Vector_t p_k;         //!< search direction
    Vector_t Ap_k;        //!< directional stiffness
    Vector_t x_k;         //!< current solution
    Vector_t y_k_prev{};  //! used to keep a copy of the residual (in the
                          //! projected space) if needed
  };

}  // namespace muSpectre

#endif  // SRC_SOLVER_KRYLOV_SOLVER_TRUST_REGION_PCG_HH_
