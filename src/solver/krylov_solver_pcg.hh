/**
 * @file   krylov_solver_pcg.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   28 Aug 2020
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

#ifndef SRC_SOLVER_KRYLOV_SOLVER_PCG_HH_
#define SRC_SOLVER_KRYLOV_SOLVER_PCG_HH_

#include "krylov_solver_base.hh"
#include "krylov_solver_preconditioned_features.hh"

namespace muSpectre {

  /**
   * implements the `muSpectre::KrylovSolverBase` interface using a conjugate
   * gradient solver with preconditioner. Broadly follows algorithm 5.3 in
   * Nocedal's Numerical Optimization (p 119), but uses the inverse of the
   * preconditioner matrix directly
   *
   *         Given x₀, preconditioner M⁻¹:
   *           Set r₀ ← Ax₀-b
   *           Set y₀ ← M⁻¹r₀
   *           Set p₀ ← -y₀, k ← 0
   *
   *           while not converged:
   *                    rᵀₖyₖ
   *             αₖ   ← ————––
   *                    pᵀₖApₖ
   *
   *             xₖ₊₁ ← xₖ + αₖpₖ
   *
   *             rₖ₊₁ ← rₖ + αₖApₖ
   *
   *             yₖ₊₁ ← M⁻¹rₖ₊₁
   *
   *                    rᵀₖ₊₁yₖ₊₁
   *             βₖ₊₁ ← ————————–
   *                      rᵀₖyₖ
   *
   *             pₖ₊₁ ← -yₖ₊₁ + βₖ₊₁pₖ
   *
   *             k    ← k + 1
   */
  class KrylovSolverPCG : public KrylovSolverBase,
                          public KrylovSolverPreconditionedFeatures {
   public:
    //! standard short-hand for base class
    using Parent = KrylovSolverBase;
    using FeaturesPC = KrylovSolverPreconditionedFeatures;

    //! Input vector for solvers
    using ConstVector_ref = typename Parent::ConstVector_ref;
    //! Output vector for solvers
    using Vector_map = typename Parent::Vector_map;

    //! Default constructor
    KrylovSolverPCG() = delete;

    //! Copy constructor
    KrylovSolverPCG(const KrylovSolverPCG & other) = delete;

    //! Move constructor
    KrylovSolverPCG(KrylovSolverPCG && other) = default;

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
    KrylovSolverPCG(std::shared_ptr<MatrixAdaptable> matrix,
                    std::shared_ptr<MatrixAdaptable> inv_preconditioner,
                    const Real & tol, const Uint & maxiter,
                    const Verbosity & verbose = Verbosity::Silent);

    /**
     * Constructor without matrix adaptable. The adaptable has to be supplied
     * usinge KrylovSolverBase::set_matrix(...) before initialisation for this
     * solver to be usable
     */
    KrylovSolverPCG(const Real & tol, const Uint & maxiter,
                    const Verbosity & verbose = Verbosity::Silent);

    //! Destructor
    virtual ~KrylovSolverPCG() = default;

    //! Copy assignment operator
    KrylovSolverPCG & operator=(const KrylovSolverPCG & other) = delete;

    //! Move assignment operator
    KrylovSolverPCG & operator=(KrylovSolverPCG && other) = delete;

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
    Vector_t r_k;   //!< residual
    Vector_t y_k;   //!< preconditioned current solution
    Vector_t p_k;   //!< search direction
    Vector_t Ap_k;  //!< directional stiffness
    Vector_t x_k;   //!< current solution
  };

}  // namespace muSpectre

#endif  // SRC_SOLVER_KRYLOV_SOLVER_PCG_HH_
