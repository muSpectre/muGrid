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

#include "solver/krylov_solver_base.hh"
#include "solver/krylov_solver_trust_region_features.hh"

namespace muSpectre {

  /**
   * implements the `muSpectre::KrylovSolverBase` interface using a
   * conjugate gradient solver with a trust region. This Krylov solver is meant
   * to be used with the nonlinear `trust_region_newton_cg` solver.
   */
  class KrylovSolverTrustRegionCG : public KrylovSolverBase,
                                    public KrylovSolverTrustRegionFeatures {
   public:
    using Parent = KrylovSolverBase;  //!< standard short-hand for base class
    using FeaturesTR = KrylovSolverTrustRegionFeatures;
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
    KrylovSolverTrustRegionCG(
        std::shared_ptr<MatrixAdaptable> matrix_holder, const Real & tol = -1.0,
        const Uint & maxiter = 1000, const Real & trust_region = 1.0,
        const Verbosity & verbose = Verbosity::Silent,
        const ResetCG & reset = ResetCG::no_reset,
        const Index_t & reset_iter_count = muGrid::Unknown);

    /**
     * Constructor without matrix adaptable. The adaptable has to be supplied
     * using KrylovSolverBase::set_matrix(...) before initialisation for this
     * solver to be usable
     */
    KrylovSolverTrustRegionCG(
        const Real & tol = -1.0, const Uint & maxiter = 1000,
        const Real & trust_region = 1.0,
        const Verbosity & verbose = Verbosity::Silent,
        const ResetCG & reset = ResetCG::no_reset,
        const Index_t & reset_iter_count = muGrid::Unknown);

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
    void initialise() override{};

    //! returns the solver's name
    std::string get_name() const override { return "TrustRegionCG"; }

    //! set the matrix
    void set_matrix(std::shared_ptr<MatrixAdaptable> matrix_adaptable) override;

    //! set the matrix
    void set_matrix(std::weak_ptr<MatrixAdaptable> matrix_adaptable) override;

    //! the actual solver
    Vector_map solve(const ConstVector_ref rhs) override;

   protected:
    // to be called in set_matrix
    void set_internal_arrays();

    //! find the minimzer on the trust region bound (To be called if necessary
    //! during the solution procedure)
    Vector_map bound(const ConstVector_ref rhs);

    Vector_t r_k;   //!< residual
    Vector_t p_k;   //!< search direction
    Vector_t Ap_k;  //!< directional stiffness
    Vector_t x_k;   //!< current solution
    //! used to keep a copy of the previous residual if needed
    Vector_t r_k_previous{};
  };

}  // namespace muSpectre

#endif  // SRC_SOLVER_KRYLOV_SOLVER_TRUST_REGION_CG_HH_
