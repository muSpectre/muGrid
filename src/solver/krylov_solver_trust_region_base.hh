/**
 * @file   krylov_solver_trust_region_base.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *         Ali Falsafi  <ali.falsafi@epfl.ch>
 *
 *
 * @date   25 July 2020
 *
 * @brief  Base class for trust region Krylov solver
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

#ifndef SRC_SOLVER_KRYLOV_SOLVER_TRUST_REGION_BASE_HH_
#define SRC_SOLVER_KRYLOV_SOLVER_TRUST_REGION_BASE_HH_

#include "solver/krylov_solver_base.hh"

namespace muSpectre {
  class KrylovSolverTrustRegionBase : public KrylovSolverBase {
   public:
    using Parent = KrylovSolverBase;  //!< standard short-hand for base class};

    //! for storage of fields
    using Vector_t = Parent::Vector_t;
    //! Input vector for solvers
    using Vector_ref = Parent::Vector_ref;
    //! Input vector for solvers
    using ConstVector_ref = Parent::ConstVector_ref;
    //! Output vector for solvers
    using Vector_map = Parent::Vector_map;

    //! Default constructor
    KrylovSolverTrustRegionBase() = delete;

    //! Copy constructor
    KrylovSolverTrustRegionBase(const KrylovSolverTrustRegionBase & other) =
        delete;

    /**
     * Constructor takes a Cell, tolerance, max number of iterations
     * , initial trust region radius ,verbosity flag, input. A
     * negative value for the tolerance tells the solver to automatically adjust
     * it.
     */
    KrylovSolverTrustRegionBase(std::shared_ptr<MatrixAdaptable> matrix_holder,
                                const Real & tol = -1.0,
                                const Uint & maxiter = 1000,
                                const Real & trust_region = 1.0,
                                const Verbosity & verbose = Verbosity::Silent);

    /**
     * Constructor without matrix adaptable. The adaptable has to be supplied
     * using KrylovSolverBase::set_matrix(...) before initialisation for this
     * solver to be usable
     */
    KrylovSolverTrustRegionBase(const Real & tol = -1.0,
                                const Uint & maxiter = 1000,
                                const Real & trust_region = 1.0,
                                const Verbosity & verbose = Verbosity::Silent);

    //! Move constructor
    KrylovSolverTrustRegionBase(KrylovSolverTrustRegionBase && other) = default;

    //! Destructor
    virtual ~KrylovSolverTrustRegionBase() = default;

    //! Copy assignment operator
    KrylovSolverTrustRegionBase &
    operator=(const KrylovSolverTrustRegionBase & other) = delete;

    //! Move assignment operator
    KrylovSolverTrustRegionBase &
    operator=(KrylovSolverTrustRegionBase && other) = delete;

    //! set size of trust region, exception if not supported
    void set_trust_region(const Real & new_trust_region);

    //! return the member variable which expresses whether the solution of the
    //! sub-problem is on the bound or not
    const bool & get_is_on_bound();

   protected:
    Real trust_region;  //!< size of trust region

    bool is_on_bound{false};  //!< Boolean showing if the solution is on the
                              //! boundary of trust region
  };
}  // namespace muSpectre

#endif  // SRC_SOLVER_KRYLOV_SOLVER_TRUST_REGION_BASE_HH_
