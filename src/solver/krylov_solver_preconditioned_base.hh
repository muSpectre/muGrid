/**
 * @file   krylov_solver_preconditioned_base.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   30 Aug 2020
 *
 * @brief  Base class for solvers with a pre-conditioner
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

#include "krylov_solver_base.hh"

#ifndef SRC_SOLVER_KRYLOV_SOLVER_PRECONDITIONED_BASE_HH_
#define SRC_SOLVER_KRYLOV_SOLVER_PRECONDITIONED_BASE_HH_

namespace muSpectre {

  class KrylovSolverPreconditionedBase : public KrylovSolverBase {
   public:
    //! standard short-hand for base class
    using Parent = KrylovSolverBase;
    //! Input vector for solvers
    using ConstVector_ref = typename Parent::ConstVector_ref;
    //! Output vector for solvers
    using Vector_map = typename Parent::Vector_map;

    //! Default constructor
    KrylovSolverPreconditionedBase() = delete;

    /**
     * Constructor takes the matrix adaptables for the system matrix and the
     * preconditioner, tolerance, max number of iterations and verbosity flag as
     * input
     */
    KrylovSolverPreconditionedBase(
        std::shared_ptr<MatrixAdaptable> matrix_adaptable,
        std::shared_ptr<MatrixAdaptable> preconditioner_adaptable,
        const Real & tol, const Uint & maxiter,
        const Verbosity & verbose = Verbosity::Silent);

    /**
     * Constructor without matrix adaptables. The adaptables have to be supplied
     * using KrylovSolverBase::set_matrix(...)  and
     * KrylovSolverPreconditionedBase::set_preconditioner(...) before
     * initialisation for this solver to be usable
     */
    KrylovSolverPreconditionedBase(
        const Real & tol, const Uint & maxiter,
        const Verbosity & verbose = Verbosity::Silent);

    //! Copy constructor
    KrylovSolverPreconditionedBase(
        const KrylovSolverPreconditionedBase & other) = delete;

    //! Move constructor
    KrylovSolverPreconditionedBase(KrylovSolverPreconditionedBase && other) =
        default;

    //! Destructor
    virtual ~KrylovSolverPreconditionedBase() = default;

    //! Copy assignment operator
    KrylovSolverPreconditionedBase &
    operator=(const KrylovSolverPreconditionedBase & other) = delete;

    //! Move assignment operator
    KrylovSolverPreconditionedBase &
    operator=(KrylovSolverPreconditionedBase && other) = delete;

    //! set the preconditioner
    virtual void set_preconditioner(
        std::shared_ptr<MatrixAdaptable> preconditioner_adaptable);

   protected:
    //! preconditioner
    std::shared_ptr<MatrixAdaptable> preconditioner_holder{nullptr};
    MatrixAdaptor preconditioner{};  //!< matrix ref for convenience
  };

}  // namespace muSpectre

#endif  // SRC_SOLVER_KRYLOV_SOLVER_PRECONDITIONED_BASE_HH_
