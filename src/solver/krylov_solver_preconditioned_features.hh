/**
 * @file   krylov_solver_preconditioned_features.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   30 Aug 2020
 *
 * @brief  Features class for solvers with a pre-conditioner
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

#ifndef SRC_SOLVER_KRYLOV_SOLVER_PRECONDITIONED_FEATURES_HH_
#define SRC_SOLVER_KRYLOV_SOLVER_PRECONDITIONED_FEATURES_HH_

namespace muSpectre {

  class KrylovSolverPreconditionedFeatures {
   public:
    //! Default constructor
    KrylovSolverPreconditionedFeatures() = default;

    /**
     * Constructor takes the matrix adaptables for the system matrix and the
     * preconditioner, tolerance, max number of iterations and verbosity flag as
     * input
     */
    KrylovSolverPreconditionedFeatures(
        std::shared_ptr<MatrixAdaptable> preconditioner_adaptable);

    /**
     * Constructor without matrix adaptables. The adaptables have to be supplied
     * using KrylovSolverFeatures::set_matrix(...)  and
     * KrylovSolverPreconditionedFeatures::set_preconditioner(...) before
     * initialisation for this solver to be usable
     */
    KrylovSolverPreconditionedFeatures(
        const Real & tol, const Uint & maxiter,
        const Verbosity & verbose = Verbosity::Silent);

    //! Copy constructor
    KrylovSolverPreconditionedFeatures(
        const KrylovSolverPreconditionedFeatures & other) = delete;

    //! Move constructor
    KrylovSolverPreconditionedFeatures(
        KrylovSolverPreconditionedFeatures && other) = default;

    //! Destructor
    virtual ~KrylovSolverPreconditionedFeatures() = default;

    //! Copy assignment operator
    KrylovSolverPreconditionedFeatures &
    operator=(const KrylovSolverPreconditionedFeatures & other) = delete;

    //! Move assignment operator
    KrylovSolverPreconditionedFeatures &
    operator=(KrylovSolverPreconditionedFeatures && other) = delete;

    //! set the preconditioner
    virtual void set_preconditioner(
        std::shared_ptr<MatrixAdaptable> preconditioner_adaptable);

   protected:
    //! preconditioner
    std::shared_ptr<MatrixAdaptable> preconditioner_holder{nullptr};
    MatrixAdaptor preconditioner{};  //!< matrix ref for convenience
  };

}  // namespace muSpectre

#endif  // SRC_SOLVER_KRYLOV_SOLVER_PRECONDITIONED_FEATURES_HH_
