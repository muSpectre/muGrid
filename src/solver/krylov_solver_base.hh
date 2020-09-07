/**
 * @file   krylov_solver_base.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   24 Apr 2018
 *
 * @brief  Base class for iterative solvers for linear systems of equations
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

#ifndef SRC_SOLVER_KRYLOV_SOLVER_BASE_HH_
#define SRC_SOLVER_KRYLOV_SOLVER_BASE_HH_

#include "solver/solver_common.hh"
#include "cell/cell.hh"

#include <Eigen/Dense>

namespace muSpectre {

  /**
   * Virtual base class for solvers. An implementation of this interface
   * can be used with the solution strategies in solvers.hh
   */
  class KrylovSolverBase {
   public:
    //! underlying vector type
    using Vector_t = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    //! Input vector for solvers
    using Vector_ref = Eigen::Ref<Vector_t>;
    //! Input vector for solvers
    using ConstVector_ref = Eigen::Ref<const Vector_t>;
    //! Output vector for solvers
    using Vector_map = Eigen::Map<const Vector_t>;
    //! Resons for convergence
    enum class Convergence {
      DidNotConverge,
      ReachedTolerance,
      HessianNotPositiveDefinite,
      ExceededTrustRegionBound
    };

    //! Default constructor
    KrylovSolverBase() = delete;

    /**
     * Constructor takes a Cell, tolerance, max number of iterations
     * and verbosity flag as input
     */
    KrylovSolverBase(Cell & cell, Real tol, Uint maxiter,
                     Verbosity verbose = Verbosity::Silent);

    //! Copy constructor
    KrylovSolverBase(const KrylovSolverBase & other) = delete;

    //! Move constructor
    KrylovSolverBase(KrylovSolverBase && other) = default;

    //! Destructor
    virtual ~KrylovSolverBase() = default;

    //! Copy assignment operator
    KrylovSolverBase & operator=(const KrylovSolverBase & other) = delete;

    //! Move assignment operator
    KrylovSolverBase & operator=(KrylovSolverBase && other) = delete;

    //! Allocate fields used during the solution
    virtual void initialise() = 0;

    //! returns whether the solver has converged
    Convergence get_convergence() const;

    //! reset the iteration counter to zero
    void reset_counter();

    //! set size of trust region, exception if not supported
    virtual void set_trust_region(Real new_trust_region);

    //! get the count of how many solve steps have been executed since
    //! construction of most recent counter reset
    Uint get_counter() const;

    //! returns the max number of iterations
    Uint get_maxiter() const;

    //! returns the solving tolerance
    Real get_tol() const;

    //! returns the solver's name (i.e. 'CG', 'GMRES', etc)
    virtual std::string get_name() const = 0;

    //! run the solve operation
    virtual Vector_map solve(const ConstVector_ref rhs) = 0;

   protected:
    Cell & cell;            //!< reference to the problem's cell
    Real tol;               //!< convergence tolerance
    Uint maxiter;           //!< maximum allowed number of iterations
    Verbosity verbose;      //!< how much information is written to the stdout
    Uint counter{0};        //!< iteration counter
    Convergence convergence{
        Convergence::DidNotConverge};  //!< whether the solver has converged
  };

}  // namespace muSpectre

#endif  // SRC_SOLVER_KRYLOV_SOLVER_BASE_HH_
