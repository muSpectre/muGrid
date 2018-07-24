/**
 * file   solver_base.hh
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

#ifndef SOLVER_BASE_H
#define SOLVER_BASE_H

#include "solver/solver_common.hh"
#include "cell/cell_base.hh"

#include <Eigen/Dense>

namespace muSpectre {

  /**
   * Virtual base class for solvers. An implementation of this interface
   * can be used with the solution strategies in solvers.hh
   */
  class SolverBase
  {
  public:

    //! underlying vector type
    using Vector_t = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    //! Input vector for solvers
    using Vector_ref = Eigen::Ref<Vector_t>;
    //! Input vector for solvers
    using ConstVector_ref = Eigen::Ref<const Vector_t>;
    //! Output vector for solvers
    using Vector_map = Eigen::Map<Vector_t>;

    //! Default constructor
    SolverBase() = delete;

    /**
     * Constructor takes a Cell, tolerance, max number of iterations
     * and verbosity flag as input
     */
    SolverBase(Cell & cell, Real tol, Uint maxiter, bool verbose=false);

    //! Copy constructor
    SolverBase(const SolverBase &other) = delete;

    //! Move constructor
    SolverBase(SolverBase &&other) = default;

    //! Destructor
    virtual ~SolverBase() = default;

    //! Copy assignment operator
    SolverBase& operator=(const SolverBase &other) = delete;

    //! Move assignment operator
    SolverBase& operator=(SolverBase &&other) = default;

    //! Allocate fields used during the solution
    virtual void initialise() = 0;

    //! returns whether the solver has converged
    bool has_converged() const ;

    //! reset the iteration counter to zero
    void reset_counter();

    //! get the count of how many solve steps have been executed since
    //! construction of most recent counter reset
    Uint get_counter() const;

    //! returns the max number of iterations
    Uint get_maxiter() const;

    //! returns the resolution tolerance
    Real get_tol() const;

    //! returns the solver's name (i.e. 'CG', 'GMRES', etc)
    virtual std::string get_name() const = 0;

    //! run the solve operation
    virtual Vector_map solve(const ConstVector_ref rhs) = 0;

  protected:
    Cell & cell;           //!< reference to the problem's cell
    Real tol;              //!< convergence tolerance
    Uint maxiter;          //!< maximum allowed number of iterations
    bool verbose;          //!< whether to write information to the stdout
    Uint counter{0};       //!< iteration counter
    bool converged{false}; //!< whether the solver has converged

  private:
  };


}  // muSpectre

#endif /* SOLVER_BASE_H */
