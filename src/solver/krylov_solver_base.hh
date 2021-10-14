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

#include "solver_common.hh"
#include "matrix_adaptor.hh"

#include <libmugrid/mapped_field.hh>

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
     * Constructor takes a Matrix adaptable, tolerance, max number of iterations
     * and verbosity flag as input. The solver takes responsibility for keeping
     * the system matrix from destruction at least until itself is destroyed.
     */
    KrylovSolverBase(std::shared_ptr<MatrixAdaptable> matrix_adaptable,
                     const Real & tol, const Uint & maxiter,
                     const Verbosity & verbose = Verbosity::Silent);

    /**
     * Constructor takes a Matrix adaptable, tolerance, max number of iterations
     * and verbosity flag as input. The solver does not take any responsibility
     * for keeping the matrix from destruction
     */
    KrylovSolverBase(std::weak_ptr<MatrixAdaptable> matrix_adaptable,
                     const Real & tol, const Uint & maxiter,
                     const Verbosity & verbose = Verbosity::Silent);

    /**
     * Constructor without matrix adaptable. The adaptable has to be supplied
     * using KrylovSolverBase::set_matrix(...) before initialisation for this
     * solver to be usable
     */
    KrylovSolverBase(const Real & tol, const Uint & maxiter,
                     const Verbosity & verbose = Verbosity::Silent);

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

    /**
     * Set the matrix. The solver will take responsibility to keep the matrix
     * from destruction
     */
    virtual void set_matrix(std::shared_ptr<MatrixAdaptable> matrix_adaptable);

    /**
     * Set the matrix. The solver will not take responsibility to keep the
     * matrix from destruction. Use this to avoid cyclic dependencies
     */
    virtual void set_matrix(std::weak_ptr<MatrixAdaptable> matrix_adaptable);

    //! returns whether the solver has converged
    Convergence get_convergence() const;

    //! reset the iteration counter to zero
    void reset_counter();

    //! get the count of how many solve steps have been executed since
    //! construction of most recent counter reset
    Uint get_counter() const;

    //! returns the max number of iterations
    Uint get_maxiter() const;

    //! returns the solving tolerance
    Real get_tol() const;

    //! return the holder of the matrix of the  cell
    std::shared_ptr<MatrixAdaptable> get_matrix_holder() const;
    std::weak_ptr<MatrixAdaptable> get_matrix_ptr() const;

    //! returns the solver's name (i.e. 'CG', 'GMRES', etc)
    virtual std::string get_name() const = 0;

    //! run the solve operation
    virtual Vector_map solve(const ConstVector_ref rhs) = 0;

    Index_t get_nb_dof() const;

    //! calculates the squared norm of a vector distributed memory safely
    Real squared_norm(const Vector_t & vec);

    //! calculates the dot product of  two vectors distributed memory safely
    Real dot(const Vector_t & vec_a, const Vector_t & vec_b);

   protected:
    std::shared_ptr<MatrixAdaptable> matrix_holder{nullptr};  //!< system matrix
    std::weak_ptr<MatrixAdaptable> matrix_ptr{};  //!< weak ref to matrix
    MatrixAdaptor matrix{};  //!< matrix ref for convenience
    muGrid::Communicator comm{};
    Real tol;                //!< convergence tolerance
    Uint maxiter;            //!< maximum allowed number of iterations
    Verbosity verbose;       //!< how much information is written to the stdout
    Uint counter{0};         //!< iteration counter
    Convergence convergence{
        Convergence::DidNotConverge};  //!< whether the solver has converged
  };

}  // namespace muSpectre

#endif  // SRC_SOLVER_KRYLOV_SOLVER_BASE_HH_
