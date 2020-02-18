/**
 * @file   krylov_solver_eigen.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   15 May 2018
 *
 * @brief  Bindings to Eigen's iterative solvers
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

#ifndef SRC_SOLVER_KRYLOV_SOLVER_EIGEN_HH_
#define SRC_SOLVER_KRYLOV_SOLVER_EIGEN_HH_

#include "solver/krylov_solver_base.hh"
#include "cell/cell.hh"
#include "cell/cell_adaptor.hh"

#include <Eigen/IterativeLinearSolvers>
#include <iostream>
#include <unsupported/Eigen/IterativeSolvers>

namespace muSpectre {

  template <class KrylovSolverType>
  class KrylovSolverEigen;

  class KrylovSolverCGEigen;

  class KrylovSolverGMRESEigen;

  class KrylovSolverBiCGSTABEigen;

  class KrylovSolverDGMRESEigen;

  class KrylovSolverMINRESEigen;

  namespace internal {

    template <class KrylovSolver>
    struct KrylovSolver_traits {};

    //! traits for the Eigen conjugate gradient solver
    template <>
    struct KrylovSolver_traits<KrylovSolverCGEigen> {
      //! Eigen Iterative KrylovSolver
      using KrylovSolver =
          Eigen::ConjugateGradient<typename Cell::Adaptor,
                                   Eigen::Lower | Eigen::Upper,
                                   Eigen::IdentityPreconditioner>;
    };

    //! traits for the Eigen GMRES solver
    template <>
    struct KrylovSolver_traits<KrylovSolverGMRESEigen> {
      //! Eigen Iterative KrylovSolver
      using KrylovSolver =
          Eigen::GMRES<typename Cell::Adaptor, Eigen::IdentityPreconditioner>;
    };

    //! traits for the Eigen BiCGSTAB solver
    template <>
    struct KrylovSolver_traits<KrylovSolverBiCGSTABEigen> {
      //! Eigen Iterative KrylovSolver
      using KrylovSolver = Eigen::BiCGSTAB<typename Cell::Adaptor,
                                           Eigen::IdentityPreconditioner>;
    };

    //! traits for the Eigen DGMRES solver
    template <>
    struct KrylovSolver_traits<KrylovSolverDGMRESEigen> {
      //! Eigen Iterative KrylovSolver
      using KrylovSolver =
          Eigen::DGMRES<typename Cell::Adaptor, Eigen::IdentityPreconditioner>;
    };

    //! traits for the Eigen MINRES solver
    template <>
    struct KrylovSolver_traits<KrylovSolverMINRESEigen> {
      //! Eigen Iterative KrylovSolver
      using KrylovSolver =
          Eigen::MINRES<typename Cell::Adaptor, Eigen::Lower | Eigen::Upper,
                        Eigen::IdentityPreconditioner>;
    };

  }  // namespace internal

  /**
   * base class for iterative solvers from Eigen
   */
  template <class KrylovSolverType>
  class KrylovSolverEigen : public KrylovSolverBase {
   public:
    using Parent = KrylovSolverBase;  //!< base class
    //! traits obtained from CRTP
    using KrylovSolver =
        typename internal::KrylovSolver_traits<KrylovSolverType>::KrylovSolver;
    //! Input vectors for solver
    using ConstVector_ref = Parent::ConstVector_ref;
    //! Output vector for solver
    using Vector_map = Parent::Vector_map;
    //! storage for output vector
    using Vector_t = Parent::Vector_t;

    //! Default constructor
    KrylovSolverEigen() = delete;

    //! Constructor with cell and solver parameters.
    KrylovSolverEigen(Cell & cell, Real tol, Uint maxiter = 0,
                      Verbosity verbose = Verbosity::Silent);

    //! Copy constructor
    KrylovSolverEigen(const KrylovSolverEigen & other) = delete;

    //! Move constructor
    KrylovSolverEigen(KrylovSolverEigen && other) = default;

    //! Destructor
    virtual ~KrylovSolverEigen() = default;

    //! Copy assignment operator
    KrylovSolverEigen & operator=(const KrylovSolverEigen & other) = delete;

    //! Move assignment operator
    KrylovSolverEigen & operator=(KrylovSolverEigen && other) = default;

    //! Allocate fields used during the solution
    void initialise() final;

    //! executes the solver
    Vector_map solve(const ConstVector_ref rhs) final;

   protected:
    Cell::Adaptor adaptor;  //!< cell handle
    KrylovSolver solver;    //!< Eigen's Iterative solver
    Vector_t result;        //!< storage for result
  };

  /**
   * Binding to Eigen's conjugate gradient solver
   */
  class KrylovSolverCGEigen : public KrylovSolverEigen<KrylovSolverCGEigen> {
   public:
    using KrylovSolverEigen<KrylovSolverCGEigen>::KrylovSolverEigen;
    std::string get_name() const final { return "CG"; }
  };

  /**
   * Binding to Eigen's GMRES solver
   */
  class KrylovSolverGMRESEigen
      : public KrylovSolverEigen<KrylovSolverGMRESEigen> {
   public:
    using KrylovSolverEigen<KrylovSolverGMRESEigen>::KrylovSolverEigen;
    std::string get_name() const final { return "GMRES"; }
  };

  /**
   * Binding to Eigen's BiCGSTAB solver
   */
  class KrylovSolverBiCGSTABEigen
      : public KrylovSolverEigen<KrylovSolverBiCGSTABEigen> {
   public:
    using KrylovSolverEigen<KrylovSolverBiCGSTABEigen>::KrylovSolverEigen;
    //! KrylovSolver's name
    std::string get_name() const final { return "BiCGSTAB"; }
  };

  /**
   * Binding to Eigen's DGMRES solver
   */
  class KrylovSolverDGMRESEigen
      : public KrylovSolverEigen<KrylovSolverDGMRESEigen> {
   public:
    using KrylovSolverEigen<KrylovSolverDGMRESEigen>::KrylovSolverEigen;
    //! KrylovSolver's name
    std::string get_name() const final { return "DGMRES"; }
  };

  /**
   * Binding to Eigen's MINRES solver
   */
  class KrylovSolverMINRESEigen
      : public KrylovSolverEigen<KrylovSolverMINRESEigen> {
   public:
    using KrylovSolverEigen<KrylovSolverMINRESEigen>::KrylovSolverEigen;
    //! KrylovSolver's name
    std::string get_name() const final { return "MINRES"; }
  };

}  // namespace muSpectre

#endif  // SRC_SOLVER_KRYLOV_SOLVER_EIGEN_HH_
