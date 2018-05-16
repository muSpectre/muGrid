/**
 * file   solver_eigen.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   15 May 2018
 *
 * @brief  Bindings to Eigen's iterative solvers
 *
 * @section LICENSE
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

#ifndef SOLVER_EIGEN_H
#define SOLVER_EIGEN_H

#include "solver/solver_base.hh"
#include "cell/cell_base.hh"

#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

namespace muSpectre {

  template <class SolverType>
  class SolverEigen;

  class SolverCGEigen;

  class SolverGMRESEigen;

  class SolverBiCGSTABEigen;

  class SolverDGMRESEigen;

  class SolverMINRESEigen;

  namespace internal {

    template <class Solver>
    struct Solver_traits {
    };

    //! traits for the Eigen conjugate gradient solver
    template<>
    struct Solver_traits<SolverCGEigen> {
      //! Eigen Iterative Solver
      using Solver =
        Eigen::ConjugateGradient<typename Cell::Adaptor,
                                 Eigen::Lower|Eigen::Upper,
                                 Eigen::IdentityPreconditioner>;
    };

    //! traits for the Eigen GMRES solver
    template<>
    struct Solver_traits<SolverGMRESEigen> {
      //! Eigen Iterative Solver
      using Solver =
        Eigen::GMRES<typename Cell::Adaptor,
                     Eigen::IdentityPreconditioner>;
    };

    //! traits for the Eigen BiCGSTAB solver
    template<>
    struct Solver_traits<SolverBiCGSTABEigen> {
      //! Eigen Iterative Solver
      using Solver =
        Eigen::BiCGSTAB<typename Cell::Adaptor,
                        Eigen::IdentityPreconditioner>;
    };

    //! traits for the Eigen DGMRES solver
    template<>
    struct Solver_traits<SolverDGMRESEigen> {
      //! Eigen Iterative Solver
      using Solver =
        Eigen::DGMRES<typename Cell::Adaptor,
                      Eigen::IdentityPreconditioner>;
    };

    //! traits for the Eigen MINRES solver
    template<>
    struct Solver_traits<SolverMINRESEigen> {
      //! Eigen Iterative Solver
      using Solver =
        Eigen::MINRES<typename Cell::Adaptor,
                      Eigen::Lower|Eigen::Upper,
                      Eigen::IdentityPreconditioner>;
    };

  }  // internal

  /**
   * base class for iterative solvers from Eigen
   */
  template <class SolverType>
  class SolverEigen: public SolverBase
  {
  public:
    using Parent = SolverBase; //!< base class
    //! traits obtained from CRTP
    using Solver = typename internal::Solver_traits<SolverType>::Solver;
    //! Input vectors for solver
    using ConstVector_ref = Parent::ConstVector_ref;
    //! Output vector for solver
    using Vector_map = Parent::Vector_map;
    //! storage for output vector
    using Vector_t = Parent::Vector_t;

    //! Default constructor
    SolverEigen() = delete;

    //! Constructor with domain resolutions, etc,
    SolverEigen(Cell& cell, Real tol, Uint maxiter=0, bool verbose =false);

    //! Copy constructor
    SolverEigen(const SolverEigen &other) = delete;

    //! Move constructor
    SolverEigen(SolverEigen &&other) = default;

    //! Destructor
    virtual ~SolverEigen() = default;

    //! Copy assignment operator
    SolverEigen& operator=(const SolverEigen &other) = delete;

    //! Move assignment operator
    SolverEigen& operator=(SolverEigen &&other) = default;

    //! Allocate fields used during the solution
    void initialise() override final;

    //! executes the solver
    Vector_map solve(const ConstVector_ref rhs) override final;


  protected:
    Cell::Adaptor adaptor; //!< cell handle
    Solver solver; //!< Eigen's Iterative solver
    Vector_t result; //!< storage for result
  };

  /**
   * Binding to Eigen's conjugate gradient solver
   */
  class SolverCGEigen:
    public SolverEigen<SolverCGEigen> {
  public:
    using SolverEigen<SolverCGEigen>::SolverEigen;
    std::string get_name() const override final {return "CG";}
  };

  /**
   * Binding to Eigen's GMRES solver
   */
  class SolverGMRESEigen:
    public SolverEigen<SolverGMRESEigen> {
  public:
    using SolverEigen<SolverGMRESEigen>::SolverEigen;
    std::string get_name() const override final {return "GMRES";}
  };

  /**
   * Binding to Eigen's BiCGSTAB solver
   */
  class SolverBiCGSTABEigen:
    public SolverEigen<SolverBiCGSTABEigen> {
  public:
    using SolverEigen<SolverBiCGSTABEigen>::SolverEigen;
    //! Solver's name
    std::string get_name() const override final {return "BiCGSTAB";}
  };

  /**
   * Binding to Eigen's DGMRES solver
   */
  class SolverDGMRESEigen:
    public SolverEigen<SolverDGMRESEigen> {
  public:
    using SolverEigen<SolverDGMRESEigen>::SolverEigen;
    //! Solver's name
    std::string get_name() const override final {return "DGMRES";}
  };

  /**
   * Binding to Eigen's MINRES solver
   */
  class SolverMINRESEigen:
    public SolverEigen<SolverMINRESEigen> {
  public:
    using SolverEigen<SolverMINRESEigen>::SolverEigen;
    //! Solver's name
    std::string get_name() const override final {return "MINRES";}
  };

}  // muSpectre

#endif /* SOLVER_EIGEN_H */
