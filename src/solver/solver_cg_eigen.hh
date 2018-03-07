/**
 * @file   solver_cg_eigen.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   19 Jan 2018
 *
 * @brief  binding to Eigen's conjugate gradient solver
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

#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

#include <type_traits>

namespace muSpectre {
  template <class SolverType, Dim_t DimS, Dim_t DimM>
  class SolverEigen;

  template <Dim_t DimS, Dim_t DimM=DimS>
  class SolverCGEigen;

  template <Dim_t DimS, Dim_t DimM=DimS>
  class SolverGMRESEigen;

  template <Dim_t DimS, Dim_t DimM=DimS>
  class SolverBiCGSTABEigen;

  template <Dim_t DimS, Dim_t DimM=DimS>
  class SolverDGMRESEigen;

  template <Dim_t DimS, Dim_t DimM=DimS>
  class SolverMINRESEigen;

  namespace internal {

    template <class Solver>
    struct Solver_traits {
    };

    //! traits for the Eigen conjugate gradient solver
    template <Dim_t DimS, Dim_t DimM>
    struct Solver_traits<SolverCGEigen<DimS, DimM>> {
      //! Eigen Iterative Solver
      using Solver =
        Eigen::ConjugateGradient<typename SolverEigen<SolverCGEigen<DimS, DimM>,
                                                      DimS, DimM>::Adaptor,
                                 Eigen::Lower|Eigen::Upper,
                                 Eigen::IdentityPreconditioner>;
    };

    //! traits for the Eigen GMRES solver
    template <Dim_t DimS, Dim_t DimM>
    struct Solver_traits<SolverGMRESEigen<DimS, DimM>> {
      //! Eigen Iterative Solver
      using Solver =
        Eigen::GMRES<typename SolverEigen<SolverGMRESEigen<DimS, DimM>,
                                          DimS, DimM>::Adaptor,
                     Eigen::IdentityPreconditioner>;
    };

    //! traits for the Eigen BiCGSTAB solver
    template <Dim_t DimS, Dim_t DimM>
    struct Solver_traits<SolverBiCGSTABEigen<DimS, DimM>> {
      //! Eigen Iterative Solver
      using Solver =
        Eigen::BiCGSTAB<typename SolverEigen<SolverBiCGSTABEigen<DimS, DimM>,
                                             DimS, DimM>::Adaptor,
                        Eigen::IdentityPreconditioner>;
    };

    //! traits for the Eigen DGMRES solver
    template <Dim_t DimS, Dim_t DimM>
    struct Solver_traits<SolverDGMRESEigen<DimS, DimM>> {
      //! Eigen Iterative Solver
      using Solver =
        Eigen::DGMRES<typename SolverEigen<SolverDGMRESEigen<DimS, DimM>,
                                           DimS, DimM>::Adaptor,
                      Eigen::IdentityPreconditioner>;
    };

    //! traits for the Eigen MINRES solver
    template <Dim_t DimS, Dim_t DimM>
    struct Solver_traits<SolverMINRESEigen<DimS, DimM>> {
      //! Eigen Iterative Solver
      using Solver =
        Eigen::MINRES<typename SolverEigen<SolverMINRESEigen<DimS, DimM>,
                                           DimS, DimM>::Adaptor,
                      Eigen::Lower|Eigen::Upper,
                      Eigen::IdentityPreconditioner>;
    };

  }  // internal

  /**
   * base class for iterative solvers from Eigen
   */
  template <class SolverType, Dim_t DimS, Dim_t DimM=DimS>
  class SolverEigen: public SolverBase<DimS, DimM>
  {
  public:
    using Parent = SolverBase<DimS, DimM>; //!< base class
    //! Input vector for solvers
    using SolvVectorIn = typename Parent::SolvVectorIn;
    //! Input vector for solvers
    using SolvVectorInC = typename Parent::SolvVectorInC;
    //! Output vector for solvers
    using SolvVectorOut = typename Parent::SolvVectorOut;
    using Cell_t = typename Parent::Cell_t; //!< cell type
    using Ccoord = typename Parent::Ccoord; //!< cell coordinates type
    //! kind of tangent that is required
    using Tg_req_t = typename Parent::TangentRequirement;
    //! handle for the cell to fit Eigen's sparse matrix interface
    using Adaptor = typename Cell_t::Adaptor;
    //! traits obtained from CRTP
    using Solver = typename internal::Solver_traits<SolverType>::Solver;

    //! All Eigen solvers need directional stiffness
    constexpr static Tg_req_t tangent_requirement{Tg_req_t::NeedEffect};

        //! Default constructor
    SolverEigen() = delete;

    //! Constructor with domain resolutions, etc,
    SolverEigen(Cell_t& cell, Real tol, Uint maxiter=0, bool verbose =false);

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

    //! returns whether the solver has converged
    bool has_converged() const override final {return this->solver.info() == Eigen::Success;}

    //! Allocate fields used during the solution
    void initialise() override final;

    //! executes the solver
    SolvVectorOut solve(const SolvVectorInC rhs, SolvVectorIn x_0) override final;


  protected:
    //! returns `muSpectre::Tg_req_t::NeedEffect`
    Tg_req_t get_tangent_req() const override final; 
    Adaptor adaptor; //!< cell handle
    Solver solver; //!< Eigen's Iterative solver

  };

  /**
   * Binding to Eigen's conjugate gradient solver
   */
  template <Dim_t DimS, Dim_t DimM>
  class SolverCGEigen:
    public SolverEigen<SolverCGEigen<DimS, DimM>, DimS, DimM> {
  public:
    using SolverEigen<SolverCGEigen<DimS, DimM>, DimS, DimM>::SolverEigen;
    std::string name() const override final {return "CG";}
  };

  /**
   * Binding to Eigen's GMRES solver
   */
  template <Dim_t DimS, Dim_t DimM>
  class SolverGMRESEigen:
    public SolverEigen<SolverGMRESEigen<DimS, DimM>, DimS, DimM> {
  public:
    using SolverEigen<SolverGMRESEigen<DimS, DimM>, DimS, DimM>::SolverEigen;
    std::string name() const override final {return "GMRES";}
  };

  /**
   * Binding to Eigen's BiCGSTAB solver
   */
  template <Dim_t DimS, Dim_t DimM>
  class SolverBiCGSTABEigen:
    public SolverEigen<SolverBiCGSTABEigen<DimS, DimM>, DimS, DimM> {
  public:
    using SolverEigen<SolverBiCGSTABEigen<DimS, DimM>, DimS, DimM>::SolverEigen;
    //! Solver's name
    std::string name() const override final {return "BiCGSTAB";}
  };

  /**
   * Binding to Eigen's DGMRES solver
   */
  template <Dim_t DimS, Dim_t DimM>
  class SolverDGMRESEigen:
    public SolverEigen<SolverDGMRESEigen<DimS, DimM>, DimS, DimM> {
  public:
    using SolverEigen<SolverDGMRESEigen<DimS, DimM>, DimS, DimM>::SolverEigen;
    //! Solver's name
    std::string name() const override final {return "DGMRES";}
  };

  /**
   * Binding to Eigen's MINRES solver
   */
  template <Dim_t DimS, Dim_t DimM>
  class SolverMINRESEigen:
    public SolverEigen<SolverMINRESEigen<DimS, DimM>, DimS, DimM> {
  public:
    using SolverEigen<SolverMINRESEigen<DimS, DimM>, DimS, DimM>::SolverEigen;
    //! Solver's name
    std::string name() const override final {return "MINRES";}
  };

} // muSpectre

#endif /* SOLVER_EIGEN_H */
