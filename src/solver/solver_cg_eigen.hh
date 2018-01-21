/**
 * file   solver_cg_eigen.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   19 Jan 2018
 *
 * @brief  binding to Eigen's conjugate gradient solver
 *
 * @section LICENCE
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


    template <Dim_t DimS, Dim_t DimM>
    struct Solver_traits<SolverCGEigen<DimS, DimM>> {
      using Solver =
        Eigen::ConjugateGradient<typename SolverEigen<SolverCGEigen<DimS, DimM>,
                                                      DimS, DimM>::Adaptor,
                                 Eigen::Lower|Eigen::Upper,
                                 Eigen::IdentityPreconditioner>;
    };

    template <Dim_t DimS, Dim_t DimM>
    struct Solver_traits<SolverGMRESEigen<DimS, DimM>> {
      using Solver =
        Eigen::GMRES<typename SolverEigen<SolverGMRESEigen<DimS, DimM>,
                                          DimS, DimM>::Adaptor,
                     Eigen::IdentityPreconditioner>;
    };

    template <Dim_t DimS, Dim_t DimM>
    struct Solver_traits<SolverBiCGSTABEigen<DimS, DimM>> {
      using Solver =
        Eigen::BiCGSTAB<typename SolverEigen<SolverBiCGSTABEigen<DimS, DimM>,
                                             DimS, DimM>::Adaptor,
                        Eigen::IdentityPreconditioner>;
    };

    template <Dim_t DimS, Dim_t DimM>
    struct Solver_traits<SolverDGMRESEigen<DimS, DimM>> {
      using Solver =
        Eigen::DGMRES<typename SolverEigen<SolverDGMRESEigen<DimS, DimM>,
                                           DimS, DimM>::Adaptor,
                      Eigen::IdentityPreconditioner>;
    };

    template <Dim_t DimS, Dim_t DimM>
    struct Solver_traits<SolverMINRESEigen<DimS, DimM>> {
      using Solver =
        Eigen::MINRES<typename SolverEigen<SolverMINRESEigen<DimS, DimM>,
                                           DimS, DimM>::Adaptor,
                      Eigen::Lower|Eigen::Upper,
                      Eigen::IdentityPreconditioner>;
    };

  }  // internal

  template <class SolverType, Dim_t DimS, Dim_t DimM=DimS>
  class SolverEigen: public SolverBase<DimS, DimM>
  {
  public:
    using Parent = SolverBase<DimS, DimM>;
    using SolvVectorIn = typename Parent::SolvVectorIn;
    using SolvVectorInC = typename Parent::SolvVectorInC;
    using SolvVectorOut = typename Parent::SolvVectorOut;
    using Sys_t = typename Parent::Sys_t;
    using Ccoord = typename Parent::Ccoord;
    using Tg_req_t = typename Parent::TangentRequirement;
    using Adaptor = typename Sys_t::Adaptor;
    using Solver = typename internal::Solver_traits<SolverType>::Solver;

    constexpr static Tg_req_t tangent_requirement{Tg_req_t::NeedEffect};

        //! Default constructor
    SolverEigen() = delete;

    //! Constructor with domain resolutions, etc,
    SolverEigen(Sys_t& sys, Real tol, Uint maxiter=0, bool verbose =false);

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

    bool has_converged() const override final {return this->solver.info() == Eigen::Success;}

    void initialise() override final;

    SolvVectorOut solve(const SolvVectorInC rhs, SolvVectorIn x_0) override final;


  protected:
    Tg_req_t get_tangent_req() const override final;
    Adaptor adaptor;
    Solver solver;

  };

  template <Dim_t DimS, Dim_t DimM>
  class SolverCGEigen:
    public SolverEigen<SolverCGEigen<DimS, DimM>, DimS, DimM> {
  public:
    using SolverEigen<SolverCGEigen<DimS, DimM>, DimS, DimM>::SolverEigen;
    std::string name() const override final {return "CG";}
  };

  template <Dim_t DimS, Dim_t DimM>
  class SolverGMRESEigen:
    public SolverEigen<SolverGMRESEigen<DimS, DimM>, DimS, DimM> {
  public:
    using SolverEigen<SolverGMRESEigen<DimS, DimM>, DimS, DimM>::SolverEigen;
    std::string name() const override final {return "GMRES";}
  };

  template <Dim_t DimS, Dim_t DimM>
  class SolverBiCGSTABEigen:
    public SolverEigen<SolverBiCGSTABEigen<DimS, DimM>, DimS, DimM> {
  public:
    using SolverEigen<SolverBiCGSTABEigen<DimS, DimM>, DimS, DimM>::SolverEigen;
    std::string name() const override final {return "BiCGSTAB";}
  };

  template <Dim_t DimS, Dim_t DimM>
  class SolverDGMRESEigen:
    public SolverEigen<SolverDGMRESEigen<DimS, DimM>, DimS, DimM> {
  public:
    using SolverEigen<SolverDGMRESEigen<DimS, DimM>, DimS, DimM>::SolverEigen;
    std::string name() const override final {return "DGMRES";}
  };

  template <Dim_t DimS, Dim_t DimM>
  class SolverMINRESEigen:
    public SolverEigen<SolverMINRESEigen<DimS, DimM>, DimS, DimM> {
  public:
    using SolverEigen<SolverMINRESEigen<DimS, DimM>, DimS, DimM>::SolverEigen;
    std::string name() const override final {return "MINRES";}
  };

} // muSpectre

#endif /* SOLVER_EIGEN_H */
