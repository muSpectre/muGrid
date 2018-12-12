/**
 * @file   deprecated_solver_cg_eigen.hh
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
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
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
 */

#ifndef SRC_SOLVER_DEPRECATED_SOLVER_CG_EIGEN_HH_
#define SRC_SOLVER_DEPRECATED_SOLVER_CG_EIGEN_HH_

#include "solver/deprecated_solver_base.hh"

#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

#include <type_traits>

namespace muSpectre {
  template <class DeprecatedSolverType, Dim_t DimS, Dim_t DimM>
  class DeprecatedSolverEigen;

  template <Dim_t DimS, Dim_t DimM = DimS> class DeprecatedSolverCGEigen;

  template <Dim_t DimS, Dim_t DimM = DimS> class DeprecatedSolverGMRESEigen;

  template <Dim_t DimS, Dim_t DimM = DimS> class DeprecatedSolverBiCGSTABEigen;

  template <Dim_t DimS, Dim_t DimM = DimS> class DeprecatedSolverDGMRESEigen;

  template <Dim_t DimS, Dim_t DimM = DimS> class DeprecatedSolverMINRESEigen;

  namespace internal {

    template <class DeprecatedSolver> struct DeprecatedSolver_traits {};

    //! traits for the Eigen conjugate gradient solver
    template <Dim_t DimS, Dim_t DimM>
    struct DeprecatedSolver_traits<DeprecatedSolverCGEigen<DimS, DimM>> {
      //! Eigen Iterative DeprecatedSolver
      using DeprecatedSolver = Eigen::ConjugateGradient<
          typename DeprecatedSolverEigen<DeprecatedSolverCGEigen<DimS, DimM>,
                                         DimS, DimM>::Adaptor,
          Eigen::Lower | Eigen::Upper, Eigen::IdentityPreconditioner>;
    };

    //! traits for the Eigen GMRES solver
    template <Dim_t DimS, Dim_t DimM>
    struct DeprecatedSolver_traits<DeprecatedSolverGMRESEigen<DimS, DimM>> {
      //! Eigen Iterative DeprecatedSolver
      using DeprecatedSolver = Eigen::GMRES<
          typename DeprecatedSolverEigen<DeprecatedSolverGMRESEigen<DimS, DimM>,
                                         DimS, DimM>::Adaptor,
          Eigen::IdentityPreconditioner>;
    };

    //! traits for the Eigen BiCGSTAB solver
    template <Dim_t DimS, Dim_t DimM>
    struct DeprecatedSolver_traits<DeprecatedSolverBiCGSTABEigen<DimS, DimM>> {
      //! Eigen Iterative DeprecatedSolver
      using DeprecatedSolver = Eigen::BiCGSTAB<
          typename DeprecatedSolverEigen<
              DeprecatedSolverBiCGSTABEigen<DimS, DimM>, DimS, DimM>::Adaptor,
          Eigen::IdentityPreconditioner>;
    };

    //! traits for the Eigen DGMRES solver
    template <Dim_t DimS, Dim_t DimM>
    struct DeprecatedSolver_traits<DeprecatedSolverDGMRESEigen<DimS, DimM>> {
      //! Eigen Iterative DeprecatedSolver
      using DeprecatedSolver = Eigen::DGMRES<
          typename DeprecatedSolverEigen<
              DeprecatedSolverDGMRESEigen<DimS, DimM>, DimS, DimM>::Adaptor,
          Eigen::IdentityPreconditioner>;
    };

    //! traits for the Eigen MINRES solver
    template <Dim_t DimS, Dim_t DimM>
    struct DeprecatedSolver_traits<DeprecatedSolverMINRESEigen<DimS, DimM>> {
      //! Eigen Iterative DeprecatedSolver
      using DeprecatedSolver = Eigen::MINRES<
          typename DeprecatedSolverEigen<
              DeprecatedSolverMINRESEigen<DimS, DimM>, DimS, DimM>::Adaptor,
          Eigen::Lower | Eigen::Upper, Eigen::IdentityPreconditioner>;
    };

  }  // namespace internal

  /**
   * base class for iterative solvers from Eigen
   */
  template <class DeprecatedSolverType, Dim_t DimS, Dim_t DimM = DimS>
  class DeprecatedSolverEigen : public DeprecatedSolverBase<DimS, DimM> {
   public:
    using Parent = DeprecatedSolverBase<DimS, DimM>;  //!< base class
    //! Input vector for solvers
    using SolvVectorIn = typename Parent::SolvVectorIn;
    //! Input vector for solvers
    using SolvVectorInC = typename Parent::SolvVectorInC;
    //! Output vector for solvers
    using SolvVectorOut = typename Parent::SolvVectorOut;
    using Cell_t = typename Parent::Cell_t;  //!< cell type
    using Ccoord = typename Parent::Ccoord;  //!< cell coordinates type
    //! kind of tangent that is required
    using Tg_req_t = typename Parent::TangentRequirement;
    //! handle for the cell to fit Eigen's sparse matrix interface
    using Adaptor = typename Cell_t::Adaptor;
    //! traits obtained from CRTP
    using DeprecatedSolver = typename internal::DeprecatedSolver_traits<
        DeprecatedSolverType>::DeprecatedSolver;

    //! All Eigen solvers need directional stiffness
    constexpr static Tg_req_t tangent_requirement{Tg_req_t::NeedEffect};

    //! Default constructor
    DeprecatedSolverEigen() = delete;

    //! Constructor with domain resolutions, etc,
    DeprecatedSolverEigen(Cell_t &cell, Real tol, Uint maxiter = 0,
                          bool verbose = false);

    //! Copy constructor
    DeprecatedSolverEigen(const DeprecatedSolverEigen &other) = delete;

    //! Move constructor
    DeprecatedSolverEigen(DeprecatedSolverEigen &&other) = default;

    //! Destructor
    virtual ~DeprecatedSolverEigen() = default;

    //! Copy assignment operator
    DeprecatedSolverEigen &
    operator=(const DeprecatedSolverEigen &other) = delete;

    //! Move assignment operator
    DeprecatedSolverEigen &operator=(DeprecatedSolverEigen &&other) = default;

    //! returns whether the solver has converged
    bool has_converged() const final {
      return this->solver.info() == Eigen::Success;
    }

    //! Allocate fields used during the solution
    void initialise() final;

    //! executes the solver
    SolvVectorOut solve(const SolvVectorInC rhs,
                        SolvVectorIn x_0) final;

   protected:
    //! returns `muSpectre::Tg_req_t::NeedEffect`
    Tg_req_t get_tangent_req() const final;
    Adaptor adaptor;          //!< cell handle
    DeprecatedSolver solver;  //!< Eigen's Iterative solver
  };

  /**
   * Binding to Eigen's conjugate gradient solver
   */
  template <Dim_t DimS, Dim_t DimM>
  class DeprecatedSolverCGEigen
      : public DeprecatedSolverEigen<DeprecatedSolverCGEigen<DimS, DimM>, DimS,
                                     DimM> {
   public:
    using DeprecatedSolverEigen<DeprecatedSolverCGEigen<DimS, DimM>, DimS,
                                DimM>::DeprecatedSolverEigen;
    std::string name() const final { return "CG"; }
  };

  /**
   * Binding to Eigen's GMRES solver
   */
  template <Dim_t DimS, Dim_t DimM>
  class DeprecatedSolverGMRESEigen
      : public DeprecatedSolverEigen<DeprecatedSolverGMRESEigen<DimS, DimM>,
                                     DimS, DimM> {
   public:
    using DeprecatedSolverEigen<DeprecatedSolverGMRESEigen<DimS, DimM>, DimS,
                                DimM>::DeprecatedSolverEigen;
    std::string name() const final { return "GMRES"; }
  };

  /**
   * Binding to Eigen's BiCGSTAB solver
   */
  template <Dim_t DimS, Dim_t DimM>
  class DeprecatedSolverBiCGSTABEigen
      : public DeprecatedSolverEigen<DeprecatedSolverBiCGSTABEigen<DimS, DimM>,
                                     DimS, DimM> {
   public:
    using DeprecatedSolverEigen<DeprecatedSolverBiCGSTABEigen<DimS, DimM>, DimS,
                                DimM>::DeprecatedSolverEigen;
    //! DeprecatedSolver's name
    std::string name() const final { return "BiCGSTAB"; }
  };

  /**
   * Binding to Eigen's DGMRES solver
   */
  template <Dim_t DimS, Dim_t DimM>
  class DeprecatedSolverDGMRESEigen
      : public DeprecatedSolverEigen<DeprecatedSolverDGMRESEigen<DimS, DimM>,
                                     DimS, DimM> {
   public:
    using DeprecatedSolverEigen<DeprecatedSolverDGMRESEigen<DimS, DimM>, DimS,
                                DimM>::DeprecatedSolverEigen;
    //! DeprecatedSolver's name
    std::string name() const final { return "DGMRES"; }
  };

  /**
   * Binding to Eigen's MINRES solver
   */
  template <Dim_t DimS, Dim_t DimM>
  class DeprecatedSolverMINRESEigen
      : public DeprecatedSolverEigen<DeprecatedSolverMINRESEigen<DimS, DimM>,
                                     DimS, DimM> {
   public:
    using DeprecatedSolverEigen<DeprecatedSolverMINRESEigen<DimS, DimM>, DimS,
                                DimM>::DeprecatedSolverEigen;
    //! DeprecatedSolver's name
    std::string name() const final { return "MINRES"; }
  };

}  // namespace muSpectre

#endif  // SRC_SOLVER_DEPRECATED_SOLVER_CG_EIGEN_HH_
