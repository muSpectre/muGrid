/**
 * @file   deprecated_solver_cg.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   20 Dec 2017
 *
 * @brief class for a simple implementation of a conjugate gradient
 *        solver. This follows algorithm 5.2 in Nocedal's Numerical
 *        Optimization (p 112)
 *
 * Copyright © 2017 Till Junge
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

#ifndef DEPRECATED_SOLVER_CG_H
#define DEPRECATED_SOLVER_CG_H

#include "solver/deprecated_solver_base.hh"
#include "common/communicator.hh"
#include "common/field.hh"

#include <functional>

namespace muSpectre {

  /**
   * implements the `muSpectre::DeprecatedSolverBase` interface using a
   * conjugate gradient solver. This particular class is useful for
   * trouble shooting, as it can be made very verbose, but for
   * production runs, it is probably better to use
   * `muSpectre::DeprecatedSolverCGEigen`.
   */
  template <Dim_t DimS, Dim_t DimM=DimS>
  class DeprecatedSolverCG: public DeprecatedSolverBase<DimS, DimM>
  {
  public:
    using Parent = DeprecatedSolverBase<DimS, DimM>; //!< base class
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
    //! cg only needs to handle fields that look like strain and stress
    using Field_t = TensorField<
      typename Parent::Collection_t, Real, secondOrder, DimM>;

    //! conjugate gradient needs directional stiffness
    constexpr static Tg_req_t tangent_requirement{Tg_req_t::NeedEffect};
    //! Default constructor
    DeprecatedSolverCG() = delete;

    //! Constructor with domain resolutions, etc,
    DeprecatedSolverCG(Cell_t& cell, Real tol, Uint maxiter=0, bool verbose=false);

    //! Copy constructor
    DeprecatedSolverCG(const DeprecatedSolverCG &other) = delete;

    //! Move constructor
    DeprecatedSolverCG(DeprecatedSolverCG &&other) = default;

    //! Destructor
    virtual ~DeprecatedSolverCG() = default;

    //! Copy assignment operator
    DeprecatedSolverCG& operator=(const DeprecatedSolverCG &other) = delete;

    //! Move assignment operator
    DeprecatedSolverCG& operator=(DeprecatedSolverCG &&other) = default;

    bool has_converged() const override final {return this->converged;}

    //! actual solver
    void solve(const Field_t & rhs,
               Field_t & x);

    // this simplistic implementation has no initialisation phase so the default is ok

    SolvVectorOut solve(const SolvVectorInC rhs, SolvVectorIn x_0) override final;

    std::string name() const override final {return "CG";}

  protected:
    //! returns `muSpectre::Tg_req_t::NeedEffect`
    Tg_req_t get_tangent_req() const override final;
    Field_t & r_k;  //!< residual
    Field_t & p_k;  //!< search direction
    Field_t & Ap_k; //!< effect of tangent on search direction
    bool converged{false}; //!< whether the solver has converged
  private:
  };

}  // muSpectre

#endif /* DEPRECATED_SOLVER_CG_H */
