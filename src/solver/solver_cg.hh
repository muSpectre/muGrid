/**
 * file   solver_cg.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   20 Dec 2017
 *
 * @brief class for a simple implementation of a conjugate gradient
 *        solver. This follows algorithm 5.2 in Nocedal's Numerical
 *        Optimization (p 112)
 *
 * @section LICENCE
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

#ifndef SOLVER_CG_H
#define SOLVER_CG_H

#include "solver/solver_base.hh"
#include "common/field.hh"

#include <functional>

namespace muSpectre {

  template <Dim_t DimS, Dim_t DimM=DimS>
  class SolverCG: public SolverBase<DimS, DimM>
  {
  public:
    using Parent = SolverBase<DimS, DimM>;
    using Ccoord = typename Parent::Ccoord;
    using Tg_req_t = typename Parent::TangentRequirement;
    // cg only needs to handle fields that look like strain and stress
    using Field_t = TensorField<
      typename Parent::Collection_t, Real, secondOrder, DimM>;
    using Fun_t = std::function<void(const Field_t& in, Field_t & out)>;

    constexpr static Tg_req_t tangent_requirement{Tg_req_t::NeedEffect};
    //! Default constructor
    SolverCG() = delete;

    //! Constructor with domain resolutions, etc,
    SolverCG(Ccoord resolutions, Real tol, Uint maxiter=0, bool verbose =false);

    //! Copy constructor
    SolverCG(const SolverCG &other) = delete;

    //! Move constructor
    SolverCG(SolverCG &&other) = default;

    //! Destructor
    virtual ~SolverCG() = default;

    //! Copy assignment operator
    SolverCG& operator=(const SolverCG &other) = delete;

    //! Move assignment operator
    SolverCG& operator=(SolverCG &&other) = default;

    //! actual solver
    void solve(const Fun_t & tangent_effect,
               const Field_t & rhs,
               Field_t & x);

    //! reset the iteration counter to zero
    void reset_counter();

    //! get the count of how many solve steps have been executed since
    //! construction of most recent counter reset
    Uint get_counter() const;



  protected:
    Tg_req_t get_tangent_req() const override final;
    Field_t & r_k; // residual
    Field_t & p_k; // search direction
    Field_t & Ap_k; // effect of tangent on search direction
    const Real tol;
    const Uint maxiter;
    const bool verbose;
    Uint counter{};
  private:
  };

}  // muSpectre

#endif /* SOLVER_CG_H */
