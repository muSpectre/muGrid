/**
 * file   solver_base.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   18 Dec 2017
 *
 * @brief  Base class for solvers
 *
 * @section LICENCE
 *
 * Copyright (C) 2017 Till Junge
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

#include <vector>

#include <Eigen/Dense>

#include "common/common.hh"
#include "system/system_base.hh"
#include "common/tensor_algebra.hh"

namespace muSpectre {
  template <Dim_t Dim>
  using Grad_t = Matrices::Tens2_t<Dim>;
  template <Dim_t Dim>
  using GradIncrements = std::vector<Grad_t<Dim>,
                                     Eigen::aligned_allocator<Grad_t<Dim>>>;

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM=DimS>
  typename SystemBase<DimS, DimM>::StrainField_t &
  newton_cg (SystemBase<DimS, DimM> & sys, const Grad_t<DimM> & delF0,
             const Real cg_tol, const Real newton_tol, const Uint maxiter=0,
             bool verbose = false){
    return newton_cg(sys, GradIncrements<DimM>{delF0},
                    cg_tol, newton_tol, maxiter, verbose);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM=DimS>
  typename SystemBase<DimS, DimM>::StrainField_t &
  newton_cg (SystemBase<DimS, DimM> & sys,
             const GradIncrements<DimM> & delF0,
             const Real cg_tol, const Real newton_tol, const Uint maxiter=0,
             bool verbose = false);

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM=DimS>
  class SolverBase
  {
  public:
    enum class TangentRequirement{NoNeed, NeedEffect, NeedTangents};
    using Ccoord = Ccoord_t<DimS>;
    using Collection_t = GlobalFieldCollection<DimS, DimM>;

    //! Default constructor
    SolverBase() = delete;

    //! Constructor with domain resolutions
    SolverBase(Ccoord resolutions): resolutions{resolutions} {}

    //! Copy constructor
    SolverBase(const SolverBase &other) = delete;

    //! Move constructor
    SolverBase(SolverBase &&other) = default;

    //! Destructor
    virtual ~SolverBase() noexcept = default;

    //! Copy assignment operator
    SolverBase& operator=(const SolverBase &other) = delete;

    //! Move assignment operator
    SolverBase& operator=(SolverBase &&other) noexcept = default;

    //! Allocate fields used during the solution
    void initialise() {this->collection.initialise(resolutions);}

    bool need_tangents() const {
      return (this->get_tangent_req() == TangentRequirement::NeedTangents);}

    bool need_effect() const {
      return (this->get_tangent_req() == TangentRequirement::NeedEffect);}

    bool no_need_tangent() const {
      return (this->get_tangent_req() == TangentRequirement::NoNeed);}

    bool has_converged() const {return this->converged;}

  protected:
    virtual TangentRequirement get_tangent_req() const = 0;
    Ccoord resolutions;
    //! storage for internal fields to avoid reallocations between calls
    Collection_t collection{};
    bool converged{false};
  private:
  };

}  // muSpectre

#endif /* SOLVER_BASE_H */