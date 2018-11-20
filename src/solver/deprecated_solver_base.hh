/**
 * @file   deprecated_solver_base.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   18 Dec 2017
 *
 * @brief  Base class for solvers
 *
 * Copyright © 2017 Till Junge
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

#ifndef DEPRECATED_SOLVER_BASE_H
#define DEPRECATED_SOLVER_BASE_H

#include "solver/solver_common.hh"
#include "common/common.hh"
#include "cell/cell_base.hh"
#include "common/tensor_algebra.hh"

#include <Eigen/Dense>

#include <vector>

namespace muSpectre {
  /* ---------------------------------------------------------------------- */
  /**
   * Virtual base class for solvers. Any implementation of this interface can be used with the solver functions prototyped in solvers.hh
   */
  template <Dim_t DimS, Dim_t DimM=DimS>
  class DeprecatedSolverBase
  {
  public:
    /**
     * Enum to describe in what kind the solver relies tangent stiffnesses
     */
    enum class TangentRequirement{NoNeed, NeedEffect, NeedTangents};
    using Cell_t = CellBase<DimS, DimM>; //!< Cell type
    using Ccoord = Ccoord_t<DimS>; //!< cell coordinates type
    //! Field collection to store temporary fields in
    using Collection_t = GlobalFieldCollection<DimS>;
    //! Input vector for solvers
    using SolvVectorIn = Eigen::Ref<Eigen::VectorXd>;
    //! Input vector for solvers
    using SolvVectorInC = Eigen::Ref<const Eigen::VectorXd>;
    //! Output vector for solvers
    using SolvVectorOut = Eigen::VectorXd;


    //! Default constructor
    DeprecatedSolverBase() = delete;

    //! Constructor with domain resolutions
    DeprecatedSolverBase(Cell_t & cell, Real tol, Uint maxiter=0, bool verbose =false);

    //! Copy constructor
    DeprecatedSolverBase(const DeprecatedSolverBase &other) = delete;

    //! Move constructor
    DeprecatedSolverBase(DeprecatedSolverBase &&other) = default;

    //! Destructor
    virtual ~DeprecatedSolverBase() = default;

    //! Copy assignment operator
    DeprecatedSolverBase& operator=(const DeprecatedSolverBase &other) = delete;

    //! Move assignment operator
    DeprecatedSolverBase& operator=(DeprecatedSolverBase &&other) = default;

    //! Allocate fields used during the solution
    virtual void initialise() {
      this->collection.initialise(this->cell.get_subdomain_resolutions(),
                                  this->cell.get_subdomain_locations());
    }

    //! determine whether this solver requires full tangent stiffnesses
    bool need_tangents() const {
      return (this->get_tangent_req() == TangentRequirement::NeedTangents);}

    //! determine whether this solver requires evaluation of directional tangent
    bool need_effect() const {
      return (this->get_tangent_req() == TangentRequirement::NeedEffect);}

    //! determine whether this solver has no need for tangents
    bool no_need_tangent() const {
      return (this->get_tangent_req() == TangentRequirement::NoNeed);}

    //! returns whether the solver has converged
    virtual bool has_converged() const = 0;

    //! reset the iteration counter to zero
    void reset_counter();

    //! get the count of how many solve steps have been executed since
    //! construction of most recent counter reset
    Uint get_counter() const;

    //! executes the solver
    virtual SolvVectorOut solve(const SolvVectorInC rhs, SolvVectorIn x_0) = 0;

    //! return a reference to the cell
    Cell_t & get_cell() {return cell;}

    //! read the current maximum number of iterations setting
    Uint get_maxiter() const {return this->maxiter;}
    //! set the maximum number of iterations
    void set_maxiter(Uint val) {this->maxiter = val;}

    //! read the current tolerance setting
    Real get_tol() const {return this->tol;}
    //! set the torelance setting
    void set_tol(Real val) {this->tol = val;}

    //! returns the name of the solver
    virtual std::string name() const = 0;

  protected:
    //! returns the tangent requirements of this solver
    virtual TangentRequirement get_tangent_req() const = 0;
    Cell_t & cell; //!< reference to the cell
    Real tol;    //!< convergence tolerance
    Uint maxiter;//!< maximum number of iterations
    bool verbose;//!< whether or not to write information to the std output
    Uint counter{0}; //!< iteration counter
    //! storage for internal fields to avoid reallocations between calls
    Collection_t collection{};
  private:
  };

}  // muSpectre

#endif /* DEPRECATED_SOLVER_BASE_H */
