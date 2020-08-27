/**
 * @file   solver_projection_based.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   22 Jul 2020
 *
 * @brief  Virtual base class for projection-based solvers
 *
 * Copyright © 2020 Till Junge
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
 * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 *
 */
#include "solver_base.hh"

#include "projection/projection_base.hh"

#ifndef SRC_SOLVER_SOLVER_PROJECTION_BASED_HH_
#define SRC_SOLVER_SOLVER_PROJECTION_BASED_HH_

namespace muSpectre {

  class SolverProjectionBased : virtual public SolverBase {
   public:
    //! Default constructor
    SolverProjectionBased() = delete;

    //! Copy constructor
    SolverProjectionBased(const SolverProjectionBased & other) = delete;

    //! Move constructor
    SolverProjectionBased(SolverProjectionBased && other) = default;

    //! Destructor
    virtual ~SolverProjectionBased() = default;

    //! Copy assignment operator
    SolverProjectionBased &
    operator=(const SolverProjectionBased & other) = delete;

    //! Move assignment operator
    SolverProjectionBased & operator=(SolverProjectionBased && other) = delete;

    void apply_projection(muGrid::TypedFieldBase<Real> & rhs);

   protected:
    //! create a mechanics projection
    void create_projection(const Formulation& form);
    //! create a generic gradient projection
    void create_projection();
    std::shared_ptr<ProjectionBase> projection{nullptr};
  };

}  // namespace muSpectre

#endif  // SRC_SOLVER_SOLVER_PROJECTION_BASED_HH_
