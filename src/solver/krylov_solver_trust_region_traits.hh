/**
 * @file   krylov_solver_trust_region_traits.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *         Ali Falsafi  <ali.falsafi@epfl.ch>
 *
 *
 * @date   25 July 2020
 *
 * @brief  Traits class for trust region Krylov solver
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

#include "solver_common.hh"

#ifndef SRC_SOLVER_KRYLOV_SOLVER_TRUST_REGION_TRAITS_HH_
#define SRC_SOLVER_KRYLOV_SOLVER_TRUST_REGION_TRAITS_HH_

namespace muSpectre {
  class KrylovSolverTrustRegionTraits {
   public:
    //! Default constructor
    KrylovSolverTrustRegionTraits() = default;

    //! Copy constructor
    KrylovSolverTrustRegionTraits(const KrylovSolverTrustRegionTraits & other) =
        delete;

    /**
     * Constructor without matrix adaptable. The adaptable has to be supplied
     * using KrylovSolverTraits::set_matrix(...) before initialisation for this
     * solver to be usable
     */
    explicit KrylovSolverTrustRegionTraits(const Real & trust_region = 1.0);

    //! Move constructor
    KrylovSolverTrustRegionTraits(KrylovSolverTrustRegionTraits && other) =
        default;

    //! Destructor
    virtual ~KrylovSolverTrustRegionTraits() = default;

    //! Copy assignment operator
    KrylovSolverTrustRegionTraits &
    operator=(const KrylovSolverTrustRegionTraits & other) = delete;

    //! Move assignment operator
    KrylovSolverTrustRegionTraits &
    operator=(KrylovSolverTrustRegionTraits && other) = delete;

    //! set size of trust region, exception if not supported
    void set_trust_region(const Real & new_trust_region);

    //! return the member variable which expresses whether the solution of the
    //! sub-problem is on the bound or not
    const bool & get_is_on_bound();

   protected:
    Real trust_region{1.0};  //!< size of trust region

    bool is_on_bound{false};  //!< Boolean showing if the solution is on the
                              //! boundary of trust region
  };
}  // namespace muSpectre

#endif  // SRC_SOLVER_KRYLOV_SOLVER_TRUST_REGION_TRAITS_HH_
