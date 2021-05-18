/**
 * @file   krylov_solver_trust_region_features.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *         Ali Falsafi  <ali.falsafi@epfl.ch>
 *
 *
 * @date   25 July 2020
 *
 * @brief  Implementation of features class for trust region Krylov solver
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

#include "solver/krylov_solver_trust_region_features.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  KrylovSolverTrustRegionFeatures::KrylovSolverTrustRegionFeatures(
      const Real & trust_region, const ResetCG & reset,
      const Index_t & reset_iter_count)
      : trust_region{trust_region}, reset{reset}, reset_iter_count{
                                                      reset_iter_count} {}

  /* ---------------------------------------------------------------------- */
  void KrylovSolverTrustRegionFeatures::set_trust_region(
      const Real & new_trust_region) {
    this->trust_region = new_trust_region;
  }

  /* ---------------------------------------------------------------------- */
  const bool & KrylovSolverTrustRegionFeatures::get_is_on_bound() {
    return this->is_on_bound;
  }

}  // namespace muSpectre
