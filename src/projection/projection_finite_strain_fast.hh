/**
 * @file   projection_finite_strain_fast.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *         Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   12 Dec 2017
 *
 * @brief  Faster alternative to ProjectionFinitestrain
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

#ifndef SRC_PROJECTION_PROJECTION_FINITE_STRAIN_FAST_HH_
#define SRC_PROJECTION_PROJECTION_FINITE_STRAIN_FAST_HH_

#include <libmugrid/field_collection.hh>
#include <libmugrid/mapped_field.hh>

#include <libmufft/derivative.hh>

#include "common/muSpectre_common.hh"
#include "projection/projection_base.hh"
#include "projection/projection_gradient.hh"

namespace muSpectre {

  /**
   * This projection used to be its own class and is now just a special case of
   * ProjectionGradient. It is kept for compatibility
   */
  template <Index_t DimS, Index_t NbQuadPts = OneQuadPt>
  using ProjectionFiniteStrainFast =
      ProjectionGradient<DimS, secondOrder, NbQuadPts>;

}  // namespace muSpectre

#endif  // SRC_PROJECTION_PROJECTION_FINITE_STRAIN_FAST_HH_
