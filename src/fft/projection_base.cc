/**
 * @file   projection_base.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   06 Dec 2017
 *
 * @brief  implementation of base class for projections
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

#include "fft/projection_base.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  ProjectionBase<DimS, DimM>::ProjectionBase(FFTEngine_ptr engine,
                                             Rcoord domain_lengths,
                                             Formulation form)
      : fft_engine{std::move(engine)}, domain_lengths{domain_lengths},
        form{form}, projection_container{
                        this->fft_engine->get_field_collection()} {
    static_assert((DimS == FFTEngine::sdim),
                  "spatial dimensions are incompatible");
    if (this->get_nb_components() != fft_engine->get_nb_components()) {
      throw ProjectionError("Incompatible number of components per pixel");
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void ProjectionBase<DimS, DimM>::initialise(FFT_PlanFlags flags) {
    fft_engine->initialise(flags);
  }

  template class ProjectionBase<twoD, twoD>;
  template class ProjectionBase<twoD, threeD>;
  template class ProjectionBase<threeD, threeD>;
}  // namespace muSpectre
