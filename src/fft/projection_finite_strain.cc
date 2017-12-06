/**
 * file   projection_finite_strain.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   05 Dec 2017
 *
 * @brief  implementation of standard finite strain projection operator
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

#include <boost/range/combine.hpp>

#include "fft/projection_finite_strain.hh"
#include "fft/fftw_engine.hh"
#include "fft/fft_utils.hh"
#include "common/field_map_matrixlike.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, class FFT_Engine>
  ProjectionFiniteStrain<DimS, DimM, FFT_Engine>::
  ProjectionFiniteStrain(Ccoord sizes)
    :Parent{sizes}, Ghat{make_field<Proj_t>("Projection Operator",
                                            this->projection_container)}
  {}

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, class FFT_Engine>
  void ProjectionFiniteStrain<DimS, DimM, FFT_Engine>::
  initialise(FFT_PlanFlags flags) {
    Parent::initialise(flags);
    FFT_freqs<DimS> fft_freqs(this->sizes);
    auto proj_map = T4MatrixFieldMap<LFieldCollection_t, Real, DimM>(Ghat);
    for (auto && tup: boost::combine(this->fft_engine, proj_map)) {
      const auto & ccoord = boost::get<0> (tup);
      auto & G = boost::get<1>(tup);
      auto xi = fft_freqs.get_unit_xi(ccoord);
      for (Dim_t im = 0; im < DimS; ++im) {
        for (Dim_t j = 0; j < DimS; ++j) {
          for (Dim_t l = 0; l < DimS; ++l) {
            get(G, im, j, l, im) = xi(j)*xi(l);
          }
        }
      }
    }
  }

  template class ProjectionFiniteStrain<twoD, twoD,
                                        FFTW_Engine<twoD, twoD>>;
  template class ProjectionFiniteStrain<twoD, threeD,
                                        FFTW_Engine<twoD, threeD>>;
  template class ProjectionFiniteStrain<threeD, threeD,
                                        FFTW_Engine<threeD, threeD>>;
}  // muSpectre
