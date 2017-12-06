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
#include "common/field_map.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  ProjectionFiniteStrain<DimS, DimM>::
  ProjectionFiniteStrain(FFT_Engine & engine)
    :Parent{engine}, Ghat{make_field<Proj_t>("Projection Operator",
                                            this->projection_container)}
  {}

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void ProjectionFiniteStrain<DimS, DimM>::
  initialise(FFT_PlanFlags flags) {
    Parent::initialise(flags);
    FFT_freqs<DimS> fft_freqs(this->sizes);
    for (auto && tup: boost::combine(this->fft_engine, this->Ghat)) {
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


  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void ProjectionFiniteStrain<DimS, DimM>::apply_projection(Field_t & field) {
    Vector_map field_map{this->fft_engine.fft(field)};
    Real factor = this->fft_engine.normalisation();
    for (auto && tup: boost::combine(this->Ghat, field_map)) {
      auto & G{boost::get<0>(tup)};
      auto & f{boost::get<1>(tup)};
      f = factor * (G*f).eval();
    }
  }

  template class ProjectionFiniteStrain<twoD,   twoD>;
  template class ProjectionFiniteStrain<twoD,   threeD>;
  template class ProjectionFiniteStrain<threeD, threeD>;
}  // muSpectre
