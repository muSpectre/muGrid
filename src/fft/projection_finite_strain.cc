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

#include "fft/projection_finite_strain.hh"
#include "fft/fftw_engine.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, class FFT_Engine>
  ProjectionFiniteStrain<DimS, DimM, FFT_Engine>::
  ProjectionFiniteStrain(Ccoord sizes)
    :Parent{sizes}, engine{sizes} {}

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, class FFT_Engine>
  void ProjectionFiniteStrain<DimS, DimM, FFT_Engine>::
  initialise(FFT_PlanFlags flags) {
    Parent::initialise(flags);

    
  }
}  // muSpectre
