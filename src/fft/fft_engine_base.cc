/**
 * file   fft_engine_base.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   03 Dec 2017
 *
 * @brief  implementation for FFT engine base class
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

#include "fft/fft_engine_base.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  FFT_Engine_base<DimS, DimM>::FFT_Engine_base(Ccoord resolutions, Rcoord lengths)
    :resolutions{resolutions}, lengths{lengths},
     work{make_field<Workspace_t>("work space", work_space_container)},
     norm_factor{1./CcoordOps::get_size(resolutions)}
  {}

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void FFT_Engine_base<DimS, DimM>::initialise(FFT_PlanFlags /*plan_flags*/) {
    this->work_space_container.initialise();
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  size_t FFT_Engine_base<DimS, DimM>::size() const {
    return CcoordOps::get_size(this->resolutions);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  size_t FFT_Engine_base<DimS, DimM>::workspace_size() const {
    return this->work_space_container.size();
  }

  template class FFT_Engine_base<twoD, twoD>;
  template class FFT_Engine_base<twoD, threeD>;
  template class FFT_Engine_base<threeD, threeD>;

}  // muSpectre
