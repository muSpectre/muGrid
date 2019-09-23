/**
 * @file   fft_utils.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   11 Dec 2017
 *
 * @brief  implementation of fft utilities
 *
 * Copyright © 2017 Till Junge
 *
 * µFFT is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µFFT is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µFFT; see the file COPYING. If not, write to the
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

#include "fft_utils.hh"

namespace muFFT {

  /* ---------------------------------------------------------------------- */
  std::valarray<Real> fft_freqs(size_t nb_samples) {
    std::valarray<Real> retval(nb_samples);
    Int N = (nb_samples - 1) / 2 + 1;  // needs to be signed int for neg freqs
    for (Int i = 0; i < N; ++i) {
      retval[i] = i;
    }
    for (Int i = N; i < Int(nb_samples); ++i) {
      retval[i] = -Int(nb_samples) / 2 + i - N;
    }
    return retval;
  }

  /* ---------------------------------------------------------------------- */
  std::valarray<Real> fft_freqs(size_t nb_samples, Real length) {
    return fft_freqs(nb_samples) / length;
  }

}  // namespace muFFT
