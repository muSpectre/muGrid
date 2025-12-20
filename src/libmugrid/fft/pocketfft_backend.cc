/**
 * @file   fft/pocketfft_backend.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   18 Dec 2024
 *
 * @brief  PocketFFT implementation of FFT1DBackend
 *
 * Copyright © 2024 Lars Pastewka
 *
 * µGrid is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µGrid is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µGrid; see the file COPYING. If not, write to the
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

#include "pocketfft_backend.hh"

// Disable pocketfft multithreading - we use MPI for parallelism
#define POCKETFFT_NO_MULTITHREADING
#include "pocketfft/pocketfft_hdronly.h"

namespace muGrid {

void PocketFFTBackend::r2c(Index_t n, Index_t batch, const Real * input,
                           Index_t in_stride, Index_t in_dist, Complex * output,
                           Index_t out_stride, Index_t out_dist) {
  pocketfft::shape_t shape{static_cast<size_t>(n)};
  pocketfft::stride_t stride_in{static_cast<ptrdiff_t>(in_stride *
                                                       sizeof(Real))};
  pocketfft::stride_t stride_out{static_cast<ptrdiff_t>(out_stride *
                                                        sizeof(Complex))};

  for (Index_t b = 0; b < batch; ++b) {
    pocketfft::r2c(shape, stride_in, stride_out,
                   0,  // axis
                   pocketfft::FORWARD, input + b * in_dist,
                   reinterpret_cast<std::complex<Real> *>(output +
                                                          b * out_dist),
                   1.0  // scale factor
    );
  }
}

void PocketFFTBackend::c2r(Index_t n, Index_t batch, const Complex * input,
                           Index_t in_stride, Index_t in_dist, Real * output,
                           Index_t out_stride, Index_t out_dist) {
  pocketfft::shape_t shape{static_cast<size_t>(n)};
  pocketfft::stride_t stride_in{static_cast<ptrdiff_t>(in_stride *
                                                       sizeof(Complex))};
  pocketfft::stride_t stride_out{static_cast<ptrdiff_t>(out_stride *
                                                        sizeof(Real))};

  for (Index_t b = 0; b < batch; ++b) {
    pocketfft::c2r(
        shape, stride_in, stride_out,
        0,  // axis
        pocketfft::BACKWARD,
        reinterpret_cast<const std::complex<Real> *>(input + b * in_dist),
        output + b * out_dist,
        1.0  // scale factor
    );
  }
}

void PocketFFTBackend::c2c_forward(Index_t n, Index_t batch,
                                   const Complex * input, Index_t in_stride,
                                   Index_t in_dist, Complex * output,
                                   Index_t out_stride, Index_t out_dist) {
  pocketfft::shape_t shape{static_cast<size_t>(n)};
  pocketfft::shape_t axes{0};  // Transform along axis 0
  pocketfft::stride_t stride_in{static_cast<ptrdiff_t>(in_stride *
                                                       sizeof(Complex))};
  pocketfft::stride_t stride_out{static_cast<ptrdiff_t>(out_stride *
                                                        sizeof(Complex))};

  for (Index_t b = 0; b < batch; ++b) {
    pocketfft::c2c(
        shape, stride_in, stride_out,
        axes,
        pocketfft::FORWARD,
        reinterpret_cast<const std::complex<Real> *>(input + b * in_dist),
        reinterpret_cast<std::complex<Real> *>(output + b * out_dist),
        1.0  // scale factor
    );
  }
}

void PocketFFTBackend::c2c_backward(Index_t n, Index_t batch,
                                    const Complex * input, Index_t in_stride,
                                    Index_t in_dist, Complex * output,
                                    Index_t out_stride, Index_t out_dist) {
  pocketfft::shape_t shape{static_cast<size_t>(n)};
  pocketfft::shape_t axes{0};  // Transform along axis 0
  pocketfft::stride_t stride_in{static_cast<ptrdiff_t>(in_stride *
                                                       sizeof(Complex))};
  pocketfft::stride_t stride_out{static_cast<ptrdiff_t>(out_stride *
                                                        sizeof(Complex))};

  for (Index_t b = 0; b < batch; ++b) {
    pocketfft::c2c(
        shape, stride_in, stride_out,
        axes,
        pocketfft::BACKWARD,
        reinterpret_cast<const std::complex<Real> *>(input + b * in_dist),
        reinterpret_cast<std::complex<Real> *>(output + b * out_dist),
        1.0  // scale factor
    );
  }
}

}  // namespace muGrid
