/**
 * @file   fft/pocketfft_backend.hh
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

#ifndef SRC_LIBMUGRID_FFT_POCKETFFT_BACKEND_HH_
#define SRC_LIBMUGRID_FFT_POCKETFFT_BACKEND_HH_

#include "fft_1d_backend.hh"

namespace muGrid {

/**
 * PocketFFT implementation of FFT1DBackend.
 *
 * Uses the embedded pocketfft library (BSD-3 licensed) for CPU-based FFT.
 * This backend is always available and serves as the fallback for host memory.
 */
class PocketFFTBackend : public FFT1DBackend {
 public:
  PocketFFTBackend() = default;
  ~PocketFFTBackend() override = default;

  void r2c(Index_t n, Index_t batch, const Real * input, Index_t in_stride,
           Index_t in_dist, Complex * output, Index_t out_stride,
           Index_t out_dist) override;

  void c2r(Index_t n, Index_t batch, const Complex * input, Index_t in_stride,
           Index_t in_dist, Real * output, Index_t out_stride,
           Index_t out_dist) override;

  void c2c_forward(Index_t n, Index_t batch, const Complex * input,
                   Index_t in_stride, Index_t in_dist, Complex * output,
                   Index_t out_stride, Index_t out_dist) override;

  void c2c_backward(Index_t n, Index_t batch, const Complex * input,
                    Index_t in_stride, Index_t in_dist, Complex * output,
                    Index_t out_stride, Index_t out_dist) override;

  bool supports_device_memory() const override { return false; }

  const char * name() const override { return "pocketfft"; }
};

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_FFT_POCKETFFT_BACKEND_HH_
