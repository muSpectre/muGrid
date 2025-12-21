/**
 * @file   fft/fft_1d_backend.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   18 Dec 2024
 *
 * @brief  Abstract interface for 1D FFT backends
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

#ifndef SRC_LIBMUGRID_FFT_FFT_1D_BACKEND_HH_
#define SRC_LIBMUGRID_FFT_FFT_1D_BACKEND_HH_

#include "core/types.hh"

#include <memory>

namespace muGrid {

/**
 * Abstract interface for 1D FFT operations.
 *
 * Implementations are selected at compile time based on GPU backend configuration
 * (MUGRID_ENABLE_CUDA or MUGRID_ENABLE_HIP). The backend operates on raw pointers -
 * the caller is responsible for ensuring the pointers are valid for the
 * backend's memory space.
 */
class FFT1DBackend {
 public:
  virtual ~FFT1DBackend() = default;

  /**
   * Batched 1D real-to-complex FFT.
   *
   * @param n          Transform size (number of real input points)
   * @param batch      Number of independent 1D transforms
   * @param input      Pointer to real input data
   * @param in_stride  Stride between consecutive input elements (in Reals)
   * @param in_dist    Distance between batches in input (in Reals)
   * @param output     Pointer to complex output data (n/2+1 complex per batch)
   * @param out_stride Stride between consecutive output elements (in Complex)
   * @param out_dist   Distance between batches in output (in Complex)
   */
  virtual void r2c(Index_t n, Index_t batch, const Real * input,
                   Index_t in_stride, Index_t in_dist, Complex * output,
                   Index_t out_stride, Index_t out_dist) = 0;

  /**
   * Batched 1D complex-to-real FFT.
   *
   * @param n          Transform size (number of real output points)
   * @param batch      Number of independent 1D transforms
   * @param input      Pointer to complex input data (n/2+1 complex per batch)
   * @param in_stride  Stride between consecutive input elements (in Complex)
   * @param in_dist    Distance between batches in input (in Complex)
   * @param output     Pointer to real output data
   * @param out_stride Stride between consecutive output elements (in Reals)
   * @param out_dist   Distance between batches in output (in Reals)
   */
  virtual void c2r(Index_t n, Index_t batch, const Complex * input,
                   Index_t in_stride, Index_t in_dist, Real * output,
                   Index_t out_stride, Index_t out_dist) = 0;

  /**
   * Batched 1D complex-to-complex forward FFT.
   *
   * @param n          Transform size
   * @param batch      Number of independent 1D transforms
   * @param input      Pointer to complex input data
   * @param in_stride  Stride between consecutive input elements (in Complex)
   * @param in_dist    Distance between batches in input (in Complex)
   * @param output     Pointer to complex output data
   * @param out_stride Stride between consecutive output elements (in Complex)
   * @param out_dist   Distance between batches in output (in Complex)
   */
  virtual void c2c_forward(Index_t n, Index_t batch, const Complex * input,
                           Index_t in_stride, Index_t in_dist, Complex * output,
                           Index_t out_stride, Index_t out_dist) = 0;

  /**
   * Batched 1D complex-to-complex backward FFT.
   *
   * @param n          Transform size
   * @param batch      Number of independent 1D transforms
   * @param input      Pointer to complex input data
   * @param in_stride  Stride between consecutive input elements (in Complex)
   * @param in_dist    Distance between batches in input (in Complex)
   * @param output     Pointer to complex output data
   * @param out_stride Stride between consecutive output elements (in Complex)
   * @param out_dist   Distance between batches in output (in Complex)
   */
  virtual void c2c_backward(Index_t n, Index_t batch, const Complex * input,
                            Index_t in_stride, Index_t in_dist,
                            Complex * output, Index_t out_stride,
                            Index_t out_dist) = 0;

  /** Returns true if this backend supports device (GPU) memory */
  virtual bool supports_device_memory() const = 0;

  /** Returns the name of this backend */
  virtual const char * name() const = 0;
};

/**
 * Get the appropriate FFT backend for host memory.
 * Returns PocketFFT (always available).
 */
std::unique_ptr<FFT1DBackend> get_host_fft_backend();

/**
 * Get the appropriate FFT backend for device memory.
 * Returns cuFFT, rocFFT, or nullptr if no GPU backend available.
 */
std::unique_ptr<FFT1DBackend> get_device_fft_backend();

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_FFT_FFT_1D_BACKEND_HH_
