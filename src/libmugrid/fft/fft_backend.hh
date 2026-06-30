/**
 * @file   fft/fft_backend.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   18 Dec 2024
 *
 * @brief  Abstract interface for FFT backends (1D primitives plus optional
 *         N-dimensional transforms)
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

#ifndef SRC_LIBMUGRID_FFT_FFT_BACKEND_HH_
#define SRC_LIBMUGRID_FFT_FFT_BACKEND_HH_

#include "core/types.hh"
#include "core/exception.hh"

#include <memory>
#include <vector>

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

  // --- Single-precision (Real32/Complex32) 1D primitives --------------------
  // Default to throwing so a backend that has not implemented fp32 reports a
  // clear error rather than silently mis-transforming. Backends that support
  // single precision (pocketfft, cuFFT, rocFFT) override these.
  virtual void r2c(Index_t /*n*/, Index_t /*batch*/, const Real32 * /*input*/,
                   Index_t /*in_stride*/, Index_t /*in_dist*/,
                   Complex32 * /*output*/, Index_t /*out_stride*/,
                   Index_t /*out_dist*/) {
    throw RuntimeError("single-precision r2c not supported by this FFT backend");
  }
  virtual void c2r(Index_t /*n*/, Index_t /*batch*/,
                   const Complex32 * /*input*/, Index_t /*in_stride*/,
                   Index_t /*in_dist*/, Real32 * /*output*/,
                   Index_t /*out_stride*/, Index_t /*out_dist*/) {
    throw RuntimeError("single-precision c2r not supported by this FFT backend");
  }
  virtual void c2c_forward(Index_t /*n*/, Index_t /*batch*/,
                           const Complex32 * /*input*/, Index_t /*in_stride*/,
                           Index_t /*in_dist*/, Complex32 * /*output*/,
                           Index_t /*out_stride*/, Index_t /*out_dist*/) {
    throw RuntimeError(
        "single-precision c2c_forward not supported by this FFT backend");
  }
  virtual void c2c_backward(Index_t /*n*/, Index_t /*batch*/,
                            const Complex32 * /*input*/, Index_t /*in_stride*/,
                            Index_t /*in_dist*/, Complex32 * /*output*/,
                            Index_t /*out_stride*/, Index_t /*out_dist*/) {
    throw RuntimeError(
        "single-precision c2c_backward not supported by this FFT backend");
  }

  /**
   * Returns true if this backend implements the N-dimensional `r2c_nd` /
   * `c2r_nd` entry points below. These let a serial (non-decomposed) engine
   * hand a whole multidimensional transform to the backend in a single planned
   * call, instead of driving it axis-by-axis through the 1D primitives. The
   * default is `false`; backends that do not override it keep the axis-by-axis
   * decomposition, which is mandatory for the MPI/pencil path anyway.
   */
  virtual bool supports_nd() const { return false; }

  /**
   * N-dimensional real-to-complex FFT, issued as a single backend call.
   *
   * The arrays are described in row-major (C) order: `shape` is the logical
   * real-space extent of each axis, `in_strides`/`out_strides` are the element
   * strides of the corresponding axis (Reals for the input, Complex for the
   * output), and `axes` lists the axes to transform. The half-complex (r2c)
   * axis must be the LAST entry of `axes`; the remaining listed axes are
   * transformed complex-to-complex. Any axis not in `axes` (e.g. a tensor
   * component) is a non-transformed batch axis. Strides let the caller fold in
   * ghost padding and component layout (AoS/SoA) without a repacking copy.
   *
   * The default throws; only backends returning true from `supports_nd()`
   * override it.
   */
  virtual void r2c_nd(const std::vector<Index_t> & /*shape*/,
                      const std::vector<Index_t> & /*axes*/,
                      const Real * /*input*/,
                      const std::vector<Index_t> & /*in_strides*/,
                      Complex * /*output*/,
                      const std::vector<Index_t> & /*out_strides*/) {
    throw RuntimeError("r2c_nd is not supported by this FFT backend");
  }

  /**
   * N-dimensional complex-to-real FFT, the inverse of `r2c_nd`. `shape` is the
   * logical real-space (output) extent; `in_strides` are Complex strides of the
   * Fourier input, `out_strides` are Real strides of the output. As in
   * `r2c_nd`, the real (c2r) axis is the last entry of `axes`.
   */
  virtual void c2r_nd(const std::vector<Index_t> & /*shape*/,
                      const std::vector<Index_t> & /*axes*/,
                      const Complex * /*input*/,
                      const std::vector<Index_t> & /*in_strides*/,
                      Real * /*output*/,
                      const std::vector<Index_t> & /*out_strides*/) {
    throw RuntimeError("c2r_nd is not supported by this FFT backend");
  }

  //! Single-precision N-dimensional r2c (see the double overload).
  virtual void r2c_nd(const std::vector<Index_t> & /*shape*/,
                      const std::vector<Index_t> & /*axes*/,
                      const Real32 * /*input*/,
                      const std::vector<Index_t> & /*in_strides*/,
                      Complex32 * /*output*/,
                      const std::vector<Index_t> & /*out_strides*/) {
    throw RuntimeError(
        "single-precision r2c_nd is not supported by this FFT backend");
  }
  //! Single-precision N-dimensional c2r (see the double overload).
  virtual void c2r_nd(const std::vector<Index_t> & /*shape*/,
                      const std::vector<Index_t> & /*axes*/,
                      const Complex32 * /*input*/,
                      const std::vector<Index_t> & /*in_strides*/,
                      Real32 * /*output*/,
                      const std::vector<Index_t> & /*out_strides*/) {
    throw RuntimeError(
        "single-precision c2r_nd is not supported by this FFT backend");
  }

  /**
   * Whether the backend can transform a real-to-complex (r2c) batch whose real
   * array has a non-unit element stride. cuFFT cannot ("Strides on the real
   * part of real-to-complex and complex-to-real transforms are not supported")
   * and overrides this to false; pocketfft and rocFFT can. The engine can query
   * this to choose a layout that avoids the limitation instead of discovering
   * it through an exception at run time.
   */
  virtual bool supports_strided_r2c() const { return true; }

  /** As supports_strided_r2c(), for complex-to-real (c2r) transforms. */
  virtual bool supports_strided_c2r() const { return true; }

  /** Returns true if this backend supports device (GPU) memory */
  virtual bool supports_device_memory() const = 0;

  /** Returns the name of this backend */
  virtual const char * name() const = 0;
};

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_FFT_FFT_BACKEND_HH_
