/**
 * @file   fft/pocketfft_backend.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   18 Dec 2025
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

// The batch of `batch` 1D transforms is handed to pocketfft as a single 2D
// array of shape {batch, n} transformed along axis 1. pocketfft then iterates
// the batch axis internally and SIMD-vectorizes across neighbouring lines,
// which a per-line loop here cannot do. The batch stride (`*_dist`) is the
// outer (axis-0) stride and the element stride (`*_stride`) is the inner
// (axis-1) stride; pocketfft wants both in bytes. An empty batch is a no-op
// (pocketfft returns early when the shape has a zero extent).
void PocketFFTBackend::r2c(Index_t n, Index_t batch, const Real * input,
                           Index_t in_stride, Index_t in_dist, Complex * output,
                           Index_t out_stride, Index_t out_dist) {
  pocketfft::shape_t shape{static_cast<size_t>(batch), static_cast<size_t>(n)};
  pocketfft::stride_t stride_in{
      static_cast<ptrdiff_t>(in_dist * sizeof(Real)),
      static_cast<ptrdiff_t>(in_stride * sizeof(Real))};
  pocketfft::stride_t stride_out{
      static_cast<ptrdiff_t>(out_dist * sizeof(Complex)),
      static_cast<ptrdiff_t>(out_stride * sizeof(Complex))};

  pocketfft::r2c(shape, stride_in, stride_out,
                 1,  // axis
                 pocketfft::FORWARD, input,
                 reinterpret_cast<std::complex<Real> *>(output),
                 1.0  // scale factor
  );
}

// Batched c2r, structured as in `r2c` (single 2D pocketfft call). Note
// pocketfft's c2r takes the *real* (output) shape {batch, n}.
void PocketFFTBackend::c2r(Index_t n, Index_t batch, const Complex * input,
                           Index_t in_stride, Index_t in_dist, Real * output,
                           Index_t out_stride, Index_t out_dist) {
  pocketfft::shape_t shape{static_cast<size_t>(batch), static_cast<size_t>(n)};
  pocketfft::stride_t stride_in{
      static_cast<ptrdiff_t>(in_dist * sizeof(Complex)),
      static_cast<ptrdiff_t>(in_stride * sizeof(Complex))};
  pocketfft::stride_t stride_out{
      static_cast<ptrdiff_t>(out_dist * sizeof(Real)),
      static_cast<ptrdiff_t>(out_stride * sizeof(Real))};

  pocketfft::c2r(
      shape, stride_in, stride_out,
      1,  // axis
      pocketfft::BACKWARD,
      reinterpret_cast<const std::complex<Real> *>(input), output,
      1.0  // scale factor
  );
}

// Batched c2c (forward), structured as in `r2c`. pocketfft supports the
// in-place case (input == output) the engine uses for the intermediate axes.
void PocketFFTBackend::c2c_forward(Index_t n, Index_t batch,
                                   const Complex * input, Index_t in_stride,
                                   Index_t in_dist, Complex * output,
                                   Index_t out_stride, Index_t out_dist) {
  pocketfft::shape_t shape{static_cast<size_t>(batch), static_cast<size_t>(n)};
  pocketfft::shape_t axes{1};  // Transform along axis 1 (the inner axis)
  pocketfft::stride_t stride_in{
      static_cast<ptrdiff_t>(in_dist * sizeof(Complex)),
      static_cast<ptrdiff_t>(in_stride * sizeof(Complex))};
  pocketfft::stride_t stride_out{
      static_cast<ptrdiff_t>(out_dist * sizeof(Complex)),
      static_cast<ptrdiff_t>(out_stride * sizeof(Complex))};

  pocketfft::c2c(
      shape, stride_in, stride_out, axes, pocketfft::FORWARD,
      reinterpret_cast<const std::complex<Real> *>(input),
      reinterpret_cast<std::complex<Real> *>(output),
      1.0  // scale factor
  );
}

// Batched c2c (backward), structured as in `c2c_forward`.
void PocketFFTBackend::c2c_backward(Index_t n, Index_t batch,
                                    const Complex * input, Index_t in_stride,
                                    Index_t in_dist, Complex * output,
                                    Index_t out_stride, Index_t out_dist) {
  pocketfft::shape_t shape{static_cast<size_t>(batch), static_cast<size_t>(n)};
  pocketfft::shape_t axes{1};  // Transform along axis 1 (the inner axis)
  pocketfft::stride_t stride_in{
      static_cast<ptrdiff_t>(in_dist * sizeof(Complex)),
      static_cast<ptrdiff_t>(in_stride * sizeof(Complex))};
  pocketfft::stride_t stride_out{
      static_cast<ptrdiff_t>(out_dist * sizeof(Complex)),
      static_cast<ptrdiff_t>(out_stride * sizeof(Complex))};

  pocketfft::c2c(
      shape, stride_in, stride_out, axes, pocketfft::BACKWARD,
      reinterpret_cast<const std::complex<Real> *>(input),
      reinterpret_cast<std::complex<Real> *>(output),
      1.0  // scale factor
  );
}

namespace {

// Translate the engine's element strides (in units of T) into pocketfft's
// byte strides, and the Index_t extents/axes into pocketfft's size_t vectors.
template <typename T>
pocketfft::stride_t byte_strides(const std::vector<Index_t> & strides) {
  pocketfft::stride_t out;
  out.reserve(strides.size());
  for (auto s : strides) {
    out.push_back(static_cast<ptrdiff_t>(s) * static_cast<ptrdiff_t>(sizeof(T)));
  }
  return out;
}

pocketfft::shape_t to_shape(const std::vector<Index_t> & v) {
  pocketfft::shape_t out;
  out.reserve(v.size());
  for (auto s : v) {
    out.push_back(static_cast<size_t>(s));
  }
  return out;
}

}  // namespace

void PocketFFTBackend::r2c_nd(const std::vector<Index_t> & shape,
                              const std::vector<Index_t> & axes,
                              const Real * input,
                              const std::vector<Index_t> & in_strides,
                              Complex * output,
                              const std::vector<Index_t> & out_strides) {
  // A single planned transform over all `axes` at once. pocketfft performs the
  // r2c along axes.back() and complex-to-complex transforms along the rest,
  // batching and SIMD-vectorizing internally over every non-transformed axis
  // (components, and the orthogonal spatial axes). This is what numpy's rfftn
  // does and is markedly faster than the per-axis, per-line decomposition.
  pocketfft::r2c(to_shape(shape), byte_strides<Real>(in_strides),
                 byte_strides<Complex>(out_strides), to_shape(axes),
                 pocketfft::FORWARD, input,
                 reinterpret_cast<std::complex<Real> *>(output), 1.0);
}

void PocketFFTBackend::c2r_nd(const std::vector<Index_t> & shape,
                              const std::vector<Index_t> & axes,
                              const Complex * input,
                              const std::vector<Index_t> & in_strides,
                              Real * output,
                              const std::vector<Index_t> & out_strides) {
  // Inverse of r2c_nd. pocketfft's multi-axis c2r transforms the non-real axes
  // into an internal temporary first, so the const input is never modified.
  // `shape` is the real-space (output) extent.
  pocketfft::c2r(to_shape(shape), byte_strides<Complex>(in_strides),
                 byte_strides<Real>(out_strides), to_shape(axes),
                 pocketfft::BACKWARD,
                 reinterpret_cast<const std::complex<Real> *>(input), output,
                 1.0);
}

}  // namespace muGrid
