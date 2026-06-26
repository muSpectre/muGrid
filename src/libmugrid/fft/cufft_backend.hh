/**
 * @file   fft/cufft_backend.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   19 Dec 2024
 *
 * @brief  cuFFT implementation of FFT1DBackend for NVIDIA GPUs
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

#ifndef SRC_LIBMUGRID_FFT_CUFFT_BACKEND_HH_
#define SRC_LIBMUGRID_FFT_CUFFT_BACKEND_HH_

#include "fft_backend.hh"

#include <cufft.h>

#include <cstddef>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace muGrid {

/**
 * cuFFT implementation of FFT1DBackend for NVIDIA GPUs.
 *
 * This backend uses NVIDIA's cuFFT library for GPU-accelerated FFT operations.
 * It operates on device memory and supports batched 1D transforms with
 * arbitrary strides.
 *
 * cuFFT plans are cached by (n, batch, stride, dist) signature to avoid
 * repeated plan creation overhead.
 */
class cuFFTBackend : public FFT1DBackend {
 public:
  cuFFTBackend();
  ~cuFFTBackend() override;

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

  //! cuFFT performs whole multidimensional r2c/c2r transforms natively
  //! (cufftPlanMany), so the serial engine hands the entire transform over in
  //! one planned call instead of driving it axis-by-axis.
  bool supports_nd() const override { return true; }

  void r2c_nd(const std::vector<Index_t> & shape,
              const std::vector<Index_t> & axes, const Real * input,
              const std::vector<Index_t> & in_strides, Complex * output,
              const std::vector<Index_t> & out_strides) override;

  void c2r_nd(const std::vector<Index_t> & shape,
              const std::vector<Index_t> & axes, const Complex * input,
              const std::vector<Index_t> & in_strides, Real * output,
              const std::vector<Index_t> & out_strides) override;

  bool supports_device_memory() const override { return true; }

  const char * name() const override { return "cufft"; }

 protected:
  /**
   * Key type for plan cache: (transform_type, n, batch, in_stride, in_dist,
   * out_stride, out_dist)
   */
  using PlanKey =
      std::tuple<int, Index_t, Index_t, Index_t, Index_t, Index_t, Index_t>;

  /**
   * Hash function for PlanKey.
   */
  struct PlanKeyHash {
    std::size_t operator()(const PlanKey & key) const {
      auto h1 = std::hash<int>{}(std::get<0>(key));
      auto h2 = std::hash<Index_t>{}(std::get<1>(key));
      auto h3 = std::hash<Index_t>{}(std::get<2>(key));
      auto h4 = std::hash<Index_t>{}(std::get<3>(key));
      auto h5 = std::hash<Index_t>{}(std::get<4>(key));
      auto h6 = std::hash<Index_t>{}(std::get<5>(key));
      auto h7 = std::hash<Index_t>{}(std::get<6>(key));
      // Combine hashes
      std::size_t result = h1;
      result ^= h2 + 0x9e3779b9 + (result << 6) + (result >> 2);
      result ^= h3 + 0x9e3779b9 + (result << 6) + (result >> 2);
      result ^= h4 + 0x9e3779b9 + (result << 6) + (result >> 2);
      result ^= h5 + 0x9e3779b9 + (result << 6) + (result >> 2);
      result ^= h6 + 0x9e3779b9 + (result << 6) + (result >> 2);
      result ^= h7 + 0x9e3779b9 + (result << 6) + (result >> 2);
      return result;
    }
  };

  /**
   * Transform type identifiers for plan caching.
   */
  enum TransformType { R2C = 0, C2R = 1, C2C = 2 };

  /**
   * Get or create a cuFFT plan for the given parameters.
   *
   * @param type       Transform type (R2C, C2R, or C2C)
   * @param n          Transform size
   * @param batch      Number of batched transforms
   * @param in_stride  Input stride
   * @param in_dist    Input distance between batches
   * @param out_stride Output stride
   * @param out_dist   Output distance between batches
   * @return cufftHandle for the requested transform
   */
  cufftHandle get_plan(TransformType type, Index_t n, Index_t batch,
                       Index_t in_stride, Index_t in_dist, Index_t out_stride,
                       Index_t out_dist);

  /**
   * Get or create a cached N-dimensional cuFFT plan (cufftPlanMany). The
   * arguments are the cuFFT advanced-data-layout parameters: `n` are the
   * logical (real-space) transform sizes, `inembed`/`onembed` the physical
   * padded extents, the strides the innermost-element strides, and the dists
   * the batch (component) strides. Cached by the full parameter signature.
   */
  cufftHandle get_nd_plan(cufftType type, const std::vector<int> & n,
                          const std::vector<int> & inembed, int istride,
                          int idist, const std::vector<int> & onembed,
                          int ostride, int odist, int batch);

  /**
   * Device scratch buffer of at least `count` complex elements (grow-only).
   * Used to stage the Fourier input of `c2r_nd`, since cuFFT's multidimensional
   * Z2D overwrites its input but the engine requires it preserved.
   */
  cufftDoubleComplex * ensure_nd_scratch(std::size_t count);

  /**
   * Check cuFFT result and throw on error.
   */
  static void check_cufft_result(cufftResult result, const char * operation);

  //! Plan cache (1D batched transforms)
  std::unordered_map<PlanKey, cufftHandle, PlanKeyHash> plan_cache;

  //! Plan cache for N-dimensional transforms, keyed by a string signature
  std::unordered_map<std::string, cufftHandle> nd_plan_cache;

  //! Grow-only scratch for c2r_nd input preservation
  cufftDoubleComplex * nd_scratch{nullptr};
  std::size_t nd_scratch_count{0};
};

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_FFT_CUFFT_BACKEND_HH_
