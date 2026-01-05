/**
 * @file   fft/rocfft_backend.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   05 Jan 2026
 *
 * @brief  Native rocFFT implementation of FFT1DBackend for AMD GPUs
 *
 * Copyright © 2024-2026 Lars Pastewka
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

#ifndef SRC_LIBMUGRID_FFT_ROCFFT_BACKEND_HH_
#define SRC_LIBMUGRID_FFT_ROCFFT_BACKEND_HH_

#include "fft_1d_backend.hh"

#include <rocfft/rocfft.h>

#include <unordered_map>
#include <tuple>

namespace muGrid {

/**
 * Native rocFFT implementation of FFT1DBackend for AMD GPUs.
 *
 * This backend uses AMD's rocFFT library directly (not via hipFFT) for
 * GPU-accelerated FFT operations. Using the native rocFFT API provides
 * better support for strided data layouts, which is essential for
 * 3D MPI-parallel FFTs.
 *
 * rocFFT plans are cached by (transform_type, n, batch, in_stride, in_dist,
 * out_stride, out_dist) signature to avoid repeated plan creation overhead.
 *
 * Key advantage over hipFFT: rocFFT's native API supports arbitrary strides
 * for all transform types including R2C and C2R, which cuFFT does not support.
 */
class rocFFTBackend : public FFT1DBackend {
 public:
  rocFFTBackend();
  ~rocFFTBackend() override;

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

  bool supports_device_memory() const override { return true; }

  const char * name() const override { return "rocfft"; }

 protected:
  /**
   * Key type for plan cache: (transform_type, direction, n, batch, in_stride,
   * in_dist, out_stride, out_dist)
   */
  using PlanKey = std::tuple<int, int, Index_t, Index_t, Index_t, Index_t,
                             Index_t, Index_t>;

  /**
   * Hash function for PlanKey.
   */
  struct PlanKeyHash {
    std::size_t operator()(const PlanKey & key) const {
      auto h1 = std::hash<int>{}(std::get<0>(key));
      auto h2 = std::hash<int>{}(std::get<1>(key));
      auto h3 = std::hash<Index_t>{}(std::get<2>(key));
      auto h4 = std::hash<Index_t>{}(std::get<3>(key));
      auto h5 = std::hash<Index_t>{}(std::get<4>(key));
      auto h6 = std::hash<Index_t>{}(std::get<5>(key));
      auto h7 = std::hash<Index_t>{}(std::get<6>(key));
      auto h8 = std::hash<Index_t>{}(std::get<7>(key));
      // Combine hashes
      std::size_t result = h1;
      result ^= h2 + 0x9e3779b9 + (result << 6) + (result >> 2);
      result ^= h3 + 0x9e3779b9 + (result << 6) + (result >> 2);
      result ^= h4 + 0x9e3779b9 + (result << 6) + (result >> 2);
      result ^= h5 + 0x9e3779b9 + (result << 6) + (result >> 2);
      result ^= h6 + 0x9e3779b9 + (result << 6) + (result >> 2);
      result ^= h7 + 0x9e3779b9 + (result << 6) + (result >> 2);
      result ^= h8 + 0x9e3779b9 + (result << 6) + (result >> 2);
      return result;
    }
  };

  /**
   * Transform type identifiers for plan caching.
   */
  enum TransformType { R2C = 0, C2R = 1, C2C = 2 };

  /**
   * Direction identifiers for C2C transforms.
   */
  enum Direction { FORWARD = 0, BACKWARD = 1 };

  /**
   * Cached plan with associated execution info.
   */
  struct CachedPlan {
    rocfft_plan plan;
    rocfft_execution_info info;
    void * work_buffer;
    size_t work_buffer_size;
  };

  /**
   * Get or create a rocFFT plan for the given parameters.
   *
   * @param type       Transform type (R2C, C2R, or C2C)
   * @param direction  Transform direction (FORWARD or BACKWARD, only for C2C)
   * @param n          Transform size
   * @param batch      Number of batched transforms
   * @param in_stride  Input stride between elements
   * @param in_dist    Input distance between batches
   * @param out_stride Output stride between elements
   * @param out_dist   Output distance between batches
   * @return CachedPlan for the requested transform
   */
  CachedPlan & get_plan(TransformType type, Direction direction, Index_t n,
                        Index_t batch, Index_t in_stride, Index_t in_dist,
                        Index_t out_stride, Index_t out_dist);

  /**
   * Check rocFFT result and throw on error.
   */
  static void check_rocfft_result(rocfft_status result, const char * operation);

  //! Plan cache
  std::unordered_map<PlanKey, CachedPlan, PlanKeyHash> plan_cache;

  //! Flag to track if rocfft_setup has been called
  static bool rocfft_initialized;
};

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_FFT_ROCFFT_BACKEND_HH_
