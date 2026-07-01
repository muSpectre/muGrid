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

#include "gpu_fft_backend.hh"

#include <rocfft/rocfft.h>

#include <cstddef>
#include <vector>

namespace muGrid {

/**
 * A cached rocFFT plan together with the execution info and work buffer it
 * needs. This is the per-plan payload stored by the GpuFFTBackend caches (the
 * `PlanT` template argument), so it lives at namespace scope rather than nested
 * in the backend.
 */
struct RocfftCachedPlan {
  rocfft_plan plan;
  rocfft_execution_info info;
  void * work_buffer;
  std::size_t work_buffer_size;
};

/**
 * Native rocFFT implementation of FFT1DBackend for AMD GPUs.
 *
 * This backend uses AMD's rocFFT library directly (not via hipFFT) for
 * GPU-accelerated FFT operations. Using the native rocFFT API provides
 * better support for strided data layouts, which is essential for
 * 3D MPI-parallel FFTs (rocFFT supports arbitrary strides for all transform
 * types including R2C and C2R, which cuFFT does not).
 *
 * The backend-agnostic machinery (the batch==0 guard, plan cache and hash,
 * alignment check, scratch buffer, synchronisation and destructor cleanup)
 * lives in GpuFFTBackend; this class supplies only the rocFFT library calls
 * (make_plan / destroy_plan / exec_*) and the N-D entry points.
 */
class rocFFTBackend : public GpuFFTBackend<rocFFTBackend, RocfftCachedPlan> {
  using Base = GpuFFTBackend<rocFFTBackend, RocfftCachedPlan>;
  friend Base;

 public:
  rocFFTBackend();
  ~rocFFTBackend() override { this->destroy_all_plans(); }

  void r2c_nd(const std::vector<Index_t> & shape,
              const std::vector<Index_t> & axes, const Real * input,
              const std::vector<Index_t> & in_strides, Complex * output,
              const std::vector<Index_t> & out_strides) override;

  void c2r_nd(const std::vector<Index_t> & shape,
              const std::vector<Index_t> & axes, const Complex * input,
              const std::vector<Index_t> & in_strides, Real * output,
              const std::vector<Index_t> & out_strides) override;

  // Single-precision (Real32/Complex32) N-D transforms. Mirror the double
  // variants with a single-precision rocFFT plan (rocfft_precision_single).
  void r2c_nd(const std::vector<Index_t> & shape,
              const std::vector<Index_t> & axes, const Real32 * input,
              const std::vector<Index_t> & in_strides, Complex32 * output,
              const std::vector<Index_t> & out_strides) override;

  void c2r_nd(const std::vector<Index_t> & shape,
              const std::vector<Index_t> & axes, const Complex32 * input,
              const std::vector<Index_t> & in_strides, Real32 * output,
              const std::vector<Index_t> & out_strides) override;

  const char * name() const override { return "rocfft"; }

 protected:
  /** Transform kind used when building a rocFFT plan from a PlanKey. */
  enum TransformType { R2C = 0, C2R = 1, C2C = 2 };

  // ---- GpuFFTBackend hooks (rocFFT library calls) ----

  RocfftCachedPlan make_plan(const PlanKey & key);
  void destroy_plan(RocfftCachedPlan & cached);

  void exec_r2c(RocfftCachedPlan & cached, const Real * input,
                Complex * output);
  void exec_c2r(RocfftCachedPlan & cached, const Complex * input,
                Real * output);
  void exec_c2c_forward(RocfftCachedPlan & cached, const Complex * input,
                        Complex * output);
  void exec_c2c_backward(RocfftCachedPlan & cached, const Complex * input,
                         Complex * output);

  // Single-precision exec hooks. rocfft_execute is precision-agnostic (it
  // takes void buffers; the precision is baked into the plan), so these differ
  // from the double overloads only in the pointer types they accept.
  void exec_r2c(RocfftCachedPlan & cached, const Real32 * input,
                Complex32 * output);
  void exec_c2r(RocfftCachedPlan & cached, const Complex32 * input,
                Real32 * output);
  void exec_c2c_forward(RocfftCachedPlan & cached, const Complex32 * input,
                        Complex32 * output);
  void exec_c2c_backward(RocfftCachedPlan & cached, const Complex32 * input,
                         Complex32 * output);

  // ---- rocFFT-specific N-D helpers ----

  /**
   * Get or create a cached N-dimensional plan (R2C or C2R), via the base's
   * nd_plan_for. `lengths` and the stride arrays are in rocFFT order
   * (fastest-varying dimension first); the `*_dist` are the batch (component)
   * strides.
   */
  RocfftCachedPlan & get_nd_plan(TransformType type, rocfft_precision precision,
                                 const std::vector<std::size_t> & lengths,
                                 const std::vector<std::size_t> & in_strides,
                                 std::size_t in_dist,
                                 const std::vector<std::size_t> & out_strides,
                                 std::size_t out_dist, std::size_t batch);

  /**
   * Build a rocFFT plan + execution info + work buffer for the given transform
   * type, precision and advanced data layout. Shared by make_plan (1D) and
   * get_nd_plan.
   */
  RocfftCachedPlan create_plan(TransformType type,
                               rocfft_transform_type transform_type,
                               rocfft_precision precision,
                               const std::vector<std::size_t> & lengths,
                               const std::vector<std::size_t> & in_strides,
                               std::size_t in_dist,
                               const std::vector<std::size_t> & out_strides,
                               std::size_t out_dist, std::size_t batch);

  /**
   * Check rocFFT result and throw on error.
   */
  static void check_rocfft_result(rocfft_status result, const char * operation);

  //! Flag to track if rocfft_setup has been called
  static bool rocfft_initialized;
};

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_FFT_ROCFFT_BACKEND_HH_
