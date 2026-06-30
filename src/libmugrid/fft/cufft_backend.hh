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

#include "gpu_fft_backend.hh"

#include <cufft.h>

#include <string>
#include <vector>

namespace muGrid {

/**
 * cuFFT implementation of FFT1DBackend for NVIDIA GPUs.
 *
 * This backend uses NVIDIA's cuFFT library for GPU-accelerated FFT operations.
 * It operates on device memory and supports batched 1D transforms with
 * arbitrary strides (except on the real side of r2c/c2r — see
 * supports_strided_r2c).
 *
 * The backend-agnostic machinery (the batch==0 guard, plan cache and hash,
 * alignment check, scratch buffer, synchronisation and destructor cleanup)
 * lives in GpuFFTBackend; this class supplies only the cuFFT library calls
 * (make_plan / destroy_plan / exec_*) and the N-D entry points.
 */
class cuFFTBackend : public GpuFFTBackend<cuFFTBackend, cufftHandle> {
  using Base = GpuFFTBackend<cuFFTBackend, cufftHandle>;
  friend Base;

 public:
  cuFFTBackend() = default;
  ~cuFFTBackend() override { this->destroy_all_plans(); }

  //! cuFFT does not support a non-unit element stride on the real side of
  //! r2c/c2r (a documented limitation); the base throws when this is violated.
  bool supports_strided_r2c() const override { return false; }
  bool supports_strided_c2r() const override { return false; }

  void r2c_nd(const std::vector<Index_t> & shape,
              const std::vector<Index_t> & axes, const Real * input,
              const std::vector<Index_t> & in_strides, Complex * output,
              const std::vector<Index_t> & out_strides) override;

  void c2r_nd(const std::vector<Index_t> & shape,
              const std::vector<Index_t> & axes, const Complex * input,
              const std::vector<Index_t> & in_strides, Real * output,
              const std::vector<Index_t> & out_strides) override;

  // Single-precision N-D transforms (CUFFT_R2C/C2R). Used by the serial-nd
  // engine path; the 1D fp32 primitives (MPI GPU) are not yet implemented.
  void r2c_nd(const std::vector<Index_t> & shape,
              const std::vector<Index_t> & axes, const Real32 * input,
              const std::vector<Index_t> & in_strides, Complex32 * output,
              const std::vector<Index_t> & out_strides) override;

  void c2r_nd(const std::vector<Index_t> & shape,
              const std::vector<Index_t> & axes, const Complex32 * input,
              const std::vector<Index_t> & in_strides, Real32 * output,
              const std::vector<Index_t> & out_strides) override;

  const char * name() const override { return "cufft"; }

 protected:
  // ---- GpuFFTBackend hooks (cuFFT library calls) ----

  //! Create a cuFFT plan for the given 1D batched transform. The C2C plan is
  //! direction-agnostic (direction is supplied at execution), so the key's
  //! direction only separates cache slots.
  cufftHandle make_plan(const PlanKey & key);
  void destroy_plan(cufftHandle & plan);

  void exec_r2c(cufftHandle & plan, const Real * input, Complex * output);
  void exec_c2r(cufftHandle & plan, const Complex * input, Real * output);
  void exec_c2c_forward(cufftHandle & plan, const Complex * input,
                        Complex * output);
  void exec_c2c_backward(cufftHandle & plan, const Complex * input,
                         Complex * output);

  // ---- cuFFT-specific N-D helpers ----

  /**
   * Get or create a cached N-dimensional cuFFT plan (cufftPlanMany), via the
   * base's nd_plan_for. The arguments are the cuFFT advanced-data-layout
   * parameters: `n` are the logical (real-space) transform sizes,
   * `inembed`/`onembed` the physical padded extents, the strides the
   * innermost-element strides, and the dists the batch (component) strides.
   */
  cufftHandle get_nd_plan(cufftType type, const std::vector<int> & n,
                          const std::vector<int> & inembed, int istride,
                          int idist, const std::vector<int> & onembed,
                          int ostride, int odist, int batch);

  /**
   * Check cuFFT result and throw on error.
   */
  static void check_cufft_result(cufftResult result, const char * operation);
};

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_FFT_CUFFT_BACKEND_HH_
