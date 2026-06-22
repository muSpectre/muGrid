/**
 * @file   fft/rocfft_backend.cc
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

#include "rocfft_backend.hh"
#include "core/exception.hh"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <sstream>

namespace muGrid {

namespace {
// muGrid pads the layout of FFT engine real-space fields so that the real
// array of every real<->complex transform is aligned like a complex array
// (16 bytes); see the FFTEngineBase constructor. rocFFT does not document
// an alignment requirement, but the layout invariant holds here regardless
// — enforce it for symmetry with the cuFFT backend so a padding regression
// fails loudly on either GPU vendor.
void check_complex_aligned(const void * ptr, const char * operation) {
  // The error branch is unreachable through the public API (the layout
  // invariant guarantees alignment), so it is excluded from coverage.
  // GCOVR_EXCL_START
  if (reinterpret_cast<std::uintptr_t>(ptr) % (2 * sizeof(Real)) != 0) {
    std::stringstream error;
    error << "The real array passed to the rocFFT " << operation
          << " transform is not aligned to the complex type (16 bytes). "
             "muGrid pads the layout of FFT engine real-space fields so "
             "that this cannot happen; this error indicates a bug in "
             "muGrid's layout padding, or a field whose memory was not "
             "allocated through a muGrid FFT engine. External arrays "
             "must be copied into an engine field before transforming.";
    throw RuntimeError(error.str());
  }
  // GCOVR_EXCL_STOP
}

// rocFFT specifies lengths and strides fastest-varying dimension first, whereas
// the engine lists the transformed axes slowest-first with the half-complex
// (r2c/c2r) axis last. Re-order a per-axis quantity into rocFFT order by
// reversing the transformed-axis list.
std::vector<size_t> rocfft_order(const std::vector<Index_t> & by_axis,
                                 const std::vector<Index_t> & axes) {
  const std::size_t rank{axes.size()};
  std::vector<size_t> out(rank);
  for (std::size_t d{0}; d < rank; ++d) {
    out[d] = static_cast<size_t>(by_axis[axes[rank - 1 - d]]);
  }
  return out;
}
}  // namespace

// Static member initialization
bool rocFFTBackend::rocfft_initialized = false;

rocFFTBackend::rocFFTBackend() : plan_cache{} {
  // Initialize rocFFT library (only once)
  if (!rocfft_initialized) {
    rocfft_status status = rocfft_setup();
    if (status != rocfft_status_success) {
      throw RuntimeError("Failed to initialize rocFFT library");
    }
    rocfft_initialized = true;
  }
}

rocFFTBackend::~rocFFTBackend() {
  // Destroy all cached plans
  for (auto & entry : this->plan_cache) {
    CachedPlan & cached = entry.second;
    if (cached.work_buffer != nullptr) {
      (void)hipFree(cached.work_buffer);
    }
    if (cached.info != nullptr) {
      rocfft_execution_info_destroy(cached.info);
    }
    if (cached.plan != nullptr) {
      rocfft_plan_destroy(cached.plan);
    }
  }
  for (auto & entry : this->nd_plan_cache) {
    CachedPlan & cached = entry.second;
    if (cached.work_buffer != nullptr) {
      (void)hipFree(cached.work_buffer);
    }
    if (cached.info != nullptr) {
      rocfft_execution_info_destroy(cached.info);
    }
    if (cached.plan != nullptr) {
      rocfft_plan_destroy(cached.plan);
    }
  }
  if (this->nd_scratch != nullptr) {
    (void)hipFree(this->nd_scratch);
  }
  // Note: We don't call rocfft_cleanup() here because other instances
  // might still be using rocFFT. In practice, cleanup happens at program exit.
}

void rocFFTBackend::check_rocfft_result(rocfft_status result,
                                        const char * operation) {
  if (result != rocfft_status_success) {
    std::stringstream error;
    error << "rocFFT error during " << operation << ": ";
    switch (result) {
    case rocfft_status_invalid_arg_value:
      error << "invalid argument value";
      break;
    case rocfft_status_invalid_dimensions:
      error << "invalid dimensions";
      break;
    case rocfft_status_invalid_array_type:
      error << "invalid array type";
      break;
    case rocfft_status_invalid_strides:
      error << "invalid strides";
      break;
    case rocfft_status_invalid_distance:
      error << "invalid distance";
      break;
    case rocfft_status_invalid_offset:
      error << "invalid offset";
      break;
    case rocfft_status_invalid_work_buffer:
      error << "invalid work buffer";
      break;
    case rocfft_status_failure:
      error << "general failure";
      break;
    default:
      error << "unknown error code " << static_cast<int>(result);
      break;
    }
    throw RuntimeError(error.str());
  }
}

rocFFTBackend::CachedPlan &
rocFFTBackend::get_plan(TransformType type, Direction direction, Index_t n,
                        Index_t batch, Index_t in_stride, Index_t in_dist,
                        Index_t out_stride, Index_t out_dist) {
  PlanKey key{type, direction, n, batch, in_stride, in_dist, out_stride,
              out_dist};

  auto it = this->plan_cache.find(key);
  if (it != this->plan_cache.end()) {
    return it->second;
  }

  // Create new plan using rocFFT native API
  CachedPlan cached{};
  cached.plan = nullptr;
  cached.info = nullptr;
  cached.work_buffer = nullptr;
  cached.work_buffer_size = 0;

  // Determine transform parameters
  rocfft_transform_type transform_type;
  rocfft_result_placement placement = rocfft_placement_notinplace;
  rocfft_array_type in_array_type, out_array_type;

  switch (type) {
  case R2C:
    transform_type = rocfft_transform_type_real_forward;
    in_array_type = rocfft_array_type_real;
    out_array_type = rocfft_array_type_hermitian_interleaved;
    break;
  case C2R:
    transform_type = rocfft_transform_type_real_inverse;
    in_array_type = rocfft_array_type_hermitian_interleaved;
    out_array_type = rocfft_array_type_real;
    break;
  case C2C:
    transform_type = (direction == FORWARD)
                         ? rocfft_transform_type_complex_forward
                         : rocfft_transform_type_complex_inverse;
    in_array_type = rocfft_array_type_complex_interleaved;
    out_array_type = rocfft_array_type_complex_interleaved;
    break;
  default:
    throw RuntimeError("Unknown transform type");
  }

  // Create plan description for advanced data layout
  rocfft_plan_description description = nullptr;
  rocfft_status status = rocfft_plan_description_create(&description);
  check_rocfft_result(status, "plan description creation");

  // Set data layout with strides
  // rocFFT uses size_t for strides and distances
  size_t lengths[1] = {static_cast<size_t>(n)};
  size_t in_strides[1] = {static_cast<size_t>(in_stride)};
  size_t out_strides[1] = {static_cast<size_t>(out_stride)};
  size_t in_distance = static_cast<size_t>(in_dist);
  size_t out_distance = static_cast<size_t>(out_dist);

  // For batch=1, distance is not used, but we need valid values
  // Set distance to transform size if not specified
  if (batch == 1) {
    if (in_distance == 0) {
      in_distance = (type == C2R) ? (n / 2 + 1) : n;
    }
    if (out_distance == 0) {
      out_distance = (type == R2C) ? (n / 2 + 1) : n;
    }
  }

  status = rocfft_plan_description_set_data_layout(
      description,
      in_array_type,   // input array type
      out_array_type,  // output array type
      nullptr,         // input offsets (not used)
      nullptr,         // output offsets (not used)
      1,               // number of dimensions for input strides
      in_strides,      // input strides
      in_distance,     // input distance between batches
      1,               // number of dimensions for output strides
      out_strides,     // output strides
      out_distance     // output distance between batches
  );
  check_rocfft_result(status, "setting data layout");

  // Create the plan
  status = rocfft_plan_create(&cached.plan, placement, transform_type,
                              rocfft_precision_double,
                              1,  // number of dimensions
                              lengths, static_cast<size_t>(batch), description);
  check_rocfft_result(status, "plan creation");

  // Destroy the description (no longer needed after plan creation)
  rocfft_plan_description_destroy(description);

  // Query work buffer size
  status = rocfft_plan_get_work_buffer_size(cached.plan,
                                            &cached.work_buffer_size);
  check_rocfft_result(status, "querying work buffer size");

  // Create execution info
  status = rocfft_execution_info_create(&cached.info);
  check_rocfft_result(status, "execution info creation");

  // Allocate work buffer if needed
  if (cached.work_buffer_size > 0) {
    hipError_t hip_status = hipMalloc(&cached.work_buffer,
                                      cached.work_buffer_size);
    if (hip_status != hipSuccess) {
      throw RuntimeError("Failed to allocate rocFFT work buffer");
    }
    status = rocfft_execution_info_set_work_buffer(cached.info,
                                                   cached.work_buffer,
                                                   cached.work_buffer_size);
    check_rocfft_result(status, "setting work buffer");
  }

  this->plan_cache[key] = cached;
  return this->plan_cache[key];
}

void rocFFTBackend::r2c(Index_t n, Index_t batch, const Real * input,
                        Index_t in_stride, Index_t in_dist, Complex * output,
                        Index_t out_stride, Index_t out_dist) {
  // Ranks with an empty subdomain (possible when a grid dimension is
  // smaller than the process grid) have nothing to transform.
  if (batch == 0) {
    return;
  }

  check_complex_aligned(input, "R2C");

  CachedPlan & cached =
      get_plan(R2C, FORWARD, n, batch, in_stride, in_dist, out_stride,
               out_dist);

  // rocFFT uses arrays of buffer pointers for in-place/out-of-place transforms
  void * in_buffer[1] = {const_cast<Real *>(input)};
  void * out_buffer[1] = {output};

  rocfft_status status =
      rocfft_execute(cached.plan, in_buffer, out_buffer, cached.info);
  check_rocfft_result(status, "R2C execution");

  // Synchronize to ensure completion
  (void)hipDeviceSynchronize();
}

void rocFFTBackend::c2r(Index_t n, Index_t batch, const Complex * input,
                        Index_t in_stride, Index_t in_dist, Real * output,
                        Index_t out_stride, Index_t out_dist) {
  // Empty subdomain: nothing to transform (see r2c)
  if (batch == 0) {
    return;
  }

  check_complex_aligned(output, "C2R");

  CachedPlan & cached =
      get_plan(C2R, BACKWARD, n, batch, in_stride, in_dist, out_stride,
               out_dist);

  // rocFFT uses arrays of buffer pointers
  // Note: rocFFT may modify the input during C2R transforms
  void * in_buffer[1] = {const_cast<Complex *>(input)};
  void * out_buffer[1] = {output};

  rocfft_status status =
      rocfft_execute(cached.plan, in_buffer, out_buffer, cached.info);
  check_rocfft_result(status, "C2R execution");

  // Synchronize to ensure completion
  (void)hipDeviceSynchronize();
}

void rocFFTBackend::c2c_forward(Index_t n, Index_t batch, const Complex * input,
                                Index_t in_stride, Index_t in_dist,
                                Complex * output, Index_t out_stride,
                                Index_t out_dist) {
  // Empty subdomain: nothing to transform (see r2c)
  if (batch == 0) {
    return;
  }

  CachedPlan & cached =
      get_plan(C2C, FORWARD, n, batch, in_stride, in_dist, out_stride,
               out_dist);

  void * in_buffer[1] = {const_cast<Complex *>(input)};
  void * out_buffer[1] = {output};

  rocfft_status status =
      rocfft_execute(cached.plan, in_buffer, out_buffer, cached.info);
  check_rocfft_result(status, "C2C forward execution");

  // Synchronize to ensure completion
  (void)hipDeviceSynchronize();
}

void rocFFTBackend::c2c_backward(Index_t n, Index_t batch,
                                 const Complex * input, Index_t in_stride,
                                 Index_t in_dist, Complex * output,
                                 Index_t out_stride, Index_t out_dist) {
  // Empty subdomain: nothing to transform (see r2c)
  if (batch == 0) {
    return;
  }

  CachedPlan & cached =
      get_plan(C2C, BACKWARD, n, batch, in_stride, in_dist, out_stride,
               out_dist);

  void * in_buffer[1] = {const_cast<Complex *>(input)};
  void * out_buffer[1] = {output};

  rocfft_status status =
      rocfft_execute(cached.plan, in_buffer, out_buffer, cached.info);
  check_rocfft_result(status, "C2C backward execution");

  // Synchronize to ensure completion
  (void)hipDeviceSynchronize();
}

rocFFTBackend::CachedPlan &
rocFFTBackend::get_nd_plan(TransformType type,
                           const std::vector<size_t> & lengths,
                           const std::vector<size_t> & in_strides,
                           size_t in_dist,
                           const std::vector<size_t> & out_strides,
                           size_t out_dist, size_t batch) {
  // String signature over every parameter that defines the plan.
  std::stringstream ss;
  ss << type << "|b:" << batch << "|id:" << in_dist << "|od:" << out_dist
     << "|l:";
  for (size_t v : lengths) ss << v << ',';
  ss << "|is:";
  for (size_t v : in_strides) ss << v << ',';
  ss << "|os:";
  for (size_t v : out_strides) ss << v << ',';
  const std::string key{ss.str()};

  auto it = this->nd_plan_cache.find(key);
  if (it != this->nd_plan_cache.end()) {
    return it->second;
  }

  CachedPlan cached{};
  cached.plan = nullptr;
  cached.info = nullptr;
  cached.work_buffer = nullptr;
  cached.work_buffer_size = 0;

  rocfft_transform_type transform_type;
  rocfft_array_type in_array_type, out_array_type;
  switch (type) {
  case R2C:
    transform_type = rocfft_transform_type_real_forward;
    in_array_type = rocfft_array_type_real;
    out_array_type = rocfft_array_type_hermitian_interleaved;
    break;
  case C2R:
    transform_type = rocfft_transform_type_real_inverse;
    in_array_type = rocfft_array_type_hermitian_interleaved;
    out_array_type = rocfft_array_type_real;
    break;
  default:
    throw RuntimeError("get_nd_plan supports only R2C and C2R transforms");
  }
  rocfft_result_placement placement = rocfft_placement_notinplace;

  rocfft_plan_description description = nullptr;
  rocfft_status status = rocfft_plan_description_create(&description);
  check_rocfft_result(status, "N-D plan description creation");

  status = rocfft_plan_description_set_data_layout(
      description, in_array_type, out_array_type,
      nullptr,  // input offsets (base pointer is pre-offset by the engine)
      nullptr,  // output offsets
      in_strides.size(), in_strides.data(), in_dist,
      out_strides.size(), out_strides.data(), out_dist);
  check_rocfft_result(status, "setting N-D data layout");

  status = rocfft_plan_create(&cached.plan, placement, transform_type,
                              rocfft_precision_double, lengths.size(),
                              lengths.data(), batch, description);
  check_rocfft_result(status, "N-D plan creation");
  rocfft_plan_description_destroy(description);

  status = rocfft_plan_get_work_buffer_size(cached.plan,
                                            &cached.work_buffer_size);
  check_rocfft_result(status, "querying N-D work buffer size");

  status = rocfft_execution_info_create(&cached.info);
  check_rocfft_result(status, "N-D execution info creation");

  if (cached.work_buffer_size > 0) {
    hipError_t hip_status =
        hipMalloc(&cached.work_buffer, cached.work_buffer_size);
    if (hip_status != hipSuccess) {
      throw RuntimeError("Failed to allocate rocFFT N-D work buffer");
    }
    status = rocfft_execution_info_set_work_buffer(
        cached.info, cached.work_buffer, cached.work_buffer_size);
    check_rocfft_result(status, "setting N-D work buffer");
  }

  this->nd_plan_cache[key] = cached;
  return this->nd_plan_cache[key];
}

void * rocFFTBackend::ensure_nd_scratch(size_t bytes) {
  if (bytes > this->nd_scratch_bytes) {
    if (this->nd_scratch != nullptr) {
      (void)hipFree(this->nd_scratch);
    }
    hipError_t err = hipMalloc(&this->nd_scratch, bytes);
    if (err != hipSuccess) {
      this->nd_scratch = nullptr;
      this->nd_scratch_bytes = 0;
      throw RuntimeError("hipMalloc failed for rocFFT N-D scratch buffer");
    }
    this->nd_scratch_bytes = bytes;
  }
  return this->nd_scratch;
}

void rocFFTBackend::r2c_nd(const std::vector<Index_t> & shape,
                           const std::vector<Index_t> & axes, const Real * input,
                           const std::vector<Index_t> & in_strides,
                           Complex * output,
                           const std::vector<Index_t> & out_strides) {
  if (shape[0] == 0) {
    return;  // empty component batch
  }
  check_complex_aligned(input, "R2C (N-D)");

  std::vector<size_t> lengths{rocfft_order(shape, axes)};
  std::vector<size_t> in_str{rocfft_order(in_strides, axes)};
  std::vector<size_t> out_str{rocfft_order(out_strides, axes)};

  CachedPlan & cached = get_nd_plan(
      R2C, lengths, in_str, static_cast<size_t>(in_strides[0]), out_str,
      static_cast<size_t>(out_strides[0]), static_cast<size_t>(shape[0]));

  void * in_buffer[1] = {const_cast<Real *>(input)};
  void * out_buffer[1] = {output};
  rocfft_status status =
      rocfft_execute(cached.plan, in_buffer, out_buffer, cached.info);
  check_rocfft_result(status, "R2C (N-D) execution");
  (void)hipDeviceSynchronize();
}

void rocFFTBackend::c2r_nd(const std::vector<Index_t> & shape,
                           const std::vector<Index_t> & axes,
                           const Complex * input,
                           const std::vector<Index_t> & in_strides,
                           Real * output,
                           const std::vector<Index_t> & out_strides) {
  if (shape[0] == 0) {
    return;
  }
  check_complex_aligned(output, "C2R (N-D)");

  std::vector<size_t> lengths{rocfft_order(shape, axes)};
  std::vector<size_t> in_str{rocfft_order(in_strides, axes)};
  std::vector<size_t> out_str{rocfft_order(out_strides, axes)};

  // rocFFT's real-inverse transform may overwrite its input, but the engine
  // requires the const Fourier input preserved (the per-axis fallback copies it
  // into a work buffer). Stage through a scratch copy. The input spans
  // batch * in_dist complex elements (the per-component Fourier buffer is
  // contiguous, so in_dist is the full component stride).
  const size_t in_dist{static_cast<size_t>(in_strides[0])};
  const size_t span{static_cast<size_t>(shape[0]) * in_dist};
  void * scratch = ensure_nd_scratch(span * sizeof(Complex));
  (void)hipMemcpy(scratch, input, span * sizeof(Complex),
                  hipMemcpyDeviceToDevice);

  CachedPlan & cached = get_nd_plan(C2R, lengths, in_str, in_dist, out_str,
                                    static_cast<size_t>(out_strides[0]),
                                    static_cast<size_t>(shape[0]));

  void * in_buffer[1] = {scratch};
  void * out_buffer[1] = {output};
  rocfft_status status =
      rocfft_execute(cached.plan, in_buffer, out_buffer, cached.info);
  check_rocfft_result(status, "C2R (N-D) execution");
  (void)hipDeviceSynchronize();
}

}  // namespace muGrid
