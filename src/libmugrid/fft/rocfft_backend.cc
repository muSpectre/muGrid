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

#include <sstream>

namespace muGrid {

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
      hipFree(cached.work_buffer);
    }
    if (cached.info != nullptr) {
      rocfft_execution_info_destroy(cached.info);
    }
    if (cached.plan != nullptr) {
      rocfft_plan_destroy(cached.plan);
    }
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
  hipDeviceSynchronize();
}

void rocFFTBackend::c2r(Index_t n, Index_t batch, const Complex * input,
                        Index_t in_stride, Index_t in_dist, Real * output,
                        Index_t out_stride, Index_t out_dist) {
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
  hipDeviceSynchronize();
}

void rocFFTBackend::c2c_forward(Index_t n, Index_t batch, const Complex * input,
                                Index_t in_stride, Index_t in_dist,
                                Complex * output, Index_t out_stride,
                                Index_t out_dist) {
  CachedPlan & cached =
      get_plan(C2C, FORWARD, n, batch, in_stride, in_dist, out_stride,
               out_dist);

  void * in_buffer[1] = {const_cast<Complex *>(input)};
  void * out_buffer[1] = {output};

  rocfft_status status =
      rocfft_execute(cached.plan, in_buffer, out_buffer, cached.info);
  check_rocfft_result(status, "C2C forward execution");

  // Synchronize to ensure completion
  hipDeviceSynchronize();
}

void rocFFTBackend::c2c_backward(Index_t n, Index_t batch,
                                 const Complex * input, Index_t in_stride,
                                 Index_t in_dist, Complex * output,
                                 Index_t out_stride, Index_t out_dist) {
  CachedPlan & cached =
      get_plan(C2C, BACKWARD, n, batch, in_stride, in_dist, out_stride,
               out_dist);

  void * in_buffer[1] = {const_cast<Complex *>(input)};
  void * out_buffer[1] = {output};

  rocfft_status status =
      rocfft_execute(cached.plan, in_buffer, out_buffer, cached.info);
  check_rocfft_result(status, "C2C backward execution");

  // Synchronize to ensure completion
  hipDeviceSynchronize();
}

}  // namespace muGrid
