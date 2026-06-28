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
#include "memory/gpu_runtime.hh"

#include <hip/hip_runtime.h>

#include <sstream>

namespace muGrid {

namespace {
// rocFFT specifies lengths and strides fastest-varying dimension first, whereas
// the engine lists the transformed axes slowest-first with the half-complex
// (r2c/c2r) axis last. Re-order a per-axis quantity into rocFFT order by
// reversing the transformed-axis list.
std::vector<std::size_t> rocfft_order(const std::vector<Index_t> & by_axis,
                                      const std::vector<Index_t> & axes) {
  const std::size_t rank{axes.size()};
  std::vector<std::size_t> out(rank);
  for (std::size_t d{0}; d < rank; ++d) {
    out[d] = static_cast<std::size_t>(by_axis[axes[rank - 1 - d]]);
  }
  return out;
}
}  // namespace

// Static member initialization
bool rocFFTBackend::rocfft_initialized = false;

rocFFTBackend::rocFFTBackend() {
  // Initialize rocFFT library (only once)
  if (!rocfft_initialized) {
    rocfft_status status = rocfft_setup();
    if (status != rocfft_status_success) {
      throw RuntimeError("Failed to initialize rocFFT library");
    }
    rocfft_initialized = true;
  }
  // Note: We don't call rocfft_cleanup() at teardown because other instances
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

RocfftCachedPlan rocFFTBackend::create_plan(
    TransformType type, rocfft_transform_type transform_type,
    const std::vector<std::size_t> & lengths,
    const std::vector<std::size_t> & in_strides, std::size_t in_dist,
    const std::vector<std::size_t> & out_strides, std::size_t out_dist,
    std::size_t batch) {
  RocfftCachedPlan cached{};
  cached.plan = nullptr;
  cached.info = nullptr;
  cached.work_buffer = nullptr;
  cached.work_buffer_size = 0;

  rocfft_array_type in_array_type, out_array_type;
  switch (type) {
  case R2C:
    in_array_type = rocfft_array_type_real;
    out_array_type = rocfft_array_type_hermitian_interleaved;
    break;
  case C2R:
    in_array_type = rocfft_array_type_hermitian_interleaved;
    out_array_type = rocfft_array_type_real;
    break;
  case C2C:
    in_array_type = rocfft_array_type_complex_interleaved;
    out_array_type = rocfft_array_type_complex_interleaved;
    break;
  default:
    throw RuntimeError("Unknown transform type");
  }
  rocfft_result_placement placement = rocfft_placement_notinplace;

  // Create plan description for advanced data layout
  rocfft_plan_description description = nullptr;
  rocfft_status status = rocfft_plan_description_create(&description);
  check_rocfft_result(status, "plan description creation");

  status = rocfft_plan_description_set_data_layout(
      description, in_array_type, out_array_type,
      nullptr,  // input offsets (base pointer is pre-offset by the engine)
      nullptr,  // output offsets
      in_strides.size(), in_strides.data(), in_dist, out_strides.size(),
      out_strides.data(), out_dist);
  check_rocfft_result(status, "setting data layout");

  status = rocfft_plan_create(&cached.plan, placement, transform_type,
                              rocfft_precision_double, lengths.size(),
                              lengths.data(), batch, description);
  check_rocfft_result(status, "plan creation");

  // Destroy the description (no longer needed after plan creation)
  rocfft_plan_description_destroy(description);

  // Query work buffer size
  status =
      rocfft_plan_get_work_buffer_size(cached.plan, &cached.work_buffer_size);
  check_rocfft_result(status, "querying work buffer size");

  // Create execution info
  status = rocfft_execution_info_create(&cached.info);
  check_rocfft_result(status, "execution info creation");

  // Allocate work buffer if needed
  if (cached.work_buffer_size > 0) {
    // Route through the device allocator chokepoint (single owner + visible
    // to the allocation profiler).
    cached.work_buffer =
        device_allocate(cached.work_buffer_size, "rocfft-work-buffer");
    status = rocfft_execution_info_set_work_buffer(
        cached.info, cached.work_buffer, cached.work_buffer_size);
    check_rocfft_result(status, "setting work buffer");
  }
  return cached;
}

RocfftCachedPlan rocFFTBackend::make_plan(const PlanKey & key) {
  TransformType type;
  rocfft_transform_type transform_type;
  switch (key.kind) {
  case Base::Kind::R2C:
    type = R2C;
    transform_type = rocfft_transform_type_real_forward;
    break;
  case Base::Kind::C2R:
    type = C2R;
    transform_type = rocfft_transform_type_real_inverse;
    break;
  case Base::Kind::C2C:
    type = C2C;
    transform_type = (key.direction == Base::Direction::FORWARD)
                         ? rocfft_transform_type_complex_forward
                         : rocfft_transform_type_complex_inverse;
    break;
  default:
    throw RuntimeError("Unknown transform type");
  }

  std::size_t in_distance = static_cast<std::size_t>(key.in_dist);
  std::size_t out_distance = static_cast<std::size_t>(key.out_dist);
  // For batch=1, distance is not used, but rocFFT needs valid values; set it to
  // the transform size if not specified.
  if (key.batch == 1) {
    if (in_distance == 0) {
      in_distance = static_cast<std::size_t>((type == C2R) ? (key.n / 2 + 1)
                                                           : key.n);
    }
    if (out_distance == 0) {
      out_distance = static_cast<std::size_t>((type == R2C) ? (key.n / 2 + 1)
                                                           : key.n);
    }
  }

  std::vector<std::size_t> lengths{static_cast<std::size_t>(key.n)};
  std::vector<std::size_t> in_strides{static_cast<std::size_t>(key.in_stride)};
  std::vector<std::size_t> out_strides{
      static_cast<std::size_t>(key.out_stride)};
  return create_plan(type, transform_type, lengths, in_strides, in_distance,
                     out_strides, out_distance,
                     static_cast<std::size_t>(key.batch));
}

void rocFFTBackend::destroy_plan(RocfftCachedPlan & cached) {
  if (cached.work_buffer != nullptr) {
    device_deallocate(cached.work_buffer);
  }
  if (cached.info != nullptr) {
    rocfft_execution_info_destroy(cached.info);
  }
  if (cached.plan != nullptr) {
    rocfft_plan_destroy(cached.plan);
  }
}

void rocFFTBackend::exec_r2c(RocfftCachedPlan & cached, const Real * input,
                             Complex * output) {
  // rocFFT uses arrays of buffer pointers for in-place/out-of-place transforms
  void * in_buffer[1] = {const_cast<Real *>(input)};
  void * out_buffer[1] = {output};
  rocfft_status status =
      rocfft_execute(cached.plan, in_buffer, out_buffer, cached.info);
  check_rocfft_result(status, "R2C execution");
}

void rocFFTBackend::exec_c2r(RocfftCachedPlan & cached, const Complex * input,
                             Real * output) {
  // Note: rocFFT may modify the input during C2R transforms; the engine stages
  // a copy where it needs the input preserved.
  void * in_buffer[1] = {const_cast<Complex *>(input)};
  void * out_buffer[1] = {output};
  rocfft_status status =
      rocfft_execute(cached.plan, in_buffer, out_buffer, cached.info);
  check_rocfft_result(status, "C2R execution");
}

void rocFFTBackend::exec_c2c_forward(RocfftCachedPlan & cached,
                                     const Complex * input, Complex * output) {
  void * in_buffer[1] = {const_cast<Complex *>(input)};
  void * out_buffer[1] = {output};
  rocfft_status status =
      rocfft_execute(cached.plan, in_buffer, out_buffer, cached.info);
  check_rocfft_result(status, "C2C forward execution");
}

void rocFFTBackend::exec_c2c_backward(RocfftCachedPlan & cached,
                                      const Complex * input, Complex * output) {
  void * in_buffer[1] = {const_cast<Complex *>(input)};
  void * out_buffer[1] = {output};
  rocfft_status status =
      rocfft_execute(cached.plan, in_buffer, out_buffer, cached.info);
  check_rocfft_result(status, "C2C backward execution");
}

RocfftCachedPlan & rocFFTBackend::get_nd_plan(
    TransformType type, const std::vector<std::size_t> & lengths,
    const std::vector<std::size_t> & in_strides, std::size_t in_dist,
    const std::vector<std::size_t> & out_strides, std::size_t out_dist,
    std::size_t batch) {
  // String signature over every parameter that defines the plan.
  std::stringstream ss;
  ss << type << "|b:" << batch << "|id:" << in_dist << "|od:" << out_dist
     << "|l:";
  for (std::size_t v : lengths) ss << v << ',';
  ss << "|is:";
  for (std::size_t v : in_strides) ss << v << ',';
  ss << "|os:";
  for (std::size_t v : out_strides) ss << v << ',';

  return this->nd_plan_for(ss.str(), [&]() -> RocfftCachedPlan {
    rocfft_transform_type transform_type;
    switch (type) {
    case R2C:
      transform_type = rocfft_transform_type_real_forward;
      break;
    case C2R:
      transform_type = rocfft_transform_type_real_inverse;
      break;
    default:
      throw RuntimeError("get_nd_plan supports only R2C and C2R transforms");
    }
    return create_plan(type, transform_type, lengths, in_strides, in_dist,
                       out_strides, out_dist, batch);
  });
}

void rocFFTBackend::r2c_nd(const std::vector<Index_t> & shape,
                           const std::vector<Index_t> & axes, const Real * input,
                           const std::vector<Index_t> & in_strides,
                           Complex * output,
                           const std::vector<Index_t> & out_strides) {
  if (shape[0] == 0) {
    return;  // empty component batch
  }
  this->check_complex_aligned(input, "r2c (N-D)");

  std::vector<std::size_t> lengths{rocfft_order(shape, axes)};
  std::vector<std::size_t> in_str{rocfft_order(in_strides, axes)};
  std::vector<std::size_t> out_str{rocfft_order(out_strides, axes)};

  RocfftCachedPlan & cached = get_nd_plan(
      R2C, lengths, in_str, static_cast<std::size_t>(in_strides[0]), out_str,
      static_cast<std::size_t>(out_strides[0]),
      static_cast<std::size_t>(shape[0]));

  void * in_buffer[1] = {const_cast<Real *>(input)};
  void * out_buffer[1] = {output};
  rocfft_status status =
      rocfft_execute(cached.plan, in_buffer, out_buffer, cached.info);
  check_rocfft_result(status, "R2C (N-D) execution");
  GPU_STREAM_SYNCHRONIZE_DEFAULT();
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
  this->check_complex_aligned(output, "c2r (N-D)");

  std::vector<std::size_t> lengths{rocfft_order(shape, axes)};
  std::vector<std::size_t> in_str{rocfft_order(in_strides, axes)};
  std::vector<std::size_t> out_str{rocfft_order(out_strides, axes)};

  // rocFFT's real-inverse transform may overwrite its input, but the engine
  // requires the const Fourier input preserved (the per-axis fallback copies it
  // into a work buffer). Stage through a scratch copy. The input spans
  // batch * in_dist complex elements (the per-component Fourier buffer is
  // contiguous, so in_dist is the full component stride).
  const std::size_t in_dist{static_cast<std::size_t>(in_strides[0])};
  const std::size_t span{static_cast<std::size_t>(shape[0]) * in_dist};
  void * scratch = this->ensure_scratch(span * sizeof(Complex));
  GPU_MEMCPY_D2D(scratch, input, span * sizeof(Complex));

  RocfftCachedPlan & cached =
      get_nd_plan(C2R, lengths, in_str, in_dist, out_str,
                  static_cast<std::size_t>(out_strides[0]),
                  static_cast<std::size_t>(shape[0]));

  void * in_buffer[1] = {scratch};
  void * out_buffer[1] = {output};
  rocfft_status status =
      rocfft_execute(cached.plan, in_buffer, out_buffer, cached.info);
  check_rocfft_result(status, "C2R (N-D) execution");
  GPU_STREAM_SYNCHRONIZE_DEFAULT();
}

}  // namespace muGrid
