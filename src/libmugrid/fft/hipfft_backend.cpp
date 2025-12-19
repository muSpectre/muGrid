/**
 * @file   fft/hipfft_backend.cpp
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   19 Dec 2024
 *
 * @brief  hipFFT implementation of FFT1DBackend for AMD GPUs
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

#include "hipfft_backend.hh"
#include "core/exception.hh"

#include <sstream>

namespace muGrid {

hipFFTBackend::hipFFTBackend() : plan_cache{} {}

hipFFTBackend::~hipFFTBackend() {
  // Destroy all cached plans
  for (auto & entry : this->plan_cache) {
    hipfftDestroy(entry.second);
  }
}

void hipFFTBackend::check_hipfft_result(hipfftResult result,
                                        const char * operation) {
  if (result != HIPFFT_SUCCESS) {
    std::stringstream error;
    error << "hipFFT error during " << operation << ": ";
    switch (result) {
    case HIPFFT_INVALID_PLAN:
      error << "HIPFFT_INVALID_PLAN";
      break;
    case HIPFFT_ALLOC_FAILED:
      error << "HIPFFT_ALLOC_FAILED";
      break;
    case HIPFFT_INVALID_TYPE:
      error << "HIPFFT_INVALID_TYPE";
      break;
    case HIPFFT_INVALID_VALUE:
      error << "HIPFFT_INVALID_VALUE";
      break;
    case HIPFFT_INTERNAL_ERROR:
      error << "HIPFFT_INTERNAL_ERROR";
      break;
    case HIPFFT_EXEC_FAILED:
      error << "HIPFFT_EXEC_FAILED";
      break;
    case HIPFFT_SETUP_FAILED:
      error << "HIPFFT_SETUP_FAILED";
      break;
    case HIPFFT_INVALID_SIZE:
      error << "HIPFFT_INVALID_SIZE";
      break;
    case HIPFFT_UNALIGNED_DATA:
      error << "HIPFFT_UNALIGNED_DATA";
      break;
    case HIPFFT_INCOMPLETE_PARAMETER_LIST:
      error << "HIPFFT_INCOMPLETE_PARAMETER_LIST";
      break;
    case HIPFFT_INVALID_DEVICE:
      error << "HIPFFT_INVALID_DEVICE";
      break;
    case HIPFFT_PARSE_ERROR:
      error << "HIPFFT_PARSE_ERROR";
      break;
    case HIPFFT_NO_WORKSPACE:
      error << "HIPFFT_NO_WORKSPACE";
      break;
    case HIPFFT_NOT_IMPLEMENTED:
      error << "HIPFFT_NOT_IMPLEMENTED";
      break;
    case HIPFFT_NOT_SUPPORTED:
      error << "HIPFFT_NOT_SUPPORTED";
      break;
    default:
      error << "Unknown error code " << result;
      break;
    }
    throw RuntimeError(error.str());
  }
}

hipfftHandle hipFFTBackend::get_plan(TransformType type, Index_t n,
                                     Index_t batch, Index_t in_stride,
                                     Index_t in_dist, Index_t out_stride,
                                     Index_t out_dist) {
  PlanKey key{type, n, batch, in_stride, in_dist, out_stride, out_dist};

  auto it = this->plan_cache.find(key);
  if (it != this->plan_cache.end()) {
    return it->second;
  }

  // Create new plan
  hipfftHandle plan;
  int rank = 1;
  int n_arr[1] = {static_cast<int>(n)};

  // For r2c: input has n reals, output has n/2+1 complex
  // For c2r: input has n/2+1 complex, output has n reals
  // For c2c: input and output both have n complex

  // Determine the embedding sizes based on transform type
  // When using strides, we need to specify the embedding to tell hipFFT
  // the actual memory layout
  int inembed[1], onembed[1];
  hipfftType hipfft_type;

  switch (type) {
  case R2C:
    // Real input with n elements, complex output with n/2+1 elements
    // The embedding describes the storage size, not the transform size
    inembed[0] = static_cast<int>(in_dist);   // Storage per batch
    onembed[0] = static_cast<int>(out_dist);  // Storage per batch
    hipfft_type = HIPFFT_D2Z;
    break;
  case C2R:
    // Complex input with n/2+1 elements, real output with n elements
    inembed[0] = static_cast<int>(in_dist);
    onembed[0] = static_cast<int>(out_dist);
    hipfft_type = HIPFFT_Z2D;
    break;
  case C2C:
    // Complex input and output with n elements each
    inembed[0] = static_cast<int>(in_dist);
    onembed[0] = static_cast<int>(out_dist);
    hipfft_type = HIPFFT_Z2Z;
    break;
  default:
    throw RuntimeError("Unknown transform type");
  }

  hipfftResult result = hipfftPlanMany(
      &plan, rank, n_arr, inembed, static_cast<int>(in_stride),
      static_cast<int>(in_dist), onembed, static_cast<int>(out_stride),
      static_cast<int>(out_dist), hipfft_type, static_cast<int>(batch));

  check_hipfft_result(result, "plan creation");

  this->plan_cache[key] = plan;
  return plan;
}

void hipFFTBackend::r2c(Index_t n, Index_t batch, const Real * input,
                        Index_t in_stride, Index_t in_dist, Complex * output,
                        Index_t out_stride, Index_t out_dist) {
  hipfftHandle plan =
      get_plan(R2C, n, batch, in_stride, in_dist, out_stride, out_dist);

  // hipFFT's D2Z expects hipfftDoubleReal* and hipfftDoubleComplex*
  // These are compatible with double* and std::complex<double>*
  hipfftResult result = hipfftExecD2Z(
      plan, const_cast<hipfftDoubleReal *>(input),
      reinterpret_cast<hipfftDoubleComplex *>(output));

  check_hipfft_result(result, "D2Z execution");
}

void hipFFTBackend::c2r(Index_t n, Index_t batch, const Complex * input,
                        Index_t in_stride, Index_t in_dist, Real * output,
                        Index_t out_stride, Index_t out_dist) {
  hipfftHandle plan =
      get_plan(C2R, n, batch, in_stride, in_dist, out_stride, out_dist);

  // Note: hipFFT may modify the input during c2r transforms
  // The const_cast is necessary but the input data may be modified
  hipfftResult result = hipfftExecZ2D(
      plan,
      const_cast<hipfftDoubleComplex *>(
          reinterpret_cast<const hipfftDoubleComplex *>(input)),
      output);

  check_hipfft_result(result, "Z2D execution");
}

void hipFFTBackend::c2c_forward(Index_t n, Index_t batch, const Complex * input,
                                Index_t in_stride, Index_t in_dist,
                                Complex * output, Index_t out_stride,
                                Index_t out_dist) {
  hipfftHandle plan =
      get_plan(C2C, n, batch, in_stride, in_dist, out_stride, out_dist);

  hipfftResult result = hipfftExecZ2Z(
      plan,
      const_cast<hipfftDoubleComplex *>(
          reinterpret_cast<const hipfftDoubleComplex *>(input)),
      reinterpret_cast<hipfftDoubleComplex *>(output), HIPFFT_FORWARD);

  check_hipfft_result(result, "Z2Z forward execution");
}

void hipFFTBackend::c2c_backward(Index_t n, Index_t batch,
                                 const Complex * input, Index_t in_stride,
                                 Index_t in_dist, Complex * output,
                                 Index_t out_stride, Index_t out_dist) {
  hipfftHandle plan =
      get_plan(C2C, n, batch, in_stride, in_dist, out_stride, out_dist);

  hipfftResult result = hipfftExecZ2Z(
      plan,
      const_cast<hipfftDoubleComplex *>(
          reinterpret_cast<const hipfftDoubleComplex *>(input)),
      reinterpret_cast<hipfftDoubleComplex *>(output), HIPFFT_BACKWARD);

  check_hipfft_result(result, "Z2Z backward execution");
}

}  // namespace muGrid
