/**
 * @file   fft/cufft_backend.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   19 Dec 2025
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

#include "cufft_backend.hh"
#include "core/exception.hh"

#include <cuda_runtime.h>

#include <cstdint>
#include <sstream>

namespace muGrid {

namespace {
// cuFFT requires the real array of D2Z/Z2D transforms to be aligned like a
// complex array; check against the complex type's size.
bool is_complex_aligned(const void * ptr) {
  return reinterpret_cast<std::uintptr_t>(ptr) % sizeof(cufftDoubleComplex) ==
         0;
}

// muGrid guarantees this alignment by construction: the FFT engine's
// real-space collection is padded to an even x-storage width and even left
// ghost count (see the FFTEngineBase constructor), so every base pointer
// the FFT engine computes is an even number of doubles from the
// cudaMalloc'ed allocation. A violation here means that layout invariant
// was broken.
void check_complex_aligned(const void * ptr, const char * operation) {
  // The error branch is unreachable through the public API (the layout
  // invariant guarantees alignment), so it is excluded from coverage.
  // GCOVR_EXCL_START
  if (!is_complex_aligned(ptr)) {
    std::stringstream error;
    error << "The real array passed to the cuFFT " << operation
          << " transform is not aligned to the complex type (16 bytes) as "
             "required by cuFFT. muGrid pads the layout of FFT engine "
             "real-space fields so that this cannot happen; this error "
             "indicates a bug in muGrid's layout padding, or a field whose "
             "memory was not allocated through a muGrid FFT engine. "
             "External arrays must be copied into an engine field before "
             "transforming.";
    throw RuntimeError(error.str());
  }
  // GCOVR_EXCL_STOP
}
}  // namespace

cuFFTBackend::cuFFTBackend() : plan_cache{} {}

cuFFTBackend::~cuFFTBackend() {
  // Destroy all cached plans
  for (auto & entry : this->plan_cache) {
    cufftDestroy(entry.second);
  }
}

void cuFFTBackend::check_cufft_result(cufftResult result,
                                      const char * operation) {
  if (result != CUFFT_SUCCESS) {
    std::stringstream error;
    error << "cuFFT error during " << operation << ": ";
    switch (result) {
    case CUFFT_INVALID_PLAN:
      error << "CUFFT_INVALID_PLAN";
      break;
    case CUFFT_ALLOC_FAILED:
      error << "CUFFT_ALLOC_FAILED";
      break;
    case CUFFT_INVALID_TYPE:
      error << "CUFFT_INVALID_TYPE";
      break;
    case CUFFT_INVALID_VALUE:
      error << "CUFFT_INVALID_VALUE";
      break;
    case CUFFT_INTERNAL_ERROR:
      error << "CUFFT_INTERNAL_ERROR";
      break;
    case CUFFT_EXEC_FAILED:
      error << "CUFFT_EXEC_FAILED";
      break;
    case CUFFT_SETUP_FAILED:
      error << "CUFFT_SETUP_FAILED";
      break;
    case CUFFT_INVALID_SIZE:
      error << "CUFFT_INVALID_SIZE";
      break;
    case CUFFT_UNALIGNED_DATA:
      error << "CUFFT_UNALIGNED_DATA";
      break;
// CUFFT_INCOMPLETE_PARAMETER_LIST and CUFFT_PARSE_ERROR were removed from the
// cufftResult enum in cuFFT 12 (CUDA 13, CUFFT_VERSION 12300).
#if !defined(CUFFT_VERSION) || CUFFT_VERSION < 12300
    case CUFFT_INCOMPLETE_PARAMETER_LIST:
      error << "CUFFT_INCOMPLETE_PARAMETER_LIST";
      break;
#endif
    case CUFFT_INVALID_DEVICE:
      error << "CUFFT_INVALID_DEVICE";
      break;
#if !defined(CUFFT_VERSION) || CUFFT_VERSION < 12300
    case CUFFT_PARSE_ERROR:
      error << "CUFFT_PARSE_ERROR";
      break;
#endif
    case CUFFT_NO_WORKSPACE:
      error << "CUFFT_NO_WORKSPACE";
      break;
    case CUFFT_NOT_IMPLEMENTED:
      error << "CUFFT_NOT_IMPLEMENTED";
      break;
    case CUFFT_NOT_SUPPORTED:
      error << "CUFFT_NOT_SUPPORTED";
      break;
    default:
      error << "Unknown error code " << result;
      break;
    }
    throw RuntimeError(error.str());
  }
}

cufftHandle cuFFTBackend::get_plan(TransformType type, Index_t n, Index_t batch,
                                   Index_t in_stride, Index_t in_dist,
                                   Index_t out_stride, Index_t out_dist) {
  PlanKey key{type, n, batch, in_stride, in_dist, out_stride, out_dist};

  auto it = this->plan_cache.find(key);
  if (it != this->plan_cache.end()) {
    return it->second;
  }

  // Create new plan
  cufftHandle plan;
  int rank = 1;
  int n_arr[1] = {static_cast<int>(n)};

  // For r2c: input has n reals, output has n/2+1 complex
  // For c2r: input has n/2+1 complex, output has n reals
  // For c2c: input and output both have n complex

  cufftType cufft_type;
  switch (type) {
  case R2C:
    cufft_type = CUFFT_D2Z;
    break;
  case C2R:
    cufft_type = CUFFT_Z2D;
    break;
  case C2C:
    cufft_type = CUFFT_Z2Z;
    break;
  default:
    throw RuntimeError("Unknown transform type");
  }

  cufftResult result;

  // For batch=1 with non-unit stride but dist=0, use simplified plan
  // cuFFT requires valid embed values (>= n) when embed is not NULL
  if (batch == 1 && in_stride == 1 && out_stride == 1) {
    // Simple contiguous transform - use basic plan
    result = cufftPlan1d(&plan, static_cast<int>(n), cufft_type, 1);
  } else if (batch == 1) {
    // Single transform with stride - use NULL embed (auto-calculated)
    // Note: This path should now rarely be used since we batch c2c transforms
    result = cufftPlanMany(
        &plan, rank, n_arr,
        nullptr, static_cast<int>(in_stride), 0,
        nullptr, static_cast<int>(out_stride), 0,
        cufft_type, 1);
  } else {
    // Batched transform - need embed values
    // embed must be >= n for the transform to be valid
    // Use dist as the embed value (storage size per batch)
    int inembed[1] = {static_cast<int>(in_dist)};
    int onembed[1] = {static_cast<int>(out_dist)};
    result = cufftPlanMany(
        &plan, rank, n_arr,
        inembed, static_cast<int>(in_stride), static_cast<int>(in_dist),
        onembed, static_cast<int>(out_stride), static_cast<int>(out_dist),
        cufft_type, static_cast<int>(batch));
  }

  check_cufft_result(result, "plan creation");

  this->plan_cache[key] = plan;
  return plan;
}

void cuFFTBackend::r2c(Index_t n, Index_t batch, const Real * input,
                       Index_t in_stride, Index_t in_dist, Complex * output,
                       Index_t out_stride, Index_t out_dist) {
  // Ranks with an empty subdomain (possible when a grid dimension is
  // smaller than the process grid) have nothing to transform; cuFFT
  // rejects batch == 0 at plan creation.
  if (batch == 0) {
    return;
  }

  // cuFFT does not support strided real-to-complex transforms.
  // This is a documented limitation: "Strides on the real part of
  // real-to-complex and complex-to-real transforms are not supported."
  // This typically occurs in 3D MPI-parallel FFTs where the data layout
  // after transpose operations results in non-unit strides.
  if (in_stride != 1) {
    throw RuntimeError(
        "cuFFT does not support strided real-to-complex (R2C) transforms. "
        "This limitation affects 3D MPI-parallel FFTs on NVIDIA GPUs. "
        "The input stride is " +
        std::to_string(in_stride) +
        " but must be 1. "
        "Workaround: Use CPU FFT backend for 3D MPI-parallel transforms, "
        "or use 2D grids which support batched transforms with unit stride.");
  }

  check_complex_aligned(input, "D2Z");

  cufftHandle plan =
      get_plan(R2C, n, batch, in_stride, in_dist, out_stride, out_dist);

  // cuFFT's D2Z expects cufftDoubleReal* and cufftDoubleComplex*
  // These are compatible with double* and std::complex<double>*
  cufftResult result = cufftExecD2Z(
      plan, const_cast<cufftDoubleReal *>(input),
      reinterpret_cast<cufftDoubleComplex *>(output));

  check_cufft_result(result, "D2Z execution");
  cudaDeviceSynchronize();
}

void cuFFTBackend::c2r(Index_t n, Index_t batch, const Complex * input,
                       Index_t in_stride, Index_t in_dist, Real * output,
                       Index_t out_stride, Index_t out_dist) {
  // Empty subdomain: nothing to transform (see r2c)
  if (batch == 0) {
    return;
  }

  // cuFFT does not support strided complex-to-real transforms.
  // This is a documented limitation: "Strides on the real part of
  // real-to-complex and complex-to-real transforms are not supported."
  // This typically occurs in 3D MPI-parallel FFTs where the data layout
  // after transpose operations results in non-unit strides.
  if (out_stride != 1) {
    throw RuntimeError(
        "cuFFT does not support strided complex-to-real (C2R) transforms. "
        "This limitation affects 3D MPI-parallel FFTs on NVIDIA GPUs. "
        "The output stride is " +
        std::to_string(out_stride) +
        " but must be 1. "
        "Workaround: Use CPU FFT backend for 3D MPI-parallel transforms, "
        "or use 2D grids which support batched transforms with unit stride.");
  }

  check_complex_aligned(output, "Z2D");

  cufftHandle plan =
      get_plan(C2R, n, batch, in_stride, in_dist, out_stride, out_dist);

  // Note: cuFFT may modify the input during c2r transforms
  // The const_cast is necessary but the input data may be modified
  cufftResult result = cufftExecZ2D(
      plan,
      const_cast<cufftDoubleComplex *>(
          reinterpret_cast<const cufftDoubleComplex *>(input)),
      output);

  check_cufft_result(result, "Z2D execution");
  // Synchronize so the result is complete before the caller hands the buffer
  // to GPU-aware MPI (which is not ordered against the cuFFT stream).
  cudaDeviceSynchronize();
}

void cuFFTBackend::c2c_forward(Index_t n, Index_t batch, const Complex * input,
                               Index_t in_stride, Index_t in_dist,
                               Complex * output, Index_t out_stride,
                               Index_t out_dist) {
  // Empty subdomain: nothing to transform (see r2c)
  if (batch == 0) {
    return;
  }

  cufftHandle plan =
      get_plan(C2C, n, batch, in_stride, in_dist, out_stride, out_dist);

  cufftResult result = cufftExecZ2Z(
      plan,
      const_cast<cufftDoubleComplex *>(
          reinterpret_cast<const cufftDoubleComplex *>(input)),
      reinterpret_cast<cufftDoubleComplex *>(output), CUFFT_FORWARD);

  check_cufft_result(result, "Z2Z forward execution");
  cudaDeviceSynchronize();
}

void cuFFTBackend::c2c_backward(Index_t n, Index_t batch, const Complex * input,
                                Index_t in_stride, Index_t in_dist,
                                Complex * output, Index_t out_stride,
                                Index_t out_dist) {
  // Empty subdomain: nothing to transform (see r2c)
  if (batch == 0) {
    return;
  }

  cufftHandle plan =
      get_plan(C2C, n, batch, in_stride, in_dist, out_stride, out_dist);

  cufftResult result = cufftExecZ2Z(
      plan,
      const_cast<cufftDoubleComplex *>(
          reinterpret_cast<const cufftDoubleComplex *>(input)),
      reinterpret_cast<cufftDoubleComplex *>(output), CUFFT_INVERSE);

  check_cufft_result(result, "Z2Z backward execution");
  // Synchronize so the result is complete before the caller hands the buffer
  // to GPU-aware MPI (which is not ordered against the cuFFT stream).
  cudaDeviceSynchronize();
}

}  // namespace muGrid
