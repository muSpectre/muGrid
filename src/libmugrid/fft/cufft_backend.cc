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

// cuFFT advanced-data-layout parameters for one side (input or output) of an
// N-D transform, derived from the per-axis element strides the engine passes.
struct NdLayout {
  std::vector<int> n;       // logical transform sizes, slowest-varying first
  std::vector<int> embed;   // physical padded extents (handles ghost padding)
  int stride;               // innermost-element stride
  int dist;                 // batch (component) stride
};

// Build the layout from `shape` (full extent incl. the component/batch axis),
// `axes` (transformed axes, half-complex axis last) and `strides` (element
// stride of every axis). cuFFT addresses element (i_0,…,i_{r-1}) within a batch
// at offset (((i_0·embed[1]+i_1)·embed[2]+…)+i_{r-1})·stride; matching that to
// the requested per-axis strides s[k] gives embed[j]=s[j-1]/s[j] and
// stride=s[r-1]. The non-transformed axis 0 is the batch (dist = strides[0]).
NdLayout derive_nd_layout(const std::vector<Index_t> & shape,
                          const std::vector<Index_t> & axes,
                          const std::vector<Index_t> & strides) {
  const std::size_t rank{axes.size()};
  NdLayout L;
  L.n.resize(rank);
  L.embed.resize(rank);
  std::vector<Index_t> s(rank);
  for (std::size_t k{0}; k < rank; ++k) {
    L.n[k] = static_cast<int>(shape[axes[k]]);
    s[k] = strides[axes[k]];
  }
  L.stride = static_cast<int>(s[rank - 1]);
  L.embed[0] = L.n[0];
  for (std::size_t j{1}; j < rank; ++j) {
    if (s[j] == 0 || s[j - 1] % s[j] != 0) {
      throw RuntimeError(
          "cuFFT N-D transform requires a nested (row-major) field layout; "
          "the per-axis strides are not multiples of the inner stride. This "
          "should not happen for muGrid FFT-engine fields.");
    }
    L.embed[j] = static_cast<int>(s[j - 1] / s[j]);
  }
  L.dist = static_cast<int>(strides[0]);
  return L;
}
}  // namespace

cuFFTBackend::cuFFTBackend() : plan_cache{} {}

cuFFTBackend::~cuFFTBackend() {
  // Destroy all cached plans
  for (auto & entry : this->plan_cache) {
    cufftDestroy(entry.second);
  }
  for (auto & entry : this->nd_plan_cache) {
    cufftDestroy(entry.second);
  }
  if (this->nd_scratch != nullptr) {
    cudaFree(this->nd_scratch);
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

cufftHandle cuFFTBackend::get_nd_plan(cufftType type, const std::vector<int> & n,
                                      const std::vector<int> & inembed,
                                      int istride, int idist,
                                      const std::vector<int> & onembed,
                                      int ostride, int odist, int batch) {
  // String signature over every parameter that defines the plan.
  std::stringstream ss;
  ss << type << '|' << batch << '|' << istride << ',' << idist << '|' << ostride
     << ',' << odist << "|n:";
  for (int v : n) ss << v << ',';
  ss << "|in:";
  for (int v : inembed) ss << v << ',';
  ss << "|on:";
  for (int v : onembed) ss << v << ',';
  const std::string key{ss.str()};

  auto it = this->nd_plan_cache.find(key);
  if (it != this->nd_plan_cache.end()) {
    return it->second;
  }

  cufftHandle plan;
  cufftResult result = cufftPlanMany(
      &plan, static_cast<int>(n.size()), const_cast<int *>(n.data()),
      const_cast<int *>(inembed.data()), istride, idist,
      const_cast<int *>(onembed.data()), ostride, odist, type, batch);
  check_cufft_result(result, "N-D plan creation");

  this->nd_plan_cache[key] = plan;
  return plan;
}

cufftDoubleComplex * cuFFTBackend::ensure_nd_scratch(std::size_t count) {
  if (count > this->nd_scratch_count) {
    if (this->nd_scratch != nullptr) {
      cudaFree(this->nd_scratch);
    }
    cudaError_t err =
        cudaMalloc(&this->nd_scratch, count * sizeof(cufftDoubleComplex));
    if (err != cudaSuccess) {
      this->nd_scratch = nullptr;
      this->nd_scratch_count = 0;
      throw RuntimeError("cudaMalloc failed for cuFFT N-D scratch buffer");
    }
    this->nd_scratch_count = count;
  }
  return this->nd_scratch;
}

void cuFFTBackend::r2c_nd(const std::vector<Index_t> & shape,
                          const std::vector<Index_t> & axes, const Real * input,
                          const std::vector<Index_t> & in_strides,
                          Complex * output,
                          const std::vector<Index_t> & out_strides) {
  if (shape[0] == 0) {
    return;  // empty component batch: nothing to do
  }
  NdLayout in{derive_nd_layout(shape, axes, in_strides)};
  NdLayout out{derive_nd_layout(shape, axes, out_strides)};

  // cuFFT requires unit innermost stride on the real side of r2c/c2r.
  if (in.stride != 1) {
    throw RuntimeError(
        "cuFFT N-D r2c requires unit innermost stride on the real input "
        "(got " +
        std::to_string(in.stride) + "); the GPU field layout must be SoA.");
  }
  check_complex_aligned(input, "D2Z (N-D)");

  cufftHandle plan =
      get_nd_plan(CUFFT_D2Z, in.n, in.embed, in.stride, in.dist, out.embed,
                  out.stride, out.dist, static_cast<int>(shape[0]));
  cufftResult result = cufftExecD2Z(
      plan, const_cast<cufftDoubleReal *>(input),
      reinterpret_cast<cufftDoubleComplex *>(output));
  check_cufft_result(result, "D2Z (N-D) execution");
  cudaDeviceSynchronize();
}

void cuFFTBackend::c2r_nd(const std::vector<Index_t> & shape,
                          const std::vector<Index_t> & axes,
                          const Complex * input,
                          const std::vector<Index_t> & in_strides, Real * output,
                          const std::vector<Index_t> & out_strides) {
  if (shape[0] == 0) {
    return;
  }
  NdLayout in{derive_nd_layout(shape, axes, in_strides)};
  NdLayout out{derive_nd_layout(shape, axes, out_strides)};

  if (out.stride != 1) {
    throw RuntimeError(
        "cuFFT N-D c2r requires unit innermost stride on the real output "
        "(got " +
        std::to_string(out.stride) + "); the GPU field layout must be SoA.");
  }
  check_complex_aligned(output, "Z2D (N-D)");

  // cuFFT's multidimensional Z2D overwrites its input, but the engine requires
  // the const Fourier input preserved (cf. the per-axis path, which copies it
  // into a work buffer). Stage through a scratch copy. The input spans
  // batch * idist complex elements (the per-component Fourier buffer is
  // contiguous, so idist is the full component stride).
  const std::size_t span{static_cast<std::size_t>(shape[0]) *
                         static_cast<std::size_t>(in.dist)};
  cufftDoubleComplex * scratch = ensure_nd_scratch(span);
  cudaMemcpy(scratch, reinterpret_cast<const cufftDoubleComplex *>(input),
             span * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice);

  cufftHandle plan =
      get_nd_plan(CUFFT_Z2D, in.n, in.embed, in.stride, in.dist, out.embed,
                  out.stride, out.dist, static_cast<int>(shape[0]));
  cufftResult result = cufftExecZ2D(
      plan, scratch, reinterpret_cast<cufftDoubleReal *>(output));
  check_cufft_result(result, "Z2D (N-D) execution");
  cudaDeviceSynchronize();
}

}  // namespace muGrid
