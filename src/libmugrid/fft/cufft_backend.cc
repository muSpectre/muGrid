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
#include "memory/gpu_runtime.hh"

#include <sstream>

namespace muGrid {

namespace {
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

cufftHandle cuFFTBackend::make_plan(const PlanKey & key) {
  cufftType cufft_type;
  switch (key.kind) {
  case Base::Kind::R2C:
    cufft_type = CUFFT_D2Z;
    break;
  case Base::Kind::C2R:
    cufft_type = CUFFT_Z2D;
    break;
  case Base::Kind::C2C:
    // Direction-agnostic; CUFFT_FORWARD/INVERSE is selected at execution.
    cufft_type = CUFFT_Z2Z;
    break;
  default:
    throw RuntimeError("Unknown transform type");
  }

  cufftHandle plan;
  int rank = 1;
  int n_arr[1] = {static_cast<int>(key.n)};
  cufftResult result;

  // For batch=1 with non-unit stride but dist=0, use simplified plan.
  // cuFFT requires valid embed values (>= n) when embed is not NULL.
  if (key.batch == 1 && key.in_stride == 1 && key.out_stride == 1) {
    // Simple contiguous transform - use basic plan
    result = cufftPlan1d(&plan, static_cast<int>(key.n), cufft_type, 1);
  } else if (key.batch == 1) {
    // Single transform with stride - use NULL embed (auto-calculated)
    result = cufftPlanMany(&plan, rank, n_arr, nullptr,
                           static_cast<int>(key.in_stride), 0, nullptr,
                           static_cast<int>(key.out_stride), 0, cufft_type, 1);
  } else {
    // Batched transform - embed must be >= n; use dist (storage size per batch)
    int inembed[1] = {static_cast<int>(key.in_dist)};
    int onembed[1] = {static_cast<int>(key.out_dist)};
    result = cufftPlanMany(
        &plan, rank, n_arr, inembed, static_cast<int>(key.in_stride),
        static_cast<int>(key.in_dist), onembed,
        static_cast<int>(key.out_stride), static_cast<int>(key.out_dist),
        cufft_type, static_cast<int>(key.batch));
  }

  check_cufft_result(result, "plan creation");
  return plan;
}

void cuFFTBackend::destroy_plan(cufftHandle & plan) { cufftDestroy(plan); }

void cuFFTBackend::exec_r2c(cufftHandle & plan, const Real * input,
                            Complex * output) {
  // cuFFT's D2Z expects cufftDoubleReal* and cufftDoubleComplex*, which are
  // compatible with double* and std::complex<double>*.
  cufftResult result =
      cufftExecD2Z(plan, const_cast<cufftDoubleReal *>(input),
                   reinterpret_cast<cufftDoubleComplex *>(output));
  check_cufft_result(result, "D2Z execution");
}

void cuFFTBackend::exec_c2r(cufftHandle & plan, const Complex * input,
                            Real * output) {
  // Note: cuFFT may modify the input during c2r transforms; the engine stages
  // a copy where it needs the input preserved.
  cufftResult result = cufftExecZ2D(
      plan,
      const_cast<cufftDoubleComplex *>(
          reinterpret_cast<const cufftDoubleComplex *>(input)),
      output);
  check_cufft_result(result, "Z2D execution");
}

void cuFFTBackend::exec_c2c_forward(cufftHandle & plan, const Complex * input,
                                    Complex * output) {
  cufftResult result = cufftExecZ2Z(
      plan,
      const_cast<cufftDoubleComplex *>(
          reinterpret_cast<const cufftDoubleComplex *>(input)),
      reinterpret_cast<cufftDoubleComplex *>(output), CUFFT_FORWARD);
  check_cufft_result(result, "Z2Z forward execution");
}

void cuFFTBackend::exec_c2c_backward(cufftHandle & plan, const Complex * input,
                                     Complex * output) {
  cufftResult result = cufftExecZ2Z(
      plan,
      const_cast<cufftDoubleComplex *>(
          reinterpret_cast<const cufftDoubleComplex *>(input)),
      reinterpret_cast<cufftDoubleComplex *>(output), CUFFT_INVERSE);
  check_cufft_result(result, "Z2Z backward execution");
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

  return this->nd_plan_for(ss.str(), [&]() -> cufftHandle {
    cufftHandle plan;
    cufftResult result = cufftPlanMany(
        &plan, static_cast<int>(n.size()), const_cast<int *>(n.data()),
        const_cast<int *>(inembed.data()), istride, idist,
        const_cast<int *>(onembed.data()), ostride, odist, type, batch);
    check_cufft_result(result, "N-D plan creation");
    return plan;
  });
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
  this->check_complex_aligned(input, "r2c (N-D)");

  cufftHandle plan =
      get_nd_plan(CUFFT_D2Z, in.n, in.embed, in.stride, in.dist, out.embed,
                  out.stride, out.dist, static_cast<int>(shape[0]));
  cufftResult result = cufftExecD2Z(
      plan, const_cast<cufftDoubleReal *>(input),
      reinterpret_cast<cufftDoubleComplex *>(output));
  check_cufft_result(result, "D2Z (N-D) execution");
  GPU_DEVICE_SYNCHRONIZE();
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
  this->check_complex_aligned(output, "c2r (N-D)");

  // cuFFT's multidimensional Z2D overwrites its input, but the engine requires
  // the const Fourier input preserved (cf. the per-axis path, which copies it
  // into a work buffer). Stage through a scratch copy. The input spans
  // batch * idist complex elements (the per-component Fourier buffer is
  // contiguous, so idist is the full component stride).
  const std::size_t span{static_cast<std::size_t>(shape[0]) *
                         static_cast<std::size_t>(in.dist)};
  auto * scratch = static_cast<cufftDoubleComplex *>(
      this->ensure_scratch(span * sizeof(cufftDoubleComplex)));
  GPU_MEMCPY_D2D(scratch, reinterpret_cast<const cufftDoubleComplex *>(input),
                 span * sizeof(cufftDoubleComplex));

  cufftHandle plan =
      get_nd_plan(CUFFT_Z2D, in.n, in.embed, in.stride, in.dist, out.embed,
                  out.stride, out.dist, static_cast<int>(shape[0]));
  cufftResult result = cufftExecZ2D(
      plan, scratch, reinterpret_cast<cufftDoubleReal *>(output));
  check_cufft_result(result, "Z2D (N-D) execution");
  GPU_DEVICE_SYNCHRONIZE();
}

}  // namespace muGrid
