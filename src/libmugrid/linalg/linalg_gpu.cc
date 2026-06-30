/**
 * @file   linalg_gpu.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   29 Dec 2025
 *
 * @brief  Unified CUDA/HIP implementations of linear algebra operations
 *
 * This file provides GPU implementations that work with both CUDA and HIP
 * backends. The kernel code is identical; only the launch mechanism and
 * runtime API differ, which are abstracted via macros.
 *
 * The Real and Complex paths share a single source. Each kernel is written
 * once as a template over a device scalar type — `Real` for real fields and
 * the lightweight `DeviceComplex` below for complex fields — and the
 * sesquilinear-vs-bilinear difference is isolated in the `conj_product` trait
 * (mirroring internal::conj_product in linalg_host.cc). The public
 * <Real, DeviceSpace> / <Complex, DeviceSpace> entry points are thin
 * delegators to one generic body per operation, so there is no per-type kernel
 * copy to keep in sync. `DeviceComplex` is a tiny `__host__ __device__`
 * aggregate (no thrust, no cuComplex) and is bit-compatible with
 * std::complex<Real>, so complex field buffers are reinterpret_cast to it.
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

#include "linalg/linalg.hh"
#include "collection/field_collection.hh"
#include "collection/field_collection_global.hh"
#include "core/exception.hh"

#include "memory/gpu_runtime.hh"
#include "memory/device_alloc.hh"

#include <type_traits>

#if defined(MUGRID_ENABLE_CUDA)
    using DeviceSpace = muGrid::CUDASpace;
#elif defined(MUGRID_ENABLE_HIP)
    using DeviceSpace = muGrid::ROCmSpace;
#endif

namespace muGrid {
namespace linalg {

namespace gpu_kernels {

// Block size for 1D kernels
constexpr int BLOCK_SIZE = 256;
constexpr int REDUCE_BLOCK_SIZE = 256;

/* ---------------------------------------------------------------------- */
/* Device scalar abstraction                                              */
/*                                                                        */
/* Device code has no std::complex (its operators are not __device__) and */
/* we deliberately avoid thrust/cuComplex so the same source compiles     */
/* under both nvcc and hipcc. DeviceComplex is a minimal aggregate, layout-*/
/* compatible with std::complex<Real> (two contiguous Reals), so complex   */
/* field data is reinterpret_cast to it. conj_product carries the          */
/* sesquilinear (complex) vs bilinear (real) choice — the single point of  */
/* difference between the two scalar types — and sq_norm returns the real  */
/* squared magnitude used by the fused norm reductions.                    */
/* ---------------------------------------------------------------------- */

// Templated on the underlying real type RT so it serves both double (Complex)
// and single (Complex32) precision; the field buffer is reinterpret_cast to it.
template <typename RT>
struct DeviceComplexT {
    RT re, im;
    // Defining member operators keeps DeviceComplexT an aggregate in C++17
    // (no user-provided constructors), so `DeviceComplexT{}` zero-initialises
    // and `DeviceComplexT{re, im}` brace-initialises as usual.
    __host__ __device__ DeviceComplexT& operator+=(DeviceComplexT o) {
        re += o.re;
        im += o.im;
        return *this;
    }
};
using DeviceComplex = DeviceComplexT<Real>;
using DeviceComplex32 = DeviceComplexT<float>;

template <typename RT>
__host__ __device__ inline DeviceComplexT<RT> operator+(DeviceComplexT<RT> a,
                                                        DeviceComplexT<RT> b) {
    return {a.re + b.re, a.im + b.im};
}
template <typename RT>
__host__ __device__ inline DeviceComplexT<RT> operator-(DeviceComplexT<RT> a,
                                                        DeviceComplexT<RT> b) {
    return {a.re - b.re, a.im - b.im};
}
template <typename RT>
__host__ __device__ inline DeviceComplexT<RT> operator*(DeviceComplexT<RT> a,
                                                        DeviceComplexT<RT> b) {
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

// Sesquilinear product: conj(a)*b for complex, a*b for real. Mirrors
// internal::conj_product in linalg_host.cc.
template <typename T>
__host__ __device__ inline T conj_product(T a, T b) {
    return a * b;  // real: bilinear
}
template <typename RT>
__host__ __device__ inline DeviceComplexT<RT>
conj_product(DeviceComplexT<RT> a, DeviceComplexT<RT> b) {
    // complex: sesquilinear conj(a)*b (Array API vecdot convention)
    return {a.re * b.re + a.im * b.im, a.re * b.im - a.im * b.re};
}

// Real squared magnitude |x|^2.
__host__ __device__ inline Real sq_norm(Real x) { return x * x; }
__host__ __device__ inline float sq_norm(float x) { return x * x; }
template <typename RT>
__host__ __device__ inline RT sq_norm(DeviceComplexT<RT> x) {
    return x.re * x.re + x.im * x.im;
}

/* ---------------------------------------------------------------------- */
/* Element-wise kernels (full buffer operations)                          */
/*                                                                         */
/* One template per operation serves both Real and DeviceComplex.          */
/* ---------------------------------------------------------------------- */

// AXPY kernel: y = alpha * x + y
template <typename T>
__global__ void axpy_kernel(T alpha, const T* x, T* y, Index_t n) {
    Index_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] += alpha * x[idx];
    }
}

// Scale kernel: x = alpha * x
template <typename T>
__global__ void scal_kernel(T alpha, T* x, Index_t n) {
    Index_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = alpha * x[idx];
    }
}

// Copy kernel: dst = src
template <typename T>
__global__ void copy_kernel(const T* src, T* dst, Index_t n) {
    Index_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// AXPBY kernel: y = alpha * x + beta * y
template <typename T>
__global__ void axpby_kernel(T alpha, const T* x, T beta, T* y, Index_t n) {
    Index_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = alpha * x[idx] + beta * y[idx];
    }
}

/**
 * Fused AXPY + norm_sq kernel: y = alpha * x + y, returns partial ||y||²
 * Each block computes AXPY for its elements AND accumulates partial norm.
 * The squared norm is real-valued for both scalar types (sq_norm), so the
 * partial sums are Real regardless of T.
 */
template <typename T>
__global__ void axpy_norm_sq_kernel(T alpha, const T* x, T* y,
                                    Real* partial_sums, Index_t n) {
    __shared__ Real shared_data[REDUCE_BLOCK_SIZE];

    Index_t tid = threadIdx.x;
    Index_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Fused: update y AND accumulate squared norm
    Real sum = 0.0;
    while (idx < n) {
        T new_y = y[idx] + alpha * x[idx];
        y[idx] = new_y;
        sum += sq_norm(new_y);
        idx += blockDim.x * gridDim.x;
    }
    shared_data[tid] = sum;
    __syncthreads();

    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_data[0];
    }
}

/* ---------------------------------------------------------------------- */
/* Reduction kernels                                                       */
/* ---------------------------------------------------------------------- */

/**
 * Dot product reduction kernel (first pass) over the full buffer.
 * Computes per-block partial sums of conj_product(a, b). For real T this is
 * the bilinear a*b; for DeviceComplex it is the sesquilinear conj(a)*b, so
 * the partial sums (and the final result) are of device-scalar type T.
 */
template <typename T>
__global__ void dot_reduce_kernel(const T* a, const T* b, T* partial_sums,
                                  Index_t n) {
    __shared__ T shared_data[REDUCE_BLOCK_SIZE];

    Index_t tid = threadIdx.x;
    Index_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and multiply
    T sum{};
    while (idx < n) {
        sum += conj_product(a[idx], b[idx]);
        idx += blockDim.x * gridDim.x;
    }
    shared_data[tid] = sum;
    __syncthreads();

    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_data[0];
    }
}

/**
 * Dot product over the interior (non-ghost) region only (1D/2D/3D).
 *
 * Sums the interior directly. Do NOT compute interior reductions as
 * full-buffer result minus ghost contribution: the ghost buffers hold large
 * stale data (stencil operators write results into ghost pixels), so the
 * subtraction cancels catastrophically once the interior values are small.
 * This destroyed converged CG residuals — the reported squared norm carried
 * an absolute error of order eps * ||ghosts||^2 and could even go negative.
 * Mirrors internal::interior_vecdot in linalg_host.cc.
 *
 * The interior box has extents (nx, ny, nz) and starts at (x0, y0, z0);
 * unused trailing dimensions degenerate to extent 1, start 0, stride 0.
 * Uses field data strides directly for correct SoA layout handling:
 * field[c, x, y, z] = base + c * stride_c + x * stride_x + y * stride_y
 *                     + z * stride_z
 *
 * As with dot_reduce_kernel, the contraction uses conj_product so the single
 * template serves both real (bilinear) and complex (sesquilinear) fields.
 */
template <typename T>
__global__ void interior_dot_kernel(
    const T* a, const T* b, T* partial_sums,
    Index_t nx, Index_t ny, Index_t nz,
    Index_t x0, Index_t y0, Index_t z0,
    Index_t stride_c, Index_t stride_x, Index_t stride_y, Index_t stride_z,
    Index_t nb_components) {

    __shared__ T shared_data[REDUCE_BLOCK_SIZE];

    Index_t tid = threadIdx.x;
    Index_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    Index_t nb_pixels = nx * ny * nz;

    T sum{};
    for (Index_t pixel_idx = global_tid; pixel_idx < nb_pixels;
         pixel_idx += blockDim.x * gridDim.x) {
        // x runs fastest (smallest stride) so consecutive threads make
        // coalesced accesses
        Index_t ix = x0 + pixel_idx % nx;
        Index_t rem = pixel_idx / nx;
        Index_t iy = y0 + rem % ny;
        Index_t iz = z0 + rem / ny;

        Index_t offset = ix * stride_x + iy * stride_y + iz * stride_z;
        for (Index_t c = 0; c < nb_components; ++c) {
            const Index_t e = offset + c * stride_c;
            sum += conj_product(a[e], b[e]);
        }
    }

    shared_data[tid] = sum;
    __syncthreads();

    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_data[0];
    }
}

/**
 * Final reduction kernel - sums partial results into data[0].
 */
template <typename T>
__global__ void final_reduce_kernel(T* data, Index_t n) {
    __shared__ T shared_data[REDUCE_BLOCK_SIZE];

    Index_t tid = threadIdx.x;

    // Load partial sums
    T sum{};
    for (Index_t i = tid; i < n; i += blockDim.x) {
        sum += data[i];
    }
    shared_data[tid] = sum;
    __syncthreads();

    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    // Write final result
    if (tid == 0) {
        data[0] = shared_data[0];
    }
}

/**
 * Fused interior reduction for pipelined CG (first pass). Reads r, u, w once
 * over the interior and writes three per-block partial sums into a single
 * buffer laid out as [ru(0..nb) | wu(nb..2nb) | rr(2nb..3nb)].
 *
 * Real-only (the pipelined CG fields are real); not templated over the scalar
 * type because there is no complex pipelined-CG instantiation.
 */
__global__ void interior_three_dots_kernel(
    const Real* r, const Real* u, const Real* w, Real* partial,
    Index_t nx, Index_t ny, Index_t nz,
    Index_t x0, Index_t y0, Index_t z0,
    Index_t stride_c, Index_t stride_x, Index_t stride_y, Index_t stride_z,
    Index_t nb_components, Index_t num_blocks) {
    __shared__ Real s_ru[REDUCE_BLOCK_SIZE];
    __shared__ Real s_wu[REDUCE_BLOCK_SIZE];
    __shared__ Real s_rr[REDUCE_BLOCK_SIZE];

    Index_t tid = threadIdx.x;
    Index_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    Index_t nb_pixels = nx * ny * nz;

    Real ru = 0.0, wu = 0.0, rr = 0.0;
    for (Index_t pixel_idx = global_tid; pixel_idx < nb_pixels;
         pixel_idx += blockDim.x * gridDim.x) {
        Index_t ix = x0 + pixel_idx % nx;
        Index_t rem = pixel_idx / nx;
        Index_t iy = y0 + rem % ny;
        Index_t iz = z0 + rem / ny;
        Index_t offset = ix * stride_x + iy * stride_y + iz * stride_z;
        for (Index_t c = 0; c < nb_components; ++c) {
            const Index_t e = offset + c * stride_c;
            const Real rv = r[e], uv = u[e], wv = w[e];
            ru += rv * uv;
            wu += wv * uv;
            rr += rv * rv;
        }
    }
    s_ru[tid] = ru;
    s_wu[tid] = wu;
    s_rr[tid] = rr;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_ru[tid] += s_ru[tid + stride];
            s_wu[tid] += s_wu[tid + stride];
            s_rr[tid] += s_rr[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = s_ru[0];
        partial[num_blocks + blockIdx.x] = s_wu[0];
        partial[2 * num_blocks + blockIdx.x] = s_rr[0];
    }
}

/**
 * Final pass for the fused three-dot reduction: reduces the three length-`nb`
 * segments of `partial` into out[0..2], so the result is one contiguous
 * three-element copy back to the host.
 */
__global__ void final_reduce3_kernel(const Real* partial, Real* out,
                                     Index_t nb) {
    __shared__ Real s[3][REDUCE_BLOCK_SIZE];
    Index_t tid = threadIdx.x;
    for (int seg = 0; seg < 3; ++seg) {
        Real sum = 0.0;
        for (Index_t i = tid; i < nb; i += blockDim.x) {
            sum += partial[seg * nb + i];
        }
        s[seg][tid] = sum;
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            for (int seg = 0; seg < 3; ++seg) {
                s[seg][tid] += s[seg][tid + stride];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[0] = s[0][0];
        out[1] = s[1][0];
        out[2] = s[2][0];
    }
}

// Scale complex x by a real per-pixel field alpha; alpha is either
// broadcast over the components of x (alpha_per_comp == false) or
// applied elementwise (alpha has x's components). Operates on the
// doubles of the complex buffer to avoid complex arithmetic in device
// code.
__global__ void field_scal_complex_kernel(Real* x2, const Real* a,
                                          Index_t npix, Index_t ncomp,
                                          bool soa, bool alpha_per_comp,
                                          Index_t n2) {
    Index_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n2) {
        Index_t elem = j >> 1;  // complex element index
        Index_t i = alpha_per_comp
                        ? elem
                        : (soa ? (elem % npix) : (elem / ncomp));
        x2[j] *= a[i];
    }
}

// Real variant of the kernel above
__global__ void field_scal_real_kernel(Real* x, const Real* a,
                                       Index_t npix, Index_t ncomp,
                                       bool soa, bool alpha_per_comp,
                                       Index_t n) {
    Index_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        Index_t i =
            alpha_per_comp ? j : (soa ? (j % npix) : (j / ncomp));
        x[j] *= a[i];
    }
}

// Per-pixel three-vector cross product out = a x b. One thread per pixel;
// `soa` selects the component stride (npix for SoA, 1 with a pixel-stride of 3
// for AoS). A single template serves both real and complex fields: the complex
// multiplies are handled by DeviceComplex's operator*/operator-, so the body
// is identical to the real case (which previously needed its own hand-unrolled
// re/im kernel).
template <typename T>
__global__ void cross_kernel(const T* a, const T* b, T* out, Index_t npix,
                             bool soa) {
    Index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < npix) {
        const Index_t cs = soa ? npix : 1;        // stride between components
        const Index_t base = soa ? i : 3 * i;     // first component of pixel i
        const T a0 = a[base], a1 = a[base + cs], a2 = a[base + 2 * cs];
        const T b0 = b[base], b1 = b[base + cs], b2 = b[base + 2 * cs];
        out[base]          = a1 * b2 - a2 * b1;
        out[base + cs]     = a2 * b0 - a0 * b2;
        out[base + 2 * cs] = a0 * b1 - a1 * b0;
    }
}

// Fused Leray projection out[c] -= k[c] * sum_d(invk[d] * N[d]) on the complex
// fields N/out (stored as interleaved re/im doubles) with real coefficient
// fields k/invk. Because the coefficients are real they scale the real and
// imaginary parts identically, so the kernel does no complex arithmetic.
__global__ void leray_kernel(const Real* k, const Real* invk, const Real* N2,
                             Real* out2, Index_t npix, bool soa) {
    Index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < npix) {
        const Index_t cs = soa ? npix : 1;     // real component stride
        const Index_t base = soa ? i : 3 * i;  // first real component
        Real s_re = 0.0, s_im = 0.0;
        for (Index_t c = 0; c < 3; ++c) {
            const Index_t r = base + c * cs;     // real coefficient index
            s_re += invk[r] * N2[2 * r];
            s_im += invk[r] * N2[2 * r + 1];
        }
        for (Index_t c = 0; c < 3; ++c) {
            const Index_t r = base + c * cs;
            out2[2 * r]     -= k[r] * s_re;
            out2[2 * r + 1] -= k[r] * s_im;
        }
    }
}

}  // namespace gpu_kernels

namespace {

using gpu_kernels::DeviceComplex;
using gpu_kernels::DeviceComplex32;

/* ---------------------------------------------------------------------- */
/* Public-type <-> device-scalar mapping                                   */
/*                                                                         */
/* The public API is templated on the field scalar (Real or Complex). On   */
/* the device, Real maps to itself and Complex maps to the layout-          */
/* compatible DeviceComplex; field buffers are reinterpret_cast to the      */
/* device scalar. These helpers centralise the cast so the generic bodies   */
/* below carry no per-type branch.                                          */
/* ---------------------------------------------------------------------- */

template <typename T>
struct device_scalar;
template <>
struct device_scalar<Real> {
    using type = Real;
};
template <>
struct device_scalar<Complex> {
    using type = DeviceComplex;
};
template <>
struct device_scalar<Real32> {
    using type = float;
};
template <>
struct device_scalar<Complex32> {
    using type = DeviceComplex32;
};
template <typename T>
using DeviceScalar = typename device_scalar<T>::type;

//! Reinterpret a field's data pointer as the device scalar type.
template <typename T>
const DeviceScalar<T>* device_ptr(const TypedField<T, DeviceSpace>& f) {
    return reinterpret_cast<const DeviceScalar<T>*>(f.view().data());
}
template <typename T>
DeviceScalar<T>* device_ptr(TypedField<T, DeviceSpace>& f) {
    return reinterpret_cast<DeviceScalar<T>*>(f.view().data());
}

//! Convert a host-side scalar coefficient to the device scalar.
inline Real to_device(Real a) { return a; }
inline DeviceComplex to_device(Complex a) {
    return DeviceComplex{a.real(), a.imag()};
}
inline float to_device(float a) { return a; }  // Real32
inline DeviceComplex32 to_device(Complex32 a) {
    return DeviceComplex32{a.real(), a.imag()};
}

//! Convert a reduced device scalar back to the public scalar type.
inline Real to_public(Real x) { return x; }
inline Complex to_public(DeviceComplex x) { return Complex{x.re, x.im}; }
inline float to_public(float x) { return x; }  // Real32
inline Complex32 to_public(DeviceComplex32 x) {
    return Complex32{x.re, x.im};
}

//! Lift a real squared-norm to the public scalar type (norms are real-valued;
//! the complex API returns it as a zero-imaginary Complex, matching the host).
template <typename T>
T norm_to_public(Real r);
template <>
Real norm_to_public<Real>(Real r) {
    return r;
}
template <>
Complex norm_to_public<Complex>(Real r) {
    return Complex{r, 0.0};
}
template <>
Real32 norm_to_public<Real32>(Real r) {
    return static_cast<Real32>(r);
}
template <>
Complex32 norm_to_public<Complex32>(Real r) {
    return Complex32{static_cast<Real32>(r), 0.0f};
}

/**
 * True if the collection carries ghost buffers in any direction.
 */
bool has_ghosts(const GlobalFieldCollection& coll) {
    const auto& nb_ghosts_left = coll.get_nb_ghosts_left();
    const auto& nb_ghosts_right = coll.get_nb_ghosts_right();
    for (Dim_t d = 0; d < coll.get_spatial_dim(); ++d) {
        if (nb_ghosts_left[d] != 0 || nb_ghosts_right[d] != 0) {
            return true;
        }
    }
    return false;
}

/**
 * Grow-only device scratch buffer for reduction partial sums.
 *
 * The reductions used to cudaMalloc/cudaFree a `d_partial` buffer on every
 * call. That allocation pair has a large fixed cost (~100 us measured) that
 * dominates the reduction itself for small/medium fields, and a CG solve issues
 * several reductions per iteration. The reductions are called serially from the
 * host and each ends in a blocking device->host copy, so by the time one
 * reduction returns its scratch is free for the next to reuse: a single cached,
 * grow-only buffer per slot backs every reduction with no per-call allocation.
 *
 * Slot 0 backs the dot/norm reductions; slot 1 backs the packed pipelined-CG
 * result. Buffers are sized in Reals; a complex reduction whose partial sums
 * are DeviceComplex requests twice as many Reals (see scratch_as). Buffers are
 * intentionally never freed (process-lifetime cache; the driver reclaims them
 * at teardown). Not thread-safe, which matches muGrid's single-threaded host
 * call sites (MPI ranks are separate processes).
 */
Real* reduction_scratch(int slot, Index_t n_reals) {
    static Real* ptr[2]{nullptr, nullptr};
    static Index_t cap[2]{0, 0};
    if (n_reals > cap[slot]) {
        if (ptr[slot]) {
            device_deallocate(ptr[slot]);
        }
        // Route through the device allocator chokepoint so this scratch shares
        // the single owner/pool and is visible to the allocation profiler.
        ptr[slot] = static_cast<Real*>(
            device_allocate(n_reals * sizeof(Real), "linalg-reduction-scratch"));
        cap[slot] = n_reals;
    }
    return ptr[slot];
}

//! Reduction scratch typed as a device scalar (sized for `count` of them).
template <typename DS>
DS* scratch_as(int slot, Index_t count) {
    // Ceiling division: a single-precision DS (float / DeviceComplex32) is
    // smaller than a Real, so the integer ratio would round down to 0 and
    // under-allocate. Round up so the Real-backed buffer always holds `count`
    // DS values.
    constexpr Index_t reals_per =
        (sizeof(DS) + sizeof(Real) - 1) / sizeof(Real);
    return reinterpret_cast<DS*>(reduction_scratch(slot, count * reals_per));
}

/**
 * Interior box (extents, starts) and pixel strides of a field on `coll`,
 * shared by all interior reductions. Strides are in field-element units (here:
 * device-scalar elements), so they are identical for Real and complex fields.
 */
struct InteriorBox {
    Index_t extent[3]{1, 1, 1};
    Index_t start[3]{0, 0, 0};
    Index_t stride[3]{0, 0, 0};
    Index_t stride_c{0};
    Index_t nb_components_per_pixel{0};
    Index_t nb_interior_pixels{0};
};

template <typename T>
InteriorBox interior_box(const TypedField<T, DeviceSpace>& f,
                         const GlobalFieldCollection& coll) {
    const auto spatial_dim = coll.get_spatial_dim();
    if (spatial_dim < 1 || spatial_dim > 3) {
        throw FieldError("interior reduction only supports 1D, 2D and 3D fields");
    }
    const auto& nb_pts = coll.get_nb_subdomain_grid_pts_with_ghosts();
    const auto& nb_ghosts_left = coll.get_nb_ghosts_left();
    const auto& nb_ghosts_right = coll.get_nb_ghosts_right();
    const auto field_strides = f.get_strides(IterUnit::Pixel);

    InteriorBox box;
    box.stride_c = field_strides[0];
    box.nb_components_per_pixel = f.get_nb_components() * f.get_nb_sub_pts();
    for (Dim_t d = 0; d < spatial_dim; ++d) {
        box.start[d] = nb_ghosts_left[d];
        box.extent[d] = nb_pts[d] - nb_ghosts_left[d] - nb_ghosts_right[d];
        box.stride[d] = field_strides[field_strides.size() - spatial_dim + d];
    }
    box.nb_interior_pixels = box.extent[0] * box.extent[1] * box.extent[2];
    return box;
}

/* ---------------------------------------------------------------------- */
/* Generic reduction launchers                                             */
/*                                                                         */
/* DS is the device scalar (Real or DeviceComplex). The conj_product trait */
/* inside the kernels picks bilinear vs sesquilinear, so a single launcher  */
/* serves both reductions.                                                  */
/* ---------------------------------------------------------------------- */

//! Full-buffer reduction of conj_product(a, b) over `n` device-scalar elements.
template <typename DS>
DS reduce_full_dot(const DS* a, const DS* b, Index_t n) {
    const int num_blocks = (n + gpu_kernels::REDUCE_BLOCK_SIZE - 1) /
                           gpu_kernels::REDUCE_BLOCK_SIZE;
    DS* d_partial = scratch_as<DS>(0, num_blocks);

    GPU_LAUNCH_KERNEL(gpu_kernels::dot_reduce_kernel<DS>,
                      num_blocks, gpu_kernels::REDUCE_BLOCK_SIZE,
                      a, b, d_partial, n);
    GPU_LAUNCH_KERNEL(gpu_kernels::final_reduce_kernel<DS>,
                      1, gpu_kernels::REDUCE_BLOCK_SIZE, d_partial, num_blocks);

    DS result;
    GPU_MEMCPY_D2H(&result, d_partial, sizeof(DS));
    return result;
}

//! Interior reduction of conj_product(a, b) (ghosts excluded). Strides come
//! from `box`; the caller guarantees a and b share that layout.
template <typename DS>
DS reduce_interior_dot(const DS* a, const DS* b, const InteriorBox& box) {
    const int num_blocks =
        (box.nb_interior_pixels + gpu_kernels::REDUCE_BLOCK_SIZE - 1) /
        gpu_kernels::REDUCE_BLOCK_SIZE;
    DS* d_partial = scratch_as<DS>(0, num_blocks);

    GPU_LAUNCH_KERNEL(gpu_kernels::interior_dot_kernel<DS>,
                      num_blocks, gpu_kernels::REDUCE_BLOCK_SIZE,
                      a, b, d_partial, box.extent[0], box.extent[1],
                      box.extent[2], box.start[0], box.start[1], box.start[2],
                      box.stride_c, box.stride[0], box.stride[1],
                      box.stride[2], box.nb_components_per_pixel);
    GPU_LAUNCH_KERNEL(gpu_kernels::final_reduce_kernel<DS>,
                      1, gpu_kernels::REDUCE_BLOCK_SIZE, d_partial, num_blocks);

    DS result;
    GPU_MEMCPY_D2H(&result, d_partial, sizeof(DS));
    return result;
}

/* ---------------------------------------------------------------------- */
/* Generic operation bodies                                                */
/*                                                                         */
/* Each operation is written once as a function template over the field    */
/* scalar type T (Real or Complex). The public <T, DeviceSpace> entry       */
/* points further down are thin delegators. This mirrors the *_host<T>      */
/* pattern in linalg_host.cc.                                               */
/* ---------------------------------------------------------------------- */

template <typename T>
T vecdot_device(const TypedField<T, DeviceSpace>& a,
                const TypedField<T, DeviceSpace>& b) {
    if (&a.get_collection() != &b.get_collection()) {
        throw FieldError("vecdot: fields must belong to the same collection");
    }
    if (a.get_nb_components() != b.get_nb_components()) {
        throw FieldError("vecdot: fields must have the same number of components");
    }
    if (a.get_nb_sub_pts() != b.get_nb_sub_pts()) {
        throw FieldError("vecdot: fields must have the same number of sub-points");
    }
    using DS = DeviceScalar<T>;

    // For GlobalFieldCollection with ghosts, sum the interior directly;
    // full-buffer-minus-ghosts cancels catastrophically when the interior
    // values are small
    const auto& coll = a.get_collection();
    if (coll.get_domain() == FieldCollection::ValidityDomain::Global) {
        const auto& global_coll = static_cast<const GlobalFieldCollection&>(coll);
        if (has_ghosts(global_coll)) {
            return to_public(reduce_interior_dot<DS>(
                device_ptr(a), device_ptr(b),
                interior_box(a, global_coll)));
        }
    }

    // No ghosts: reduce the full buffer (get_nb_entries already counts
    // sub-points, so multiply only by the number of components)
    const Index_t n = a.get_nb_entries() * a.get_nb_components();
    return to_public(reduce_full_dot<DS>(device_ptr(a), device_ptr(b), n));
}

template <typename T>
T norm_sq_device(const TypedField<T, DeviceSpace>& x) {
    // norm_sq(x) == vecdot(x, x): conj_product(x, x) = |x|^2 for complex
    // (its imaginary part is exactly zero, x_r*x_i - x_i*x_r) and x^2 for real.
    using DS = DeviceScalar<T>;
    const auto& coll = x.get_collection();
    if (coll.get_domain() == FieldCollection::ValidityDomain::Global) {
        const auto& global_coll = static_cast<const GlobalFieldCollection&>(coll);
        if (has_ghosts(global_coll)) {
            return to_public(reduce_interior_dot<DS>(
                device_ptr(x), device_ptr(x), interior_box(x, global_coll)));
        }
    }
    const Index_t n = x.get_nb_entries() * x.get_nb_components();
    return to_public(reduce_full_dot<DS>(device_ptr(x), device_ptr(x), n));
}

template <typename T>
void axpy_device(T alpha, const TypedField<T, DeviceSpace>& x,
                 TypedField<T, DeviceSpace>& y) {
    if (&x.get_collection() != &y.get_collection()) {
        throw FieldError("axpy: fields must belong to the same collection");
    }
    if (x.get_nb_entries() != y.get_nb_entries() ||
        x.get_nb_components() != y.get_nb_components()) {
        throw FieldError("axpy: fields must have the same number of entries");
    }
    using DS = DeviceScalar<T>;
    // Total scalar elements in the full buffer (get_nb_entries already counts
    // sub-points, so multiply only by the number of components)
    const Index_t n = x.get_nb_entries() * x.get_nb_components();
    const int num_blocks = (n + gpu_kernels::BLOCK_SIZE - 1) /
                           gpu_kernels::BLOCK_SIZE;

    GPU_LAUNCH_KERNEL(gpu_kernels::axpy_kernel<DS>,
                      num_blocks, gpu_kernels::BLOCK_SIZE,
                      to_device(alpha), device_ptr(x), device_ptr(y), n);
}

template <typename T>
void scal_device(T alpha, TypedField<T, DeviceSpace>& x) {
    using DS = DeviceScalar<T>;
    const Index_t n = x.get_nb_entries() * x.get_nb_components();
    const int num_blocks = (n + gpu_kernels::BLOCK_SIZE - 1) /
                           gpu_kernels::BLOCK_SIZE;

    GPU_LAUNCH_KERNEL(gpu_kernels::scal_kernel<DS>,
                      num_blocks, gpu_kernels::BLOCK_SIZE,
                      to_device(alpha), device_ptr(x), n);
}

template <typename T>
void axpby_device(T alpha, const TypedField<T, DeviceSpace>& x, T beta,
                  TypedField<T, DeviceSpace>& y) {
    if (&x.get_collection() != &y.get_collection()) {
        throw FieldError("axpby: fields must belong to the same collection");
    }
    if (x.get_nb_entries() != y.get_nb_entries() ||
        x.get_nb_components() != y.get_nb_components()) {
        throw FieldError("axpby: fields must have the same number of entries");
    }
    using DS = DeviceScalar<T>;
    const Index_t n = x.get_nb_entries() * x.get_nb_components();
    const int num_blocks = (n + gpu_kernels::BLOCK_SIZE - 1) /
                           gpu_kernels::BLOCK_SIZE;

    GPU_LAUNCH_KERNEL(gpu_kernels::axpby_kernel<DS>,
                      num_blocks, gpu_kernels::BLOCK_SIZE,
                      to_device(alpha), device_ptr(x), to_device(beta),
                      device_ptr(y), n);
}

template <typename T>
void copy_device(const TypedField<T, DeviceSpace>& src,
                 TypedField<T, DeviceSpace>& dst) {
    if (&src.get_collection() != &dst.get_collection()) {
        throw FieldError("copy: fields must belong to the same collection");
    }
    if (src.get_nb_entries() != dst.get_nb_entries() ||
        src.get_nb_components() != dst.get_nb_components()) {
        throw FieldError("copy: fields must have the same number of entries");
    }
    using DS = DeviceScalar<T>;
    const Index_t n = src.get_nb_entries() * src.get_nb_components();
    const int num_blocks = (n + gpu_kernels::BLOCK_SIZE - 1) /
                           gpu_kernels::BLOCK_SIZE;

    GPU_LAUNCH_KERNEL(gpu_kernels::copy_kernel<DS>,
                      num_blocks, gpu_kernels::BLOCK_SIZE,
                      device_ptr(src), device_ptr(dst), n);
}

template <typename T>
T axpy_norm_sq_device(T alpha, const TypedField<T, DeviceSpace>& x,
                      TypedField<T, DeviceSpace>& y) {
    if (&x.get_collection() != &y.get_collection()) {
        throw FieldError("axpy_norm_sq: fields must belong to the same collection");
    }
    if (x.get_nb_entries() != y.get_nb_entries() ||
        x.get_nb_components() != y.get_nb_components()) {
        throw FieldError("axpy_norm_sq: fields must have the same number of entries");
    }
    using DS = DeviceScalar<T>;
    const auto& coll = x.get_collection();
    // Total scalar elements in the full buffer (get_nb_entries already counts
    // sub-points, so multiply only by the number of components)
    const Index_t n = x.get_nb_entries() * x.get_nb_components();

    // For GlobalFieldCollection with ghosts: update y over the full buffer
    // (ghost pixels included, keeping the buffer consistent), then sum the
    // interior directly. Full-buffer-minus-ghosts cancels catastrophically
    // when the interior values are small.
    if (coll.get_domain() == FieldCollection::ValidityDomain::Global) {
        const auto& global_coll = static_cast<const GlobalFieldCollection&>(coll);
        if (has_ghosts(global_coll)) {
            const int num_blocks = (n + gpu_kernels::BLOCK_SIZE - 1) /
                                   gpu_kernels::BLOCK_SIZE;
            GPU_LAUNCH_KERNEL(gpu_kernels::axpy_kernel<DS>,
                              num_blocks, gpu_kernels::BLOCK_SIZE,
                              to_device(alpha), device_ptr(x), device_ptr(y), n);
            // Kernels on the default stream serialize, so the interior
            // reduction sees the updated y. norm_sq(y) == vecdot(y, y).
            return to_public(reduce_interior_dot<DS>(
                device_ptr(y), device_ptr(y), interior_box(y, global_coll)));
        }
    }

    // No ghosts: fused single-pass AXPY + norm_sq (real-valued partial sums).
    const int num_blocks = (n + gpu_kernels::REDUCE_BLOCK_SIZE - 1) /
                           gpu_kernels::REDUCE_BLOCK_SIZE;
    Real* d_partial = reduction_scratch(0, num_blocks);

    GPU_LAUNCH_KERNEL(gpu_kernels::axpy_norm_sq_kernel<DS>,
                      num_blocks, gpu_kernels::REDUCE_BLOCK_SIZE,
                      to_device(alpha), device_ptr(x), device_ptr(y),
                      d_partial, n);
    GPU_LAUNCH_KERNEL(gpu_kernels::final_reduce_kernel<Real>,
                      1, gpu_kernels::REDUCE_BLOCK_SIZE, d_partial, num_blocks);

    Real result;
    GPU_MEMCPY_D2H(&result, d_partial, sizeof(Real));
    return norm_to_public<T>(result);
}

template <typename T>
void cross_device(const TypedField<T, DeviceSpace>& a,
                  const TypedField<T, DeviceSpace>& b,
                  TypedField<T, DeviceSpace>& out) {
    const auto& coll = a.get_collection();
    internal::check_three_vector("cross", a, coll);
    internal::check_three_vector("cross", b, coll);
    internal::check_three_vector("cross", out, coll);
    const Index_t npix = a.get_nb_entries();
    // An empty subdomain (e.g. an MPI rank with no local pixels) has nothing to
    // compute. Skip it, including the aliasing check below: empty fields share a
    // null data pointer, so that check would otherwise fire spuriously.
    if (npix == 0) {
        return;
    }
    if (out.view().data() == a.view().data() ||
        out.view().data() == b.view().data()) {
        throw FieldError(
            "cross: output must be a field distinct from both inputs");
    }
    using DS = DeviceScalar<T>;
    const bool soa =
        (out.get_storage_order() == StorageOrder::StructureOfArrays);
    const int num_blocks =
        (npix + gpu_kernels::BLOCK_SIZE - 1) / gpu_kernels::BLOCK_SIZE;
    GPU_LAUNCH_KERNEL(gpu_kernels::cross_kernel<DS>, num_blocks,
                      gpu_kernels::BLOCK_SIZE, device_ptr(a), device_ptr(b),
                      device_ptr(out), npix, soa);
}

}  // namespace

/* ---------------------------------------------------------------------- */
/* Public API: thin <T, DeviceSpace> delegators to the generic bodies.     */
/* These specializations are the device ABI surface that linalg.hh         */
/* declares; the Real and Complex paths share one body each.               */
/* ---------------------------------------------------------------------- */

template <>
Real vecdot<Real, DeviceSpace>(const TypedField<Real, DeviceSpace>& a,
                               const TypedField<Real, DeviceSpace>& b) {
    return vecdot_device(a, b);
}
template <>
Complex vecdot<Complex, DeviceSpace>(const TypedField<Complex, DeviceSpace>& a,
                                     const TypedField<Complex, DeviceSpace>& b) {
    return vecdot_device(a, b);
}

template <>
Real norm_sq<Real, DeviceSpace>(const TypedField<Real, DeviceSpace>& x) {
    return norm_sq_device(x);
}
template <>
Complex norm_sq<Complex, DeviceSpace>(const TypedField<Complex, DeviceSpace>& x) {
    return norm_sq_device(x);
}

template <>
void axpy<Real, DeviceSpace>(Real alpha, const TypedField<Real, DeviceSpace>& x,
                             TypedField<Real, DeviceSpace>& y) {
    axpy_device(alpha, x, y);
}
template <>
void axpy<Complex, DeviceSpace>(Complex alpha,
                                const TypedField<Complex, DeviceSpace>& x,
                                TypedField<Complex, DeviceSpace>& y) {
    axpy_device(alpha, x, y);
}

template <>
void scal<Real, DeviceSpace>(Real alpha, TypedField<Real, DeviceSpace>& x) {
    scal_device(alpha, x);
}
template <>
void scal<Complex, DeviceSpace>(Complex alpha,
                                TypedField<Complex, DeviceSpace>& x) {
    scal_device(alpha, x);
}

template <>
void axpby<Real, DeviceSpace>(Real alpha, const TypedField<Real, DeviceSpace>& x,
                              Real beta, TypedField<Real, DeviceSpace>& y) {
    axpby_device(alpha, x, beta, y);
}
template <>
void axpby<Complex, DeviceSpace>(Complex alpha,
                                 const TypedField<Complex, DeviceSpace>& x,
                                 Complex beta,
                                 TypedField<Complex, DeviceSpace>& y) {
    axpby_device(alpha, x, beta, y);
}

template <>
void copy<Real, DeviceSpace>(const TypedField<Real, DeviceSpace>& src,
                             TypedField<Real, DeviceSpace>& dst) {
    copy_device(src, dst);
}
template <>
void copy<Complex, DeviceSpace>(const TypedField<Complex, DeviceSpace>& src,
                                TypedField<Complex, DeviceSpace>& dst) {
    copy_device(src, dst);
}

template <>
Real axpy_norm_sq<Real, DeviceSpace>(Real alpha,
                                     const TypedField<Real, DeviceSpace>& x,
                                     TypedField<Real, DeviceSpace>& y) {
    return axpy_norm_sq_device(alpha, x, y);
}
template <>
Complex axpy_norm_sq<Complex, DeviceSpace>(
    Complex alpha, const TypedField<Complex, DeviceSpace>& x,
    TypedField<Complex, DeviceSpace>& y) {
    return axpy_norm_sq_device(alpha, x, y);
}

template <>
void cross<Real, DeviceSpace>(const TypedField<Real, DeviceSpace>& a,
                              const TypedField<Real, DeviceSpace>& b,
                              TypedField<Real, DeviceSpace>& out) {
    cross_device(a, b, out);
}
template <>
void cross<Complex, DeviceSpace>(const TypedField<Complex, DeviceSpace>& a,
                                 const TypedField<Complex, DeviceSpace>& b,
                                 TypedField<Complex, DeviceSpace>& out) {
    cross_device(a, b, out);
}

/* -- Single-precision (Real32 / Complex32) device CG building blocks ------ */
/* These go through the generic *_device<T> bodies + DeviceComplex32 layer.  */
/* The field-valued scal / leray_project / pipelined_cg_dots use custom      */
/* double kernels and are not yet provided in single precision on device.    */
#define MUGRID_LINALG_DEVICE_SPECIALIZATIONS(T)                                \
    template <>                                                                \
    T vecdot<T, DeviceSpace>(const TypedField<T, DeviceSpace>& a,              \
                             const TypedField<T, DeviceSpace>& b) {            \
        return vecdot_device(a, b);                                            \
    }                                                                          \
    template <>                                                                \
    T norm_sq<T, DeviceSpace>(const TypedField<T, DeviceSpace>& x) {           \
        return norm_sq_device(x);                                             \
    }                                                                          \
    template <>                                                                \
    void axpy<T, DeviceSpace>(T alpha, const TypedField<T, DeviceSpace>& x,    \
                              TypedField<T, DeviceSpace>& y) {                 \
        axpy_device(alpha, x, y);                                             \
    }                                                                          \
    template <>                                                                \
    void scal<T, DeviceSpace>(T alpha, TypedField<T, DeviceSpace>& x) {        \
        scal_device(alpha, x);                                               \
    }                                                                          \
    template <>                                                                \
    void axpby<T, DeviceSpace>(T alpha, const TypedField<T, DeviceSpace>& x,   \
                               T beta, TypedField<T, DeviceSpace>& y) {        \
        axpby_device(alpha, x, beta, y);                                      \
    }                                                                          \
    template <>                                                                \
    void copy<T, DeviceSpace>(const TypedField<T, DeviceSpace>& src,           \
                              TypedField<T, DeviceSpace>& dst) {               \
        copy_device(src, dst);                                               \
    }                                                                          \
    template <>                                                                \
    T axpy_norm_sq<T, DeviceSpace>(T alpha, const TypedField<T, DeviceSpace>& x,\
                                   TypedField<T, DeviceSpace>& y) {            \
        return axpy_norm_sq_device(alpha, x, y);                              \
    }                                                                          \
    template <>                                                                \
    void cross<T, DeviceSpace>(const TypedField<T, DeviceSpace>& a,            \
                               const TypedField<T, DeviceSpace>& b,            \
                               TypedField<T, DeviceSpace>& out) {              \
        cross_device(a, b, out);                                             \
    }
MUGRID_LINALG_DEVICE_SPECIALIZATIONS(Real32)
MUGRID_LINALG_DEVICE_SPECIALIZATIONS(Complex32)
#undef MUGRID_LINALG_DEVICE_SPECIALIZATIONS

/* ---------------------------------------------------------------------- */
/* pipelined_cg_dots (Real only): fused {(r,u), (w,u), (r,r)} reduction.   */
/* ---------------------------------------------------------------------- */

template <>
std::array<Real, 3> pipelined_cg_dots<Real, DeviceSpace>(
    const TypedField<Real, DeviceSpace>& r,
    const TypedField<Real, DeviceSpace>& u,
    const TypedField<Real, DeviceSpace>& w) {
    const auto& coll = r.get_collection();

    // The pipelined CG fields live on a decomposition (Global collection). For
    // anything else, fall back to three separate reductions (correctness over
    // the single-sync fast path).
    if (coll.get_domain() != FieldCollection::ValidityDomain::Global) {
        return {vecdot<Real, DeviceSpace>(r, u), vecdot<Real, DeviceSpace>(w, u),
                norm_sq<Real, DeviceSpace>(r)};
    }
    const auto& global_coll = static_cast<const GlobalFieldCollection&>(coll);
    const InteriorBox box = interior_box(r, global_coll);
    if (box.nb_interior_pixels <= 0) {
        return {0.0, 0.0, 0.0};
    }

    const int num_blocks =
        (box.nb_interior_pixels + gpu_kernels::REDUCE_BLOCK_SIZE - 1) /
        gpu_kernels::REDUCE_BLOCK_SIZE;
    // Slot 0: 3*num_blocks partial sums; slot 1: the 3 packed results.
    Real* d_partial = reduction_scratch(0, 3 * num_blocks);
    Real* d_out = reduction_scratch(1, 3);

    GPU_LAUNCH_KERNEL(gpu_kernels::interior_three_dots_kernel,
                      num_blocks, gpu_kernels::REDUCE_BLOCK_SIZE,
                      r.view().data(), u.view().data(), w.view().data(),
                      d_partial, box.extent[0], box.extent[1], box.extent[2],
                      box.start[0], box.start[1], box.start[2], box.stride_c,
                      box.stride[0], box.stride[1], box.stride[2],
                      box.nb_components_per_pixel,
                      static_cast<Index_t>(num_blocks));
    GPU_LAUNCH_KERNEL(gpu_kernels::final_reduce3_kernel, 1,
                      gpu_kernels::REDUCE_BLOCK_SIZE, d_partial, d_out,
                      static_cast<Index_t>(num_blocks));

    Real h[3];
    GPU_MEMCPY_D2H(h, d_out, 3 * sizeof(Real));
    return {h[0], h[1], h[2]};
}

/* ---------------------------------------------------------------------- */
/* Field-valued scal: x[c, i] *= alpha[c, i] with a real coefficient field.*/
/* alpha is real, so it scales the real and imaginary parts of a complex x  */
/* identically; the kernels work on the underlying reals.                   */
/* ---------------------------------------------------------------------- */

template <>
void scal<DeviceSpace>(const TypedField<Real, DeviceSpace>& alpha,
                       TypedField<Complex, DeviceSpace>& x) {
    internal::check_field_alpha(alpha, x);

    const Index_t npix = x.get_nb_entries();
    const Index_t ncomp = x.get_nb_components();
    const Index_t n2 = npix * ncomp * 2;
    const bool soa =
        (x.get_storage_order() == StorageOrder::StructureOfArrays);
    const bool alpha_per_comp = (alpha.get_nb_components() == ncomp &&
                                 ncomp != 1);
    const int num_blocks = (n2 + gpu_kernels::BLOCK_SIZE - 1) /
                           gpu_kernels::BLOCK_SIZE;

    GPU_LAUNCH_KERNEL(gpu_kernels::field_scal_complex_kernel,
                      num_blocks, gpu_kernels::BLOCK_SIZE,
                      reinterpret_cast<Real*>(x.view().data()),
                      alpha.view().data(), npix, ncomp, soa,
                      alpha_per_comp, n2);
}

template <>
void scal<DeviceSpace>(const TypedField<Real, DeviceSpace>& alpha,
                       TypedField<Real, DeviceSpace>& x) {
    internal::check_field_alpha(alpha, x);

    const Index_t npix = x.get_nb_entries();
    const Index_t ncomp = x.get_nb_components();
    const Index_t n = npix * ncomp;
    const bool soa =
        (x.get_storage_order() == StorageOrder::StructureOfArrays);
    const bool alpha_per_comp = (alpha.get_nb_components() == ncomp &&
                                 ncomp != 1);
    const int num_blocks = (n + gpu_kernels::BLOCK_SIZE - 1) /
                           gpu_kernels::BLOCK_SIZE;

    GPU_LAUNCH_KERNEL(gpu_kernels::field_scal_real_kernel,
                      num_blocks, gpu_kernels::BLOCK_SIZE,
                      x.view().data(), alpha.view().data(), npix, ncomp,
                      soa, alpha_per_comp, n);
}

/* ---------------------------------------------------------------------- */
/* Fused Leray (Helmholtz) projection (complex field, real coefficients).  */
/* ---------------------------------------------------------------------- */

template <>
void leray_project<DeviceSpace>(const TypedField<Real, DeviceSpace>& k,
                                const TypedField<Real, DeviceSpace>& invk,
                                const TypedField<Complex, DeviceSpace>& N,
                                TypedField<Complex, DeviceSpace>& out) {
    const auto& coll = out.get_collection();
    internal::check_three_vector("leray_project", k, coll);
    internal::check_three_vector("leray_project", invk, coll);
    internal::check_three_vector("leray_project", N, coll);
    internal::check_three_vector("leray_project", out, coll);
    const Index_t npix = out.get_nb_entries();
    const bool soa =
        (out.get_storage_order() == StorageOrder::StructureOfArrays);
    const int num_blocks =
        (npix + gpu_kernels::BLOCK_SIZE - 1) / gpu_kernels::BLOCK_SIZE;
    GPU_LAUNCH_KERNEL(
        gpu_kernels::leray_kernel, num_blocks, gpu_kernels::BLOCK_SIZE,
        k.view().data(), invk.view().data(),
        reinterpret_cast<const Real*>(N.view().data()),
        reinterpret_cast<Real*>(out.view().data()), npix, soa);
}

}  // namespace linalg
}  // namespace muGrid
