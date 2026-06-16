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
/* Element-wise kernels (full buffer operations)                          */
/* ---------------------------------------------------------------------- */

/**
 * AXPY kernel: y = alpha * x + y
 */
__global__ void axpy_kernel(Real alpha, const Real* x, Real* y, Index_t n) {
    Index_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] += alpha * x[idx];
    }
}

/**
 * Scale kernel: x = alpha * x
 */
__global__ void scal_kernel(Real alpha, Real* x, Index_t n) {
    Index_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] *= alpha;
    }
}

/**
 * Copy kernel: dst = src
 */
__global__ void copy_kernel(const Real* src, Real* dst, Index_t n) {
    Index_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

/**
 * AXPBY kernel: y = alpha * x + beta * y
 */
__global__ void axpby_kernel(Real alpha, const Real* x, Real beta, Real* y, Index_t n) {
    Index_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = alpha * x[idx] + beta * y[idx];
    }
}

/**
 * Fused AXPY + norm_sq kernel: y = alpha * x + y, returns partial ||y||²
 * Each block computes AXPY for its elements AND accumulates partial norm.
 */
__global__ void axpy_norm_sq_kernel(Real alpha, const Real* x, Real* y,
                                    Real* partial_sums, Index_t n) {
    __shared__ Real shared_data[REDUCE_BLOCK_SIZE];

    Index_t tid = threadIdx.x;
    Index_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Fused: update y AND accumulate squared norm
    Real sum = 0.0;
    while (idx < n) {
        Real new_y = y[idx] + alpha * x[idx];
        y[idx] = new_y;
        sum += new_y * new_y;
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
 * Dot product reduction kernel (first pass).
 * Computes partial sums per block.
 */
__global__ void dot_reduce_kernel(const Real* a, const Real* b,
                                  Real* partial_sums, Index_t n) {
    __shared__ Real shared_data[REDUCE_BLOCK_SIZE];

    Index_t tid = threadIdx.x;
    Index_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and multiply
    Real sum = 0.0;
    while (idx < n) {
        sum += a[idx] * b[idx];
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
 * Squared norm reduction kernel (first pass).
 * Computes partial sums per block.
 */
__global__ void norm_sq_reduce_kernel(const Real* x, Real* partial_sums,
                                      Index_t n) {
    __shared__ Real shared_data[REDUCE_BLOCK_SIZE];

    Index_t tid = threadIdx.x;
    Index_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and square
    Real sum = 0.0;
    while (idx < n) {
        Real val = x[idx];
        sum += val * val;
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
 */
__global__ void interior_dot_kernel(
    const Real* a, const Real* b, Real* partial_sums,
    Index_t nx, Index_t ny, Index_t nz,
    Index_t x0, Index_t y0, Index_t z0,
    Index_t stride_c, Index_t stride_x, Index_t stride_y, Index_t stride_z,
    Index_t nb_components) {

    __shared__ Real shared_data[REDUCE_BLOCK_SIZE];

    Index_t tid = threadIdx.x;
    Index_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    Index_t nb_pixels = nx * ny * nz;

    Real sum = 0.0;
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
            sum += a[offset + c * stride_c] * b[offset + c * stride_c];
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
 * Final reduction kernel - sums partial results.
 */
__global__ void final_reduce_kernel(Real* data, Index_t n) {
    __shared__ Real shared_data[REDUCE_BLOCK_SIZE];

    Index_t tid = threadIdx.x;

    // Load partial sums
    Real sum = 0.0;
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

// Per-pixel three-vector cross product out = a x b on real buffers. One thread
// per pixel; `soa` selects the component stride (npix for SoA, 1 with a
// pixel-stride of 3 for AoS), matching the host kernel.
__global__ void cross_real_kernel(const Real* a, const Real* b, Real* out,
                                  Index_t npix, bool soa) {
    Index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < npix) {
        const Index_t cs = soa ? npix : 1;        // stride between components
        const Index_t base = soa ? i : 3 * i;     // first component of pixel i
        const Real a0 = a[base], a1 = a[base + cs], a2 = a[base + 2 * cs];
        const Real b0 = b[base], b1 = b[base + cs], b2 = b[base + 2 * cs];
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

/* ---------------------------------------------------------------------- */
/* Complex element-wise kernels                                            */
/*                                                                         */
/* The complex buffers are addressed as their underlying reals (re, im at  */
/* 2*i and 2*i+1); scalar complex coefficients are passed as (re, im) pairs */
/* so the kernels need no device complex type. One thread per complex      */
/* element, `n` complex elements total.                                    */
/* ---------------------------------------------------------------------- */

// x = alpha * x
__global__ void scal_complex_scalar_kernel(Real ar, Real ai, Real* x,
                                           Index_t n) {
    Index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const Real xr = x[2 * i], xi = x[2 * i + 1];
        x[2 * i]     = ar * xr - ai * xi;
        x[2 * i + 1] = ar * xi + ai * xr;
    }
}

// y = alpha * x + y
__global__ void axpy_complex_scalar_kernel(Real ar, Real ai, const Real* x,
                                           Real* y, Index_t n) {
    Index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const Real xr = x[2 * i], xi = x[2 * i + 1];
        y[2 * i]     += ar * xr - ai * xi;
        y[2 * i + 1] += ar * xi + ai * xr;
    }
}

// y = alpha * x + beta * y
__global__ void axpby_complex_scalar_kernel(Real ar, Real ai, const Real* x,
                                            Real br, Real bi, Real* y,
                                            Index_t n) {
    Index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const Real xr = x[2 * i], xi = x[2 * i + 1];
        const Real yr = y[2 * i], yi = y[2 * i + 1];
        y[2 * i]     = ar * xr - ai * xi + br * yr - bi * yi;
        y[2 * i + 1] = ar * xi + ai * xr + br * yi + bi * yr;
    }
}

// Per-pixel three-vector cross product out = a x b for complex fields, on the
// underlying reals (each multiply is a complex multiply). Mirrors
// cross_real_kernel; one thread per pixel.
__global__ void cross_complex_kernel(const Real* a, const Real* b, Real* out,
                                     Index_t npix, bool soa) {
    Index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < npix) {
        const Index_t cs = soa ? npix : 1;     // complex component stride
        const Index_t base = soa ? i : 3 * i;  // first component of pixel i
        // load the three complex components of a and b (re, im)
        const Real a0r = a[2 * base],            a0i = a[2 * base + 1];
        const Real a1r = a[2 * (base + cs)],     a1i = a[2 * (base + cs) + 1];
        const Real a2r = a[2 * (base + 2 * cs)], a2i = a[2 * (base + 2 * cs) + 1];
        const Real b0r = b[2 * base],            b0i = b[2 * base + 1];
        const Real b1r = b[2 * (base + cs)],     b1i = b[2 * (base + cs) + 1];
        const Real b2r = b[2 * (base + 2 * cs)], b2i = b[2 * (base + 2 * cs) + 1];
        // out0 = a1*b2 - a2*b1
        out[2 * base]              = (a1r * b2r - a1i * b2i) - (a2r * b1r - a2i * b1i);
        out[2 * base + 1]          = (a1r * b2i + a1i * b2r) - (a2r * b1i + a2i * b1r);
        // out1 = a2*b0 - a0*b2
        out[2 * (base + cs)]       = (a2r * b0r - a2i * b0i) - (a0r * b2r - a0i * b2i);
        out[2 * (base + cs) + 1]   = (a2r * b0i + a2i * b0r) - (a0r * b2i + a0i * b2r);
        // out2 = a0*b1 - a1*b0
        out[2 * (base + 2 * cs)]     = (a0r * b1r - a0i * b1i) - (a1r * b0r - a1i * b0i);
        out[2 * (base + 2 * cs) + 1] = (a0r * b1i + a0i * b1r) - (a1r * b0i + a1i * b0r);
    }
}

/* ---------------------------------------------------------------------- */
/* Complex reduction kernels                                               */
/* ---------------------------------------------------------------------- */

// Full-buffer sesquilinear dot conj(a).b: separate Re/Im partial sums per
// block. Re(conj(a)*b) = ar*br + ai*bi; Im = ar*bi - ai*br.
__global__ void dot_reduce_complex_kernel(const Real* a, const Real* b,
                                          Real* partial_re, Real* partial_im,
                                          Index_t n) {
    __shared__ Real sh_re[REDUCE_BLOCK_SIZE];
    __shared__ Real sh_im[REDUCE_BLOCK_SIZE];
    Index_t tid = threadIdx.x;
    Index_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    Real sre = 0.0, sim = 0.0;
    while (idx < n) {
        const Real ar = a[2 * idx], ai = a[2 * idx + 1];
        const Real br = b[2 * idx], bi = b[2 * idx + 1];
        sre += ar * br + ai * bi;
        sim += ar * bi - ai * br;
        idx += blockDim.x * gridDim.x;
    }
    sh_re[tid] = sre;
    sh_im[tid] = sim;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_re[tid] += sh_re[tid + stride];
            sh_im[tid] += sh_im[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial_re[blockIdx.x] = sh_re[0];
        partial_im[blockIdx.x] = sh_im[0];
    }
}

// Interior (ghost-excluded) sesquilinear dot conj(a).b on complex fields.
// Mirrors interior_dot_kernel but reads complex elements and accumulates the
// real and imaginary parts of the contraction separately.
__global__ void interior_dot_complex_kernel(
    const Real* a, const Real* b, Real* partial_re, Real* partial_im,
    Index_t nx, Index_t ny, Index_t nz, Index_t x0, Index_t y0, Index_t z0,
    Index_t stride_c, Index_t stride_x, Index_t stride_y, Index_t stride_z,
    Index_t nb_components) {
    __shared__ Real sh_re[REDUCE_BLOCK_SIZE];
    __shared__ Real sh_im[REDUCE_BLOCK_SIZE];
    Index_t tid = threadIdx.x;
    Index_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    Index_t nb_pixels = nx * ny * nz;
    Real sre = 0.0, sim = 0.0;
    for (Index_t pixel_idx = global_tid; pixel_idx < nb_pixels;
         pixel_idx += blockDim.x * gridDim.x) {
        Index_t ix = x0 + pixel_idx % nx;
        Index_t rem = pixel_idx / nx;
        Index_t iy = y0 + rem % ny;
        Index_t iz = z0 + rem / ny;
        Index_t offset = ix * stride_x + iy * stride_y + iz * stride_z;
        for (Index_t c = 0; c < nb_components; ++c) {
            const Index_t e = offset + c * stride_c;  // complex element index
            const Real ar = a[2 * e], ai = a[2 * e + 1];
            const Real br = b[2 * e], bi = b[2 * e + 1];
            sre += ar * br + ai * bi;
            sim += ar * bi - ai * br;
        }
    }
    sh_re[tid] = sre;
    sh_im[tid] = sim;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_re[tid] += sh_re[tid + stride];
            sh_im[tid] += sh_im[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial_re[blockIdx.x] = sh_re[0];
        partial_im[blockIdx.x] = sh_im[0];
    }
}

// Interior (ghost-excluded) squared norm sum |x|^2 on a complex field.
__global__ void interior_norm_sq_complex_kernel(
    const Real* x, Real* partial_sums,
    Index_t nx, Index_t ny, Index_t nz, Index_t x0, Index_t y0, Index_t z0,
    Index_t stride_c, Index_t stride_x, Index_t stride_y, Index_t stride_z,
    Index_t nb_components) {
    __shared__ Real shared_data[REDUCE_BLOCK_SIZE];
    Index_t tid = threadIdx.x;
    Index_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    Index_t nb_pixels = nx * ny * nz;
    Real sum = 0.0;
    for (Index_t pixel_idx = global_tid; pixel_idx < nb_pixels;
         pixel_idx += blockDim.x * gridDim.x) {
        Index_t ix = x0 + pixel_idx % nx;
        Index_t rem = pixel_idx / nx;
        Index_t iy = y0 + rem % ny;
        Index_t iz = z0 + rem / ny;
        Index_t offset = ix * stride_x + iy * stride_y + iz * stride_z;
        for (Index_t c = 0; c < nb_components; ++c) {
            const Index_t e = offset + c * stride_c;
            const Real xr = x[2 * e], xi = x[2 * e + 1];
            sum += xr * xr + xi * xi;
        }
    }
    shared_data[tid] = sum;
    __syncthreads();
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

// Fused complex AXPY + squared norm: y = alpha*x + y, accumulate |y|^2 (real).
__global__ void axpy_norm_sq_complex_kernel(Real ar, Real ai, const Real* x,
                                            Real* y, Real* partial_sums,
                                            Index_t n) {
    __shared__ Real shared_data[REDUCE_BLOCK_SIZE];
    Index_t tid = threadIdx.x;
    Index_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    Real sum = 0.0;
    while (idx < n) {
        const Real xr = x[2 * idx], xi = x[2 * idx + 1];
        const Real new_yr = y[2 * idx] + (ar * xr - ai * xi);
        const Real new_yi = y[2 * idx + 1] + (ar * xi + ai * xr);
        y[2 * idx]     = new_yr;
        y[2 * idx + 1] = new_yi;
        sum += new_yr * new_yr + new_yi * new_yi;
        idx += blockDim.x * gridDim.x;
    }
    shared_data[tid] = sum;
    __syncthreads();
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

}  // namespace gpu_kernels

namespace {

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
 * Launch the interior dot-product reduction for two fields of `coll` and
 * return the result. Strides are taken from `a`; the caller guarantees that
 * `b` shares the same layout (fields of the same collection with the same
 * number of components and sub-points).
 */
Real interior_dot(const TypedField<Real, DeviceSpace>& a,
                  const TypedField<Real, DeviceSpace>& b,
                  const GlobalFieldCollection& coll) {
    const auto spatial_dim = coll.get_spatial_dim();
    if (spatial_dim < 1 || spatial_dim > 3) {
        throw FieldError("interior_dot only supports 1D, 2D and 3D fields");
    }
    const auto& nb_pts = coll.get_nb_subdomain_grid_pts_with_ghosts();
    const auto& nb_ghosts_left = coll.get_nb_ghosts_left();
    const auto& nb_ghosts_right = coll.get_nb_ghosts_right();
    const auto field_strides = a.get_strides(IterUnit::Pixel);
    const Index_t stride_c = field_strides[0];
    const Index_t nb_components_per_pixel =
        a.get_nb_components() * a.get_nb_sub_pts();

    // Interior bounds; unused trailing dimensions degenerate to one pass
    Index_t extent[3]{1, 1, 1};
    Index_t start[3]{0, 0, 0};
    Index_t stride[3]{0, 0, 0};
    for (Dim_t d = 0; d < spatial_dim; ++d) {
        start[d] = nb_ghosts_left[d];
        extent[d] = nb_pts[d] - nb_ghosts_left[d] - nb_ghosts_right[d];
        stride[d] = field_strides[field_strides.size() - spatial_dim + d];
    }

    const Index_t nb_interior_pixels = extent[0] * extent[1] * extent[2];
    if (nb_interior_pixels <= 0) {
        return 0.0;
    }

    const int num_blocks =
        (nb_interior_pixels + gpu_kernels::REDUCE_BLOCK_SIZE - 1) /
        gpu_kernels::REDUCE_BLOCK_SIZE;
    Real* d_partial;
    GPU_MALLOC(&d_partial, num_blocks * sizeof(Real));

    GPU_LAUNCH_KERNEL(gpu_kernels::interior_dot_kernel,
                      num_blocks, gpu_kernels::REDUCE_BLOCK_SIZE,
                      a.view().data(), b.view().data(), d_partial,
                      extent[0], extent[1], extent[2],
                      start[0], start[1], start[2],
                      stride_c, stride[0], stride[1], stride[2],
                      nb_components_per_pixel);

    GPU_LAUNCH_KERNEL(gpu_kernels::final_reduce_kernel,
                      1, gpu_kernels::REDUCE_BLOCK_SIZE,
                      d_partial, num_blocks);

    Real result;
    GPU_MEMCPY_D2H(&result, d_partial, sizeof(Real));
    GPU_FREE(d_partial);
    return result;
}

/**
 * Extract the interior box (extents, starts) and pixel strides of a field on
 * `coll`, shared by the complex interior reductions below. Strides are in
 * field-element (here: complex-element) units.
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

/**
 * Interior sesquilinear dot conj(a).b for complex fields (ghosts excluded).
 */
Complex interior_dot_complex(const TypedField<Complex, DeviceSpace>& a,
                             const TypedField<Complex, DeviceSpace>& b,
                             const GlobalFieldCollection& coll) {
    const InteriorBox box = interior_box(a, coll);
    if (box.nb_interior_pixels <= 0) {
        return Complex{0.0, 0.0};
    }
    const int num_blocks =
        (box.nb_interior_pixels + gpu_kernels::REDUCE_BLOCK_SIZE - 1) /
        gpu_kernels::REDUCE_BLOCK_SIZE;
    Real* d_re;
    Real* d_im;
    GPU_MALLOC(&d_re, num_blocks * sizeof(Real));
    GPU_MALLOC(&d_im, num_blocks * sizeof(Real));

    GPU_LAUNCH_KERNEL(gpu_kernels::interior_dot_complex_kernel,
                      num_blocks, gpu_kernels::REDUCE_BLOCK_SIZE,
                      reinterpret_cast<const Real*>(a.view().data()),
                      reinterpret_cast<const Real*>(b.view().data()),
                      d_re, d_im, box.extent[0], box.extent[1], box.extent[2],
                      box.start[0], box.start[1], box.start[2], box.stride_c,
                      box.stride[0], box.stride[1], box.stride[2],
                      box.nb_components_per_pixel);
    GPU_LAUNCH_KERNEL(gpu_kernels::final_reduce_kernel, 1,
                      gpu_kernels::REDUCE_BLOCK_SIZE, d_re, num_blocks);
    GPU_LAUNCH_KERNEL(gpu_kernels::final_reduce_kernel, 1,
                      gpu_kernels::REDUCE_BLOCK_SIZE, d_im, num_blocks);

    Real re, im;
    GPU_MEMCPY_D2H(&re, d_re, sizeof(Real));
    GPU_MEMCPY_D2H(&im, d_im, sizeof(Real));
    GPU_FREE(d_re);
    GPU_FREE(d_im);
    return Complex{re, im};
}

/**
 * Interior squared norm sum |x|^2 for a complex field (ghosts excluded).
 */
Real interior_norm_sq_complex(const TypedField<Complex, DeviceSpace>& x,
                              const GlobalFieldCollection& coll) {
    const InteriorBox box = interior_box(x, coll);
    if (box.nb_interior_pixels <= 0) {
        return 0.0;
    }
    const int num_blocks =
        (box.nb_interior_pixels + gpu_kernels::REDUCE_BLOCK_SIZE - 1) /
        gpu_kernels::REDUCE_BLOCK_SIZE;
    Real* d_partial;
    GPU_MALLOC(&d_partial, num_blocks * sizeof(Real));

    GPU_LAUNCH_KERNEL(gpu_kernels::interior_norm_sq_complex_kernel,
                      num_blocks, gpu_kernels::REDUCE_BLOCK_SIZE,
                      reinterpret_cast<const Real*>(x.view().data()),
                      d_partial, box.extent[0], box.extent[1], box.extent[2],
                      box.start[0], box.start[1], box.start[2], box.stride_c,
                      box.stride[0], box.stride[1], box.stride[2],
                      box.nb_components_per_pixel);
    GPU_LAUNCH_KERNEL(gpu_kernels::final_reduce_kernel, 1,
                      gpu_kernels::REDUCE_BLOCK_SIZE, d_partial, num_blocks);

    Real result;
    GPU_MEMCPY_D2H(&result, d_partial, sizeof(Real));
    GPU_FREE(d_partial);
    return result;
}

}  // namespace

/* ---------------------------------------------------------------------- */
/* Public API implementations                                              */
/* ---------------------------------------------------------------------- */

template <>
Real vecdot<Real, DeviceSpace>(const TypedField<Real, DeviceSpace>& a,
                                const TypedField<Real, DeviceSpace>& b) {
    // Verify fields are compatible
    if (&a.get_collection() != &b.get_collection()) {
        throw FieldError("vecdot: fields must belong to the same collection");
    }
    if (a.get_nb_components() != b.get_nb_components()) {
        throw FieldError("vecdot: fields must have the same number of components");
    }
    if (a.get_nb_sub_pts() != b.get_nb_sub_pts()) {
        throw FieldError("vecdot: fields must have the same number of sub-points");
    }

    const auto& coll = a.get_collection();

    // For GlobalFieldCollection with ghosts, sum the interior directly;
    // full-buffer-minus-ghosts cancels catastrophically when the interior
    // values are small
    if (coll.get_domain() == FieldCollection::ValidityDomain::Global) {
        const auto& global_coll = static_cast<const GlobalFieldCollection&>(coll);
        if (has_ghosts(global_coll)) {
            return interior_dot(a, b, global_coll);
        }
    }

    // No ghosts: reduce the full buffer (get_nb_entries already counts
    // sub-points, so multiply only by the number of components)
    const Index_t n = a.get_nb_entries() * a.get_nb_components();
    const int num_blocks = (n + gpu_kernels::REDUCE_BLOCK_SIZE - 1) /
                           gpu_kernels::REDUCE_BLOCK_SIZE;
    Real* d_partial;
    GPU_MALLOC(&d_partial, num_blocks * sizeof(Real));

    GPU_LAUNCH_KERNEL(gpu_kernels::dot_reduce_kernel,
                      num_blocks, gpu_kernels::REDUCE_BLOCK_SIZE,
                      a.view().data(), b.view().data(), d_partial, n);

    GPU_LAUNCH_KERNEL(gpu_kernels::final_reduce_kernel,
                      1, gpu_kernels::REDUCE_BLOCK_SIZE,
                      d_partial, num_blocks);

    Real result;
    GPU_MEMCPY_D2H(&result, d_partial, sizeof(Real));
    GPU_FREE(d_partial);
    return result;
}

template <>
Real norm_sq<Real, DeviceSpace>(const TypedField<Real, DeviceSpace>& x) {
    const auto& coll = x.get_collection();

    // For GlobalFieldCollection with ghosts, sum the interior directly;
    // full-buffer-minus-ghosts cancels catastrophically when the interior
    // values are small
    if (coll.get_domain() == FieldCollection::ValidityDomain::Global) {
        const auto& global_coll = static_cast<const GlobalFieldCollection&>(coll);
        if (has_ghosts(global_coll)) {
            return interior_dot(x, x, global_coll);
        }
    }

    // No ghosts: reduce the full buffer (get_nb_entries already counts
    // sub-points, so multiply only by the number of components)
    const Index_t n = x.get_nb_entries() * x.get_nb_components();
    const int num_blocks = (n + gpu_kernels::REDUCE_BLOCK_SIZE - 1) /
                           gpu_kernels::REDUCE_BLOCK_SIZE;
    Real* d_partial;
    GPU_MALLOC(&d_partial, num_blocks * sizeof(Real));

    GPU_LAUNCH_KERNEL(gpu_kernels::norm_sq_reduce_kernel,
                      num_blocks, gpu_kernels::REDUCE_BLOCK_SIZE,
                      x.view().data(), d_partial, n);

    GPU_LAUNCH_KERNEL(gpu_kernels::final_reduce_kernel,
                      1, gpu_kernels::REDUCE_BLOCK_SIZE,
                      d_partial, num_blocks);

    Real result;
    GPU_MEMCPY_D2H(&result, d_partial, sizeof(Real));
    GPU_FREE(d_partial);
    return result;
}

template <>
void axpy<Real, DeviceSpace>(Real alpha,
                              const TypedField<Real, DeviceSpace>& x,
                              TypedField<Real, DeviceSpace>& y) {
    if (&x.get_collection() != &y.get_collection()) {
        throw FieldError("axpy: fields must belong to the same collection");
    }
    if (x.get_nb_entries() != y.get_nb_entries() ||
        x.get_nb_components() != y.get_nb_components()) {
        throw FieldError("axpy: fields must have the same number of entries");
    }

    // Total scalar elements in the full buffer (get_nb_entries already counts
    // sub-points, so multiply only by the number of components)
    const Index_t n = x.get_nb_entries() * x.get_nb_components();
    const int num_blocks = (n + gpu_kernels::BLOCK_SIZE - 1) /
                           gpu_kernels::BLOCK_SIZE;

    GPU_LAUNCH_KERNEL(gpu_kernels::axpy_kernel,
                      num_blocks, gpu_kernels::BLOCK_SIZE,
                      alpha, x.view().data(), y.view().data(), n);
    GPU_DEVICE_SYNCHRONIZE();
}

template <>
void scal<Real, DeviceSpace>(Real alpha, TypedField<Real, DeviceSpace>& x) {
    // Total scalar elements in the full buffer (get_nb_entries already counts
    // sub-points, so multiply only by the number of components)
    const Index_t n = x.get_nb_entries() * x.get_nb_components();
    const int num_blocks = (n + gpu_kernels::BLOCK_SIZE - 1) /
                           gpu_kernels::BLOCK_SIZE;

    GPU_LAUNCH_KERNEL(gpu_kernels::scal_kernel,
                      num_blocks, gpu_kernels::BLOCK_SIZE,
                      alpha, x.view().data(), n);
    GPU_DEVICE_SYNCHRONIZE();
}

template <>
void axpby<Real, DeviceSpace>(Real alpha,
                               const TypedField<Real, DeviceSpace>& x,
                               Real beta,
                               TypedField<Real, DeviceSpace>& y) {
    if (&x.get_collection() != &y.get_collection()) {
        throw FieldError("axpby: fields must belong to the same collection");
    }
    if (x.get_nb_entries() != y.get_nb_entries() ||
        x.get_nb_components() != y.get_nb_components()) {
        throw FieldError("axpby: fields must have the same number of entries");
    }

    // Total scalar elements in the full buffer (get_nb_entries already counts
    // sub-points, so multiply only by the number of components)
    const Index_t n = x.get_nb_entries() * x.get_nb_components();
    const int num_blocks = (n + gpu_kernels::BLOCK_SIZE - 1) /
                           gpu_kernels::BLOCK_SIZE;

    GPU_LAUNCH_KERNEL(gpu_kernels::axpby_kernel,
                      num_blocks, gpu_kernels::BLOCK_SIZE,
                      alpha, x.view().data(), beta, y.view().data(), n);
    GPU_DEVICE_SYNCHRONIZE();
}

template <>
void copy<Real, DeviceSpace>(const TypedField<Real, DeviceSpace>& src,
                              TypedField<Real, DeviceSpace>& dst) {
    if (&src.get_collection() != &dst.get_collection()) {
        throw FieldError("copy: fields must belong to the same collection");
    }
    if (src.get_nb_entries() != dst.get_nb_entries() ||
        src.get_nb_components() != dst.get_nb_components()) {
        throw FieldError("copy: fields must have the same number of entries");
    }

    // Total scalar elements in the full buffer (get_nb_entries already counts
    // sub-points, so multiply only by the number of components)
    const Index_t n = src.get_nb_entries() * src.get_nb_components();
    const int num_blocks = (n + gpu_kernels::BLOCK_SIZE - 1) /
                           gpu_kernels::BLOCK_SIZE;

    GPU_LAUNCH_KERNEL(gpu_kernels::copy_kernel,
                      num_blocks, gpu_kernels::BLOCK_SIZE,
                      src.view().data(), dst.view().data(), n);
    GPU_DEVICE_SYNCHRONIZE();
}

template <>
Real axpy_norm_sq<Real, DeviceSpace>(Real alpha,
                                      const TypedField<Real, DeviceSpace>& x,
                                      TypedField<Real, DeviceSpace>& y) {
    if (&x.get_collection() != &y.get_collection()) {
        throw FieldError("axpy_norm_sq: fields must belong to the same collection");
    }
    if (x.get_nb_entries() != y.get_nb_entries() ||
        x.get_nb_components() != y.get_nb_components()) {
        throw FieldError("axpy_norm_sq: fields must have the same number of entries");
    }

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
            GPU_LAUNCH_KERNEL(gpu_kernels::axpy_kernel,
                              num_blocks, gpu_kernels::BLOCK_SIZE,
                              alpha, x.view().data(), y.view().data(), n);
            // Kernels on the default stream serialize, so the interior
            // reduction sees the updated y
            return interior_dot(y, y, global_coll);
        }
    }

    // No ghosts: fused single-pass AXPY + norm_sq
    const int num_blocks = (n + gpu_kernels::REDUCE_BLOCK_SIZE - 1) /
                           gpu_kernels::REDUCE_BLOCK_SIZE;
    Real* d_partial;
    GPU_MALLOC(&d_partial, num_blocks * sizeof(Real));

    GPU_LAUNCH_KERNEL(gpu_kernels::axpy_norm_sq_kernel,
                      num_blocks, gpu_kernels::REDUCE_BLOCK_SIZE,
                      alpha, x.view().data(), y.view().data(), d_partial, n);

    GPU_LAUNCH_KERNEL(gpu_kernels::final_reduce_kernel,
                      1, gpu_kernels::REDUCE_BLOCK_SIZE,
                      d_partial, num_blocks);

    Real result;
    GPU_MEMCPY_D2H(&result, d_partial, sizeof(Real));
    GPU_FREE(d_partial);
    return result;
}

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
    GPU_DEVICE_SYNCHRONIZE();
}

/* ---------------------------------------------------------------------- */
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
    GPU_DEVICE_SYNCHRONIZE();
}

/* ---------------------------------------------------------------------- */
template <>
void cross<Real, DeviceSpace>(const TypedField<Real, DeviceSpace>& a,
                              const TypedField<Real, DeviceSpace>& b,
                              TypedField<Real, DeviceSpace>& out) {
    const auto& coll = a.get_collection();
    internal::check_three_vector("cross", a, coll);
    internal::check_three_vector("cross", b, coll);
    internal::check_three_vector("cross", out, coll);
    if (out.view().data() == a.view().data() ||
        out.view().data() == b.view().data()) {
        throw FieldError(
            "cross: output must be a field distinct from both inputs");
    }
    const Index_t npix = a.get_nb_entries();
    const bool soa =
        (out.get_storage_order() == StorageOrder::StructureOfArrays);
    const int num_blocks =
        (npix + gpu_kernels::BLOCK_SIZE - 1) / gpu_kernels::BLOCK_SIZE;
    GPU_LAUNCH_KERNEL(gpu_kernels::cross_real_kernel, num_blocks,
                      gpu_kernels::BLOCK_SIZE, a.view().data(), b.view().data(),
                      out.view().data(), npix, soa);
    GPU_DEVICE_SYNCHRONIZE();
}

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
    GPU_DEVICE_SYNCHRONIZE();
}

/* ---------------------------------------------------------------------- */
/* Complex device operations                                               */
/*                                                                         */
/* The complex buffers are addressed as their underlying reals; scalar     */
/* complex coefficients are split into (real, imag) for the kernels.       */
/* Reductions exclude ghost regions for GlobalFieldCollections, matching   */
/* the Real implementations and host semantics.                            */
/* ---------------------------------------------------------------------- */

template <>
Complex vecdot<Complex, DeviceSpace>(const TypedField<Complex, DeviceSpace>& a,
                                      const TypedField<Complex, DeviceSpace>& b) {
    if (&a.get_collection() != &b.get_collection()) {
        throw FieldError("vecdot: fields must belong to the same collection");
    }
    if (a.get_nb_components() != b.get_nb_components()) {
        throw FieldError("vecdot: fields must have the same number of components");
    }
    if (a.get_nb_sub_pts() != b.get_nb_sub_pts()) {
        throw FieldError("vecdot: fields must have the same number of sub-points");
    }

    const auto& coll = a.get_collection();
    if (coll.get_domain() == FieldCollection::ValidityDomain::Global) {
        const auto& global_coll = static_cast<const GlobalFieldCollection&>(coll);
        if (has_ghosts(global_coll)) {
            return interior_dot_complex(a, b, global_coll);
        }
    }

    // No ghosts: reduce the full buffer (n complex elements)
    const Index_t n = a.get_nb_entries() * a.get_nb_components();
    const int num_blocks = (n + gpu_kernels::REDUCE_BLOCK_SIZE - 1) /
                           gpu_kernels::REDUCE_BLOCK_SIZE;
    Real* d_re;
    Real* d_im;
    GPU_MALLOC(&d_re, num_blocks * sizeof(Real));
    GPU_MALLOC(&d_im, num_blocks * sizeof(Real));

    GPU_LAUNCH_KERNEL(gpu_kernels::dot_reduce_complex_kernel,
                      num_blocks, gpu_kernels::REDUCE_BLOCK_SIZE,
                      reinterpret_cast<const Real*>(a.view().data()),
                      reinterpret_cast<const Real*>(b.view().data()),
                      d_re, d_im, n);
    GPU_LAUNCH_KERNEL(gpu_kernels::final_reduce_kernel, 1,
                      gpu_kernels::REDUCE_BLOCK_SIZE, d_re, num_blocks);
    GPU_LAUNCH_KERNEL(gpu_kernels::final_reduce_kernel, 1,
                      gpu_kernels::REDUCE_BLOCK_SIZE, d_im, num_blocks);

    Real re, im;
    GPU_MEMCPY_D2H(&re, d_re, sizeof(Real));
    GPU_MEMCPY_D2H(&im, d_im, sizeof(Real));
    GPU_FREE(d_re);
    GPU_FREE(d_im);
    return Complex{re, im};
}

template <>
Complex norm_sq<Complex, DeviceSpace>(const TypedField<Complex, DeviceSpace>& x) {
    const auto& coll = x.get_collection();
    if (coll.get_domain() == FieldCollection::ValidityDomain::Global) {
        const auto& global_coll = static_cast<const GlobalFieldCollection&>(coll);
        if (has_ghosts(global_coll)) {
            return Complex{interior_norm_sq_complex(x, global_coll), 0.0};
        }
    }

    // No ghosts: sum |x|^2 over the full buffer. The squared magnitude sums to
    // the squared norm of the underlying 2n reals, so reuse the real kernel.
    const Index_t n2 = x.get_nb_entries() * x.get_nb_components() * 2;
    const int num_blocks = (n2 + gpu_kernels::REDUCE_BLOCK_SIZE - 1) /
                           gpu_kernels::REDUCE_BLOCK_SIZE;
    Real* d_partial;
    GPU_MALLOC(&d_partial, num_blocks * sizeof(Real));

    GPU_LAUNCH_KERNEL(gpu_kernels::norm_sq_reduce_kernel,
                      num_blocks, gpu_kernels::REDUCE_BLOCK_SIZE,
                      reinterpret_cast<const Real*>(x.view().data()),
                      d_partial, n2);
    GPU_LAUNCH_KERNEL(gpu_kernels::final_reduce_kernel, 1,
                      gpu_kernels::REDUCE_BLOCK_SIZE, d_partial, num_blocks);

    Real result;
    GPU_MEMCPY_D2H(&result, d_partial, sizeof(Real));
    GPU_FREE(d_partial);
    return Complex{result, 0.0};
}

template <>
void axpy<Complex, DeviceSpace>(Complex alpha,
                                 const TypedField<Complex, DeviceSpace>& x,
                                 TypedField<Complex, DeviceSpace>& y) {
    if (&x.get_collection() != &y.get_collection()) {
        throw FieldError("axpy: fields must belong to the same collection");
    }
    if (x.get_nb_entries() != y.get_nb_entries() ||
        x.get_nb_components() != y.get_nb_components()) {
        throw FieldError("axpy: fields must have the same number of entries");
    }
    const Index_t n = x.get_nb_entries() * x.get_nb_components();
    const int num_blocks = (n + gpu_kernels::BLOCK_SIZE - 1) /
                           gpu_kernels::BLOCK_SIZE;
    GPU_LAUNCH_KERNEL(gpu_kernels::axpy_complex_scalar_kernel,
                      num_blocks, gpu_kernels::BLOCK_SIZE,
                      alpha.real(), alpha.imag(),
                      reinterpret_cast<const Real*>(x.view().data()),
                      reinterpret_cast<Real*>(y.view().data()), n);
    GPU_DEVICE_SYNCHRONIZE();
}

template <>
void scal<Complex, DeviceSpace>(Complex alpha,
                                 TypedField<Complex, DeviceSpace>& x) {
    const Index_t n = x.get_nb_entries() * x.get_nb_components();
    const int num_blocks = (n + gpu_kernels::BLOCK_SIZE - 1) /
                           gpu_kernels::BLOCK_SIZE;
    GPU_LAUNCH_KERNEL(gpu_kernels::scal_complex_scalar_kernel,
                      num_blocks, gpu_kernels::BLOCK_SIZE,
                      alpha.real(), alpha.imag(),
                      reinterpret_cast<Real*>(x.view().data()), n);
    GPU_DEVICE_SYNCHRONIZE();
}

template <>
void axpby<Complex, DeviceSpace>(Complex alpha,
                                  const TypedField<Complex, DeviceSpace>& x,
                                  Complex beta,
                                  TypedField<Complex, DeviceSpace>& y) {
    if (&x.get_collection() != &y.get_collection()) {
        throw FieldError("axpby: fields must belong to the same collection");
    }
    if (x.get_nb_entries() != y.get_nb_entries() ||
        x.get_nb_components() != y.get_nb_components()) {
        throw FieldError("axpby: fields must have the same number of entries");
    }
    const Index_t n = x.get_nb_entries() * x.get_nb_components();
    const int num_blocks = (n + gpu_kernels::BLOCK_SIZE - 1) /
                           gpu_kernels::BLOCK_SIZE;
    GPU_LAUNCH_KERNEL(gpu_kernels::axpby_complex_scalar_kernel,
                      num_blocks, gpu_kernels::BLOCK_SIZE,
                      alpha.real(), alpha.imag(),
                      reinterpret_cast<const Real*>(x.view().data()),
                      beta.real(), beta.imag(),
                      reinterpret_cast<Real*>(y.view().data()), n);
    GPU_DEVICE_SYNCHRONIZE();
}

template <>
void copy<Complex, DeviceSpace>(const TypedField<Complex, DeviceSpace>& src,
                                 TypedField<Complex, DeviceSpace>& dst) {
    if (&src.get_collection() != &dst.get_collection()) {
        throw FieldError("copy: fields must belong to the same collection");
    }
    if (src.get_nb_entries() != dst.get_nb_entries() ||
        src.get_nb_components() != dst.get_nb_components()) {
        throw FieldError("copy: fields must have the same number of entries");
    }
    // dst = src is a plain bitwise copy of the underlying 2n reals.
    const Index_t n2 = src.get_nb_entries() * src.get_nb_components() * 2;
    const int num_blocks = (n2 + gpu_kernels::BLOCK_SIZE - 1) /
                           gpu_kernels::BLOCK_SIZE;
    GPU_LAUNCH_KERNEL(gpu_kernels::copy_kernel,
                      num_blocks, gpu_kernels::BLOCK_SIZE,
                      reinterpret_cast<const Real*>(src.view().data()),
                      reinterpret_cast<Real*>(dst.view().data()), n2);
    GPU_DEVICE_SYNCHRONIZE();
}

template <>
Complex axpy_norm_sq<Complex, DeviceSpace>(Complex alpha,
                                            const TypedField<Complex, DeviceSpace>& x,
                                            TypedField<Complex, DeviceSpace>& y) {
    if (&x.get_collection() != &y.get_collection()) {
        throw FieldError("axpy_norm_sq: fields must belong to the same collection");
    }
    if (x.get_nb_entries() != y.get_nb_entries() ||
        x.get_nb_components() != y.get_nb_components()) {
        throw FieldError("axpy_norm_sq: fields must have the same number of entries");
    }
    const auto& coll = x.get_collection();
    const Index_t n = x.get_nb_entries() * x.get_nb_components();

    // With ghosts: update y on the full buffer, then sum |y|^2 over the
    // interior directly (full-buffer-minus-ghosts cancels catastrophically).
    if (coll.get_domain() == FieldCollection::ValidityDomain::Global) {
        const auto& global_coll = static_cast<const GlobalFieldCollection&>(coll);
        if (has_ghosts(global_coll)) {
            const int num_blocks = (n + gpu_kernels::BLOCK_SIZE - 1) /
                                   gpu_kernels::BLOCK_SIZE;
            GPU_LAUNCH_KERNEL(gpu_kernels::axpy_complex_scalar_kernel,
                              num_blocks, gpu_kernels::BLOCK_SIZE,
                              alpha.real(), alpha.imag(),
                              reinterpret_cast<const Real*>(x.view().data()),
                              reinterpret_cast<Real*>(y.view().data()), n);
            return Complex{interior_norm_sq_complex(y, global_coll), 0.0};
        }
    }

    // No ghosts: fused single-pass complex AXPY + squared norm.
    const int num_blocks = (n + gpu_kernels::REDUCE_BLOCK_SIZE - 1) /
                           gpu_kernels::REDUCE_BLOCK_SIZE;
    Real* d_partial;
    GPU_MALLOC(&d_partial, num_blocks * sizeof(Real));

    GPU_LAUNCH_KERNEL(gpu_kernels::axpy_norm_sq_complex_kernel,
                      num_blocks, gpu_kernels::REDUCE_BLOCK_SIZE,
                      alpha.real(), alpha.imag(),
                      reinterpret_cast<const Real*>(x.view().data()),
                      reinterpret_cast<Real*>(y.view().data()), d_partial, n);
    GPU_LAUNCH_KERNEL(gpu_kernels::final_reduce_kernel, 1,
                      gpu_kernels::REDUCE_BLOCK_SIZE, d_partial, num_blocks);

    Real result;
    GPU_MEMCPY_D2H(&result, d_partial, sizeof(Real));
    GPU_FREE(d_partial);
    return Complex{result, 0.0};
}

template <>
void cross<Complex, DeviceSpace>(const TypedField<Complex, DeviceSpace>& a,
                                 const TypedField<Complex, DeviceSpace>& b,
                                 TypedField<Complex, DeviceSpace>& out) {
    const auto& coll = a.get_collection();
    internal::check_three_vector("cross", a, coll);
    internal::check_three_vector("cross", b, coll);
    internal::check_three_vector("cross", out, coll);
    if (out.view().data() == a.view().data() ||
        out.view().data() == b.view().data()) {
        throw FieldError(
            "cross: output must be a field distinct from both inputs");
    }
    const Index_t npix = a.get_nb_entries();
    const bool soa =
        (out.get_storage_order() == StorageOrder::StructureOfArrays);
    const int num_blocks =
        (npix + gpu_kernels::BLOCK_SIZE - 1) / gpu_kernels::BLOCK_SIZE;
    GPU_LAUNCH_KERNEL(gpu_kernels::cross_complex_kernel, num_blocks,
                      gpu_kernels::BLOCK_SIZE,
                      reinterpret_cast<const Real*>(a.view().data()),
                      reinterpret_cast<const Real*>(b.view().data()),
                      reinterpret_cast<Real*>(out.view().data()), npix, soa);
    GPU_DEVICE_SYNCHRONIZE();
}

}  // namespace linalg
}  // namespace muGrid
