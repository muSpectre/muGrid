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

// Unified GPU abstraction macros
#if defined(MUGRID_ENABLE_CUDA)
    #include <cuda_runtime.h>
    #define GPU_LAUNCH_KERNEL(kernel, grid, block, ...) \
        kernel<<<grid, block>>>(__VA_ARGS__)
    #define GPU_DEVICE_SYNCHRONIZE() (void)cudaDeviceSynchronize()
    #define GPU_MALLOC(ptr, size) (void)cudaMalloc(ptr, size)
    #define GPU_FREE(ptr) (void)cudaFree(ptr)
    #define GPU_MEMCPY_D2H(dst, src, size) \
        (void)cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost)
    #define GPU_MEMSET(ptr, value, size) (void)cudaMemset(ptr, value, size)
    using DeviceSpace = muGrid::CUDASpace;
#elif defined(MUGRID_ENABLE_HIP)
    #include <hip/hip_runtime.h>
    #define GPU_LAUNCH_KERNEL(kernel, grid, block, ...) \
        hipLaunchKernelGGL(kernel, grid, block, 0, 0, __VA_ARGS__)
    #define GPU_DEVICE_SYNCHRONIZE() (void)hipDeviceSynchronize()
    #define GPU_MALLOC(ptr, size) (void)hipMalloc(ptr, size)
    #define GPU_FREE(ptr) (void)hipFree(ptr)
    #define GPU_MEMCPY_D2H(dst, src, size) \
        (void)hipMemcpy(dst, src, size, hipMemcpyDeviceToHost)
    #define GPU_MEMSET(ptr, value, size) (void)hipMemset(ptr, value, size)
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

// Complex versions would follow the same pattern but with Complex type
// For now, provide stubs that throw

template <>
Complex vecdot<Complex, DeviceSpace>(const TypedField<Complex, DeviceSpace>& a,
                                      const TypedField<Complex, DeviceSpace>& b) {
    throw FieldError("Complex GPU linalg not yet implemented");
}

template <>
Complex norm_sq<Complex, DeviceSpace>(const TypedField<Complex, DeviceSpace>& x) {
    throw FieldError("Complex GPU linalg not yet implemented");
}

template <>
void axpy<Complex, DeviceSpace>(Complex alpha,
                                 const TypedField<Complex, DeviceSpace>& x,
                                 TypedField<Complex, DeviceSpace>& y) {
    throw FieldError("Complex GPU linalg not yet implemented");
}

template <>
void scal<Complex, DeviceSpace>(Complex alpha,
                                 TypedField<Complex, DeviceSpace>& x) {
    throw FieldError("Complex GPU linalg not yet implemented");
}

template <>
void axpby<Complex, DeviceSpace>(Complex alpha,
                                  const TypedField<Complex, DeviceSpace>& x,
                                  Complex beta,
                                  TypedField<Complex, DeviceSpace>& y) {
    throw FieldError("Complex GPU linalg not yet implemented");
}

template <>
void copy<Complex, DeviceSpace>(const TypedField<Complex, DeviceSpace>& src,
                                 TypedField<Complex, DeviceSpace>& dst) {
    throw FieldError("Complex GPU linalg not yet implemented");
}

template <>
Complex axpy_norm_sq<Complex, DeviceSpace>(Complex alpha,
                                            const TypedField<Complex, DeviceSpace>& x,
                                            TypedField<Complex, DeviceSpace>& y) {
    throw FieldError("Complex GPU linalg not yet implemented");
}

}  // namespace linalg
}  // namespace muGrid
