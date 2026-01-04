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
    using DeviceSpace = muGrid::CudaSpace;
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
    using DeviceSpace = muGrid::HIPSpace;
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

/* ---------------------------------------------------------------------- */
/* Ghost region kernels for subtraction                                    */
/* ---------------------------------------------------------------------- */

/**
 * Compute dot product over ghost region only (2D).
 * Ghost region = left/right columns + top/bottom rows (excluding corners).
 * Uses field data strides directly for correct SoA layout handling.
 */
__global__ void ghost_dot_2d_kernel(
    const Real* a, const Real* b, Real* partial_sums,
    Index_t nx_total, Index_t ny_total,
    Index_t gx_left, Index_t gx_right,
    Index_t gy_left, Index_t gy_right,
    Index_t field_stride_c, Index_t field_stride_x, Index_t field_stride_y,
    Index_t nb_components) {

    __shared__ Real shared_data[REDUCE_BLOCK_SIZE];

    Index_t tid = threadIdx.x;
    Index_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Interior bounds
    Index_t x_start = gx_left;
    Index_t x_end = nx_total - gx_right;
    Index_t y_end = ny_total - gy_right;

    // Count total ghost pixels
    Index_t total_ghost_pixels =
        gx_left * ny_total +                          // Left columns
        gx_right * ny_total +                         // Right columns
        (x_end - x_start) * gy_left +                 // Top rows (excl corners)
        (x_end - x_start) * gy_right;                 // Bottom rows (excl corners)

    Real sum = 0.0;

    // Each thread processes multiple ghost pixels
    for (Index_t ghost_idx = global_tid; ghost_idx < total_ghost_pixels;
         ghost_idx += blockDim.x * gridDim.x) {

        Index_t ix, iy;

        // Map ghost_idx to (ix, iy) - this is the tricky part
        if (ghost_idx < gx_left * ny_total) {
            // Left columns
            ix = ghost_idx / ny_total;
            iy = ghost_idx % ny_total;
        } else if (ghost_idx < (gx_left + gx_right) * ny_total) {
            // Right columns
            Index_t local_idx = ghost_idx - gx_left * ny_total;
            ix = nx_total - gx_right + local_idx / ny_total;
            iy = local_idx % ny_total;
        } else if (ghost_idx < (gx_left + gx_right) * ny_total +
                               (x_end - x_start) * gy_left) {
            // Top rows (excluding corners)
            Index_t local_idx = ghost_idx - (gx_left + gx_right) * ny_total;
            ix = x_start + local_idx / gy_left;
            iy = local_idx % gy_left;
        } else {
            // Bottom rows (excluding corners)
            Index_t local_idx = ghost_idx - (gx_left + gx_right) * ny_total -
                                (x_end - x_start) * gy_left;
            ix = x_start + local_idx / gy_right;
            iy = y_end + local_idx % gy_right;
        }

        // Use field strides for correct SoA layout handling
        // field[c, x, y] = base + c * field_stride_c + x * field_stride_x + y * field_stride_y
        for (Index_t c = 0; c < nb_components; ++c) {
            Index_t offset = c * field_stride_c + ix * field_stride_x + iy * field_stride_y;
            sum += a[offset] * b[offset];
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
 * Compute squared norm over ghost region only (2D).
 * Uses field data strides directly for correct SoA layout handling.
 */
__global__ void ghost_norm_sq_2d_kernel(
    const Real* x, Real* partial_sums,
    Index_t nx_total, Index_t ny_total,
    Index_t gx_left, Index_t gx_right,
    Index_t gy_left, Index_t gy_right,
    Index_t field_stride_c, Index_t field_stride_x, Index_t field_stride_y,
    Index_t nb_components) {

    __shared__ Real shared_data[REDUCE_BLOCK_SIZE];

    Index_t tid = threadIdx.x;
    Index_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Interior bounds
    Index_t x_start = gx_left;
    Index_t x_end = nx_total - gx_right;

    // Count total ghost pixels
    Index_t total_ghost_pixels =
        gx_left * ny_total +
        gx_right * ny_total +
        (x_end - x_start) * gy_left +
        (x_end - x_start) * gy_right;

    Real sum = 0.0;

    for (Index_t ghost_idx = global_tid; ghost_idx < total_ghost_pixels;
         ghost_idx += blockDim.x * gridDim.x) {

        Index_t ix, iy;
        Index_t y_end = ny_total - gy_right;

        if (ghost_idx < gx_left * ny_total) {
            ix = ghost_idx / ny_total;
            iy = ghost_idx % ny_total;
        } else if (ghost_idx < (gx_left + gx_right) * ny_total) {
            Index_t local_idx = ghost_idx - gx_left * ny_total;
            ix = nx_total - gx_right + local_idx / ny_total;
            iy = local_idx % ny_total;
        } else if (ghost_idx < (gx_left + gx_right) * ny_total +
                               (x_end - x_start) * gy_left) {
            Index_t local_idx = ghost_idx - (gx_left + gx_right) * ny_total;
            ix = x_start + local_idx / gy_left;
            iy = local_idx % gy_left;
        } else {
            Index_t local_idx = ghost_idx - (gx_left + gx_right) * ny_total -
                                (x_end - x_start) * gy_left;
            ix = x_start + local_idx / gy_right;
            iy = y_end + local_idx % gy_right;
        }

        // Use field strides for correct SoA layout handling
        // field[c, x, y] = base + c * field_stride_c + x * field_stride_x + y * field_stride_y
        for (Index_t c = 0; c < nb_components; ++c) {
            Index_t offset = c * field_stride_c + ix * field_stride_x + iy * field_stride_y;
            Real val = x[offset];
            sum += val * val;
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

/**
 * Compute dot product over ghost region only (3D).
 * Ghost region = all pixels outside the interior box.
 * Uses a simple approach: iterate all pixels, skip interior ones.
 * Uses field strides for correct SoA layout handling.
 */
__global__ void ghost_dot_3d_kernel(
    const Real* a, const Real* b, Real* partial_sums,
    Index_t nx_total, Index_t ny_total, Index_t nz_total,
    Index_t gx_left, Index_t gx_right,
    Index_t gy_left, Index_t gy_right,
    Index_t gz_left, Index_t gz_right,
    Index_t field_stride_c, Index_t field_stride_x,
    Index_t field_stride_y, Index_t field_stride_z,
    Index_t nb_components) {

    __shared__ Real shared_data[REDUCE_BLOCK_SIZE];

    Index_t tid = threadIdx.x;
    Index_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Interior bounds
    Index_t x_start = gx_left;
    Index_t x_end = nx_total - gx_right;
    Index_t y_start = gy_left;
    Index_t y_end = ny_total - gy_right;
    Index_t z_start = gz_left;
    Index_t z_end = nz_total - gz_right;

    // Total pixels in the domain
    Index_t total_pixels = nx_total * ny_total * nz_total;

    Real sum = 0.0;

    // Each thread processes multiple pixels
    for (Index_t pixel_idx = global_tid; pixel_idx < total_pixels;
         pixel_idx += blockDim.x * gridDim.x) {

        Index_t ix = pixel_idx / (ny_total * nz_total);
        Index_t rem = pixel_idx % (ny_total * nz_total);
        Index_t iy = rem / nz_total;
        Index_t iz = rem % nz_total;

        // Skip interior pixels
        if (ix >= x_start && ix < x_end &&
            iy >= y_start && iy < y_end &&
            iz >= z_start && iz < z_end) {
            continue;
        }

        // Use field strides for correct SoA layout handling
        // field[c, x, y, z] = base + c * stride_c + x * stride_x + y * stride_y + z * stride_z
        for (Index_t c = 0; c < nb_components; ++c) {
            Index_t offset = c * field_stride_c +
                             ix * field_stride_x +
                             iy * field_stride_y +
                             iz * field_stride_z;
            sum += a[offset] * b[offset];
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
 * Compute squared norm over ghost region only (3D).
 * Uses a simple approach: iterate all pixels, skip interior ones.
 * Uses field strides for correct SoA layout handling.
 */
__global__ void ghost_norm_sq_3d_kernel(
    const Real* x, Real* partial_sums,
    Index_t nx_total, Index_t ny_total, Index_t nz_total,
    Index_t gx_left, Index_t gx_right,
    Index_t gy_left, Index_t gy_right,
    Index_t gz_left, Index_t gz_right,
    Index_t field_stride_c, Index_t field_stride_x,
    Index_t field_stride_y, Index_t field_stride_z,
    Index_t nb_components) {

    __shared__ Real shared_data[REDUCE_BLOCK_SIZE];

    Index_t tid = threadIdx.x;
    Index_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Interior bounds
    Index_t x_start = gx_left;
    Index_t x_end = nx_total - gx_right;
    Index_t y_start = gy_left;
    Index_t y_end = ny_total - gy_right;
    Index_t z_start = gz_left;
    Index_t z_end = nz_total - gz_right;

    // Total pixels in the domain
    Index_t total_pixels = nx_total * ny_total * nz_total;

    Real sum = 0.0;

    for (Index_t pixel_idx = global_tid; pixel_idx < total_pixels;
         pixel_idx += blockDim.x * gridDim.x) {

        Index_t ix = pixel_idx / (ny_total * nz_total);
        Index_t rem = pixel_idx % (ny_total * nz_total);
        Index_t iy = rem / nz_total;
        Index_t iz = rem % nz_total;

        // Skip interior pixels
        if (ix >= x_start && ix < x_end &&
            iy >= y_start && iy < y_end &&
            iz >= z_start && iz < z_end) {
            continue;
        }

        // Use field strides for correct SoA layout handling
        // field[c, x, y, z] = base + c * stride_c + x * stride_x + y * stride_y + z * stride_z
        for (Index_t c = 0; c < nb_components; ++c) {
            Index_t offset = c * field_stride_c +
                             ix * field_stride_x +
                             iy * field_stride_y +
                             iz * field_stride_z;
            Real val = x[offset];
            sum += val * val;
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

}  // namespace gpu_kernels

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
    // Total number of scalar elements = pixels * components * sub_pts
    const Index_t nb_components_per_pixel =
        a.get_nb_components() * a.get_nb_sub_pts();
    const Index_t n = a.get_nb_entries() * nb_components_per_pixel;

    // Allocate device memory for partial sums
    const int num_blocks = (n + gpu_kernels::REDUCE_BLOCK_SIZE - 1) /
                           gpu_kernels::REDUCE_BLOCK_SIZE;
    Real* d_partial;
    GPU_MALLOC(&d_partial, num_blocks * sizeof(Real));

    // Full buffer reduction
    GPU_LAUNCH_KERNEL(gpu_kernels::dot_reduce_kernel,
                      num_blocks, gpu_kernels::REDUCE_BLOCK_SIZE,
                      a.view().data(), b.view().data(), d_partial, n);

    // Final reduction
    GPU_LAUNCH_KERNEL(gpu_kernels::final_reduce_kernel,
                      1, gpu_kernels::REDUCE_BLOCK_SIZE,
                      d_partial, num_blocks);

    // Copy result back
    Real full_dot;
    GPU_MEMCPY_D2H(&full_dot, d_partial, sizeof(Real));

    // For GlobalFieldCollection, subtract ghost contributions
    if (coll.get_domain() == FieldCollection::ValidityDomain::Global) {
        const auto& global_coll = static_cast<const GlobalFieldCollection&>(coll);
        const auto spatial_dim = global_coll.get_spatial_dim();

        if (spatial_dim == 2) {
            const auto& nb_pts = global_coll.get_nb_subdomain_grid_pts_with_ghosts();
            const auto& nb_ghosts_left = global_coll.get_nb_ghosts_left();
            const auto& nb_ghosts_right = global_coll.get_nb_ghosts_right();
            // Get field data strides (for SoA layout)
            const auto field_strides = a.get_strides(IterUnit::Pixel);
            const Index_t field_stride_c = field_strides[0];
            const Index_t field_stride_x = field_strides[field_strides.size() - 2];
            const Index_t field_stride_y = field_strides[field_strides.size() - 1];

            // Count ghost pixels for block sizing
            Index_t x_start = nb_ghosts_left[0];
            Index_t x_end = nb_pts[0] - nb_ghosts_right[0];
            Index_t ghost_pixels =
                nb_ghosts_left[0] * nb_pts[1] +
                nb_ghosts_right[0] * nb_pts[1] +
                (x_end - x_start) * nb_ghosts_left[1] +
                (x_end - x_start) * nb_ghosts_right[1];

            if (ghost_pixels > 0) {
                int ghost_blocks = (ghost_pixels + gpu_kernels::REDUCE_BLOCK_SIZE - 1) /
                                   gpu_kernels::REDUCE_BLOCK_SIZE;

                GPU_LAUNCH_KERNEL(gpu_kernels::ghost_dot_2d_kernel,
                                  ghost_blocks, gpu_kernels::REDUCE_BLOCK_SIZE,
                                  a.view().data(), b.view().data(), d_partial,
                                  nb_pts[0], nb_pts[1],
                                  nb_ghosts_left[0], nb_ghosts_right[0],
                                  nb_ghosts_left[1], nb_ghosts_right[1],
                                  field_stride_c, field_stride_x, field_stride_y,
                                  nb_components_per_pixel);

                GPU_LAUNCH_KERNEL(gpu_kernels::final_reduce_kernel,
                                  1, gpu_kernels::REDUCE_BLOCK_SIZE,
                                  d_partial, ghost_blocks);

                Real ghost_dot;
                GPU_MEMCPY_D2H(&ghost_dot, d_partial, sizeof(Real));
                full_dot -= ghost_dot;
            }
        } else if (spatial_dim == 3) {
            const auto& nb_pts = global_coll.get_nb_subdomain_grid_pts_with_ghosts();
            const auto& nb_ghosts_left = global_coll.get_nb_ghosts_left();
            const auto& nb_ghosts_right = global_coll.get_nb_ghosts_right();
            // Get field data strides (for SoA layout)
            const auto field_strides = a.get_strides(IterUnit::Pixel);
            const Index_t field_stride_c = field_strides[0];
            const Index_t field_stride_x = field_strides[field_strides.size() - 3];
            const Index_t field_stride_y = field_strides[field_strides.size() - 2];
            const Index_t field_stride_z = field_strides[field_strides.size() - 1];

            // Total pixels for block count
            Index_t total_pixels = nb_pts[0] * nb_pts[1] * nb_pts[2];
            int ghost_blocks = (total_pixels + gpu_kernels::REDUCE_BLOCK_SIZE - 1) /
                               gpu_kernels::REDUCE_BLOCK_SIZE;

            GPU_LAUNCH_KERNEL(gpu_kernels::ghost_dot_3d_kernel,
                              ghost_blocks, gpu_kernels::REDUCE_BLOCK_SIZE,
                              a.view().data(), b.view().data(), d_partial,
                              nb_pts[0], nb_pts[1], nb_pts[2],
                              nb_ghosts_left[0], nb_ghosts_right[0],
                              nb_ghosts_left[1], nb_ghosts_right[1],
                              nb_ghosts_left[2], nb_ghosts_right[2],
                              field_stride_c, field_stride_x,
                              field_stride_y, field_stride_z,
                              nb_components_per_pixel);

            GPU_LAUNCH_KERNEL(gpu_kernels::final_reduce_kernel,
                              1, gpu_kernels::REDUCE_BLOCK_SIZE,
                              d_partial, ghost_blocks);

            Real ghost_dot;
            GPU_MEMCPY_D2H(&ghost_dot, d_partial, sizeof(Real));
            full_dot -= ghost_dot;
        }
    }

    GPU_FREE(d_partial);
    return full_dot;
}

template <>
Real norm_sq<Real, DeviceSpace>(const TypedField<Real, DeviceSpace>& x) {
    const auto& coll = x.get_collection();
    // Total number of scalar elements = pixels * components * sub_pts
    const Index_t nb_components_per_pixel =
        x.get_nb_components() * x.get_nb_sub_pts();
    const Index_t n = x.get_nb_entries() * nb_components_per_pixel;

    // Allocate device memory for partial sums
    const int num_blocks = (n + gpu_kernels::REDUCE_BLOCK_SIZE - 1) /
                           gpu_kernels::REDUCE_BLOCK_SIZE;
    Real* d_partial;
    GPU_MALLOC(&d_partial, num_blocks * sizeof(Real));

    // Full buffer reduction
    GPU_LAUNCH_KERNEL(gpu_kernels::norm_sq_reduce_kernel,
                      num_blocks, gpu_kernels::REDUCE_BLOCK_SIZE,
                      x.view().data(), d_partial, n);

    // Final reduction
    GPU_LAUNCH_KERNEL(gpu_kernels::final_reduce_kernel,
                      1, gpu_kernels::REDUCE_BLOCK_SIZE,
                      d_partial, num_blocks);

    // Copy result back
    Real full_norm;
    GPU_MEMCPY_D2H(&full_norm, d_partial, sizeof(Real));

    // For GlobalFieldCollection, subtract ghost contributions
    if (coll.get_domain() == FieldCollection::ValidityDomain::Global) {
        const auto& global_coll = static_cast<const GlobalFieldCollection&>(coll);
        const auto spatial_dim = global_coll.get_spatial_dim();

        if (spatial_dim == 2) {
            const auto& nb_pts = global_coll.get_nb_subdomain_grid_pts_with_ghosts();
            const auto& nb_ghosts_left = global_coll.get_nb_ghosts_left();
            const auto& nb_ghosts_right = global_coll.get_nb_ghosts_right();
            // Get field data strides (for SoA layout: [comp_stride, sub_pt_stride, x_stride, y_stride])
            const auto field_strides = x.get_strides(IterUnit::Pixel);
            // Component stride is first, spatial strides come after component and sub_pt dimensions
            const Index_t field_stride_c = field_strides[0];  // component stride
            const Index_t field_stride_x = field_strides[field_strides.size() - 2];  // x stride
            const Index_t field_stride_y = field_strides[field_strides.size() - 1];  // y stride

            Index_t x_start = nb_ghosts_left[0];
            Index_t x_end = nb_pts[0] - nb_ghosts_right[0];
            Index_t ghost_pixels =
                nb_ghosts_left[0] * nb_pts[1] +
                nb_ghosts_right[0] * nb_pts[1] +
                (x_end - x_start) * nb_ghosts_left[1] +
                (x_end - x_start) * nb_ghosts_right[1];

            if (ghost_pixels > 0) {
                int ghost_blocks = (ghost_pixels + gpu_kernels::REDUCE_BLOCK_SIZE - 1) /
                                   gpu_kernels::REDUCE_BLOCK_SIZE;

                GPU_LAUNCH_KERNEL(gpu_kernels::ghost_norm_sq_2d_kernel,
                                  ghost_blocks, gpu_kernels::REDUCE_BLOCK_SIZE,
                                  x.view().data(), d_partial,
                                  nb_pts[0], nb_pts[1],
                                  nb_ghosts_left[0], nb_ghosts_right[0],
                                  nb_ghosts_left[1], nb_ghosts_right[1],
                                  field_stride_c, field_stride_x, field_stride_y,
                                  nb_components_per_pixel);

                GPU_LAUNCH_KERNEL(gpu_kernels::final_reduce_kernel,
                                  1, gpu_kernels::REDUCE_BLOCK_SIZE,
                                  d_partial, ghost_blocks);

                Real ghost_norm;
                GPU_MEMCPY_D2H(&ghost_norm, d_partial, sizeof(Real));
                full_norm -= ghost_norm;
            }
        } else if (spatial_dim == 3) {
            const auto& nb_pts = global_coll.get_nb_subdomain_grid_pts_with_ghosts();
            const auto& nb_ghosts_left = global_coll.get_nb_ghosts_left();
            const auto& nb_ghosts_right = global_coll.get_nb_ghosts_right();
            // Get field data strides (for SoA layout)
            const auto field_strides = x.get_strides(IterUnit::Pixel);
            const Index_t field_stride_c = field_strides[0];
            const Index_t field_stride_x = field_strides[field_strides.size() - 3];
            const Index_t field_stride_y = field_strides[field_strides.size() - 2];
            const Index_t field_stride_z = field_strides[field_strides.size() - 1];

            // Total pixels for block count
            Index_t total_pixels = nb_pts[0] * nb_pts[1] * nb_pts[2];
            int ghost_blocks = (total_pixels + gpu_kernels::REDUCE_BLOCK_SIZE - 1) /
                               gpu_kernels::REDUCE_BLOCK_SIZE;

            GPU_LAUNCH_KERNEL(gpu_kernels::ghost_norm_sq_3d_kernel,
                              ghost_blocks, gpu_kernels::REDUCE_BLOCK_SIZE,
                              x.view().data(), d_partial,
                              nb_pts[0], nb_pts[1], nb_pts[2],
                              nb_ghosts_left[0], nb_ghosts_right[0],
                              nb_ghosts_left[1], nb_ghosts_right[1],
                              nb_ghosts_left[2], nb_ghosts_right[2],
                              field_stride_c, field_stride_x,
                              field_stride_y, field_stride_z,
                              nb_components_per_pixel);

            GPU_LAUNCH_KERNEL(gpu_kernels::final_reduce_kernel,
                              1, gpu_kernels::REDUCE_BLOCK_SIZE,
                              d_partial, ghost_blocks);

            Real ghost_norm;
            GPU_MEMCPY_D2H(&ghost_norm, d_partial, sizeof(Real));
            full_norm -= ghost_norm;
        }
    }

    GPU_FREE(d_partial);
    return full_norm;
}

template <>
void axpy<Real, DeviceSpace>(Real alpha,
                              const TypedField<Real, DeviceSpace>& x,
                              TypedField<Real, DeviceSpace>& y) {
    if (&x.get_collection() != &y.get_collection()) {
        throw FieldError("axpy: fields must belong to the same collection");
    }
    if (x.get_nb_entries() != y.get_nb_entries()) {
        throw FieldError("axpy: fields must have the same number of entries");
    }

    // Total number of scalar elements = pixels * components * sub_pts
    const Index_t n = x.get_nb_entries() * x.get_nb_components() * x.get_nb_sub_pts();
    const int num_blocks = (n + gpu_kernels::BLOCK_SIZE - 1) /
                           gpu_kernels::BLOCK_SIZE;

    GPU_LAUNCH_KERNEL(gpu_kernels::axpy_kernel,
                      num_blocks, gpu_kernels::BLOCK_SIZE,
                      alpha, x.view().data(), y.view().data(), n);
    GPU_DEVICE_SYNCHRONIZE();
}

template <>
void scal<Real, DeviceSpace>(Real alpha, TypedField<Real, DeviceSpace>& x) {
    // Total number of scalar elements = pixels * components * sub_pts
    const Index_t n = x.get_nb_entries() * x.get_nb_components() * x.get_nb_sub_pts();
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
    if (x.get_nb_entries() != y.get_nb_entries()) {
        throw FieldError("axpby: fields must have the same number of entries");
    }

    // Total number of scalar elements = pixels * components * sub_pts
    const Index_t n = x.get_nb_entries() * x.get_nb_components() * x.get_nb_sub_pts();
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
    if (src.get_nb_entries() != dst.get_nb_entries()) {
        throw FieldError("copy: fields must have the same number of entries");
    }

    // Total number of scalar elements = pixels * components * sub_pts
    const Index_t n = src.get_nb_entries() * src.get_nb_components() * src.get_nb_sub_pts();
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
    if (x.get_nb_entries() != y.get_nb_entries()) {
        throw FieldError("axpy_norm_sq: fields must have the same number of entries");
    }

    const auto& coll = x.get_collection();
    // Total number of scalar elements = pixels * components * sub_pts
    const Index_t nb_components_per_pixel =
        x.get_nb_components() * x.get_nb_sub_pts();
    const Index_t n = x.get_nb_entries() * nb_components_per_pixel;

    // Allocate device memory for partial sums
    const int num_blocks = (n + gpu_kernels::REDUCE_BLOCK_SIZE - 1) /
                           gpu_kernels::REDUCE_BLOCK_SIZE;
    Real* d_partial;
    GPU_MALLOC(&d_partial, num_blocks * sizeof(Real));

    // Fused AXPY + norm_sq kernel
    GPU_LAUNCH_KERNEL(gpu_kernels::axpy_norm_sq_kernel,
                      num_blocks, gpu_kernels::REDUCE_BLOCK_SIZE,
                      alpha, x.view().data(), y.view().data(), d_partial, n);

    // Final reduction
    GPU_LAUNCH_KERNEL(gpu_kernels::final_reduce_kernel,
                      1, gpu_kernels::REDUCE_BLOCK_SIZE,
                      d_partial, num_blocks);

    // Copy result back
    Real full_norm;
    GPU_MEMCPY_D2H(&full_norm, d_partial, sizeof(Real));

    // For GlobalFieldCollection, subtract ghost contributions
    if (coll.get_domain() == FieldCollection::ValidityDomain::Global) {
        const auto& global_coll = static_cast<const GlobalFieldCollection&>(coll);
        const auto spatial_dim = global_coll.get_spatial_dim();

        if (spatial_dim == 2) {
            const auto& nb_pts = global_coll.get_nb_subdomain_grid_pts_with_ghosts();
            const auto& nb_ghosts_left = global_coll.get_nb_ghosts_left();
            const auto& nb_ghosts_right = global_coll.get_nb_ghosts_right();
            // Get field data strides from y (for SoA layout)
            const auto field_strides = y.get_strides(IterUnit::Pixel);
            const Index_t field_stride_c = field_strides[0];
            const Index_t field_stride_x = field_strides[field_strides.size() - 2];
            const Index_t field_stride_y = field_strides[field_strides.size() - 1];

            Index_t x_start = nb_ghosts_left[0];
            Index_t x_end = nb_pts[0] - nb_ghosts_right[0];
            Index_t ghost_pixels =
                nb_ghosts_left[0] * nb_pts[1] +
                nb_ghosts_right[0] * nb_pts[1] +
                (x_end - x_start) * nb_ghosts_left[1] +
                (x_end - x_start) * nb_ghosts_right[1];

            if (ghost_pixels > 0) {
                int ghost_blocks = (ghost_pixels + gpu_kernels::REDUCE_BLOCK_SIZE - 1) /
                                   gpu_kernels::REDUCE_BLOCK_SIZE;

                GPU_LAUNCH_KERNEL(gpu_kernels::ghost_norm_sq_2d_kernel,
                                  ghost_blocks, gpu_kernels::REDUCE_BLOCK_SIZE,
                                  y.view().data(), d_partial,
                                  nb_pts[0], nb_pts[1],
                                  nb_ghosts_left[0], nb_ghosts_right[0],
                                  nb_ghosts_left[1], nb_ghosts_right[1],
                                  field_stride_c, field_stride_x, field_stride_y,
                                  nb_components_per_pixel);

                GPU_LAUNCH_KERNEL(gpu_kernels::final_reduce_kernel,
                                  1, gpu_kernels::REDUCE_BLOCK_SIZE,
                                  d_partial, ghost_blocks);

                Real ghost_norm;
                GPU_MEMCPY_D2H(&ghost_norm, d_partial, sizeof(Real));
                full_norm -= ghost_norm;
            }
        } else if (spatial_dim == 3) {
            const auto& nb_pts = global_coll.get_nb_subdomain_grid_pts_with_ghosts();
            const auto& nb_ghosts_left = global_coll.get_nb_ghosts_left();
            const auto& nb_ghosts_right = global_coll.get_nb_ghosts_right();
            // Get field data strides from y (for SoA layout)
            const auto field_strides = y.get_strides(IterUnit::Pixel);
            const Index_t field_stride_c = field_strides[0];
            const Index_t field_stride_x = field_strides[field_strides.size() - 3];
            const Index_t field_stride_y = field_strides[field_strides.size() - 2];
            const Index_t field_stride_z = field_strides[field_strides.size() - 1];

            // Total pixels for block count
            Index_t total_pixels = nb_pts[0] * nb_pts[1] * nb_pts[2];
            int ghost_blocks = (total_pixels + gpu_kernels::REDUCE_BLOCK_SIZE - 1) /
                               gpu_kernels::REDUCE_BLOCK_SIZE;

            GPU_LAUNCH_KERNEL(gpu_kernels::ghost_norm_sq_3d_kernel,
                              ghost_blocks, gpu_kernels::REDUCE_BLOCK_SIZE,
                              y.view().data(), d_partial,
                              nb_pts[0], nb_pts[1], nb_pts[2],
                              nb_ghosts_left[0], nb_ghosts_right[0],
                              nb_ghosts_left[1], nb_ghosts_right[1],
                              nb_ghosts_left[2], nb_ghosts_right[2],
                              field_stride_c, field_stride_x,
                              field_stride_y, field_stride_z,
                              nb_components_per_pixel);

            GPU_LAUNCH_KERNEL(gpu_kernels::final_reduce_kernel,
                              1, gpu_kernels::REDUCE_BLOCK_SIZE,
                              d_partial, ghost_blocks);

            Real ghost_norm;
            GPU_MEMCPY_D2H(&ghost_norm, d_partial, sizeof(Real));
            full_norm -= ghost_norm;
        }
    }

    GPU_FREE(d_partial);
    return full_norm;
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
