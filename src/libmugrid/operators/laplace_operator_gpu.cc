/**
 * @file   laplace_operator_gpu.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   25 Dec 2025
 *
 * @brief  Unified CUDA/HIP implementation of hard-coded Laplace operator
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

#include "laplace_operator.hh"

#include <vector>

// Unified GPU abstraction macros
#if defined(MUGRID_ENABLE_CUDA)
    #include <cuda_runtime.h>
    #define GPU_LAUNCH_KERNEL(kernel, grid, block, ...) \
        kernel<<<grid, block>>>(__VA_ARGS__)
    #define GPU_LAUNCH_KERNEL_SHMEM(kernel, grid, block, shmem, ...) \
        kernel<<<grid, block, shmem>>>(__VA_ARGS__)
    #define GPU_DEVICE_SYNCHRONIZE() (void)cudaDeviceSynchronize()
#elif defined(MUGRID_ENABLE_HIP)
    #include <hip/hip_runtime.h>
    #define GPU_LAUNCH_KERNEL(kernel, grid, block, ...) \
        hipLaunchKernelGGL(kernel, grid, block, 0, 0, __VA_ARGS__)
    #define GPU_LAUNCH_KERNEL_SHMEM(kernel, grid, block, shmem, ...) \
        hipLaunchKernelGGL(kernel, grid, block, shmem, 0, __VA_ARGS__)
    #define GPU_DEVICE_SYNCHRONIZE() (void)hipDeviceSynchronize()
#endif

namespace muGrid {
namespace laplace_kernels {

// Block sizes for GPU kernels
constexpr int BLOCK_SIZE_2D = 16;
constexpr int BLOCK_SIZE_3D = 8;

/**
 * GPU kernel for 2D Laplace operator with 5-point stencil.
 * Works with both CUDA and HIP - the __global__ keyword is the same.
 */
__global__ void laplace_2d_kernel(
    const Real* MUGRID_RESTRICT input,
    Real* MUGRID_RESTRICT output,
    Index_t nx, Index_t ny,
    Index_t stride_x, Index_t stride_y,
    Real scale,
    bool increment) {

    // Thread indices (offset by 1 for ghost layer)
    Index_t ix = blockIdx.x * blockDim.x + threadIdx.x + 1;
    Index_t iy = blockIdx.y * blockDim.y + threadIdx.y + 1;

    // Check bounds (excluding ghost layers)
    if (ix < nx - 1 && iy < ny - 1) {
        Index_t idx = ix * stride_x + iy * stride_y;

        // 5-point stencil: [0,1,0; 1,-4,1; 0,1,0]
        Real center = input[idx];
        Real left   = input[idx - stride_x];
        Real right  = input[idx + stride_x];
        Real down   = input[idx - stride_y];
        Real up     = input[idx + stride_y];

        Real result = scale * (left + right + down + up - 4.0 * center);
        if (increment) {
            output[idx] += result;
        } else {
            output[idx] = result;
        }
    }
}

/**
 * GPU kernel for 3D Laplace operator with 7-point stencil.
 * Works with both CUDA and HIP.
 */
__global__ void laplace_3d_kernel(
    const Real* MUGRID_RESTRICT input,
    Real* MUGRID_RESTRICT output,
    Index_t nx, Index_t ny, Index_t nz,
    Index_t stride_x, Index_t stride_y, Index_t stride_z,
    Real scale,
    bool increment) {

    // Thread indices (offset by 1 for ghost layer)
    Index_t ix = blockIdx.x * blockDim.x + threadIdx.x + 1;
    Index_t iy = blockIdx.y * blockDim.y + threadIdx.y + 1;
    Index_t iz = blockIdx.z * blockDim.z + threadIdx.z + 1;

    // Check bounds (excluding ghost layers)
    if (ix < nx - 1 && iy < ny - 1 && iz < nz - 1) {
        Index_t idx = ix * stride_x + iy * stride_y + iz * stride_z;

        // 7-point stencil: center=-6, neighbors=+1
        Real center = input[idx];
        Real xm = input[idx - stride_x];
        Real xp = input[idx + stride_x];
        Real ym = input[idx - stride_y];
        Real yp = input[idx + stride_y];
        Real zm = input[idx - stride_z];
        Real zp = input[idx + stride_z];

        Real result = scale * (xm + xp + ym + yp + zm + zp - 6.0 * center);
        if (increment) {
            output[idx] += result;
        } else {
            output[idx] = result;
        }
    }
}

// Launch wrapper for 2D - uses unified macro
#if defined(MUGRID_ENABLE_CUDA)
void laplace_2d_cuda(
#elif defined(MUGRID_ENABLE_HIP)
void laplace_2d_hip(
#endif
    const Real* input,
    Real* output,
    Index_t nx, Index_t ny,
    Index_t stride_x, Index_t stride_y,
    Real scale,
    bool increment) {

    // Compute grid dimensions (for interior points only)
    Index_t interior_nx = nx - 2;  // Exclude ghost layers
    Index_t interior_ny = ny - 2;

    dim3 block(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    dim3 grid(
        (interior_nx + block.x - 1) / block.x,
        (interior_ny + block.y - 1) / block.y
    );

    GPU_LAUNCH_KERNEL(laplace_2d_kernel, grid, block,
        input, output, nx, ny, stride_x, stride_y, scale, increment);

    GPU_DEVICE_SYNCHRONIZE();
}

// Launch wrapper for 3D - uses unified macro
#if defined(MUGRID_ENABLE_CUDA)
void laplace_3d_cuda(
#elif defined(MUGRID_ENABLE_HIP)
void laplace_3d_hip(
#endif
    const Real* input,
    Real* output,
    Index_t nx, Index_t ny, Index_t nz,
    Index_t stride_x, Index_t stride_y, Index_t stride_z,
    Real scale,
    bool increment) {

    // Compute grid dimensions (for interior points only)
    Index_t interior_nx = nx - 2;
    Index_t interior_ny = ny - 2;
    Index_t interior_nz = nz - 2;

    dim3 block(BLOCK_SIZE_3D, BLOCK_SIZE_3D, BLOCK_SIZE_3D);
    dim3 grid(
        (interior_nx + block.x - 1) / block.x,
        (interior_ny + block.y - 1) / block.y,
        (interior_nz + block.z - 1) / block.z
    );

    GPU_LAUNCH_KERNEL(laplace_3d_kernel, grid, block,
        input, output, nx, ny, nz, stride_x, stride_y, stride_z, scale, increment);

    GPU_DEVICE_SYNCHRONIZE();
}

/**
 * Fused GPU kernel for 2D Laplace operator with dot product reduction.
 * Computes output = scale * Laplace(input) AND returns input · output.
 * Uses block-level reduction with shared memory.
 */
__global__ void laplace_2d_vecdot_kernel(
    const Real* MUGRID_RESTRICT input,
    Real* MUGRID_RESTRICT output,
    Real* MUGRID_RESTRICT partial_sums,
    Index_t nx, Index_t ny,
    Index_t stride_x, Index_t stride_y,
    Real scale) {

    extern __shared__ Real shared_data[];

    // Thread indices (offset by 1 for ghost layer)
    Index_t ix = blockIdx.x * blockDim.x + threadIdx.x + 1;
    Index_t iy = blockIdx.y * blockDim.y + threadIdx.y + 1;
    Index_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    Index_t block_size = blockDim.x * blockDim.y;

    // Compute Laplace and dot product contribution in one pass
    Real local_sum = 0.0;
    if (ix < nx - 1 && iy < ny - 1) {
        Index_t idx = ix * stride_x + iy * stride_y;

        // 5-point stencil: [0,1,0; 1,-4,1; 0,1,0]
        Real center = input[idx];
        Real left   = input[idx - stride_x];
        Real right  = input[idx + stride_x];
        Real down   = input[idx - stride_y];
        Real up     = input[idx + stride_y];

        Real result = scale * (left + right + down + up - 4.0 * center);
        output[idx] = result;

        // Compute dot product contribution: input[idx] * output[idx]
        local_sum = center * result;
    }
    shared_data[tid] = local_sum;
    __syncthreads();

    // Block-level reduction
    for (Index_t s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        Index_t block_id = blockIdx.y * gridDim.x + blockIdx.x;
        partial_sums[block_id] = shared_data[0];
    }
}

/**
 * Fused GPU kernel for 3D Laplace operator with dot product reduction.
 * Computes output = scale * Laplace(input) AND returns input · output.
 */
__global__ void laplace_3d_vecdot_kernel(
    const Real* MUGRID_RESTRICT input,
    Real* MUGRID_RESTRICT output,
    Real* MUGRID_RESTRICT partial_sums,
    Index_t nx, Index_t ny, Index_t nz,
    Index_t stride_x, Index_t stride_y, Index_t stride_z,
    Real scale) {

    extern __shared__ Real shared_data[];

    // Thread indices (offset by 1 for ghost layer)
    Index_t ix = blockIdx.x * blockDim.x + threadIdx.x + 1;
    Index_t iy = blockIdx.y * blockDim.y + threadIdx.y + 1;
    Index_t iz = blockIdx.z * blockDim.z + threadIdx.z + 1;
    Index_t tid = threadIdx.z * (blockDim.x * blockDim.y) +
                  threadIdx.y * blockDim.x + threadIdx.x;
    Index_t block_size = blockDim.x * blockDim.y * blockDim.z;

    // Compute Laplace and dot product contribution in one pass
    Real local_sum = 0.0;
    if (ix < nx - 1 && iy < ny - 1 && iz < nz - 1) {
        Index_t idx = ix * stride_x + iy * stride_y + iz * stride_z;

        // 7-point stencil: center=-6, neighbors=+1
        Real center = input[idx];
        Real xm = input[idx - stride_x];
        Real xp = input[idx + stride_x];
        Real ym = input[idx - stride_y];
        Real yp = input[idx + stride_y];
        Real zm = input[idx - stride_z];
        Real zp = input[idx + stride_z];

        Real result = scale * (xm + xp + ym + yp + zm + zp - 6.0 * center);
        output[idx] = result;

        // Compute dot product contribution: input[idx] * output[idx]
        local_sum = center * result;
    }
    shared_data[tid] = local_sum;
    __syncthreads();

    // Block-level reduction
    for (Index_t s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        Index_t block_id = blockIdx.z * (gridDim.x * gridDim.y) +
                           blockIdx.y * gridDim.x + blockIdx.x;
        partial_sums[block_id] = shared_data[0];
    }
}

/**
 * GPU kernel for interior dot product reduction (2D).
 * Uses block-level reduction with shared memory.
 * (Kept for potential future use when apply and vecdot need to be separate)
 */
__global__ void vecdot_2d_kernel(
    const Real* MUGRID_RESTRICT input,
    const Real* MUGRID_RESTRICT output,
    Real* MUGRID_RESTRICT partial_sums,
    Index_t nx, Index_t ny,
    Index_t stride_x, Index_t stride_y) {

    extern __shared__ Real shared_data[];

    // Thread indices (offset by 1 for ghost layer)
    Index_t ix = blockIdx.x * blockDim.x + threadIdx.x + 1;
    Index_t iy = blockIdx.y * blockDim.y + threadIdx.y + 1;
    Index_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    Index_t block_size = blockDim.x * blockDim.y;

    // Load and compute local contribution
    Real local_sum = 0.0;
    if (ix < nx - 1 && iy < ny - 1) {
        Index_t idx = ix * stride_x + iy * stride_y;
        local_sum = input[idx] * output[idx];
    }
    shared_data[tid] = local_sum;
    __syncthreads();

    // Block-level reduction
    for (Index_t s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        Index_t block_id = blockIdx.y * gridDim.x + blockIdx.x;
        partial_sums[block_id] = shared_data[0];
    }
}

/**
 * GPU kernel for interior dot product reduction (3D).
 */
__global__ void vecdot_3d_kernel(
    const Real* MUGRID_RESTRICT input,
    const Real* MUGRID_RESTRICT output,
    Real* MUGRID_RESTRICT partial_sums,
    Index_t nx, Index_t ny, Index_t nz,
    Index_t stride_x, Index_t stride_y, Index_t stride_z) {

    extern __shared__ Real shared_data[];

    // Thread indices (offset by 1 for ghost layer)
    Index_t ix = blockIdx.x * blockDim.x + threadIdx.x + 1;
    Index_t iy = blockIdx.y * blockDim.y + threadIdx.y + 1;
    Index_t iz = blockIdx.z * blockDim.z + threadIdx.z + 1;
    Index_t tid = threadIdx.z * (blockDim.x * blockDim.y) +
                  threadIdx.y * blockDim.x + threadIdx.x;
    Index_t block_size = blockDim.x * blockDim.y * blockDim.z;

    // Load and compute local contribution
    Real local_sum = 0.0;
    if (ix < nx - 1 && iy < ny - 1 && iz < nz - 1) {
        Index_t idx = ix * stride_x + iy * stride_y + iz * stride_z;
        local_sum = input[idx] * output[idx];
    }
    shared_data[tid] = local_sum;
    __syncthreads();

    // Block-level reduction
    for (Index_t s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        Index_t block_id = blockIdx.z * (gridDim.x * gridDim.y) +
                           blockIdx.y * gridDim.x + blockIdx.x;
        partial_sums[block_id] = shared_data[0];
    }
}

// Fused launch wrapper for 2D apply_vecdot
#if defined(MUGRID_ENABLE_CUDA)
Real laplace_2d_apply_vecdot_cuda(
#elif defined(MUGRID_ENABLE_HIP)
Real laplace_2d_apply_vecdot_hip(
#endif
    const Real* input,
    Real* output,
    Index_t nx, Index_t ny,
    Index_t stride_x, Index_t stride_y,
    Real scale) {

    // Compute grid dimensions (for interior points only)
    Index_t interior_nx = nx - 2;
    Index_t interior_ny = ny - 2;

    dim3 block(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    dim3 grid(
        (interior_nx + block.x - 1) / block.x,
        (interior_ny + block.y - 1) / block.y
    );

    Index_t num_blocks = grid.x * grid.y;
    Index_t shared_mem_size = block.x * block.y * sizeof(Real);

    // Allocate device memory for partial sums
    Real* d_partial_sums;
#if defined(MUGRID_ENABLE_CUDA)
    (void)cudaMalloc(&d_partial_sums, num_blocks * sizeof(Real));
#elif defined(MUGRID_ENABLE_HIP)
    (void)hipMalloc(&d_partial_sums, num_blocks * sizeof(Real));
#endif

    GPU_LAUNCH_KERNEL_SHMEM(laplace_2d_vecdot_kernel, grid, block, shared_mem_size,
        input, output, d_partial_sums, nx, ny, stride_x, stride_y, scale);

    // Copy partial sums to host and accumulate
    std::vector<Real> h_partial_sums(num_blocks);
#if defined(MUGRID_ENABLE_CUDA)
    (void)cudaMemcpy(h_partial_sums.data(), d_partial_sums,
                     num_blocks * sizeof(Real), cudaMemcpyDeviceToHost);
    (void)cudaFree(d_partial_sums);
#elif defined(MUGRID_ENABLE_HIP)
    (void)hipMemcpy(h_partial_sums.data(), d_partial_sums,
                    num_blocks * sizeof(Real), hipMemcpyDeviceToHost);
    (void)hipFree(d_partial_sums);
#endif

    Real result = 0.0;
    for (Index_t i = 0; i < num_blocks; ++i) {
        result += h_partial_sums[i];
    }
    return result;
}

// Fused launch wrapper for 3D apply_vecdot
#if defined(MUGRID_ENABLE_CUDA)
Real laplace_3d_apply_vecdot_cuda(
#elif defined(MUGRID_ENABLE_HIP)
Real laplace_3d_apply_vecdot_hip(
#endif
    const Real* input,
    Real* output,
    Index_t nx, Index_t ny, Index_t nz,
    Index_t stride_x, Index_t stride_y, Index_t stride_z,
    Real scale) {

    // Compute grid dimensions (for interior points only)
    Index_t interior_nx = nx - 2;
    Index_t interior_ny = ny - 2;
    Index_t interior_nz = nz - 2;

    dim3 block(BLOCK_SIZE_3D, BLOCK_SIZE_3D, BLOCK_SIZE_3D);
    dim3 grid(
        (interior_nx + block.x - 1) / block.x,
        (interior_ny + block.y - 1) / block.y,
        (interior_nz + block.z - 1) / block.z
    );

    Index_t num_blocks = grid.x * grid.y * grid.z;
    Index_t shared_mem_size = block.x * block.y * block.z * sizeof(Real);

    // Allocate device memory for partial sums
    Real* d_partial_sums;
#if defined(MUGRID_ENABLE_CUDA)
    (void)cudaMalloc(&d_partial_sums, num_blocks * sizeof(Real));
#elif defined(MUGRID_ENABLE_HIP)
    (void)hipMalloc(&d_partial_sums, num_blocks * sizeof(Real));
#endif

    GPU_LAUNCH_KERNEL_SHMEM(laplace_3d_vecdot_kernel, grid, block, shared_mem_size,
        input, output, d_partial_sums, nx, ny, nz, stride_x, stride_y, stride_z, scale);

    // Copy partial sums to host and accumulate
    std::vector<Real> h_partial_sums(num_blocks);
#if defined(MUGRID_ENABLE_CUDA)
    (void)cudaMemcpy(h_partial_sums.data(), d_partial_sums,
                     num_blocks * sizeof(Real), cudaMemcpyDeviceToHost);
    (void)cudaFree(d_partial_sums);
#elif defined(MUGRID_ENABLE_HIP)
    (void)hipMemcpy(h_partial_sums.data(), d_partial_sums,
                    num_blocks * sizeof(Real), hipMemcpyDeviceToHost);
    (void)hipFree(d_partial_sums);
#endif

    Real result = 0.0;
    for (Index_t i = 0; i < num_blocks; ++i) {
        result += h_partial_sums[i];
    }
    return result;
}

// Launch wrapper for 2D vecdot (separate, for when apply was already done)
#if defined(MUGRID_ENABLE_CUDA)
Real laplace_2d_vecdot_cuda(
#elif defined(MUGRID_ENABLE_HIP)
Real laplace_2d_vecdot_hip(
#endif
    const Real* input,
    const Real* output,
    Index_t nx, Index_t ny,
    Index_t stride_x, Index_t stride_y) {

    // Compute grid dimensions (for interior points only)
    Index_t interior_nx = nx - 2;
    Index_t interior_ny = ny - 2;

    dim3 block(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    dim3 grid(
        (interior_nx + block.x - 1) / block.x,
        (interior_ny + block.y - 1) / block.y
    );

    Index_t num_blocks = grid.x * grid.y;
    Index_t shared_mem_size = block.x * block.y * sizeof(Real);

    // Allocate device memory for partial sums
    Real* d_partial_sums;
#if defined(MUGRID_ENABLE_CUDA)
    (void)cudaMalloc(&d_partial_sums, num_blocks * sizeof(Real));
#elif defined(MUGRID_ENABLE_HIP)
    (void)hipMalloc(&d_partial_sums, num_blocks * sizeof(Real));
#endif

    GPU_LAUNCH_KERNEL_SHMEM(vecdot_2d_kernel, grid, block, shared_mem_size,
        input, output, d_partial_sums, nx, ny, stride_x, stride_y);

    // Copy partial sums to host and accumulate
    std::vector<Real> h_partial_sums(num_blocks);
#if defined(MUGRID_ENABLE_CUDA)
    (void)cudaMemcpy(h_partial_sums.data(), d_partial_sums,
                     num_blocks * sizeof(Real), cudaMemcpyDeviceToHost);
    (void)cudaFree(d_partial_sums);
#elif defined(MUGRID_ENABLE_HIP)
    (void)hipMemcpy(h_partial_sums.data(), d_partial_sums,
                    num_blocks * sizeof(Real), hipMemcpyDeviceToHost);
    (void)hipFree(d_partial_sums);
#endif

    Real result = 0.0;
    for (Index_t i = 0; i < num_blocks; ++i) {
        result += h_partial_sums[i];
    }
    return result;
}

// Launch wrapper for 3D vecdot
#if defined(MUGRID_ENABLE_CUDA)
Real laplace_3d_vecdot_cuda(
#elif defined(MUGRID_ENABLE_HIP)
Real laplace_3d_vecdot_hip(
#endif
    const Real* input,
    const Real* output,
    Index_t nx, Index_t ny, Index_t nz,
    Index_t stride_x, Index_t stride_y, Index_t stride_z) {

    // Compute grid dimensions (for interior points only)
    Index_t interior_nx = nx - 2;
    Index_t interior_ny = ny - 2;
    Index_t interior_nz = nz - 2;

    dim3 block(BLOCK_SIZE_3D, BLOCK_SIZE_3D, BLOCK_SIZE_3D);
    dim3 grid(
        (interior_nx + block.x - 1) / block.x,
        (interior_ny + block.y - 1) / block.y,
        (interior_nz + block.z - 1) / block.z
    );

    Index_t num_blocks = grid.x * grid.y * grid.z;
    Index_t shared_mem_size = block.x * block.y * block.z * sizeof(Real);

    // Allocate device memory for partial sums
    Real* d_partial_sums;
#if defined(MUGRID_ENABLE_CUDA)
    (void)cudaMalloc(&d_partial_sums, num_blocks * sizeof(Real));
#elif defined(MUGRID_ENABLE_HIP)
    (void)hipMalloc(&d_partial_sums, num_blocks * sizeof(Real));
#endif

    GPU_LAUNCH_KERNEL_SHMEM(vecdot_3d_kernel, grid, block, shared_mem_size,
        input, output, d_partial_sums, nx, ny, nz, stride_x, stride_y, stride_z);

    // Copy partial sums to host and accumulate
    std::vector<Real> h_partial_sums(num_blocks);
#if defined(MUGRID_ENABLE_CUDA)
    (void)cudaMemcpy(h_partial_sums.data(), d_partial_sums,
                     num_blocks * sizeof(Real), cudaMemcpyDeviceToHost);
    (void)cudaFree(d_partial_sums);
#elif defined(MUGRID_ENABLE_HIP)
    (void)hipMemcpy(h_partial_sums.data(), d_partial_sums,
                    num_blocks * sizeof(Real), hipMemcpyDeviceToHost);
    (void)hipFree(d_partial_sums);
#endif

    Real result = 0.0;
    for (Index_t i = 0; i < num_blocks; ++i) {
        result += h_partial_sums[i];
    }
    return result;
}

}  // namespace laplace_kernels
}  // namespace muGrid
