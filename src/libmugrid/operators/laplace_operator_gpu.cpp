/**
 * @file   laplace_operator_gpu.cpp
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   25 Dec 2024
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

// Unified GPU abstraction macros
#if defined(MUGRID_ENABLE_CUDA)
    #include <cuda_runtime.h>
    #define GPU_LAUNCH_KERNEL(kernel, grid, block, ...) \
        kernel<<<grid, block>>>(__VA_ARGS__)
    #define GPU_DEVICE_SYNCHRONIZE() cudaDeviceSynchronize()
#elif defined(MUGRID_ENABLE_HIP)
    #include <hip/hip_runtime.h>
    #define GPU_LAUNCH_KERNEL(kernel, grid, block, ...) \
        hipLaunchKernelGGL(kernel, grid, block, 0, 0, __VA_ARGS__)
    #define GPU_DEVICE_SYNCHRONIZE() hipDeviceSynchronize()
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

}  // namespace laplace_kernels
}  // namespace muGrid
