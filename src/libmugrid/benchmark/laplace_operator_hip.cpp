/**
 * @file   laplace_operator_hip.cpp
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   24 Dec 2024
 *
 * @brief  HIP implementation of hard-coded Laplace operators
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
#include <hip/hip_runtime.h>

namespace muGrid {
namespace benchmark_kernels {

// Block size for HIP kernels
constexpr int BLOCK_SIZE_2D = 16;
constexpr int BLOCK_SIZE_3D = 8;

/**
 * HIP kernel for 2D Laplace operator with 5-point stencil.
 */
__global__ void laplace_2d_kernel(
    const Real* __restrict__ input,
    Real* __restrict__ output,
    Index_t nx, Index_t ny,
    Index_t stride_x, Index_t stride_y) {

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

        output[idx] = left + right + down + up - 4.0 * center;
    }
}

/**
 * HIP kernel for 3D Laplace operator with 7-point stencil.
 */
__global__ void laplace_3d_kernel(
    const Real* __restrict__ input,
    Real* __restrict__ output,
    Index_t nx, Index_t ny, Index_t nz,
    Index_t stride_x, Index_t stride_y, Index_t stride_z) {

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

        output[idx] = xm + xp + ym + yp + zm + zp - 6.0 * center;
    }
}

void laplace_2d_hip(
    const Real* input,
    Real* output,
    Index_t nx, Index_t ny,
    Index_t stride_x, Index_t stride_y) {

    // Compute grid dimensions (for interior points only)
    Index_t interior_nx = nx - 2;  // Exclude ghost layers
    Index_t interior_ny = ny - 2;

    dim3 block(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    dim3 grid(
        (interior_nx + block.x - 1) / block.x,
        (interior_ny + block.y - 1) / block.y
    );

    hipLaunchKernelGGL(laplace_2d_kernel, grid, block, 0, 0,
        input, output, nx, ny, stride_x, stride_y);

    // Synchronize to ensure kernel completion
    hipDeviceSynchronize();
}

void laplace_3d_hip(
    const Real* input,
    Real* output,
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

    hipLaunchKernelGGL(laplace_3d_kernel, grid, block, 0, 0,
        input, output, nx, ny, nz, stride_x, stride_y, stride_z);

    // Synchronize to ensure kernel completion
    hipDeviceSynchronize();
}

}  // namespace benchmark_kernels
}  // namespace muGrid
