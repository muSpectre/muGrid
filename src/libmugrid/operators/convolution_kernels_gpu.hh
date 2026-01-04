/**
 * @file   convolution_kernels_gpu.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   25 Dec 2024
 *
 * @brief  Unified CUDA/HIP implementations of convolution kernels
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

#ifndef SRC_LIBMUGRID_OPERATORS_CONVOLUTION_KERNELS_GPU_HPP_
#define SRC_LIBMUGRID_OPERATORS_CONVOLUTION_KERNELS_GPU_HPP_

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)

#include "core/types.hh"
#include "convolution_kernels_cpu.hh"  // For GridTraversalParams

// Unified GPU abstraction
#if defined(MUGRID_ENABLE_CUDA)
    #include <cuda_runtime.h>
    #define GPU_LAUNCH_KERNEL(kernel, grid, block, stream, ...) \
        kernel<<<grid, block, 0, stream>>>(__VA_ARGS__)
    using gpuStream_t = cudaStream_t;
#elif defined(MUGRID_ENABLE_HIP)
    #include <hip/hip_runtime.h>
    #define GPU_LAUNCH_KERNEL(kernel, grid, block, stream, ...) \
        hipLaunchKernelGGL(kernel, grid, block, 0, stream, __VA_ARGS__)
    using gpuStream_t = hipStream_t;
#endif

namespace muGrid {

// Use a single namespace for GPU kernels - the dispatcher will select
// based on memory space, not on CUDA vs HIP
namespace gpu {

// GPU kernel implementations - only compiled when using nvcc or hipcc
#if defined(__CUDACC__) || defined(__HIPCC__)

    /**
     * @brief GPU kernel for forward convolution
     *
     * Computes: quad_data[quad_offset + quad_indices[i]] +=
     *           alpha * nodal_data[nodal_offset + nodal_indices[i]] * op_values[i]
     *
     * Works with both CUDA and HIP backends.
     */
    __global__ void apply_convolution_kernel_impl(
        const Real* MUGRID_RESTRICT nodal_data,
        Real* MUGRID_RESTRICT quad_data,
        const Real alpha,
        const Index_t nx, const Index_t ny, const Index_t nz,
        const Index_t nodal_base, const Index_t quad_base,
        const Index_t nodal_stride_x, const Index_t nodal_stride_y,
        const Index_t nodal_stride_z,
        const Index_t quad_stride_x, const Index_t quad_stride_y,
        const Index_t quad_stride_z,
        const Index_t* MUGRID_RESTRICT quad_indices,
        const Index_t* MUGRID_RESTRICT nodal_indices,
        const Real* MUGRID_RESTRICT op_values,
        const Index_t nnz) {

        const Index_t x = blockIdx.x * blockDim.x + threadIdx.x;
        const Index_t y = blockIdx.y * blockDim.y + threadIdx.y;
        const Index_t z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x < nx && y < ny && z < nz) {
            const Index_t nodal_offset = nodal_base +
                z * nodal_stride_z + y * nodal_stride_y + x * nodal_stride_x;
            const Index_t quad_offset = quad_base +
                z * quad_stride_z + y * quad_stride_y + x * quad_stride_x;

            for (Index_t i = 0; i < nnz; ++i) {
                quad_data[quad_offset + quad_indices[i]] +=
                    alpha * nodal_data[nodal_offset + nodal_indices[i]] *
                    op_values[i];
            }
        }
    }

    /**
     * @brief GPU kernel for transpose convolution
     *
     * Computes: nodal_data[nodal_offset + nodal_indices[i]] +=
     *           alpha * quad_data[quad_offset + quad_indices[i]] * op_values[i]
     *
     * Works with both CUDA and HIP backends.
     */
    __global__ void transpose_convolution_kernel_impl(
        const Real* MUGRID_RESTRICT quad_data,
        Real* MUGRID_RESTRICT nodal_data,
        const Real alpha,
        const Index_t nx, const Index_t ny, const Index_t nz,
        const Index_t nodal_base, const Index_t quad_base,
        const Index_t nodal_stride_x, const Index_t nodal_stride_y,
        const Index_t nodal_stride_z,
        const Index_t quad_stride_x, const Index_t quad_stride_y,
        const Index_t quad_stride_z,
        const Index_t* MUGRID_RESTRICT quad_indices,
        const Index_t* MUGRID_RESTRICT nodal_indices,
        const Real* MUGRID_RESTRICT op_values,
        const Index_t nnz) {

        const Index_t x = blockIdx.x * blockDim.x + threadIdx.x;
        const Index_t y = blockIdx.y * blockDim.y + threadIdx.y;
        const Index_t z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x < nx && y < ny && z < nz) {
            const Index_t nodal_offset = nodal_base +
                z * nodal_stride_z + y * nodal_stride_y + x * nodal_stride_x;
            const Index_t quad_offset = quad_base +
                z * quad_stride_z + y * quad_stride_y + x * quad_stride_x;

            for (Index_t i = 0; i < nnz; ++i) {
                nodal_data[nodal_offset + nodal_indices[i]] +=
                    alpha * quad_data[quad_offset + quad_indices[i]] *
                    op_values[i];
            }
        }
    }

    /**
     * @brief Launch forward convolution kernel
     *
     * Uses unified GPU_LAUNCH_KERNEL macro that works for both CUDA and HIP.
     */
    inline void apply_convolution_kernel(
        const Real* nodal_data,
        Real* quad_data,
        const Real alpha,
        const GridTraversalParams& params,
        const Index_t* quad_indices,
        const Index_t* nodal_indices,
        const Real* op_values,
        const Index_t nnz,
        gpuStream_t stream = 0) {

        const Index_t nodal_base = params.start_pixel_index *
                                   params.nodal_elems_per_pixel;
        const Index_t quad_base = params.start_pixel_index *
                                  params.quad_elems_per_pixel;

        // Thread block size (8x8x8 = 512 threads)
        dim3 block(8, 8, 8);
        dim3 grid(
            (params.nx + block.x - 1) / block.x,
            (params.ny + block.y - 1) / block.y,
            (params.nz + block.z - 1) / block.z
        );

        GPU_LAUNCH_KERNEL(apply_convolution_kernel_impl, grid, block, stream,
            nodal_data, quad_data, alpha,
            params.nx, params.ny, params.nz,
            nodal_base, quad_base,
            params.nodal_stride_x, params.nodal_stride_y, params.nodal_stride_z,
            params.quad_stride_x, params.quad_stride_y, params.quad_stride_z,
            quad_indices, nodal_indices, op_values, nnz
        );
    }

    /**
     * @brief Launch transpose convolution kernel
     *
     * Uses unified GPU_LAUNCH_KERNEL macro that works for both CUDA and HIP.
     */
    inline void transpose_convolution_kernel(
        const Real* quad_data,
        Real* nodal_data,
        const Real alpha,
        const GridTraversalParams& params,
        const Index_t* quad_indices,
        const Index_t* nodal_indices,
        const Real* op_values,
        const Index_t nnz,
        gpuStream_t stream = 0) {

        const Index_t nodal_base = params.start_pixel_index *
                                   params.nodal_elems_per_pixel;
        const Index_t quad_base = params.start_pixel_index *
                                  params.quad_elems_per_pixel;

        // Thread block size (8x8x8 = 512 threads)
        dim3 block(8, 8, 8);
        dim3 grid(
            (params.nx + block.x - 1) / block.x,
            (params.ny + block.y - 1) / block.y,
            (params.nz + block.z - 1) / block.z
        );

        GPU_LAUNCH_KERNEL(transpose_convolution_kernel_impl, grid, block, stream,
            quad_data, nodal_data, alpha,
            params.nx, params.ny, params.nz,
            nodal_base, quad_base,
            params.nodal_stride_x, params.nodal_stride_y, params.nodal_stride_z,
            params.quad_stride_x, params.quad_stride_y, params.quad_stride_z,
            quad_indices, nodal_indices, op_values, nnz
        );
    }

#endif  // __CUDACC__ || __HIPCC__

}  // namespace gpu
}  // namespace muGrid

#endif  // MUGRID_ENABLE_CUDA || MUGRID_ENABLE_HIP

#endif  // SRC_LIBMUGRID_OPERATORS_CONVOLUTION_KERNELS_GPU_HPP_
