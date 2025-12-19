/**
 * @file   convolution_kernels_cuda.cuh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   19 Dec 2024
 *
 * @brief  CUDA implementations of convolution kernels
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

#ifndef SRC_LIBMUGRID_OPERATORS_CONVOLUTION_KERNELS_CUDA_CUH_
#define SRC_LIBMUGRID_OPERATORS_CONVOLUTION_KERNELS_CUDA_CUH_

#ifdef MUGRID_ENABLE_CUDA

#include <cuda_runtime.h>
#include "core/grid_common.hh"
#include "convolution_kernels_cpu.hh"  // For GridTraversalParams

namespace muGrid {
namespace cuda {

// CUDA kernel implementations - only compiled when using nvcc (__CUDACC__)
#ifdef __CUDACC__

    /**
     * @brief CUDA kernel for forward convolution
     */
    __global__ void apply_convolution_kernel_impl(
        const Real* __restrict__ nodal_data,
        Real* __restrict__ quad_data,
        const Real alpha,
        const Index_t nx, const Index_t ny, const Index_t nz,
        const Index_t nodal_base, const Index_t quad_base,
        const Index_t nodal_stride_x, const Index_t nodal_stride_y,
        const Index_t nodal_stride_z,
        const Index_t quad_stride_x, const Index_t quad_stride_y,
        const Index_t quad_stride_z,
        const Index_t* __restrict__ quad_indices,
        const Index_t* __restrict__ nodal_indices,
        const Real* __restrict__ op_values,
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
     * @brief CUDA kernel for transpose convolution
     */
    __global__ void transpose_convolution_kernel_impl(
        const Real* __restrict__ quad_data,
        Real* __restrict__ nodal_data,
        const Real alpha,
        const Index_t nx, const Index_t ny, const Index_t nz,
        const Index_t nodal_base, const Index_t quad_base,
        const Index_t nodal_stride_x, const Index_t nodal_stride_y,
        const Index_t nodal_stride_z,
        const Index_t quad_stride_x, const Index_t quad_stride_y,
        const Index_t quad_stride_z,
        const Index_t* __restrict__ quad_indices,
        const Index_t* __restrict__ nodal_indices,
        const Real* __restrict__ op_values,
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
        cudaStream_t stream = 0) {

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

        apply_convolution_kernel_impl<<<grid, block, 0, stream>>>(
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
        cudaStream_t stream = 0) {

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

        transpose_convolution_kernel_impl<<<grid, block, 0, stream>>>(
            quad_data, nodal_data, alpha,
            params.nx, params.ny, params.nz,
            nodal_base, quad_base,
            params.nodal_stride_x, params.nodal_stride_y, params.nodal_stride_z,
            params.quad_stride_x, params.quad_stride_y, params.quad_stride_z,
            quad_indices, nodal_indices, op_values, nnz
        );
    }

#endif  // __CUDACC__

}  // namespace cuda
}  // namespace muGrid

#endif  // MUGRID_ENABLE_CUDA

#endif  // SRC_LIBMUGRID_OPERATORS_CONVOLUTION_KERNELS_CUDA_CUH_
