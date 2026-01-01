/**
 * @file   convolution_kernels.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   19 Dec 2024
 *
 * @brief  Unified kernel interface - dispatches to backend-specific implementations
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

#ifndef SRC_LIBMUGRID_OPERATORS_CONVOLUTION_KERNELS_HH_
#define SRC_LIBMUGRID_OPERATORS_CONVOLUTION_KERNELS_HH_

#include "memory/memory_space.hh"
#include "memory/array.hh"
#include "convolution_kernels_cpu.hh"

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
#include "convolution_kernels_gpu.hh"
#endif

namespace muGrid {

    /**
     * @brief Dispatch convolution kernel to appropriate backend based on memory space
     *
     * @tparam MemorySpace The memory space where the data resides
     */
    template<typename MemorySpace>
    struct KernelDispatcher {
        /**
         * @brief Forward convolution: quad_data += alpha * Op * nodal_data
         */
        static void apply_convolution(
            const Real* nodal_data,
            Real* quad_data,
            const Real alpha,
            const GridTraversalParams& params,
            const SparseOperatorSoA<MemorySpace>& sparse_op) {

            if constexpr (is_host_space_v<MemorySpace>) {
                cpu::apply_convolution_kernel(
                    nodal_data, quad_data, alpha, params,
                    sparse_op.quad_indices.data(),
                    sparse_op.nodal_indices.data(),
                    sparse_op.values.data(),
                    sparse_op.size);
            }
#if (defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)) && \
    (defined(__CUDACC__) || defined(__HIPCC__))
            else if constexpr (is_device_space_v<MemorySpace>) {
                gpu::apply_convolution_kernel(
                    nodal_data, quad_data, alpha, params,
                    sparse_op.quad_indices.data(),
                    sparse_op.nodal_indices.data(),
                    sparse_op.values.data(),
                    sparse_op.size);
            }
#endif
            else {
                static_assert(is_host_space_v<MemorySpace>,
                             "Unsupported memory space for convolution kernel");
            }
        }

        /**
         * @brief Transpose convolution: nodal_data += alpha * Op^T * quad_data
         */
        static void transpose_convolution(
            const Real* quad_data,
            Real* nodal_data,
            const Real alpha,
            const GridTraversalParams& params,
            const SparseOperatorSoA<MemorySpace>& sparse_op) {

            if constexpr (is_host_space_v<MemorySpace>) {
                cpu::transpose_convolution_kernel(
                    quad_data, nodal_data, alpha, params,
                    sparse_op.quad_indices.data(),
                    sparse_op.nodal_indices.data(),
                    sparse_op.values.data(),
                    sparse_op.size);
            }
#if (defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)) && \
    (defined(__CUDACC__) || defined(__HIPCC__))
            else if constexpr (is_device_space_v<MemorySpace>) {
                gpu::transpose_convolution_kernel(
                    quad_data, nodal_data, alpha, params,
                    sparse_op.quad_indices.data(),
                    sparse_op.nodal_indices.data(),
                    sparse_op.values.data(),
                    sparse_op.size);
            }
#endif
            else {
                static_assert(is_host_space_v<MemorySpace>,
                             "Unsupported memory space for transpose convolution kernel");
            }
        }
    };

    /**
     * @brief Deep copy sparse operator between memory spaces
     */
    template<typename DstSpace, typename SrcSpace>
    SparseOperatorSoA<DstSpace> deep_copy_sparse_operator(
        const SparseOperatorSoA<SrcSpace>& src) {

        SparseOperatorSoA<DstSpace> dst;
        dst.size = src.size;

        dst.quad_indices.resize(src.quad_indices.size());
        dst.nodal_indices.resize(src.nodal_indices.size());
        dst.values.resize(src.values.size());

        deep_copy(dst.quad_indices, src.quad_indices);
        deep_copy(dst.nodal_indices, src.nodal_indices);
        deep_copy(dst.values, src.values);

        return dst;
    }

    /**
     * @brief Synchronize device (no-op for CPU)
     */
    inline void device_synchronize() {
#if defined(MUGRID_ENABLE_CUDA) && defined(__CUDACC__)
        (void)cudaDeviceSynchronize();
#elif defined(MUGRID_ENABLE_HIP) && defined(__HIPCC__)
        (void)hipDeviceSynchronize();
#endif
        // No-op for CPU
    }

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_OPERATORS_CONVOLUTION_KERNELS_HH_
