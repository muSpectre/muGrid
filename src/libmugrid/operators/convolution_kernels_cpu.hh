/**
 * @file   convolution_kernels_cpu.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   19 Dec 2024
 *
 * @brief  CPU implementations of convolution kernels
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

#ifndef SRC_LIBMUGRID_OPERATORS_CONVOLUTION_KERNELS_CPU_HH_
#define SRC_LIBMUGRID_OPERATORS_CONVOLUTION_KERNELS_CPU_HH_

#include "core/types.hh"
#include "memory/array.hh"

namespace muGrid {

    /**
     * Parameters for grid traversal in convolution kernels
     */
    struct GridTraversalParams {
        Index_t nx, ny, nz;                 // Grid dimensions (without ghosts)
        Index_t total_pixels;               // nx * ny * nz
        Index_t row_width;                  // nx + ghosts in x direction
        Index_t start_pixel_index;          // Starting pixel offset
        Index_t nodal_elems_per_pixel;      // DOFs per pixel for nodal field
        Index_t quad_elems_per_pixel;       // DOFs per pixel for quad field
        Index_t nodal_stride_x, nodal_stride_y, nodal_stride_z;
        Index_t quad_stride_x, quad_stride_y, quad_stride_z;

        // Total elements in fields (needed for SoA indexing)
        Index_t total_nodal_elements;       // Total nodal elements in buffer
        Index_t total_quad_elements;        // Total quad elements in buffer
    };

    /**
     * Sparse operator in Structure-of-Arrays format
     */
    template<typename MemorySpace>
    struct SparseOperatorSoA {
        Index_t size{0};  // Number of non-zeros
        Array<Index_t, MemorySpace> quad_indices;
        Array<Index_t, MemorySpace> nodal_indices;
        Array<Real, MemorySpace> values;

        //! Default constructor - creates empty operator
        SparseOperatorSoA() = default;

        //! Constructor that allocates arrays of given size
        explicit SparseOperatorSoA(Index_t n)
            : size{n}, quad_indices(n), nodal_indices(n), values(n) {}

        //! Check if operator is empty
        bool empty() const { return size == 0; }
    };

    namespace cpu {

        /**
         * @brief Forward convolution kernel for CPU
         *
         * Applies: quad_data += alpha * Op * nodal_data
         *
         * Loop structure is optimized for SIMD vectorization:
         * - Stencil entries are in the outermost loop (constant per x-sweep)
         * - X-dimension is innermost for contiguous memory access
         * - Compiler can vectorize the x-loop when strides are 1
         */
        inline void apply_convolution_kernel(
            const Real* MUGRID_RESTRICT nodal_data,
            Real* MUGRID_RESTRICT quad_data,
            const Real alpha,
            const GridTraversalParams& params,
            const Index_t* MUGRID_RESTRICT quad_indices,
            const Index_t* MUGRID_RESTRICT nodal_indices,
            const Real* MUGRID_RESTRICT op_values,
            const Index_t nnz) {

            const Index_t nx = params.nx;
            const Index_t ny = params.ny;
            const Index_t nz = params.nz;
            const Index_t nodal_base = params.start_pixel_index *
                                       params.nodal_elems_per_pixel;
            const Index_t quad_base = params.start_pixel_index *
                                      params.quad_elems_per_pixel;
            const Index_t nodal_stride_x = params.nodal_stride_x;
            const Index_t nodal_stride_y = params.nodal_stride_y;
            const Index_t nodal_stride_z = params.nodal_stride_z;
            const Index_t quad_stride_x = params.quad_stride_x;
            const Index_t quad_stride_y = params.quad_stride_y;
            const Index_t quad_stride_z = params.quad_stride_z;

            // Restructured loop: stencil entries outside, x inside for SIMD
            // This allows the compiler to:
            // 1. Hoist op_values[i] and index offsets out of the x-loop
            // 2. Vectorize the x-loop with predictable stride access
            for (Index_t i = 0; i < nnz; ++i) {
                const Index_t nodal_idx = nodal_indices[i];
                const Index_t quad_idx = quad_indices[i];
                const Real scaled_op_val = alpha * op_values[i];

                for (Index_t z = 0; z < nz; ++z) {
                    const Index_t nodal_z = nodal_base + z * nodal_stride_z + nodal_idx;
                    const Index_t quad_z = quad_base + z * quad_stride_z + quad_idx;

                    for (Index_t y = 0; y < ny; ++y) {
                        const Index_t nodal_yz = nodal_z + y * nodal_stride_y;
                        const Index_t quad_yz = quad_z + y * quad_stride_y;

                        // Innermost loop over x - vectorizable with stride access
                        #if defined(_MSC_VER)
                        #pragma loop(ivdep)
                        #elif defined(__clang__)
                        #pragma clang loop vectorize(enable) interleave(enable)
                        #elif defined(__GNUC__)
                        #pragma GCC ivdep
                        #endif
                        for (Index_t x = 0; x < nx; ++x) {
                            quad_data[quad_yz + x * quad_stride_x] +=
                                scaled_op_val *
                                nodal_data[nodal_yz + x * nodal_stride_x];
                        }
                    }
                }
            }
        }

        /**
         * @brief Transpose convolution kernel for CPU
         *
         * Applies: nodal_data += alpha * Op^T * quad_data
         *
         * Note: This kernel accumulates to nodal_data, which may have
         * overlapping writes if stencils overlap. For CPU, no atomics
         * are needed since we process sequentially.
         *
         * Loop structure is optimized for SIMD vectorization (same as apply).
         */
        inline void transpose_convolution_kernel(
            const Real* MUGRID_RESTRICT quad_data,
            Real* MUGRID_RESTRICT nodal_data,
            const Real alpha,
            const GridTraversalParams& params,
            const Index_t* MUGRID_RESTRICT quad_indices,
            const Index_t* MUGRID_RESTRICT nodal_indices,
            const Real* MUGRID_RESTRICT op_values,
            const Index_t nnz) {

            const Index_t nx = params.nx;
            const Index_t ny = params.ny;
            const Index_t nz = params.nz;
            const Index_t nodal_base = params.start_pixel_index *
                                       params.nodal_elems_per_pixel;
            const Index_t quad_base = params.start_pixel_index *
                                      params.quad_elems_per_pixel;
            const Index_t nodal_stride_x = params.nodal_stride_x;
            const Index_t nodal_stride_y = params.nodal_stride_y;
            const Index_t nodal_stride_z = params.nodal_stride_z;
            const Index_t quad_stride_x = params.quad_stride_x;
            const Index_t quad_stride_y = params.quad_stride_y;
            const Index_t quad_stride_z = params.quad_stride_z;

            // Restructured loop: stencil entries outside, x inside for SIMD
            for (Index_t i = 0; i < nnz; ++i) {
                const Index_t nodal_idx = nodal_indices[i];
                const Index_t quad_idx = quad_indices[i];
                const Real scaled_op_val = alpha * op_values[i];

                for (Index_t z = 0; z < nz; ++z) {
                    const Index_t nodal_z = nodal_base + z * nodal_stride_z + nodal_idx;
                    const Index_t quad_z = quad_base + z * quad_stride_z + quad_idx;

                    for (Index_t y = 0; y < ny; ++y) {
                        const Index_t nodal_yz = nodal_z + y * nodal_stride_y;
                        const Index_t quad_yz = quad_z + y * quad_stride_y;

                        // Innermost loop over x - vectorizable with stride access
                        #if defined(_MSC_VER)
                        #pragma loop(ivdep)
                        #elif defined(__clang__)
                        #pragma clang loop vectorize(enable) interleave(enable)
                        #elif defined(__GNUC__)
                        #pragma GCC ivdep
                        #endif
                        for (Index_t x = 0; x < nx; ++x) {
                            nodal_data[nodal_yz + x * nodal_stride_x] +=
                                scaled_op_val *
                                quad_data[quad_yz + x * quad_stride_x];
                        }
                    }
                }
            }
        }

    }  // namespace cpu

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_OPERATORS_CONVOLUTION_KERNELS_CPU_HH_
