/**
 * @file   fem_gradient_operator_2d.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   01 Jan 2026
 *
 * @brief  Host implementation of hard-coded 2D linear FEM gradient operator
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

#include "fem_gradient.hh"
#include "collection/field_collection_global.hh"
#include "core/exception.hh"

#if defined(MUGRID_ENABLE_CUDA)
#include <cuda_runtime.h>
#elif defined(MUGRID_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace muGrid {

    // =========================================================================
    // 2D Kernel Implementations
    // =========================================================================

    namespace fem_gradient_kernels {

        void fem_gradient_2d_host(const Real * MUGRID_RESTRICT nodal_input,
                                  Real * MUGRID_RESTRICT gradient_output,
                                  Index_t nx, Index_t ny,
                                  Index_t nodal_stride_x,
                                  Index_t nodal_stride_y,
                                  Index_t nodal_stride_n, Index_t grad_stride_x,
                                  Index_t grad_stride_y, Index_t grad_stride_q,
                                  Index_t grad_stride_d, Real hx, Real hy,
                                  Real alpha, bool increment) {

            // Scale factors for shape function gradients
            const Real inv_hx = alpha / hx;
            const Real inv_hy = alpha / hy;

            // Process all computable pixels based on stencil requirements.
            // Each pixel uses 4 nodes at corners [ix, ix+1] x [iy, iy+1],
            // so the stencil needs 0 left, 1 right ghost. We iterate [0, n-1)
            // in each dimension, computing for all pixels where nodal data is valid.
            for (Index_t iy = 0; iy < ny - 1; ++iy) {
#if defined(_MSC_VER)
#pragma loop(ivdep)
#elif defined(__clang__)
                // clang auto-vectorizes via -O and MUGRID_RESTRICT; we do not
                // force vectorize(enable) here because this heavy body (strided
                // scatter over quad points) cannot be vectorized profitably and
                // forcing it only emits -Wpass-failed.
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
                for (Index_t ix = 0; ix < nx - 1; ++ix) {
                    // Base indices for this pixel
                    Index_t nodal_base =
                        ix * nodal_stride_x + iy * nodal_stride_y;
                    Index_t grad_base = ix * grad_stride_x + iy * grad_stride_y;

                    // Get nodal values at pixel corners
                    // Node 0: (ix, iy), Node 1: (ix+1, iy)
                    // Node 2: (ix, iy+1), Node 3: (ix+1, iy+1)
                    Real n0 = nodal_input[nodal_base + 0 * nodal_stride_n];
                    Real n1 = nodal_input[nodal_base + nodal_stride_x +
                                          0 * nodal_stride_n];
                    Real n2 = nodal_input[nodal_base + nodal_stride_y +
                                          0 * nodal_stride_n];
                    Real n3 = nodal_input[nodal_base + nodal_stride_x +
                                          nodal_stride_y + 0 * nodal_stride_n];

                    // Triangle 0 (lower-left): nodes 0, 1, 2
                    // grad_x = (-n0 + n1) / hx
                    // grad_y = (-n0 + n2) / hy
                    Real grad_x_t0 = inv_hx * (-n0 + n1);
                    Real grad_y_t0 = inv_hy * (-n0 + n2);

                    // Triangle 1 (upper-right): nodes 1, 3, 2
                    // grad_x = (-n2 + n3) / hx
                    // grad_y = (-n1 + n3) / hy
                    Real grad_x_t1 = inv_hx * (-n2 + n3);
                    Real grad_y_t1 = inv_hy * (-n1 + n3);

                    // Store gradients
                    if (increment) {
                        // Quad 0 (Triangle 0)
                        gradient_output[grad_base + 0 * grad_stride_d +
                                        0 * grad_stride_q] += grad_x_t0;
                        gradient_output[grad_base + 1 * grad_stride_d +
                                        0 * grad_stride_q] += grad_y_t0;
                        // Quad 1 (Triangle 1)
                        gradient_output[grad_base + 0 * grad_stride_d +
                                        1 * grad_stride_q] += grad_x_t1;
                        gradient_output[grad_base + 1 * grad_stride_d +
                                        1 * grad_stride_q] += grad_y_t1;
                    } else {
                        gradient_output[grad_base + 0 * grad_stride_d +
                                        0 * grad_stride_q] = grad_x_t0;
                        gradient_output[grad_base + 1 * grad_stride_d +
                                        0 * grad_stride_q] = grad_y_t0;
                        gradient_output[grad_base + 0 * grad_stride_d +
                                        1 * grad_stride_q] = grad_x_t1;
                        gradient_output[grad_base + 1 * grad_stride_d +
                                        1 * grad_stride_q] = grad_y_t1;
                    }
                }
            }
        }

        void fem_divergence_2d_host(
            const Real * MUGRID_RESTRICT gradient_input,
            Real * MUGRID_RESTRICT nodal_output, Index_t nx, Index_t ny,
            Index_t grad_stride_x, Index_t grad_stride_y, Index_t grad_stride_q,
            Index_t grad_stride_d, Index_t nodal_stride_x,
            Index_t nodal_stride_y, Index_t nodal_stride_n, Real hx, Real hy,
            const Real * quad_weights, Real alpha, bool increment) {

            // Scale factors including quadrature weights
            const Real w0_inv_hx = alpha * quad_weights[0] / hx;
            const Real w0_inv_hy = alpha * quad_weights[0] / hy;
            const Real w1_inv_hx = alpha * quad_weights[1] / hx;
            const Real w1_inv_hy = alpha * quad_weights[1] / hy;

            // Initialize output if not incrementing
            if (!increment) {
                for (Index_t iy = 0; iy < ny; ++iy) {
                    for (Index_t ix = 0; ix < nx; ++ix) {
                        nodal_output[ix * nodal_stride_x +
                                     iy * nodal_stride_y] = 0.0;
                    }
                }
            }

            // Process all computable pixels based on stencil requirements.
            // The transpose scatters from pixels to their 4 corner nodes, writing
            // to nodes at [ix, ix+1] x [iy, iy+1]. We iterate [0, n-1)
            // in each dimension, which covers all pixels in the gradient field.
            for (Index_t iy = 0; iy < ny - 1; ++iy) {
                for (Index_t ix = 0; ix < nx - 1; ++ix) {
                    Index_t grad_base = ix * grad_stride_x + iy * grad_stride_y;
                    Index_t nodal_base =
                        ix * nodal_stride_x + iy * nodal_stride_y;

                    // Get gradient values at quadrature points
                    Real gx_t0 = gradient_input[grad_base + 0 * grad_stride_d +
                                                0 * grad_stride_q];
                    Real gy_t0 = gradient_input[grad_base + 1 * grad_stride_d +
                                                0 * grad_stride_q];
                    Real gx_t1 = gradient_input[grad_base + 0 * grad_stride_d +
                                                1 * grad_stride_q];
                    Real gy_t1 = gradient_input[grad_base + 1 * grad_stride_d +
                                                1 * grad_stride_q];

                    // Triangle 0 contributions: B^T * sigma
                    Real contrib_n0_t0 =
                        w0_inv_hx * (-gx_t0) + w0_inv_hy * (-gy_t0);
                    Real contrib_n1_t0 = w0_inv_hx * (gx_t0);
                    Real contrib_n2_t0 = w0_inv_hy * (gy_t0);

                    // Triangle 1 contributions
                    Real contrib_n1_t1 = w1_inv_hy * (-gy_t1);
                    Real contrib_n2_t1 = w1_inv_hx * (-gx_t1);
                    Real contrib_n3_t1 =
                        w1_inv_hx * (gx_t1) + w1_inv_hy * (gy_t1);

                    // Accumulate to nodal points
                    nodal_output[nodal_base] += contrib_n0_t0;
                    nodal_output[nodal_base + nodal_stride_x] +=
                        contrib_n1_t0 + contrib_n1_t1;
                    nodal_output[nodal_base + nodal_stride_y] +=
                        contrib_n2_t0 + contrib_n2_t1;
                    nodal_output[nodal_base + nodal_stride_x +
                                 nodal_stride_y] += contrib_n3_t1;
                }
            }
        }

    }  // namespace fem_gradient_kernels

}  // namespace muGrid
