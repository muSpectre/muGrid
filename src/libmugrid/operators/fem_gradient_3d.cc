/**
 * @file   fem_gradient_operator_3d.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   01 Jan 2026
 *
 * @brief  Host implementation of hard-coded 3D linear FEM gradient operator
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
    // 3D Shape Function Gradients Definition
    // =========================================================================
    // These are computed for the 5-tet Kuhn triangulation of a unit cube.
    // =========================================================================

    namespace fem_gradient_kernels {

        // 3D shape function gradients [dim][quad][node]
        const Real B_3D_REF[DIM_3D][NB_QUAD_3D][NB_NODES_3D] = {
            // d/dx gradients (scaled by 1/hx at runtime)
            {// Tet 0: Central tetrahedron (nodes 1,2,4,7)
             {0.0, 0.5, -0.5, 0.0, -0.5, 0.0, 0.0, 0.5},
             // Tet 1: Corner at (0,0,0) - nodes 0,1,2,4
             {-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
             // Tet 2: Corner at (1,1,0) - nodes 1,2,3,7
             {0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0},
             // Tet 3: Corner at (1,0,1) - nodes 1,4,5,7
             {0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0},
             // Tet 4: Corner at (0,1,1) - nodes 2,4,6,7
             {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0}},
            // d/dy gradients (scaled by 1/hy at runtime)
            {// Tet 0: Central tetrahedron (nodes 1,2,4,7)
             {0.0, -0.5, 0.5, 0.0, -0.5, 0.0, 0.0, 0.5},
             // Tet 1: Corner at (0,0,0) - nodes 0,1,2,4
             {-1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
             // Tet 2: Corner at (1,1,0) - nodes 1,2,3,7
             {0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
             // Tet 3: Corner at (1,0,1) - nodes 1,4,5,7
             {0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0},
             // Tet 4: Corner at (0,1,1) - nodes 2,4,6,7
             {0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0}},
            // d/dz gradients (scaled by 1/hz at runtime)
            {// Tet 0: Central tetrahedron (nodes 1,2,4,7)
             {0.0, -0.5, -0.5, 0.0, 0.5, 0.0, 0.0, 0.5},
             // Tet 1: Corner at (0,0,0) - nodes 0,1,2,4
             {-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
             // Tet 2: Corner at (1,1,0) - nodes 1,2,3,7
             {0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0},
             // Tet 3: Corner at (1,0,1) - nodes 1,4,5,7
             {0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
             // Tet 4: Corner at (0,1,1) - nodes 2,4,6,7
             {0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0}}};

    }  // namespace fem_gradient_kernels

    // =========================================================================
    // 3D Kernel Implementations
    // =========================================================================

    namespace fem_gradient_kernels {

        void fem_gradient_3d_host(
            const Real * MUGRID_RESTRICT nodal_input,
            Real * MUGRID_RESTRICT gradient_output, Index_t nx, Index_t ny,
            Index_t nz, Index_t nodal_stride_x, Index_t nodal_stride_y,
            Index_t nodal_stride_z, Index_t nodal_stride_n,
            Index_t grad_stride_x, Index_t grad_stride_y, Index_t grad_stride_z,
            Index_t grad_stride_q, Index_t grad_stride_d, Real hx, Real hy,
            Real hz, Real alpha, bool increment) {

            const Real inv_hx = alpha / hx;
            const Real inv_hy = alpha / hy;
            const Real inv_hz = alpha / hz;

            // Process all computable voxels based on stencil requirements.
            // Each voxel uses 8 nodes at corners [ix, ix+1] x [iy, iy+1] x [iz, iz+1],
            // so the stencil needs 0 left, 1 right ghost. We iterate [0, n-1) in each
            // dimension, computing for all voxels where nodal data is valid.
            for (Index_t iz = 0; iz < nz - 1; ++iz) {
                for (Index_t iy = 0; iy < ny - 1; ++iy) {
#if defined(_MSC_VER)
#pragma loop(ivdep)
#elif defined(__clang__)
                    // clang auto-vectorizes via -O and MUGRID_RESTRICT; we do
                    // not force vectorize(enable) here because this heavy body
                    // (strided scatter over quad points) cannot be vectorized
                    // profitably and forcing it only emits -Wpass-failed.
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
                    for (Index_t ix = 0; ix < nx - 1; ++ix) {
                        Index_t nodal_base = ix * nodal_stride_x +
                                             iy * nodal_stride_y +
                                             iz * nodal_stride_z;
                        Index_t grad_base = ix * grad_stride_x +
                                            iy * grad_stride_y +
                                            iz * grad_stride_z;

                        // Get all 8 nodal values
                        Real n[8];
                        for (Index_t node = 0; node < 8; ++node) {
                            Index_t ox = NODE_OFFSET_3D[node][0];
                            Index_t oy = NODE_OFFSET_3D[node][1];
                            Index_t oz = NODE_OFFSET_3D[node][2];
                            n[node] =
                                nodal_input[nodal_base + ox * nodal_stride_x +
                                            oy * nodal_stride_y +
                                            oz * nodal_stride_z];
                        }

                        // Compute gradients for each tetrahedron
                        for (Index_t q = 0; q < NB_QUAD_3D; ++q) {
                            Real grad_x = 0.0, grad_y = 0.0, grad_z = 0.0;
                            for (Index_t node = 0; node < 8; ++node) {
                                grad_x += B_3D_REF[0][q][node] * n[node];
                                grad_y += B_3D_REF[1][q][node] * n[node];
                                grad_z += B_3D_REF[2][q][node] * n[node];
                            }
                            grad_x *= inv_hx;
                            grad_y *= inv_hy;
                            grad_z *= inv_hz;

                            Index_t grad_idx = grad_base + q * grad_stride_q;
                            if (increment) {
                                gradient_output[grad_idx + 0 * grad_stride_d] +=
                                    grad_x;
                                gradient_output[grad_idx + 1 * grad_stride_d] +=
                                    grad_y;
                                gradient_output[grad_idx + 2 * grad_stride_d] +=
                                    grad_z;
                            } else {
                                gradient_output[grad_idx + 0 * grad_stride_d] =
                                    grad_x;
                                gradient_output[grad_idx + 1 * grad_stride_d] =
                                    grad_y;
                                gradient_output[grad_idx + 2 * grad_stride_d] =
                                    grad_z;
                            }
                        }
                    }
                }
            }
        }

        void fem_divergence_3d_host(
            const Real * MUGRID_RESTRICT gradient_input,
            Real * MUGRID_RESTRICT nodal_output, Index_t nx, Index_t ny,
            Index_t nz, Index_t grad_stride_x, Index_t grad_stride_y,
            Index_t grad_stride_z, Index_t grad_stride_q, Index_t grad_stride_d,
            Index_t nodal_stride_x, Index_t nodal_stride_y,
            Index_t nodal_stride_z, Index_t nodal_stride_n, Real hx, Real hy,
            Real hz, const Real * quad_weights, Real alpha, bool increment) {

            // Initialize output if not incrementing
            if (!increment) {
                for (Index_t iz = 0; iz < nz; ++iz) {
                    for (Index_t iy = 0; iy < ny; ++iy) {
                        for (Index_t ix = 0; ix < nx; ++ix) {
                            nodal_output[ix * nodal_stride_x +
                                         iy * nodal_stride_y +
                                         iz * nodal_stride_z] = 0.0;
                        }
                    }
                }
            }

            const Real inv_hx = alpha / hx;
            const Real inv_hy = alpha / hy;
            const Real inv_hz = alpha / hz;

            // Process all computable voxels based on stencil requirements.
            // The transpose scatters from voxels to their 8 corner nodes, writing
            // to nodes at [ix, ix+1] x [iy, iy+1] x [iz, iz+1]. We iterate [0, n-1)
            // in each dimension, which covers all voxels in the gradient field.
            for (Index_t iz = 0; iz < nz - 1; ++iz) {
                for (Index_t iy = 0; iy < ny - 1; ++iy) {
                    for (Index_t ix = 0; ix < nx - 1; ++ix) {
                        Index_t grad_base = ix * grad_stride_x +
                                            iy * grad_stride_y +
                                            iz * grad_stride_z;
                        Index_t nodal_base = ix * nodal_stride_x +
                                             iy * nodal_stride_y +
                                             iz * nodal_stride_z;

                        // For each quadrature point
                        for (Index_t q = 0; q < NB_QUAD_3D; ++q) {
                            Real w = quad_weights[q];
                            Index_t grad_idx = grad_base + q * grad_stride_q;
                            Real gx =
                                gradient_input[grad_idx + 0 * grad_stride_d];
                            Real gy =
                                gradient_input[grad_idx + 1 * grad_stride_d];
                            Real gz =
                                gradient_input[grad_idx + 2 * grad_stride_d];

                            // Accumulate B^T * g to each node
                            for (Index_t node = 0; node < 8; ++node) {
                                Real contrib =
                                    w * (B_3D_REF[0][q][node] * inv_hx * gx +
                                         B_3D_REF[1][q][node] * inv_hy * gy +
                                         B_3D_REF[2][q][node] * inv_hz * gz);
                                Index_t ox = NODE_OFFSET_3D[node][0];
                                Index_t oy = NODE_OFFSET_3D[node][1];
                                Index_t oz = NODE_OFFSET_3D[node][2];
                                nodal_output[nodal_base + ox * nodal_stride_x +
                                             oy * nodal_stride_y +
                                             oz * nodal_stride_z] += contrib;
                            }
                        }
                    }
                }
            }
        }

    }  // namespace fem_gradient_kernels

}  // namespace muGrid
