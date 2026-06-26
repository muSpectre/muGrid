/**
 * @file   laplace.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   26 Jun 2026
 *
 * @brief  Host stencil kernels for the dimension-templated Laplace operator
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

#include "laplace.hh"

namespace muGrid {

    namespace laplace_kernels {

        void laplace_2d_host(
            const Real* MUGRID_RESTRICT input,
            Real* MUGRID_RESTRICT output,
            Index_t nx, Index_t ny,
            Index_t stride_x, Index_t stride_y,
            Real scale,
            bool increment) {

            // Process all computable points based on stencil requirements.
            // The 5-point stencil needs 1 neighbor on each side, so we iterate
            // [1, n-1) in each dimension. This computes for all points where
            // the stencil has valid input data, including ghost points beyond
            // the minimum stencil requirement.
            for (Index_t iy = 1; iy < ny - 1; ++iy) {
                #if defined(_MSC_VER)
                #pragma loop(ivdep)
                #elif defined(__clang__)
                #pragma clang loop vectorize(enable) interleave(enable)
                #elif defined(__GNUC__)
                #pragma GCC ivdep
                #endif
                for (Index_t ix = 1; ix < nx - 1; ++ix) {
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
        }

        void laplace_3d_host(
            const Real* MUGRID_RESTRICT input,
            Real* MUGRID_RESTRICT output,
            Index_t nx, Index_t ny, Index_t nz,
            Index_t stride_x, Index_t stride_y, Index_t stride_z,
            Real scale,
            bool increment) {

            // Process all computable points based on stencil requirements.
            // The 7-point stencil needs 1 neighbor on each side, so we iterate
            // [1, n-1) in each dimension. This computes for all points where
            // the stencil has valid input data, including ghost points beyond
            // the minimum stencil requirement.
            for (Index_t iz = 1; iz < nz - 1; ++iz) {
                for (Index_t iy = 1; iy < ny - 1; ++iy) {
                    #if defined(_MSC_VER)
                    #pragma loop(ivdep)
                    #elif defined(__clang__)
                    #pragma clang loop vectorize(enable) interleave(enable)
                    #elif defined(__GNUC__)
                    #pragma GCC ivdep
                    #endif
                    for (Index_t ix = 1; ix < nx - 1; ++ix) {
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
            }
        }

    }  // namespace laplace_kernels

}  // namespace muGrid
