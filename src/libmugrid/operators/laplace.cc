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

        template <typename T>
        void laplace_2d_host(
            const T* MUGRID_RESTRICT input,
            T* MUGRID_RESTRICT output,
            Index_t nx, Index_t ny,
            Index_t stride_x, Index_t stride_y,
            T scale,
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
                #elif defined(__GNUC__) && !defined(__NVCOMPILER)
                #pragma GCC ivdep
                #endif
                for (Index_t ix = 1; ix < nx - 1; ++ix) {
                    Index_t idx = ix * stride_x + iy * stride_y;

                    // 5-point stencil: [0,1,0; 1,-4,1; 0,1,0]
                    T center = input[idx];
                    T left   = input[idx - stride_x];
                    T right  = input[idx + stride_x];
                    T down   = input[idx - stride_y];
                    T up     = input[idx + stride_y];

                    T result = scale * (left + right + down + up -
                                        static_cast<T>(4) * center);
                    if (increment) {
                        output[idx] += result;
                    } else {
                        output[idx] = result;
                    }
                }
            }
        }

        template <typename T>
        void laplace_3d_host(
            const T* MUGRID_RESTRICT input,
            T* MUGRID_RESTRICT output,
            Index_t nx, Index_t ny, Index_t nz,
            Index_t stride_x, Index_t stride_y, Index_t stride_z,
            T scale,
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
                    #elif defined(__GNUC__) && !defined(__NVCOMPILER)
                    #pragma GCC ivdep
                    #endif
                    for (Index_t ix = 1; ix < nx - 1; ++ix) {
                        Index_t idx = ix * stride_x + iy * stride_y + iz * stride_z;

                        // 7-point stencil: center=-6, neighbors=+1
                        T center = input[idx];
                        T xm = input[idx - stride_x];
                        T xp = input[idx + stride_x];
                        T ym = input[idx - stride_y];
                        T yp = input[idx + stride_y];
                        T zm = input[idx - stride_z];
                        T zp = input[idx + stride_z];

                        T result = scale * (xm + xp + ym + yp + zm + zp -
                                            static_cast<T>(6) * center);
                        if (increment) {
                            output[idx] += result;
                        } else {
                            output[idx] = result;
                        }
                    }
                }
            }
        }

        // Explicit instantiations for double and single precision.
        template void laplace_2d_host<Real>(const Real *, Real *, Index_t,
                                            Index_t, Index_t, Index_t, Real,
                                            bool);
        template void laplace_3d_host<Real>(const Real *, Real *, Index_t,
                                            Index_t, Index_t, Index_t, Index_t,
                                            Index_t, Real, bool);
        template void laplace_2d_host<Real32>(const Real32 *, Real32 *, Index_t,
                                              Index_t, Index_t, Index_t, Real32,
                                              bool);
        template void laplace_3d_host<Real32>(const Real32 *, Real32 *, Index_t,
                                              Index_t, Index_t, Index_t,
                                              Index_t, Index_t, Real32, bool);

    }  // namespace laplace_kernels

}  // namespace muGrid
