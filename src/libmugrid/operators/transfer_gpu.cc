/**
 * @file   transfer_gpu.cc
 *
 * @author Martin Ladecký <martin.ladecky@imtek.uni-freiburg.de>
 *
 * @date   16 Nov 2026
 *
 * @brief  Device kernels of the multigrid grid transfer (CUDA/HIP)
 *
 * Copyright © 2026 Martin Ladecký
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

#include "core/types.hh"
#include "operators/transfer.hh"

#include "memory/gpu_runtime.hh"

namespace muGrid {
    namespace transfer_kernels {

        namespace {

            constexpr int BLOCK_SIZE{256};

            /**
             * One thread per coarse node and dof (coarse x fastest). Writes the
             * 2^Dim fine nodes 2c + o of the coarse cell, as prolong_*_host:
             * f(2c + o) = 2^-|o| sum_{s <= o} u(c + s). In 2D the z extent is
             * 1 and the z strides are unused.
             */
            template <typename T>
            __global__ void prolong_kernel(
                const T * MUGRID_RESTRICT coarse, T * MUGRID_RESTRICT fine,
                Index_t ncx, Index_t ncy, Index_t ncz, Index_t nb_dof,
                Index_t csx, Index_t csy, Index_t csz, Index_t csd,
                Index_t fsx, Index_t fsy, Index_t fsz, Index_t fsd,
                bool three_d) {
                const Index_t index{static_cast<Index_t>(blockIdx.x) *
                                        blockDim.x +
                                    threadIdx.x};
                if (index >= ncx * ncy * ncz * nb_dof) {
                    return;
                }
                const Index_t cx{index % ncx};
                Index_t rest{index / ncx};
                const Index_t cy{rest % ncy};
                rest /= ncy;
                const Index_t cz{rest % ncz};
                const Index_t dof{rest / ncz};
                const T * c{coarse + dof * csd};
                T * f{fine + dof * fsd};
                const T half{static_cast<T>(0.5)};
                const T quarter{static_cast<T>(0.25)};
                const Index_t ci{cx * csx + cy * csy + cz * csz};
                const Index_t fi{2 * (cx * fsx + cy * fsy + cz * fsz)};
                const T c000{c[ci]};
                const T c100{c[ci + csx]};
                const T c010{c[ci + csy]};
                const T c110{c[ci + csx + csy]};
                f[fi] = c000;
                f[fi + fsx] = half * (c000 + c100);
                f[fi + fsy] = half * (c000 + c010);
                f[fi + fsx + fsy] = quarter * (c000 + c100 + c010 + c110);
                if (three_d) {
                    const T eighth{static_cast<T>(0.125)};
                    const T c001{c[ci + csz]};
                    const T c101{c[ci + csx + csz]};
                    const T c011{c[ci + csy + csz]};
                    const T c111{c[ci + csx + csy + csz]};
                    f[fi + fsz] = half * (c000 + c001);
                    f[fi + fsx + fsz] = quarter * (c000 + c100 + c001 + c101);
                    f[fi + fsy + fsz] = quarter * (c000 + c010 + c001 + c011);
                    f[fi + fsx + fsy + fsz] =
                        eighth * (c000 + c100 + c010 + c001 + c110 + c101 +
                                  c011 + c111);
                }
            }

            /**
             * One thread per coarse node and dof: the [1/2, 1, 1/2]^Dim
             * weighted sum of the 3^Dim fine nodes around 2c, as
             * restrict_*_host.
             */
            template <typename T>
            __global__ void restrict_kernel(
                const T * MUGRID_RESTRICT fine, T * MUGRID_RESTRICT coarse,
                Index_t ncx, Index_t ncy, Index_t ncz, Index_t nb_dof,
                Index_t fsx, Index_t fsy, Index_t fsz, Index_t fsd,
                Index_t csx, Index_t csy, Index_t csz, Index_t csd,
                bool three_d) {
                const Index_t index{static_cast<Index_t>(blockIdx.x) *
                                        blockDim.x +
                                    threadIdx.x};
                if (index >= ncx * ncy * ncz * nb_dof) {
                    return;
                }
                const Index_t cx{index % ncx};
                Index_t rest{index / ncx};
                const Index_t cy{rest % ncy};
                rest /= ncy;
                const Index_t cz{rest % ncz};
                const Index_t dof{rest / ncz};
                const T * f{fine + dof * fsd};
                const Index_t fi{2 * (cx * fsx + cy * fsy + cz * fsz)};
                const T w[3]{static_cast<T>(0.5), static_cast<T>(1),
                             static_cast<T>(0.5)};
                const int z_reach{three_d ? 1 : 0};
                T acc{0};
                for (int oz{-z_reach}; oz <= z_reach; ++oz) {
                    const T wz{three_d ? w[oz + 1] : static_cast<T>(1)};
                    for (int oy{-1}; oy <= 1; ++oy) {
                        const T wyz{w[oy + 1] * wz};
                        for (int ox{-1}; ox <= 1; ++ox) {
                            acc += w[ox + 1] * wyz *
                                   f[fi + ox * fsx + oy * fsy + oz * fsz];
                        }
                    }
                }
                coarse[dof * csd + cx * csx + cy * csy + cz * csz] = acc;
            }

            dim3 grid_for(Index_t nb_threads) {
                return dim3(static_cast<unsigned int>(
                    (nb_threads + BLOCK_SIZE - 1) / BLOCK_SIZE));
            }

        }  // namespace

        template <typename T>
        void prolong_gpu(const T * coarse, T * fine, Index_t ncx, Index_t ncy,
                         Index_t ncz, Index_t nb_dof, Index_t csx,
                         Index_t csy, Index_t csz, Index_t csd, Index_t fsx,
                         Index_t fsy, Index_t fsz, Index_t fsd,
                         bool three_d) {
            const Index_t nb_threads{ncx * ncy * ncz * nb_dof};
            if (nb_threads == 0) {
                return;
            }
            auto kernel = prolong_kernel<T>;
            GPU_LAUNCH_KERNEL(kernel, grid_for(nb_threads), dim3(BLOCK_SIZE),
                              coarse, fine, ncx, ncy, ncz, nb_dof, csx, csy,
                              csz, csd, fsx, fsy, fsz, fsd, three_d);
        }

        template <typename T>
        void restrict_gpu(const T * fine, T * coarse, Index_t ncx, Index_t ncy,
                          Index_t ncz, Index_t nb_dof, Index_t fsx,
                          Index_t fsy, Index_t fsz, Index_t fsd, Index_t csx,
                          Index_t csy, Index_t csz, Index_t csd,
                          bool three_d) {
            const Index_t nb_threads{ncx * ncy * ncz * nb_dof};
            if (nb_threads == 0) {
                return;
            }
            auto kernel = restrict_kernel<T>;
            GPU_LAUNCH_KERNEL(kernel, grid_for(nb_threads), dim3(BLOCK_SIZE),
                              fine, coarse, ncx, ncy, ncz, nb_dof, fsx, fsy,
                              fsz, fsd, csx, csy, csz, csd, three_d);
        }

#define MUGRID_TRANSFER_GPU_INSTANTIATE(T)                                     \
    template void prolong_gpu<T>(const T *, T *, Index_t, Index_t, Index_t,    \
                                 Index_t, Index_t, Index_t, Index_t, Index_t,  \
                                 Index_t, Index_t, Index_t, Index_t, bool);    \
    template void restrict_gpu<T>(const T *, T *, Index_t, Index_t, Index_t,   \
                                  Index_t, Index_t, Index_t, Index_t, Index_t, \
                                  Index_t, Index_t, Index_t, Index_t, bool);
        MUGRID_TRANSFER_GPU_INSTANTIATE(Real)
        MUGRID_TRANSFER_GPU_INSTANTIATE(Real32)
#undef MUGRID_TRANSFER_GPU_INSTANTIATE

    }  // namespace transfer_kernels
}  // namespace muGrid
