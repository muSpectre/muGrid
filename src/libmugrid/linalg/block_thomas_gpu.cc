/**
 * @file   block_thomas_gpu.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   19 Sep 2026
 *
 * @brief  Device implementation of the fused block-Thomas sweep
 *
 * Copyright © 2026 Lars Pastewka
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
 * ---------------------------------------------------------------------------
 *
 * The device counterpart of `block_thomas.cc`, and the two differ in loop order
 * on purpose. The host runs plane-outer and mode-inner so that a single thread
 * streams contiguous memory. A device gives every mode its own thread and
 * marches the axis in registers, so a wavefront wants consecutive modes at a
 * fixed plane -- which the z-major layout provides. Assigning one thread to
 * walk one mode instead would stride by `nb_modes * Dim` between plane steps,
 * the trap `cuSPARSE`'s `gtsv2StridedBatch` documents.
 *
 * The sweep is serial in the axis by nature; the parallelism is the mode count,
 * `nx * (ny/2 + 1)`, which is ample.
 */

#include "linalg/block_thomas.hh"

#include "core/exception.hh"
#include "memory/gpu_runtime.hh"

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)

namespace muGrid {

    namespace block_thomas {

        namespace {

            /**
             * Minimal complex aggregate for device code.
             *
             * Device code has no `std::complex` -- its operators are not
             * `__device__` -- and thrust/cuComplex are deliberately avoided so
             * that one source compiles under both nvcc and hipcc. This mirrors
             * `DeviceComplexT` in `linalg_gpu.cc`; it is repeated rather than
             * shared because that one lives in a translation unit, not a
             * header. Being layout-compatible with `std::complex<T>` (two
             * contiguous reals) lets the buffers be reinterpret_cast.
             */
            template <typename T>
            struct DevCplx {
                T re, im;
            };

            template <typename T>
            __device__ __forceinline__ DevCplx<T> cmul(DevCplx<T> a,
                                                       DevCplx<T> b) {
                return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
            }

            template <typename T>
            __device__ __forceinline__ DevCplx<T> csub(DevCplx<T> a,
                                                       DevCplx<T> b) {
                return {a.re - b.re, a.im - b.im};
            }

            //! Diagonal inverse for this plane and mode; see the host version.
            template <typename T>
            __device__ __forceinline__ const DevCplx<T> *
            diagonal_inverse(const DevCplx<T> * head, const DevCplx<T> * exc,
                             int slot, Index_t plane, Index_t mode,
                             Index_t nb_modes, Index_t nb_head, Index_t nb_exc,
                             Index_t block) {
                if (slot >= 0) {
                    return exc + (plane * nb_exc + slot) * block;
                }
                const Index_t clamped{plane < nb_head ? plane : nb_head - 1};
                return head + (clamped * nb_modes + mode) * block;
            }

            constexpr int BLOCK_SIZE{256};

            template <Dim_t Dim, typename T>
            __global__ void block_thomas_kernel(
                const DevCplx<T> * MUGRID_RESTRICT rhs,
                const DevCplx<T> * MUGRID_RESTRICT head,
                const DevCplx<T> * MUGRID_RESTRICT exc,
                const int * MUGRID_RESTRICT exc_index,
                const DevCplx<T> * MUGRID_RESTRICT A0,
                const DevCplx<T> * MUGRID_RESTRICT A2,
                DevCplx<T> * MUGRID_RESTRICT y, DevCplx<T> * MUGRID_RESTRICT out,
                Index_t nz, Index_t nb_modes, Index_t nb_head, Index_t nb_exc) {
                const Index_t m{static_cast<Index_t>(blockIdx.x) * blockDim.x +
                                threadIdx.x};
                if (m >= nb_modes) {
                    return;
                }
                constexpr Index_t BLOCK{Dim * Dim};
                const int slot{exc_index[m]};

                // The coupling blocks do not vary along the axis, so they are
                // loaded once and kept in registers for the whole march.
                DevCplx<T> a0[BLOCK], a2[BLOCK];
                for (Index_t i{0}; i < BLOCK; ++i) {
                    a0[i] = A0[m * BLOCK + i];
                    a2[i] = A2[m * BLOCK + i];
                }

                DevCplx<T> prev[Dim], cur[Dim];

                // Forward: y_0 = rhs_0, y_k = rhs_k - A2 (Dinv_{k-1} y_{k-1}).
                for (Index_t i{0}; i < Dim; ++i) {
                    const DevCplx<T> v{rhs[m * Dim + i]};
                    y[m * Dim + i] = v;
                    prev[i] = v;
                }
                for (Index_t k{1}; k < nz; ++k) {
                    const DevCplx<T> * dinv{diagonal_inverse<T>(
                        head, exc, slot, k - 1, m, nb_modes, nb_head, nb_exc,
                        BLOCK)};
                    DevCplx<T> tmp[Dim];
                    for (Index_t i{0}; i < Dim; ++i) {
                        DevCplx<T> acc{0, 0};
                        for (Index_t j{0}; j < Dim; ++j) {
                            const DevCplx<T> t{
                                cmul(dinv[i * Dim + j], prev[j])};
                            acc.re += t.re;
                            acc.im += t.im;
                        }
                        tmp[i] = acc;
                    }
                    const Index_t off{(k * nb_modes + m) * Dim};
                    for (Index_t i{0}; i < Dim; ++i) {
                        DevCplx<T> acc{rhs[off + i]};
                        for (Index_t j{0}; j < Dim; ++j) {
                            acc = csub(acc, cmul(a2[i * Dim + j], tmp[j]));
                        }
                        cur[i] = acc;
                    }
                    for (Index_t i{0}; i < Dim; ++i) {
                        y[off + i] = cur[i];
                        prev[i] = cur[i];
                    }
                }

                // Backward: out_k = Dinv_k (y_k - A0 out_{k+1}).
                for (Index_t k{nz - 1}; k >= 0; --k) {
                    const Index_t off{(k * nb_modes + m) * Dim};
                    const DevCplx<T> * dinv{diagonal_inverse<T>(
                        head, exc, slot, k, m, nb_modes, nb_head, nb_exc,
                        BLOCK)};
                    DevCplx<T> rhs_k[Dim];
                    for (Index_t i{0}; i < Dim; ++i) {
                        DevCplx<T> acc{y[off + i]};
                        if (k < nz - 1) {
                            for (Index_t j{0}; j < Dim; ++j) {
                                acc = csub(acc, cmul(a0[i * Dim + j], prev[j]));
                            }
                        }
                        rhs_k[i] = acc;
                    }
                    for (Index_t i{0}; i < Dim; ++i) {
                        DevCplx<T> acc{0, 0};
                        for (Index_t j{0}; j < Dim; ++j) {
                            const DevCplx<T> t{
                                cmul(dinv[i * Dim + j], rhs_k[j])};
                            acc.re += t.re;
                            acc.im += t.im;
                        }
                        cur[i] = acc;
                    }
                    for (Index_t i{0}; i < Dim; ++i) {
                        out[off + i] = cur[i];
                        prev[i] = cur[i];
                    }
                }
            }

        }  // namespace

        template <Dim_t Dim, typename T>
        void sweep_gpu(const std::complex<T> * rhs, const std::complex<T> * head,
                       const std::complex<T> * exc, const int * exc_index,
                       const std::complex<T> * A0, const std::complex<T> * A2,
                       std::complex<T> * y, std::complex<T> * out, Index_t nz,
                       Index_t nb_modes, Index_t nb_head, Index_t nb_exc) {
            using DC = DevCplx<T>;
            const Index_t nb_blocks{(nb_modes + BLOCK_SIZE - 1) / BLOCK_SIZE};
            GPU_LAUNCH_KERNEL(
                (block_thomas_kernel<Dim, T>), nb_blocks, BLOCK_SIZE,
                reinterpret_cast<const DC *>(rhs),
                reinterpret_cast<const DC *>(head),
                reinterpret_cast<const DC *>(exc), exc_index,
                reinterpret_cast<const DC *>(A0),
                reinterpret_cast<const DC *>(A2), reinterpret_cast<DC *>(y),
                reinterpret_cast<DC *>(out), nz, nb_modes, nb_head, nb_exc);
            GPU_KERNEL_DEBUG_SYNC("block_thomas");
        }

        template void sweep_gpu<2, Real>(
            const std::complex<Real> *, const std::complex<Real> *,
            const std::complex<Real> *, const int *,
            const std::complex<Real> *, const std::complex<Real> *,
            std::complex<Real> *, std::complex<Real> *, Index_t, Index_t,
            Index_t, Index_t);
        template void sweep_gpu<3, Real>(
            const std::complex<Real> *, const std::complex<Real> *,
            const std::complex<Real> *, const int *,
            const std::complex<Real> *, const std::complex<Real> *,
            std::complex<Real> *, std::complex<Real> *, Index_t, Index_t,
            Index_t, Index_t);
        template void sweep_gpu<2, Real32>(
            const std::complex<Real32> *, const std::complex<Real32> *,
            const std::complex<Real32> *, const int *,
            const std::complex<Real32> *, const std::complex<Real32> *,
            std::complex<Real32> *, std::complex<Real32> *, Index_t, Index_t,
            Index_t, Index_t);
        template void sweep_gpu<3, Real32>(
            const std::complex<Real32> *, const std::complex<Real32> *,
            const std::complex<Real32> *, const int *,
            const std::complex<Real32> *, const std::complex<Real32> *,
            std::complex<Real32> *, std::complex<Real32> *, Index_t, Index_t,
            Index_t, Index_t);

    }  // namespace block_thomas

}  // namespace muGrid

#endif  // MUGRID_ENABLE_CUDA || MUGRID_ENABLE_HIP
