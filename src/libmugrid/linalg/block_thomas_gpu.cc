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
 * The sweep is serial in the axis by nature; the parallelism is the mode
 * count, the number of local Fourier modes in the two transformed axes
 * (`(N_0/2 + 1) * N_1` for an undistributed grid, axis 0 being the
 * half-complex one), which is ample.
 */

#include "linalg/block_thomas.hh"

#include "core/exception.hh"
#include "memory/gpu_runtime.hh"

#include <limits>

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

            /**
             * One forward step from registers:
             * `prev <- r - A2 (D prev)` with `D = Dinv_{k-1}`, i.e.
             * `y_k = rhs_k - A2 Dinv_{k-1} y_{k-1}`.
             */
            template <Dim_t Dim, typename T>
            __device__ __forceinline__ void
            forward_step(const DevCplx<T> * r, const DevCplx<T> * d,
                         const DevCplx<T> * a2, DevCplx<T> * prev) {
                DevCplx<T> tmp[Dim];
                for (int i{0}; i < Dim; ++i) {
                    DevCplx<T> acc{0, 0};
                    for (int j{0}; j < Dim; ++j) {
                        const DevCplx<T> t{cmul(d[i * Dim + j], prev[j])};
                        acc.re += t.re;
                        acc.im += t.im;
                    }
                    tmp[i] = acc;
                }
                for (int i{0}; i < Dim; ++i) {
                    DevCplx<T> acc{r[i]};
                    for (int j{0}; j < Dim; ++j) {
                        acc = csub(acc, cmul(a2[i * Dim + j], tmp[j]));
                    }
                    prev[i] = acc;
                }
            }

            /**
             * One backward step from registers:
             * `prev <- D (yk - A0 prev)` with `D = Dinv_k`, i.e.
             * `out_k = Dinv_k (y_k - A0 out_{k+1})`.
             */
            template <Dim_t Dim, typename T>
            __device__ __forceinline__ void
            backward_step(const DevCplx<T> * yk, const DevCplx<T> * d,
                          const DevCplx<T> * a0, DevCplx<T> * prev) {
                DevCplx<T> rk[Dim];
                for (int i{0}; i < Dim; ++i) {
                    DevCplx<T> acc{yk[i]};
                    for (int j{0}; j < Dim; ++j) {
                        acc = csub(acc, cmul(a0[i * Dim + j], prev[j]));
                    }
                    rk[i] = acc;
                }
                for (int i{0}; i < Dim; ++i) {
                    DevCplx<T> acc{0, 0};
                    for (int j{0}; j < Dim; ++j) {
                        const DevCplx<T> t{cmul(d[i * Dim + j], rk[j])};
                        acc.re += t.re;
                        acc.im += t.im;
                    }
                    prev[i] = acc;
                }
            }

            /**
             * The sweep, one thread per mode `m`, with the two constant
             * coupling blocks in registers for the whole march.
             *
             * The loads of a plane do not depend on the recurrence, only the
             * arithmetic does, so the planes are processed in chunks of
             * `CHUNK`: every load of the chunk is issued first, then its
             * `CHUNK` steps run from registers, so the memory latency of a
             * chunk overlaps instead of serialising with each step. The
             * chunk loops are unrolled so the arrays stay in registers; a
             * tail of fewer than `CHUNK` planes goes step by step.
             *
             * `__launch_bounds__(THREADS)` declares the block size the kernel
             * is launched with. Without it the compiler must allow for
             * 1024-thread blocks, which caps a thread at 128 registers on
             * CDNA3; the chunked sweep needs more and then spills (measured on
             * MI300A: 516 B of spills at CHUNK = 4, running 2x slower).
             *
             * Plane and mode indices are `int`, offsets `Index_t`: the
             * narrower loop state keeps register pressure down, and the caller
             * guarantees `nz` and `nb_modes` fit.
             */
            template <Dim_t Dim, typename T, int CHUNK, int THREADS>
            __global__ void __launch_bounds__(THREADS) block_thomas_kernel(
                const DevCplx<T> * MUGRID_RESTRICT rhs,
                const DevCplx<T> * MUGRID_RESTRICT head,
                const DevCplx<T> * MUGRID_RESTRICT exc,
                const int * MUGRID_RESTRICT exc_index,
                const DevCplx<T> * MUGRID_RESTRICT A0,
                const DevCplx<T> * MUGRID_RESTRICT A2,
                DevCplx<T> * MUGRID_RESTRICT y, DevCplx<T> * MUGRID_RESTRICT out,
                int nz, int nb_modes, int nb_head, int nb_exc) {
                const int m{static_cast<int>(blockIdx.x * blockDim.x +
                                             threadIdx.x)};
                if (m >= nb_modes) {
                    return;
                }
                constexpr int BLOCK{Dim * Dim};
                const int slot{exc_index[m]};
                auto dinv = [&](int plane) {
                    return diagonal_inverse<T>(head, exc, slot, plane, m,
                                               nb_modes, nb_head, nb_exc,
                                               BLOCK);
                };
                auto offset = [&](int plane) {
                    return (static_cast<Index_t>(plane) * nb_modes + m) * Dim;
                };

                DevCplx<T> a0[BLOCK], a2[BLOCK], prev[Dim];
                for (int i{0}; i < BLOCK; ++i) {
                    a0[i] = A0[static_cast<Index_t>(m) * BLOCK + i];
                    a2[i] = A2[static_cast<Index_t>(m) * BLOCK + i];
                }

                // Forward: y_0 = rhs_0, y_k = rhs_k - A2 Dinv_{k-1} y_{k-1}.
                for (int i{0}; i < Dim; ++i) {
                    const DevCplx<T> v{rhs[offset(0) + i]};
                    y[offset(0) + i] = v;
                    prev[i] = v;
                }
                int k{1};
                for (; k + CHUNK <= nz; k += CHUNK) {
                    DevCplx<T> r[CHUNK][Dim], d[CHUNK][BLOCK];
#pragma unroll
                    for (int c{0}; c < CHUNK; ++c) {
                        const DevCplx<T> * D{dinv(k + c - 1)};
                        for (int i{0}; i < BLOCK; ++i) {
                            d[c][i] = D[i];
                        }
                        for (int i{0}; i < Dim; ++i) {
                            r[c][i] = rhs[offset(k + c) + i];
                        }
                    }
#pragma unroll
                    for (int c{0}; c < CHUNK; ++c) {
                        forward_step<Dim>(r[c], d[c], a2, prev);
                        for (int i{0}; i < Dim; ++i) {
                            y[offset(k + c) + i] = prev[i];
                        }
                    }
                }
                for (; k < nz; ++k) {
                    DevCplx<T> r[Dim], d[BLOCK];
                    const DevCplx<T> * D{dinv(k - 1)};
                    for (int i{0}; i < BLOCK; ++i) {
                        d[i] = D[i];
                    }
                    for (int i{0}; i < Dim; ++i) {
                        r[i] = rhs[offset(k) + i];
                    }
                    forward_step<Dim>(r, d, a2, prev);
                    for (int i{0}; i < Dim; ++i) {
                        y[offset(k) + i] = prev[i];
                    }
                }

                // Backward: out_k = Dinv_k (y_k - A0 out_{k+1}). The last
                // plane has no out_{k+1}; a zero `prev` expresses that without
                // a branch per step.
                for (int i{0}; i < Dim; ++i) {
                    prev[i] = DevCplx<T>{0, 0};
                }
                k = nz - 1;
                for (; k + 1 - CHUNK >= 0; k -= CHUNK) {
                    DevCplx<T> yk[CHUNK][Dim], d[CHUNK][BLOCK];
#pragma unroll
                    for (int c{0}; c < CHUNK; ++c) {
                        const DevCplx<T> * D{dinv(k - c)};
                        for (int i{0}; i < BLOCK; ++i) {
                            d[c][i] = D[i];
                        }
                        for (int i{0}; i < Dim; ++i) {
                            yk[c][i] = y[offset(k - c) + i];
                        }
                    }
#pragma unroll
                    for (int c{0}; c < CHUNK; ++c) {
                        backward_step<Dim>(yk[c], d[c], a0, prev);
                        for (int i{0}; i < Dim; ++i) {
                            out[offset(k - c) + i] = prev[i];
                        }
                    }
                }
                for (; k >= 0; --k) {
                    DevCplx<T> yk[Dim], d[BLOCK];
                    const DevCplx<T> * D{dinv(k)};
                    for (int i{0}; i < BLOCK; ++i) {
                        d[i] = D[i];
                    }
                    for (int i{0}; i < Dim; ++i) {
                        yk[i] = y[offset(k) + i];
                    }
                    backward_step<Dim>(yk, d, a0, prev);
                    for (int i{0}; i < Dim; ++i) {
                        out[offset(k) + i] = prev[i];
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
            constexpr Index_t int_max{std::numeric_limits<int>::max()};
            if (nz > int_max || nb_modes > int_max || nb_head > int_max ||
                nb_exc > int_max) {
                throw RuntimeError(
                    "block_thomas::sweep_gpu: nz, nb_modes, nb_head and "
                    "nb_exc must fit in an int");
            }
            if (nz == 0 || nb_modes == 0) {
                return;
            }
            const Index_t nb_blocks{(nb_modes + DEVICE_THREADS - 1) /
                                    DEVICE_THREADS};
            GPU_LAUNCH_KERNEL(
                (block_thomas_kernel<Dim, T, DEVICE_CHUNK, DEVICE_THREADS>),
                nb_blocks, DEVICE_THREADS,
                reinterpret_cast<const DC *>(rhs),
                reinterpret_cast<const DC *>(head),
                reinterpret_cast<const DC *>(exc), exc_index,
                reinterpret_cast<const DC *>(A0),
                reinterpret_cast<const DC *>(A2), reinterpret_cast<DC *>(y),
                reinterpret_cast<DC *>(out), static_cast<int>(nz),
                static_cast<int>(nb_modes), static_cast<int>(nb_head),
                static_cast<int>(nb_exc));
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
