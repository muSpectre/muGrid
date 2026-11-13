/**
 * @file   pcr_gpu.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   13 Nov 2026
 *
 * @brief  Device fused parallel-cyclic-reduction step
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
 */

#include "linalg/pcr.hh"

#include "core/exception.hh"
#include "memory/gpu_runtime.hh"

#include <string>

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)

namespace muGrid {

    namespace pcr {

        namespace {

            //! Complex aggregate for device code; see block_matvec_gpu.cc.
            template <typename T>
            struct alignas(2 * sizeof(T)) DevCplx {
                T re, im;
            };

            //! acc += a * b
            template <typename T>
            __device__ __forceinline__ void cfma(DevCplx<T> & acc,
                                                 DevCplx<T> a, DevCplx<T> b) {
                acc.re += a.re * b.re - a.im * b.im;
                acc.im += a.re * b.im + a.im * b.re;
            }

            //! acc -= a * b
            template <typename T>
            __device__ __forceinline__ void cfms(DevCplx<T> & acc,
                                                 DevCplx<T> a, DevCplx<T> b) {
                acc.re -= a.re * b.re - a.im * b.im;
                acc.im -= a.re * b.im + a.im * b.re;
            }

            //! Memory-bound per-mode work, as for block_matvec.
            constexpr int BLOCK_SIZE{256};

            /**
             * y = D^-1 x - Ã lo - C̃ hi for mode `k`, in registers. `D` is
             * the spatial dimension, `2D` the interface unknowns per mode.
             */
            template <int D, typename T>
            __device__ __forceinline__ void
            level(Index_t k, Index_t M, const DevCplx<T> * MUGRID_RESTRICT dinv,
                  const DevCplx<T> * MUGRID_RESTRICT a,
                  const DevCplx<T> * MUGRID_RESTRICT c,
                  const DevCplx<T> * x,
                  const DevCplx<T> * MUGRID_RESTRICT from_low,
                  const DevCplx<T> * MUGRID_RESTRICT from_high,
                  DevCplx<T> (&y)[2 * D]) {
                constexpr int m{2 * D};
                DevCplx<T> xv[m], lo[D], hi[D];
                for (int i{0}; i < m; ++i) {
                    xv[i] = x[i * M + k];
                }
                for (int i{0}; i < D; ++i) {
                    lo[i] = from_low[i * M + k];
                    hi[i] = from_high[i * M + k];
                }
                for (int i{0}; i < m; ++i) {
                    DevCplx<T> acc{T(0), T(0)};
                    for (int j{0}; j < m; ++j) {
                        cfma(acc, dinv[(i * m + j) * M + k], xv[j]);
                    }
                    for (int j{0}; j < D; ++j) {
                        cfms(acc, a[(i * D + j) * M + k], lo[j]);
                        cfms(acc, c[(i * D + j) * M + k], hi[j]);
                    }
                    y[i] = acc;
                }
            }

            template <int D, typename T>
            __global__ void step_kernel(const DevCplx<T> * MUGRID_RESTRICT
                                            dinv,
                                        const DevCplx<T> * MUGRID_RESTRICT a,
                                        const DevCplx<T> * MUGRID_RESTRICT c,
                                        const DevCplx<T> * x,
                                        const DevCplx<T> * MUGRID_RESTRICT
                                            from_low,
                                        const DevCplx<T> * MUGRID_RESTRICT
                                            from_high,
                                        DevCplx<T> * x_out, Index_t M) {
                const Index_t k{global_thread_x()};
                if (k >= M) {
                    return;
                }
                DevCplx<T> y[2 * D];
                level<D, T>(k, M, dinv, a, c, x, from_low, from_high, y);
                // x and x_out may alias: every read of this mode is done
                for (int i{0}; i < 2 * D; ++i) {
                    x_out[i * M + k] = y[i];
                }
            }

            template <int D, typename T>
            __global__ void
            final_step_kernel(const DevCplx<T> * MUGRID_RESTRICT dinv,
                              const DevCplx<T> * MUGRID_RESTRICT a,
                              const DevCplx<T> * MUGRID_RESTRICT c,
                              const DevCplx<T> * MUGRID_RESTRICT x,
                              const DevCplx<T> * MUGRID_RESTRICT from_low,
                              const DevCplx<T> * MUGRID_RESTRICT from_high,
                              const DevCplx<T> * MUGRID_RESTRICT op_right,
                              const DevCplx<T> * MUGRID_RESTRICT op_left,
                              DevCplx<T> * MUGRID_RESTRICT to_right,
                              DevCplx<T> * MUGRID_RESTRICT to_left,
                              Index_t M) {
                const Index_t k{global_thread_x()};
                if (k >= M) {
                    return;
                }
                constexpr int m{2 * D};
                DevCplx<T> y[m];
                level<D, T>(k, M, dinv, a, c, x, from_low, from_high, y);
                for (int i{0}; i < D; ++i) {
                    DevCplx<T> r{T(0), T(0)}, l{T(0), T(0)};
                    for (int j{0}; j < m; ++j) {
                        cfma(r, op_right[(i * m + j) * M + k], y[j]);
                        cfma(l, op_left[(i * m + j) * M + k], y[j]);
                    }
                    to_right[i * M + k] = r;
                    to_left[i * M + k] = l;
                }
            }

            void check_dim(Index_t dim, const char * name) {
                if (dim != 2 && dim != 3) {
                    throw RuntimeError(std::string(name) +
                                       ": dim must be 2 or 3, got " +
                                       std::to_string(dim));
                }
            }

            Index_t nb_blocks(Index_t nb_modes) {
                return (nb_modes + BLOCK_SIZE - 1) / BLOCK_SIZE;
            }

        }  // namespace

        template <typename T>
        void step_gpu(Index_t dim, Index_t nb_modes,
                      const std::complex<T> * dinv,
                      const std::complex<T> * a, const std::complex<T> * c,
                      const std::complex<T> * x,
                      const std::complex<T> * from_low,
                      const std::complex<T> * from_high,
                      std::complex<T> * x_out) {
            check_dim(dim, "pcr::step_gpu");
            if (nb_modes == 0) {
                return;
            }
            using DC = DevCplx<T>;
            const auto * di{reinterpret_cast<const DC *>(dinv)};
            const auto * aa{reinterpret_cast<const DC *>(a)};
            const auto * cc{reinterpret_cast<const DC *>(c)};
            const auto * xx{reinterpret_cast<const DC *>(x)};
            const auto * lo{reinterpret_cast<const DC *>(from_low)};
            const auto * hi{reinterpret_cast<const DC *>(from_high)};
            auto * out{reinterpret_cast<DC *>(x_out)};
            const Index_t grid{nb_blocks(nb_modes)};
            if (dim == 2) {
                GPU_LAUNCH_KERNEL((step_kernel<2, T>), grid, BLOCK_SIZE, di,
                                  aa, cc, xx, lo, hi, out, nb_modes);
            } else {
                GPU_LAUNCH_KERNEL((step_kernel<3, T>), grid, BLOCK_SIZE, di,
                                  aa, cc, xx, lo, hi, out, nb_modes);
            }
            GPU_KERNEL_DEBUG_SYNC("pcr::step");
        }

        template <typename T>
        void final_step_gpu(Index_t dim, Index_t nb_modes,
                            const std::complex<T> * dinv,
                            const std::complex<T> * a,
                            const std::complex<T> * c,
                            const std::complex<T> * x,
                            const std::complex<T> * from_low,
                            const std::complex<T> * from_high,
                            const std::complex<T> * op_right,
                            const std::complex<T> * op_left,
                            std::complex<T> * to_right,
                            std::complex<T> * to_left) {
            check_dim(dim, "pcr::final_step_gpu");
            if (nb_modes == 0) {
                return;
            }
            using DC = DevCplx<T>;
            const auto * di{reinterpret_cast<const DC *>(dinv)};
            const auto * aa{reinterpret_cast<const DC *>(a)};
            const auto * cc{reinterpret_cast<const DC *>(c)};
            const auto * xx{reinterpret_cast<const DC *>(x)};
            const auto * lo{reinterpret_cast<const DC *>(from_low)};
            const auto * hi{reinterpret_cast<const DC *>(from_high)};
            const auto * opr{reinterpret_cast<const DC *>(op_right)};
            const auto * opl{reinterpret_cast<const DC *>(op_left)};
            auto * tr{reinterpret_cast<DC *>(to_right)};
            auto * tl{reinterpret_cast<DC *>(to_left)};
            const Index_t grid{nb_blocks(nb_modes)};
            if (dim == 2) {
                GPU_LAUNCH_KERNEL((final_step_kernel<2, T>), grid, BLOCK_SIZE,
                                  di, aa, cc, xx, lo, hi, opr, opl, tr, tl,
                                  nb_modes);
            } else {
                GPU_LAUNCH_KERNEL((final_step_kernel<3, T>), grid, BLOCK_SIZE,
                                  di, aa, cc, xx, lo, hi, opr, opl, tr, tl,
                                  nb_modes);
            }
            GPU_KERNEL_DEBUG_SYNC("pcr::final_step");
        }

        template void step_gpu<Real>(Index_t, Index_t,
                                     const std::complex<Real> *,
                                     const std::complex<Real> *,
                                     const std::complex<Real> *,
                                     const std::complex<Real> *,
                                     const std::complex<Real> *,
                                     const std::complex<Real> *,
                                     std::complex<Real> *);
        template void step_gpu<Real32>(Index_t, Index_t,
                                       const std::complex<Real32> *,
                                       const std::complex<Real32> *,
                                       const std::complex<Real32> *,
                                       const std::complex<Real32> *,
                                       const std::complex<Real32> *,
                                       const std::complex<Real32> *,
                                       std::complex<Real32> *);
        template void final_step_gpu<Real>(
            Index_t, Index_t, const std::complex<Real> *,
            const std::complex<Real> *, const std::complex<Real> *,
            const std::complex<Real> *, const std::complex<Real> *,
            const std::complex<Real> *, const std::complex<Real> *,
            const std::complex<Real> *, std::complex<Real> *,
            std::complex<Real> *);
        template void final_step_gpu<Real32>(
            Index_t, Index_t, const std::complex<Real32> *,
            const std::complex<Real32> *, const std::complex<Real32> *,
            const std::complex<Real32> *, const std::complex<Real32> *,
            const std::complex<Real32> *, const std::complex<Real32> *,
            const std::complex<Real32> *, std::complex<Real32> *,
            std::complex<Real32> *);

    }  // namespace pcr

}  // namespace muGrid

#endif  // MUGRID_ENABLE_CUDA || MUGRID_ENABLE_HIP
