/**
 * @file   block_matvec_gpu.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   18 Oct 2026
 *
 * @brief  Device per-mode n x n block product of a Fourier field
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

#include "linalg/block_matvec.hh"

#include "core/exception.hh"
#include "memory/gpu_runtime.hh"

#include <string>

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)

namespace muGrid {

    namespace block_matvec {

        namespace {

            /**
             * Minimal complex aggregate for device code (device code has no
             * `std::complex` operators). Aligned to its size, like
             * `float2`/`double2`, so a value moves as one vector load;
             * layout-compatible with `std::complex<T>`.
             */
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

            //! acc += conj(a) * b
            template <typename T>
            __device__ __forceinline__ void cfma_conj(DevCplx<T> & acc,
                                                      DevCplx<T> a,
                                                      DevCplx<T> b) {
                acc.re += a.re * b.re + a.im * b.im;
                acc.im += a.re * b.im - a.im * b.re;
            }

            //! Memory-bound elementwise work: a block size that keeps the
            //! device busy for any realistic mode count.
            constexpr int BLOCK_SIZE{256};

            template <int N, typename T>
            __global__ void hermitian_kernel(const T * MUGRID_RESTRICT diag,
                                             const DevCplx<T> * MUGRID_RESTRICT
                                                 upper,
                                             DevCplx<T> * field,
                                             Index_t component_stride,
                                             Index_t nb_modes) {
                const Index_t m{global_thread_x()};
                if (m >= nb_modes) {
                    return;
                }
                DevCplx<T> s[N], z[N];
                for (int i{0}; i < N; ++i) {
                    s[i] = field[i * component_stride + m];
                }
                for (int i{0}; i < N; ++i) {
                    const T d{diag[i * nb_modes + m]};
                    z[i] = DevCplx<T>{d * s[i].re, d * s[i].im};
                }
                // Each stored K_ij (i < j) contributes K_ij s_j to z_i and
                // conj(K_ij) s_i to z_j, so it is read once.
                int p{0};
                for (int i{0}; i < N; ++i) {
                    for (int j{i + 1}; j < N; ++j, ++p) {
                        const DevCplx<T> k{upper[p * nb_modes + m]};
                        cfma(z[i], k, s[j]);
                        cfma_conj(z[j], k, s[i]);
                    }
                }
                for (int i{0}; i < N; ++i) {
                    field[i * component_stride + m] = z[i];
                }
            }

            template <int N, typename T>
            __global__ void dense_kernel(const DevCplx<T> * MUGRID_RESTRICT
                                             blocks,
                                         DevCplx<T> * field,
                                         Index_t component_stride,
                                         Index_t nb_modes) {
                const Index_t m{global_thread_x()};
                if (m >= nb_modes) {
                    return;
                }
                DevCplx<T> s[N], z[N];
                for (int i{0}; i < N; ++i) {
                    s[i] = field[i * component_stride + m];
                    z[i] = DevCplx<T>{T(0), T(0)};
                }
                for (int i{0}; i < N; ++i) {
                    for (int j{0}; j < N; ++j) {
                        cfma(z[i], blocks[(i * N + j) * nb_modes + m], s[j]);
                    }
                }
                for (int i{0}; i < N; ++i) {
                    field[i * component_stride + m] = z[i];
                }
            }

            void check(Index_t n, Index_t component_stride, Index_t nb_modes,
                       const char * name) {
                if (n < 1 || n > MAX_COMPONENTS) {
                    throw RuntimeError(
                        std::string(name) + ": n must be between 1 and " +
                        std::to_string(MAX_COMPONENTS) + ", got " +
                        std::to_string(n));
                }
                if (n > 1 && component_stride < nb_modes) {
                    throw RuntimeError(
                        std::string(name) + ": the component stride " +
                        std::to_string(component_stride) +
                        " is smaller than the mode count " +
                        std::to_string(nb_modes) +
                        "; components would overlap");
                }
            }

            Index_t nb_blocks(Index_t nb_modes) {
                return (nb_modes + BLOCK_SIZE - 1) / BLOCK_SIZE;
            }

        }  // namespace

        template <typename T>
        void hermitian_gpu(Index_t n, const T * diag,
                           const std::complex<T> * upper,
                           std::complex<T> * field, Index_t component_stride,
                           Index_t nb_modes) {
            check(n, component_stride, nb_modes, "block_matvec::hermitian_gpu");
            if (nb_modes == 0) {
                return;
            }
            using DC = DevCplx<T>;
            const auto * u{reinterpret_cast<const DC *>(upper)};
            auto * f{reinterpret_cast<DC *>(field)};
            const Index_t blocks{nb_blocks(nb_modes)};
            switch (n) {
            case 1:
                GPU_LAUNCH_KERNEL((hermitian_kernel<1, T>), blocks, BLOCK_SIZE,
                                  diag, u, f, component_stride, nb_modes);
                break;
            case 2:
                GPU_LAUNCH_KERNEL((hermitian_kernel<2, T>), blocks, BLOCK_SIZE,
                                  diag, u, f, component_stride, nb_modes);
                break;
            case 3:
                GPU_LAUNCH_KERNEL((hermitian_kernel<3, T>), blocks, BLOCK_SIZE,
                                  diag, u, f, component_stride, nb_modes);
                break;
            default:
                GPU_LAUNCH_KERNEL((hermitian_kernel<4, T>), blocks, BLOCK_SIZE,
                                  diag, u, f, component_stride, nb_modes);
                break;
            }
            GPU_KERNEL_DEBUG_SYNC("block_matvec::hermitian");
        }

        template <typename T>
        void dense_gpu(Index_t n, const std::complex<T> * blocks,
                       std::complex<T> * field, Index_t component_stride,
                       Index_t nb_modes) {
            check(n, component_stride, nb_modes, "block_matvec::dense_gpu");
            if (nb_modes == 0) {
                return;
            }
            using DC = DevCplx<T>;
            const auto * b{reinterpret_cast<const DC *>(blocks)};
            auto * f{reinterpret_cast<DC *>(field)};
            const Index_t grid{nb_blocks(nb_modes)};
            switch (n) {
            case 1:
                GPU_LAUNCH_KERNEL((dense_kernel<1, T>), grid, BLOCK_SIZE, b, f,
                                  component_stride, nb_modes);
                break;
            case 2:
                GPU_LAUNCH_KERNEL((dense_kernel<2, T>), grid, BLOCK_SIZE, b, f,
                                  component_stride, nb_modes);
                break;
            case 3:
                GPU_LAUNCH_KERNEL((dense_kernel<3, T>), grid, BLOCK_SIZE, b, f,
                                  component_stride, nb_modes);
                break;
            default:
                GPU_LAUNCH_KERNEL((dense_kernel<4, T>), grid, BLOCK_SIZE, b, f,
                                  component_stride, nb_modes);
                break;
            }
            GPU_KERNEL_DEBUG_SYNC("block_matvec::dense");
        }

        template void hermitian_gpu<Real>(Index_t, const Real *,
                                          const std::complex<Real> *,
                                          std::complex<Real> *, Index_t,
                                          Index_t);
        template void hermitian_gpu<Real32>(Index_t, const Real32 *,
                                            const std::complex<Real32> *,
                                            std::complex<Real32> *, Index_t,
                                            Index_t);
        template void dense_gpu<Real>(Index_t, const std::complex<Real> *,
                                      std::complex<Real> *, Index_t, Index_t);
        template void dense_gpu<Real32>(Index_t, const std::complex<Real32> *,
                                        std::complex<Real32> *, Index_t,
                                        Index_t);

    }  // namespace block_matvec

}  // namespace muGrid

#endif  // MUGRID_ENABLE_CUDA || MUGRID_ENABLE_HIP
