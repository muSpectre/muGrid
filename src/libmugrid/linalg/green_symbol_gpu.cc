/**
 * @file   linalg/green_symbol_gpu.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   24 Sep 2026
 *
 * @brief  Device implementation of the per-mode inverse-symbol application
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
 * The device counterpart of `green_symbol.cc`. It is not a transcription of
 * the host loop, because the host's one optimisation that matters -- hoisting
 * the sums over the slower axes out of the axis-0 loop -- is a loop invariant
 * of a *serial* walk, and one thread per mode has no walk to hoist it out of.
 *
 * Without the hoist the symbol build is `3^Dim * Dim^2` complex FMAs per mode
 * instead of `3 * Dim^2`, and the kernel stops being competitive with reading a
 * stored symbol. So the work is split in two levels:
 *
 *  - A block covers `BLOCK_SIZE` consecutive modes. Axis 0 is fastest, so they
 *    lie on a handful of consecutive axis-0 lines -- two for a 512^3 grid,
 *    whose lines are 257 modes long. The block builds the hoisted partial
 *    sums `P[d0]` of every line it touches once, cooperatively, into shared
 *    memory.
 *  - Each thread then finishes its own mode in registers: the three-term sum
 *    over `d0`, the cofactor inverse and the matvec.
 *
 * Consecutive threads take consecutive modes, which is what keeps the field
 * access coalesced in the device's structure-of-arrays layout.
 *
 * When axis 0 is so short that a block would touch more than `MAX_LINES`
 * lines, each thread builds its own `P` instead. That is the naive per-mode
 * cost, but it only arises for lines a few modes long, where the hoist would
 * be amortised over almost nothing anyway.
 */

#include "linalg/green_symbol.hh"

#include "core/exception.hh"
#include "memory/gpu_runtime.hh"

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)

#include <limits>
#include <string>

namespace muGrid {

    namespace green_symbol {

        namespace {

            /**
             * Minimal complex aggregate for device code; see `DevCplx` in
             * `block_thomas_gpu.cc` for why this is not `std::complex`,
             * thrust or cuComplex. Layout-compatible with `std::complex<T>`.
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

            //! a * b, accumulated into acc
            template <typename T>
            __device__ __forceinline__ void cfma(DevCplx<T> & acc,
                                                 DevCplx<T> a, DevCplx<T> b) {
                acc.re += a.re * b.re - a.im * b.im;
                acc.im += a.re * b.im + a.im * b.re;
            }

            __device__ __forceinline__ void sincos_pi(float x, float * s,
                                                      float * c) {
                sincospif(x, s, c);
            }

            __device__ __forceinline__ void sincos_pi(double x, double * s,
                                                      double * c) {
                sincospi(x, s, c);
            }

            /**
             * exp(-2 pi i f), the phase of a unit offset at frequency f.
             *
             * `sincospi` rather than `sincos(2 * M_PI * f)`: the argument
             * reduction is exact for a multiple of pi, and the frequencies are
             * exactly representable fractions of the grid size.
             */
            template <typename T>
            __device__ __forceinline__ DevCplx<T> unit_phase(T f) {
                T s, c;
                sincos_pi(T(2) * f, &s, &c);
                return {c, -s};
            }

            //! Phase of offset index d (0, 1, 2 for -1, 0, +1) given the +1
            //! phase p; the -1 phase is its conjugate.
            template <typename T>
            __device__ __forceinline__ DevCplx<T> offset_phase(DevCplx<T> p,
                                                               int d) {
                return d == 0 ? DevCplx<T>{p.re, -p.im}
                              : (d == 1 ? DevCplx<T>{T(1), T(0)} : p);
            }

            /**
             * The stencil, passed to the kernel by value.
             *
             * 243 reals in 3D, under 2 KB even in double, which fits in the
             * kernel parameter space. That space *is* constant memory -- the
             * same cache and broadcast as a `__constant__` array -- but it
             * belongs to the launch rather than to the module, so two
             * preconditioners with different stencils cannot overwrite each
             * other's copy, and there is no per-apply memcpy to a symbol.
             */
            template <Dim_t Dim, typename T>
            struct Stencil {
                static constexpr Index_t NB_OFFSETS{(Dim == 2) ? 9 : 27};
                T s[NB_OFFSETS * Dim * Dim];

                __device__ __forceinline__ T operator()(int d0, int d1,
                                                        int d2, int a,
                                                        int b) const {
                    int off{d0};
                    if constexpr (Dim > 1) off = off * 3 + d1;
                    if constexpr (Dim > 2) off = off * 3 + d2;
                    return s[(off * Dim + a) * Dim + b];
                }
            };

            //! Per-axis frequency tables (device pointers) and mode counts.
            template <typename T>
            struct Axes {
                const T * q[3];
                int n[3];
            };

            constexpr int BLOCK_SIZE{256};
            //! Lines whose hoisted sums a block keeps in shared memory. A block
            //! of BLOCK_SIZE modes touches at most BLOCK_SIZE / n0 + 2 lines,
            //! so this covers every axis-0 length from about 40 modes up.
            constexpr int MAX_LINES{8};

            /**
             * One entry `P[d0][a][b] = sum_{d1,d2} phase * S[d0,d1,d2,a,b]` of
             * the hoisted partial sum, given the +1 phases of axes 1 and 2.
             */
            template <Dim_t Dim, typename T>
            __device__ __forceinline__ DevCplx<T>
            hoisted_entry(const Stencil<Dim, T> & S, DevCplx<T> p1,
                          DevCplx<T> p2, int d0, int a, int b) {
                DevCplx<T> acc{T(0), T(0)};
                for (int d1{0}; d1 < ((Dim > 1) ? 3 : 1); ++d1) {
                    for (int d2{0}; d2 < ((Dim > 2) ? 3 : 1); ++d2) {
                        DevCplx<T> w{T(1), T(0)};
                        if constexpr (Dim > 1) w = offset_phase(p1, d1);
                        if constexpr (Dim > 2)
                            w = cmul(w, offset_phase(p2, d2));
                        const T s{S(d0, d1, d2, a, b)};
                        acc.re += w.re * s;
                        acc.im += w.im * s;
                    }
                }
                return acc;
            }

            /**
             * Finish one mode: `K = sum_d0 phase0 * P[d0]`, invert by
             * cofactors, and replace `v` by `normalisation * K^-1 v`.
             *
             * `P` is an accessor, `P(d0, a, b)`, so the same code reads the
             * block's shared copy or the thread's own registers.
             */
            template <Dim_t Dim, typename T, class Get>
            __device__ __forceinline__ void
            finish_mode(DevCplx<T> * MUGRID_RESTRICT v,
                        Index_t stride_component, DevCplx<T> p0, Get P,
                        T normalisation) {
                DevCplx<T> K[Dim][Dim];
                for (int a{0}; a < Dim; ++a) {
                    for (int b{0}; b < Dim; ++b) {
                        DevCplx<T> acc{T(0), T(0)};
                        for (int d0{0}; d0 < 3; ++d0) {
                            cfma(acc, offset_phase(p0, d0), P(d0, a, b));
                        }
                        K[a][b] = acc;
                    }
                }

                // Cofactor inverse, as on the host.
                DevCplx<T> det{T(0), T(0)};
                if constexpr (Dim == 2) {
                    det = csub(cmul(K[0][0], K[1][1]), cmul(K[0][1], K[1][0]));
                } else {
                    for (int j{0}; j < 3; ++j) {
                        const int j1{(j + 1) % 3}, j2{(j + 2) % 3};
                        cfma(det, K[0][j],
                             csub(cmul(K[1][j1], K[2][j2]),
                                  cmul(K[1][j2], K[2][j1])));
                    }
                }
                const T mag{det.re * det.re + det.im * det.im};
                DevCplx<T> in[Dim];
                for (int a{0}; a < Dim; ++a) {
                    in[a] = v[a * stride_component];
                }
                if (!(mag > T(0))) {
                    for (int a{0}; a < Dim; ++a) {
                        v[a * stride_component] = DevCplx<T>{T(0), T(0)};
                    }
                    return;
                }
                // normalisation / det, folded into one scale
                const DevCplx<T> scale{normalisation * det.re / mag,
                                       -normalisation * det.im / mag};

                DevCplx<T> Kinv[Dim][Dim];
                if constexpr (Dim == 2) {
                    Kinv[0][0] = K[1][1];
                    Kinv[0][1] = DevCplx<T>{-K[0][1].re, -K[0][1].im};
                    Kinv[1][0] = DevCplx<T>{-K[1][0].re, -K[1][0].im};
                    Kinv[1][1] = K[0][0];
                } else {
                    for (int i{0}; i < 3; ++i) {
                        for (int j{0}; j < 3; ++j) {
                            const int i1{(i + 1) % 3}, i2{(i + 2) % 3};
                            const int j1{(j + 1) % 3}, j2{(j + 2) % 3};
                            // inverse is the *transposed* cofactor matrix
                            Kinv[i][j] = csub(cmul(K[j1][i1], K[j2][i2]),
                                              cmul(K[j1][i2], K[j2][i1]));
                        }
                    }
                }
                for (int a{0}; a < Dim; ++a) {
                    DevCplx<T> acc{T(0), T(0)};
                    for (int b{0}; b < Dim; ++b) {
                        cfma(acc, Kinv[a][b], in[b]);
                    }
                    v[a * stride_component] = cmul(acc, scale);
                }
            }

            template <Dim_t Dim, typename T>
            __global__ void __launch_bounds__(BLOCK_SIZE)
                green_symbol_kernel(DevCplx<T> * MUGRID_RESTRICT field,
                                    Index_t stride_component,
                                    Index_t stride_mode,
                                    const Stencil<Dim, T> S, const Axes<T> ax,
                                    T normalisation, Index_t nb_modes) {
                constexpr int NP{3 * Dim * Dim};
                __shared__ DevCplx<T> P_sh[MAX_LINES][NP];
                __shared__ bool zero_12_sh[MAX_LINES];

                // Line = flat index over axes 1 and 2. The host guarantees
                // n0 and the line count fit in an int, so the divisions below
                // are 32-bit; 64-bit division is a long instruction sequence
                // on a GPU and would sit on every thread.
                const int n0{ax.n[0]}, n1{ax.n[1]};
                const Index_t first{static_cast<Index_t>(blockIdx.x) *
                                    blockDim.x};
                const Index_t last{
                    (first + blockDim.x < nb_modes ? first + blockDim.x
                                                   : nb_modes) -
                    1};
                const int line_lo{static_cast<int>(first / n0)};
                const int nb_lines{static_cast<int>(last / n0) - line_lo + 1};
                // Block-uniform, so the __syncthreads below is safe.
                const bool shared{nb_lines <= MAX_LINES};

                if (shared) {
                    for (int w{static_cast<int>(threadIdx.x)};
                         w < nb_lines * NP; w += blockDim.x) {
                        const int j{w / NP}, e{w % NP};
                        const int line{line_lo + j};
                        const int i1{line % n1}, i2{line / n1};
                        const T f1{(Dim > 1) ? ax.q[1][i1] : T(0)};
                        const T f2{(Dim > 2) ? ax.q[2][i2] : T(0)};
                        P_sh[j][e] = hoisted_entry<Dim, T>(
                            S, unit_phase(f1), unit_phase(f2), e / (Dim * Dim),
                            (e / Dim) % Dim, e % Dim);
                        if (e == 0) {
                            zero_12_sh[j] = (f1 == T(0)) && (f2 == T(0));
                        }
                    }
                    __syncthreads();
                }

                const Index_t m{global_thread_x()};
                if (m >= nb_modes) {
                    return;
                }
                DevCplx<T> * v{field + m * stride_mode};

                // Position within the block's first line; small, so 32-bit.
                const int off{
                    static_cast<int>(first - static_cast<Index_t>(line_lo) * n0) +
                    static_cast<int>(threadIdx.x)};
                const int j{off / n0};
                const int i0{off - j * n0};
                const T f0{ax.q[0][i0]};
                const DevCplx<T> p0{unit_phase(f0)};

                if (shared) {
                    // The q = 0 block is the rigid-body null space; its
                    // pseudo-inverse is zero, as in the assembled path.
                    if (zero_12_sh[j] && f0 == T(0)) {
                        for (int a{0}; a < Dim; ++a) {
                            v[a * stride_component] = DevCplx<T>{T(0), T(0)};
                        }
                        return;
                    }
                    const DevCplx<T> * P{P_sh[j]};
                    finish_mode<Dim, T>(
                        v, stride_component, p0,
                        [P](int d0, int a, int b) {
                            return P[(d0 * Dim + a) * Dim + b];
                        },
                        normalisation);
                } else {
                    const int line{line_lo + j};
                    const int i1{line % n1}, i2{line / n1};
                    const T f1{(Dim > 1) ? ax.q[1][i1] : T(0)};
                    const T f2{(Dim > 2) ? ax.q[2][i2] : T(0)};
                    if (f0 == T(0) && f1 == T(0) && f2 == T(0)) {
                        for (int a{0}; a < Dim; ++a) {
                            v[a * stride_component] = DevCplx<T>{T(0), T(0)};
                        }
                        return;
                    }
                    const DevCplx<T> p1{unit_phase(f1)}, p2{unit_phase(f2)};
                    DevCplx<T> P[NP];
                    for (int e{0}; e < NP; ++e) {
                        P[e] = hoisted_entry<Dim, T>(S, p1, p2, e / (Dim * Dim),
                                                     (e / Dim) % Dim, e % Dim);
                    }
                    finish_mode<Dim, T>(
                        v, stride_component, p0,
                        [&P](int d0, int a, int b) {
                            return P[(d0 * Dim + a) * Dim + b];
                        },
                        normalisation);
                }
            }

        }  // namespace

        template <Dim_t Dim, typename T>
        void apply_inverse_gpu(std::complex<T> * field,
                               Index_t stride_component, Index_t stride_mode,
                               const T * stencil, const T * const * q,
                               const Index_t * nb_fourier_grid_pts,
                               T normalisation, Index_t nb_modes) {
            if (nb_modes == 0) {
                return;
            }
            Stencil<Dim, T> S;
            for (Index_t i{0}; i < Stencil<Dim, T>::NB_OFFSETS * Dim * Dim;
                 ++i) {
                S.s[i] = stencil[i];
            }
            Axes<T> ax{};
            Index_t nb_lines{1};
            for (Dim_t d{0}; d < 3; ++d) {
                const Index_t n{d < Dim ? nb_fourier_grid_pts[d] : 1};
                if (n > std::numeric_limits<int>::max()) {
                    throw RuntimeError("apply_inverse_gpu: axis " +
                                       std::to_string(d) + " has " +
                                       std::to_string(n) +
                                       " modes, more than an int holds");
                }
                ax.q[d] = d < Dim ? q[d] : nullptr;
                ax.n[d] = static_cast<int>(n);
                if (d > 0) {
                    nb_lines *= n;
                }
            }
            if (nb_lines > std::numeric_limits<int>::max()) {
                throw RuntimeError(
                    "apply_inverse_gpu: more axis-0 lines than an int holds");
            }
            const Index_t nb_blocks{(nb_modes + BLOCK_SIZE - 1) / BLOCK_SIZE};
            GPU_LAUNCH_KERNEL((green_symbol_kernel<Dim, T>), nb_blocks,
                              BLOCK_SIZE, reinterpret_cast<DevCplx<T> *>(field),
                              stride_component, stride_mode, S, ax,
                              normalisation, nb_modes);
        }

        template void apply_inverse_gpu<2, Real>(std::complex<Real> *, Index_t,
                                                 Index_t, const Real *,
                                                 const Real * const *,
                                                 const Index_t *, Real,
                                                 Index_t);
        template void apply_inverse_gpu<3, Real>(std::complex<Real> *, Index_t,
                                                 Index_t, const Real *,
                                                 const Real * const *,
                                                 const Index_t *, Real,
                                                 Index_t);
        template void apply_inverse_gpu<2, Real32>(std::complex<Real32> *,
                                                   Index_t, Index_t,
                                                   const Real32 *,
                                                   const Real32 * const *,
                                                   const Index_t *, Real32,
                                                   Index_t);
        template void apply_inverse_gpu<3, Real32>(std::complex<Real32> *,
                                                   Index_t, Index_t,
                                                   const Real32 *,
                                                   const Real32 * const *,
                                                   const Index_t *, Real32,
                                                   Index_t);

    }  // namespace green_symbol

}  // namespace muGrid

#endif  // MUGRID_ENABLE_CUDA || MUGRID_ENABLE_HIP
