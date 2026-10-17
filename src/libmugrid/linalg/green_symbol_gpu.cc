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
 * For a float64 field the symbol is built and inverted in double. For a
 * float32 field every mode is first finished in single precision, which is
 * what consumer GPUs with 1:64 double throughput need, and the attempt is
 * kept only where an a-posteriori test on the block shows it accurate to
 * 1e-4; elsewhere -- the long wavelengths, where the phase sum cancels (see
 * `green_symbol.hh`) -- the mode is redone in double. For an isotropic
 * reference medium that is about 0.05% of the modes.
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

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>

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
             * Accuracy a single-precision attempt must reach to be kept, as a
             * relative error of the mode's result. A preconditioner this close
             * to the exact inverse changes the CG iteration count by O(1e-4);
             * the attainable accuracy is set by the operator, not by it.
             */
            constexpr Real FLOAT_PATH_TOLERANCE{1e-4};
            /**
             * Margin on the error model of the single-precision attempt. The
             * relative error of a mode was measured at up to 0.4 E / b (see
             * `finish_mode`) over isotropic, high-contrast (lambda/mu = 100)
             * and anisotropic (h ratio 10) stencils in 2D and 3D, so this is a
             * factor of five.
             */
            constexpr Real FLOAT_PATH_SAFETY{2};

            /**
             * One entry `P[d0][a][b] = sum_{d1,d2} phase * S[d0,d1,d2,a,b]` of
             * the hoisted partial sum, given the +1 phases of axes 1 and 2.
             */
            template <Dim_t Dim, typename C>
            __device__ __forceinline__ DevCplx<C>
            hoisted_entry(const Stencil<Dim, C> & S, DevCplx<C> p1,
                          DevCplx<C> p2, int d0, int a, int b) {
                DevCplx<C> acc{C(0), C(0)};
                for (int d1{0}; d1 < ((Dim > 1) ? 3 : 1); ++d1) {
                    for (int d2{0}; d2 < ((Dim > 2) ? 3 : 1); ++d2) {
                        DevCplx<C> w{C(1), C(0)};
                        if constexpr (Dim > 1) w = offset_phase(p1, d1);
                        if constexpr (Dim > 2)
                            w = cmul(w, offset_phase(p2, d2));
                        const C s{S(d0, d1, d2, a, b)};
                        acc.re += w.re * s;
                        acc.im += w.im * s;
                    }
                }
                return acc;
            }

            //! Write zero to every component of a mode.
            template <Dim_t Dim, typename T>
            __device__ __forceinline__ void
            zero_mode(DevCplx<T> * MUGRID_RESTRICT v,
                      Index_t stride_component) {
                for (int a{0}; a < Dim; ++a) {
                    v[a * stride_component] = DevCplx<T>{T(0), T(0)};
                }
            }

            /**
             * Finish one mode in precision `C`: `K = sum_d0 phase0 * P[d0]`,
             * invert by cofactors, and replace `v` by
             * `normalisation * K^-1 v`.
             *
             * `P` is an accessor, `P(d0, a, b)`, so the same code reads the
             * block's shared copy or the thread's own registers.
             *
             * With `threshold < 0` this is the final attempt: it always writes
             * `v`, and a block too small to invert is treated as the null
             * space and zeroed. Otherwise it is a single-precision attempt
             * that writes `v` -- and returns true -- only if the result is
             * accurate to about `FLOAT_PATH_TOLERANCE`, and leaves `v`
             * untouched otherwise.
             *
             * The test is a posteriori, on the K just built. Rounding the
             * phase sum leaves an absolute error of about
             * `E = eps * max_ab sum_d |S[d][a][b]|` in K, and a relative error
             * in the result of up to 0.4 E / lambda_min(K) (measured; see
             * FLOAT_PATH_SAFETY). For a Hermitian positive definite block,
             * `lambda_min >= b = det / e2` with `e2` the sum of the principal
             * minors -- the diagonal cofactors in 3D, the trace in 2D -- which
             * the inversion forms anyway, and `b` is within a factor of `Dim`
             * of `lambda_min`. `threshold` is `SAFETY * E / TOLERANCE`, so the
             * attempt is kept iff the float K is positive definite and
             * `b >= threshold`. A block whose float K is too inaccurate to be
             * trusted cannot pass: either it is indefinite, or its
             * `lambda_min` is within E of the true one, which is then below
             * E, and `b` with it.
             */
            template <Dim_t Dim, typename C, typename T, class Get>
            __device__ __forceinline__ bool
            finish_mode(DevCplx<T> * MUGRID_RESTRICT v,
                        Index_t stride_component, DevCplx<C> p0, Get P,
                        C normalisation, C threshold) {
                const bool final_attempt{!(threshold >= C(0))};
                DevCplx<C> K[Dim][Dim];
                for (int a{0}; a < Dim; ++a) {
                    for (int b{0}; b < Dim; ++b) {
                        DevCplx<C> acc{C(0), C(0)};
                        for (int d0{0}; d0 < 3; ++d0) {
                            cfma(acc, offset_phase(p0, d0), P(d0, a, b));
                        }
                        K[a][b] = acc;
                    }
                }

                // Cofactor inverse, as on the host: scale the block by its
                // largest entry first, so |det|^2 cannot underflow at low
                // frequency, where the entries are O(S q^2).
                C kscale{0};
                for (int a{0}; a < Dim; ++a) {
                    for (int b{0}; b < Dim; ++b) {
                        kscale = fmax(kscale, fmax(fabs(K[a][b].re),
                                                   fabs(K[a][b].im)));
                    }
                }
                if (!(kscale > C(0))) {
                    if (final_attempt) {
                        zero_mode<Dim>(v, stride_component);
                    }
                    return final_attempt;
                }
                const C inv_kscale{C(1) / kscale};
                for (int a{0}; a < Dim; ++a) {
                    for (int b{0}; b < Dim; ++b) {
                        K[a][b].re *= inv_kscale;
                        K[a][b].im *= inv_kscale;
                    }
                }

                // The inverse is the *transposed* cofactor matrix.
                DevCplx<C> Kinv[Dim][Dim];
                DevCplx<C> det{C(0), C(0)};
                if constexpr (Dim == 2) {
                    Kinv[0][0] = K[1][1];
                    Kinv[0][1] = DevCplx<C>{-K[0][1].re, -K[0][1].im};
                    Kinv[1][0] = DevCplx<C>{-K[1][0].re, -K[1][0].im};
                    Kinv[1][1] = K[0][0];
                    det = csub(cmul(K[0][0], K[1][1]), cmul(K[0][1], K[1][0]));
                } else {
                    for (int i{0}; i < 3; ++i) {
                        for (int j{0}; j < 3; ++j) {
                            const int i1{(i + 1) % 3}, i2{(i + 2) % 3};
                            const int j1{(j + 1) % 3}, j2{(j + 2) % 3};
                            Kinv[i][j] = csub(cmul(K[j1][i1], K[j2][i2]),
                                              cmul(K[j1][i2], K[j2][i1]));
                        }
                    }
                    for (int j{0}; j < 3; ++j) {
                        cfma(det, K[0][j], Kinv[j][0]);
                    }
                }
                const C mag{det.re * det.re + det.im * det.im};
                if (!(mag > C(0))) {
                    if (final_attempt) {
                        zero_mode<Dim>(v, stride_component);
                    }
                    return final_attempt;
                }
                if (!final_attempt) {
                    // The bound needs a positive definite block, checked by
                    // Sylvester's criterion on the leading minors. Without
                    // it, a float K wrong enough to have two negative
                    // eigenvalues still has det > 0 and can have e2 > 0 and
                    // a large det / e2.
                    bool definite{K[0][0].re > C(0) && det.re > C(0)};
                    if constexpr (Dim == 3) {
                        definite = definite && Kinv[2][2].re > C(0);
                    }
                    // b = kscale * det / e2 in the scaled block's terms
                    C e2{0};
                    if constexpr (Dim == 2) {
                        e2 = K[0][0].re + K[1][1].re;
                    } else {
                        e2 = Kinv[0][0].re + Kinv[1][1].re + Kinv[2][2].re;
                    }
                    if (!(definite && det.re * kscale >= threshold * e2)) {
                        return false;
                    }
                }

                // normalisation / (kscale * det), folded into one scale
                const C f{normalisation * inv_kscale / mag};
                const DevCplx<C> scale{det.re * f, -det.im * f};
                DevCplx<T> in[Dim];
                for (int a{0}; a < Dim; ++a) {
                    in[a] = v[a * stride_component];
                }
                for (int a{0}; a < Dim; ++a) {
                    DevCplx<C> acc{C(0), C(0)};
                    for (int b{0}; b < Dim; ++b) {
                        cfma(acc, Kinv[a][b],
                             DevCplx<C>{static_cast<C>(in[b].re),
                                        static_cast<C>(in[b].im)});
                    }
                    const DevCplx<C> out{cmul(acc, scale)};
                    v[a * stride_component] = DevCplx<T>{
                        static_cast<T>(out.re), static_cast<T>(out.im)};
                }
                return true;
            }

            /**
             * Single-precision data for the first attempt on a float32 field:
             * the stencil, the acceptance threshold and the normalisation, all
             * rounded on the host so no thread converts from double for them.
             */
            template <Dim_t Dim>
            struct FloatPath {
                Stencil<Dim, Real32> S;
                Real32 threshold;
                Real32 normalisation;
            };

            /**
             * For a float64 field every mode is finished in double. For a
             * float32 field every mode is first finished in single precision
             * and kept if `finish_mode` accepts it; the rest -- the long
             * wavelengths, where the phase sum cancels -- are redone in
             * double. Those are few and lie at the start of few axis-0 lines,
             * so the double work lands in few blocks and few warps.
             *
             * `SHARED` selects how the hoisted sums are built: once per line
             * into shared memory, or -- when axis 0 is so short that a block
             * could touch more than `MAX_LINES` lines -- by every thread for
             * itself. It depends only on the axis-0 length, so the host picks
             * it per launch, and the two paths do not share a register
             * allocation: the per-thread arrays of the second would otherwise
             * cost the first occupancy.
             */
            template <Dim_t Dim, typename T, bool SHARED>
            __global__ void __launch_bounds__(BLOCK_SIZE)
                green_symbol_kernel(DevCplx<T> * MUGRID_RESTRICT field,
                                    Index_t stride_component,
                                    Index_t stride_mode,
                                    const Stencil<Dim, Real> S,
                                    const FloatPath<Dim> F,
                                    const Axes<Real> ax, Real normalisation,
                                    Index_t nb_modes) {
                constexpr bool MIXED{std::is_same_v<T, Real32>};
                //! precision of the first attempt
                using W = std::conditional_t<MIXED, Real32, Real>;
                constexpr int NP{3 * Dim * Dim};
                constexpr int NL{SHARED ? MAX_LINES : 1};
                __shared__ DevCplx<W> P_sh[NL][NP];
                // hoisted sums in double for the lines that need a retry
                __shared__ DevCplx<Real> Pd_sh[MIXED ? NL : 1][NP];
                __shared__ bool zero_12_sh[NL];
                __shared__ bool retry_sh[NL];

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

                const Index_t m{global_thread_x()};
                const bool active{m < nb_modes};
                DevCplx<T> * v{field + (active ? m : 0) * stride_mode};
                // Position within the block's first line; small, so 32-bit.
                const int off{
                    static_cast<int>(first - static_cast<Index_t>(line_lo) * n0) +
                    static_cast<int>(threadIdx.x)};
                const int j{off / n0};
                const int i0{off - j * n0};
                const Real f0{active ? ax.q[0][i0] : Real(0)};

                if constexpr (SHARED) {
                    for (int w{static_cast<int>(threadIdx.x)};
                         w < nb_lines * NP; w += blockDim.x) {
                        const int jj{w / NP}, e{w % NP};
                        const int line{line_lo + jj};
                        const int i1{line % n1}, i2{line / n1};
                        const Real f1{(Dim > 1) ? ax.q[1][i1] : Real(0)};
                        const Real f2{(Dim > 2) ? ax.q[2][i2] : Real(0)};
                        if constexpr (MIXED) {
                            P_sh[jj][e] = hoisted_entry<Dim, Real32>(
                                F.S, unit_phase(static_cast<Real32>(f1)),
                                unit_phase(static_cast<Real32>(f2)),
                                e / (Dim * Dim), (e / Dim) % Dim, e % Dim);
                        } else {
                            P_sh[jj][e] = hoisted_entry<Dim, Real>(
                                S, unit_phase(f1), unit_phase(f2),
                                e / (Dim * Dim), (e / Dim) % Dim, e % Dim);
                        }
                        if (e == 0) {
                            zero_12_sh[jj] = (f1 == Real(0)) && (f2 == Real(0));
                            retry_sh[jj] = false;
                        }
                    }
                    __syncthreads();

                    bool retry{false};
                    // The q = 0 block is the rigid-body null space; its
                    // pseudo-inverse is zero, as in the assembled path.
                    if (active && zero_12_sh[j] && f0 == Real(0)) {
                        zero_mode<Dim>(v, stride_component);
                    } else if (active) {
                        const DevCplx<W> * P{P_sh[j]};
                        auto get = [P](int d0, int a, int b) {
                            return P[(d0 * Dim + a) * Dim + b];
                        };
                        if constexpr (MIXED) {
                            retry = !finish_mode<Dim, Real32>(
                                v, stride_component,
                                unit_phase(static_cast<Real32>(f0)), get,
                                F.normalisation, F.threshold);
                            if (retry) {
                                retry_sh[j] = true;
                            }
                        } else {
                            finish_mode<Dim, Real>(v, stride_component,
                                                   unit_phase(f0), get,
                                                   normalisation, Real(-1));
                        }
                    }

                    if constexpr (MIXED) {
                        // Block-uniform: true in the few blocks that hold a
                        // mode the single-precision attempt rejected.
                        if (__syncthreads_or(retry)) {
                            for (int w{static_cast<int>(threadIdx.x)};
                                 w < nb_lines * NP; w += blockDim.x) {
                                const int jj{w / NP}, e{w % NP};
                                if (!retry_sh[jj]) {
                                    continue;
                                }
                                const int line{line_lo + jj};
                                const int i1{line % n1}, i2{line / n1};
                                const Real f1{(Dim > 1) ? ax.q[1][i1]
                                                        : Real(0)};
                                const Real f2{(Dim > 2) ? ax.q[2][i2]
                                                        : Real(0)};
                                Pd_sh[jj][e] = hoisted_entry<Dim, Real>(
                                    S, unit_phase(f1), unit_phase(f2),
                                    e / (Dim * Dim), (e / Dim) % Dim, e % Dim);
                            }
                            __syncthreads();
                            if (retry) {
                                const DevCplx<Real> * P{Pd_sh[j]};
                                finish_mode<Dim, Real>(
                                    v, stride_component, unit_phase(f0),
                                    [P](int d0, int a, int b) {
                                        return P[(d0 * Dim + a) * Dim + b];
                                    },
                                    normalisation, Real(-1));
                            }
                        }
                    }
                } else {
                    // Each thread builds its own hoisted sums, first in the
                    // attempt's precision.
                    if (!active) {
                        return;
                    }
                    const int line{line_lo + j};
                    const int i1{line % n1}, i2{line / n1};
                    const Real f1{(Dim > 1) ? ax.q[1][i1] : Real(0)};
                    const Real f2{(Dim > 2) ? ax.q[2][i2] : Real(0)};
                    if (f0 == Real(0) && f1 == Real(0) && f2 == Real(0)) {
                        zero_mode<Dim>(v, stride_component);
                        return;
                    }
                    if constexpr (MIXED) {
                        const DevCplx<Real32> p1{
                            unit_phase(static_cast<Real32>(f1))},
                            p2{unit_phase(static_cast<Real32>(f2))};
                        DevCplx<Real32> P[NP];
                        for (int e{0}; e < NP; ++e) {
                            P[e] = hoisted_entry<Dim, Real32>(
                                F.S, p1, p2, e / (Dim * Dim), (e / Dim) % Dim,
                                e % Dim);
                        }
                        if (finish_mode<Dim, Real32>(
                                v, stride_component,
                                unit_phase(static_cast<Real32>(f0)),
                                [&P](int d0, int a, int b) {
                                    return P[(d0 * Dim + a) * Dim + b];
                                },
                                F.normalisation, F.threshold)) {
                            return;
                        }
                    }
                    const DevCplx<Real> p1{unit_phase(f1)}, p2{unit_phase(f2)};
                    DevCplx<Real> P[NP];
                    for (int e{0}; e < NP; ++e) {
                        P[e] = hoisted_entry<Dim, Real>(S, p1, p2, e / (Dim * Dim),
                                                        (e / Dim) % Dim, e % Dim);
                    }
                    finish_mode<Dim, Real>(
                        v, stride_component, unit_phase(f0),
                        [&P](int d0, int a, int b) {
                            return P[(d0 * Dim + a) * Dim + b];
                        },
                        normalisation, Real(-1));
                }
            }

        }  // namespace

        template <Dim_t Dim, typename T>
        void apply_inverse_gpu(std::complex<T> * field,
                               Index_t stride_component, Index_t stride_mode,
                               const Real * stencil, const Real * const * q,
                               const Index_t * nb_fourier_grid_pts,
                               Real normalisation, Index_t nb_modes) {
            if (nb_modes == 0) {
                return;
            }
            Stencil<Dim, Real> S;
            FloatPath<Dim> F;
            constexpr Index_t NB_OFFSETS{Stencil<Dim, Real>::NB_OFFSETS};
            for (Index_t i{0}; i < NB_OFFSETS * Dim * Dim; ++i) {
                S.s[i] = stencil[i];
                F.S.s[i] = static_cast<Real32>(stencil[i]);
            }
            // E / eps: the largest absolute sum of a block entry over the
            // offsets, which bounds the rounding error of the phase sum
            Real abs_sum_max{0};
            for (Index_t ab{0}; ab < Dim * Dim; ++ab) {
                Real abs_sum{0};
                for (Index_t d{0}; d < NB_OFFSETS; ++d) {
                    abs_sum += std::abs(stencil[d * Dim * Dim + ab]);
                }
                abs_sum_max = std::max(abs_sum_max, abs_sum);
            }
            F.threshold = static_cast<Real32>(
                FLOAT_PATH_SAFETY * std::numeric_limits<Real32>::epsilon() *
                abs_sum_max / FLOAT_PATH_TOLERANCE);
            F.normalisation = static_cast<Real32>(normalisation);

            Axes<Real> ax{};
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
            // Most lines a block of BLOCK_SIZE consecutive modes can touch
            const Index_t n0{ax.n[0]};
            const bool shared{(BLOCK_SIZE - 1) / n0 + 2 <= MAX_LINES};
            auto * out{reinterpret_cast<DevCplx<T> *>(field)};
            if (shared) {
                GPU_LAUNCH_KERNEL((green_symbol_kernel<Dim, T, true>),
                                  nb_blocks, BLOCK_SIZE, out, stride_component,
                                  stride_mode, S, F, ax, normalisation,
                                  nb_modes);
            } else {
                GPU_LAUNCH_KERNEL((green_symbol_kernel<Dim, T, false>),
                                  nb_blocks, BLOCK_SIZE, out, stride_component,
                                  stride_mode, S, F, ax, normalisation,
                                  nb_modes);
            }
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
                                                   const Real *,
                                                   const Real * const *,
                                                   const Index_t *, Real,
                                                   Index_t);
        template void apply_inverse_gpu<3, Real32>(std::complex<Real32> *,
                                                   Index_t, Index_t,
                                                   const Real *,
                                                   const Real * const *,
                                                   const Index_t *, Real,
                                                   Index_t);

    }  // namespace green_symbol

}  // namespace muGrid

#endif  // MUGRID_ENABLE_CUDA || MUGRID_ENABLE_HIP
