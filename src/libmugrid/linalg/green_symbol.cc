/**
 * @file   linalg/green_symbol.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   24 Sep 2026
 *
 * @brief  Host implementation of the per-mode inverse-symbol application
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
 */

#include "green_symbol.hh"

#include <cmath>

namespace muGrid {

    namespace green_symbol {

        namespace {

            /**
             * Complex arithmetic on the real and imaginary parts, for the same
             * reason as in block_thomas.cc: `std::complex<T>::operator*`
             * carries the Annex G infinity and NaN handling, which compiles to
             * a `__mul?c3` call. That call would sit in the innermost loop.
             */
            template <typename T>
            struct Cplx {
                T re{}, im{};
            };

            template <typename T>
            inline Cplx<T> cmul(Cplx<T> a, Cplx<T> b) {
                return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
            }

            template <typename T>
            inline Cplx<T> cadd(Cplx<T> a, Cplx<T> b) {
                return {a.re + b.re, a.im + b.im};
            }

            template <typename T>
            inline Cplx<T> csub(Cplx<T> a, Cplx<T> b) {
                return {a.re - b.re, a.im - b.im};
            }

            template <typename T>
            inline Cplx<T> cscale(Cplx<T> a, T s) {
                return {a.re * s, a.im * s};
            }

            //! a * b, accumulated into acc
            template <typename T>
            inline void cfma(Cplx<T> & acc, Cplx<T> a, Cplx<T> b) {
                acc.re += a.re * b.re - a.im * b.im;
                acc.im += a.re * b.im + a.im * b.re;
            }

            /**
             * Inverse of a Dim x Dim complex matrix by cofactors.
             *
             * Dim is 2 or 3, so a closed form is both shorter and faster than
             * any factorisation, and it keeps the whole thing in registers --
             * which is the point of evaluating the symbol per mode rather than
             * storing it. Returns false if the matrix is singular to working
             * precision, which the caller treats as the q = 0 null space.
             */
            template <Dim_t Dim, typename T>
            inline bool invert(const Cplx<T> (&m)[Dim][Dim],
                               Cplx<T> (&out)[Dim][Dim]) {
                Cplx<T> det{};
                if constexpr (Dim == 2) {
                    det = csub(cmul(m[0][0], m[1][1]), cmul(m[0][1], m[1][0]));
                } else {
                    for (Dim_t j = 0; j < 3; ++j) {
                        const Dim_t j1 = (j + 1) % 3, j2 = (j + 2) % 3;
                        const Cplx<T> cof =
                            csub(cmul(m[1][j1], m[2][j2]),
                                 cmul(m[1][j2], m[2][j1]));
                        cfma(det, m[0][j], cof);
                    }
                }
                const T mag = det.re * det.re + det.im * det.im;
                if (!(mag > T(0))) {
                    return false;
                }
                // 1 / det, by the conjugate over the squared magnitude
                const Cplx<T> inv_det{det.re / mag, -det.im / mag};

                if constexpr (Dim == 2) {
                    out[0][0] = cmul(m[1][1], inv_det);
                    out[0][1] = cmul(Cplx<T>{-m[0][1].re, -m[0][1].im},
                                     inv_det);
                    out[1][0] = cmul(Cplx<T>{-m[1][0].re, -m[1][0].im},
                                     inv_det);
                    out[1][1] = cmul(m[0][0], inv_det);
                } else {
                    for (Dim_t i = 0; i < 3; ++i) {
                        for (Dim_t j = 0; j < 3; ++j) {
                            const Dim_t i1 = (i + 1) % 3, i2 = (i + 2) % 3;
                            const Dim_t j1 = (j + 1) % 3, j2 = (j + 2) % 3;
                            // inverse is the *transposed* cofactor matrix
                            const Cplx<T> cof =
                                csub(cmul(m[j1][i1], m[j2][i2]),
                                     cmul(m[j1][i2], m[j2][i1]));
                            out[i][j] = cmul(cof, inv_det);
                        }
                    }
                }
                return true;
            }

            //! exp(-2 pi i f), the phase of a unit offset at frequency f
            template <typename T>
            inline Cplx<T> unit_phase(T f) {
                const T theta = T(-2) * T(M_PI) * f;
                return {std::cos(theta), std::sin(theta)};
            }

        }  // namespace

        template <Dim_t Dim, typename T>
        void apply_inverse(std::complex<T> * field, Index_t stride_component,
                           Index_t stride_mode, const T * stencil,
                           const T * const * q,
                           const Index_t * nb_fourier_grid_pts,
                           T normalisation, Index_t nb_modes) {
            constexpr Dim_t NB_OFF = (Dim == 2) ? 9 : 27;
            (void)NB_OFF;
            const Index_t n0 = nb_fourier_grid_pts[0];
            const Index_t n1 = (Dim > 1) ? nb_fourier_grid_pts[1] : 1;
            const Index_t n2 = (Dim > 2) ? nb_fourier_grid_pts[2] : 1;

            // Stencil entry S[d0][d1][d2][a][b], C order, offsets slowest.
            auto S = [stencil](Dim_t d0, Dim_t d1, Dim_t d2, Dim_t a,
                               Dim_t b) -> T {
                Index_t off = d0;
                if constexpr (Dim > 1) off = off * 3 + d1;
                if constexpr (Dim > 2) off = off * 3 + d2;
                return stencil[(off * Dim + a) * Dim + b];
            };

            // The modes along axis 0 share q1 and q2, so the sums over d1 and
            // d2 are loop-invariant there. Hoisting them into P[d0] turns the
            // per-mode symbol build from 3^Dim * Dim^2 complex FMAs into
            // 3 * Dim^2 -- a factor 9 in 3D, and the difference between this
            // kernel being competitive with reading a stored symbol and being
            // several times slower than it.
            Cplx<T> P[3][Dim][Dim];

            for (Index_t i2 = 0; i2 < n2; ++i2) {
                const T f2 = (Dim > 2) ? q[2][i2] : T(0);
                const Cplx<T> p2 = unit_phase(f2);
                for (Index_t i1 = 0; i1 < n1; ++i1) {
                    const T f1 = (Dim > 1) ? q[1][i1] : T(0);
                    const Cplx<T> p1 = unit_phase(f1);

                    // phases for d = -1, 0, +1 along each of the hoisted axes;
                    // d = 0 is unity and d = -1 is the conjugate of d = +1.
                    const Cplx<T> ph1[3] = {{p1.re, -p1.im}, {T(1), T(0)}, p1};
                    const Cplx<T> ph2[3] = {{p2.re, -p2.im}, {T(1), T(0)}, p2};

                    for (Dim_t d0 = 0; d0 < 3; ++d0) {
                        for (Dim_t a = 0; a < Dim; ++a) {
                            for (Dim_t b = 0; b < Dim; ++b) {
                                P[d0][a][b] = Cplx<T>{};
                            }
                        }
                        for (Dim_t d1 = 0; d1 < ((Dim > 1) ? 3 : 1); ++d1) {
                            for (Dim_t d2 = 0; d2 < ((Dim > 2) ? 3 : 1);
                                 ++d2) {
                                Cplx<T> w{T(1), T(0)};
                                if constexpr (Dim > 1) w = cmul(w, ph1[d1]);
                                if constexpr (Dim > 2) w = cmul(w, ph2[d2]);
                                for (Dim_t a = 0; a < Dim; ++a) {
                                    for (Dim_t b = 0; b < Dim; ++b) {
                                        const T s = S(d0, d1, d2, a, b);
                                        P[d0][a][b].re += w.re * s;
                                        P[d0][a][b].im += w.im * s;
                                    }
                                }
                            }
                        }
                    }

                    const bool zero_12 = (f1 == T(0)) && (f2 == T(0));
                    for (Index_t i0 = 0; i0 < n0; ++i0) {
                        const T f0 = q[0][i0];
                        const Cplx<T> p0 = unit_phase(f0);
                        const Cplx<T> ph0[3] = {
                            {p0.re, -p0.im}, {T(1), T(0)}, p0};

                        const Index_t mode = i0 + n0 * (i1 + n1 * i2);
                        std::complex<T> * v = field + mode * stride_mode;

                        // The q = 0 block is the operator's rigid-body null
                        // space; its pseudo-inverse is zero, matching what the
                        // assembled path stores.
                        if (zero_12 && f0 == T(0)) {
                            for (Dim_t a = 0; a < Dim; ++a) {
                                v[a * stride_component] = std::complex<T>{};
                            }
                            continue;
                        }

                        Cplx<T> K[Dim][Dim];
                        for (Dim_t a = 0; a < Dim; ++a) {
                            for (Dim_t b = 0; b < Dim; ++b) {
                                Cplx<T> acc{};
                                for (Dim_t d0 = 0; d0 < 3; ++d0) {
                                    cfma(acc, ph0[d0], P[d0][a][b]);
                                }
                                K[a][b] = acc;
                            }
                        }

                        Cplx<T> Kinv[Dim][Dim];
                        if (!invert<Dim, T>(K, Kinv)) {
                            for (Dim_t a = 0; a < Dim; ++a) {
                                v[a * stride_component] = std::complex<T>{};
                            }
                            continue;
                        }

                        Cplx<T> in[Dim];
                        for (Dim_t a = 0; a < Dim; ++a) {
                            const std::complex<T> & z =
                                v[a * stride_component];
                            in[a] = Cplx<T>{z.real(), z.imag()};
                        }
                        for (Dim_t a = 0; a < Dim; ++a) {
                            Cplx<T> acc{};
                            for (Dim_t b = 0; b < Dim; ++b) {
                                cfma(acc, Kinv[a][b], in[b]);
                            }
                            acc = cscale(acc, normalisation);
                            v[a * stride_component] =
                                std::complex<T>{acc.re, acc.im};
                        }
                    }
                }
            }
            (void)nb_modes;
        }

        /* ---- explicit instantiations --------------------------------- */

        template void apply_inverse<2, Real>(std::complex<Real> *, Index_t,
                                             Index_t, const Real *,
                                             const Real * const *,
                                             const Index_t *, Real, Index_t);
        template void apply_inverse<3, Real>(std::complex<Real> *, Index_t,
                                             Index_t, const Real *,
                                             const Real * const *,
                                             const Index_t *, Real, Index_t);
        template void apply_inverse<2, Real32>(std::complex<Real32> *, Index_t,
                                               Index_t, const Real32 *,
                                               const Real32 * const *,
                                               const Index_t *, Real32,
                                               Index_t);
        template void apply_inverse<3, Real32>(std::complex<Real32> *, Index_t,
                                               Index_t, const Real32 *,
                                               const Real32 * const *,
                                               const Index_t *, Real32,
                                               Index_t);

    }  // namespace green_symbol

}  // namespace muGrid
