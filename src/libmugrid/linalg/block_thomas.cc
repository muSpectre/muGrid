/**
 * @file   block_thomas.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   19 Sep 2026
 *
 * @brief  Host implementation of the fused block-Thomas sweep
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

#include "block_thomas.hh"

#include <algorithm>

namespace muGrid {

    namespace block_thomas {

        namespace {

            /**
             * Complex arithmetic written out on the real and imaginary parts.
             *
             * `std::complex<T>::operator*` carries the Annex G infinity and
             * NaN handling, which compiles to a call to `__mul?c3` unless the
             * translation unit opts out. That call sits in the innermost loop
             * here, so the products are written by hand instead.
             */
            template <typename T>
            struct Cplx {
                T re{}, im{};
            };

            template <typename T>
            inline Cplx<T> load(const std::complex<T> & z) {
                return Cplx<T>{z.real(), z.imag()};
            }

            template <typename T>
            inline void store(std::complex<T> & z, const Cplx<T> & c) {
                z.real(c.re);
                z.imag(c.im);
            }

            //! `acc -= a * b`
            template <typename T>
            inline void fnma(Cplx<T> & acc, const Cplx<T> & a,
                             const Cplx<T> & b) {
                acc.re -= a.re * b.re - a.im * b.im;
                acc.im -= a.re * b.im + a.im * b.re;
            }

            //! `acc += a * b`
            template <typename T>
            inline void fma(Cplx<T> & acc, const Cplx<T> & a,
                            const Cplx<T> & b) {
                acc.re += a.re * b.re - a.im * b.im;
                acc.im += a.re * b.im + a.im * b.re;
            }

            /**
             * Diagonal inverse of plane @p plane for mode @p mode.
             *
             * Modes that reached the factor recurrence's fixed point read the
             * compressed head, clamped at its last entry; the slow-converging
             * minority keep a full line in the exception table.
             */
            template <typename T>
            inline const std::complex<T> *
            diagonal_inverse(const std::complex<T> * head,
                             const std::complex<T> * exc, int slot,
                             Index_t plane, Index_t mode, Index_t nb_modes,
                             Index_t nb_head, Index_t nb_exc, Index_t block) {
                if (slot >= 0) {
                    return exc + (plane * nb_exc + slot) * block;
                }
                const Index_t clamped{std::min(plane, nb_head - 1)};
                return head + (clamped * nb_modes + mode) * block;
            }

        }  // namespace

        template <Dim_t Dim, typename T>
        void sweep(const std::complex<T> * MUGRID_RESTRICT rhs,
                   const std::complex<T> * MUGRID_RESTRICT head,
                   const std::complex<T> * MUGRID_RESTRICT exc,
                   const int * MUGRID_RESTRICT exc_index,
                   const std::complex<T> * MUGRID_RESTRICT A0,
                   const std::complex<T> * MUGRID_RESTRICT A2,
                   std::complex<T> * MUGRID_RESTRICT y,
                   std::complex<T> * MUGRID_RESTRICT out,
                   Index_t nz, Index_t nb_modes, Index_t nb_head,
                   Index_t nb_exc) {
            constexpr Index_t BLOCK{Dim * Dim};

            // Forward: y_0 = rhs_0, y_k = rhs_k - A2 (Dinv_{k-1} y_{k-1}).
            std::copy(rhs, rhs + nb_modes * Dim, y);

            for (Index_t k{1}; k < nz; ++k) {
                const Index_t prev_plane{(k - 1) * nb_modes * Dim};
                const Index_t plane{k * nb_modes * Dim};
                for (Index_t m{0}; m < nb_modes; ++m) {
                    const int slot{exc_index[m]};
                    const std::complex<T> * dinv{diagonal_inverse<T>(
                        head, exc, slot, k - 1, m, nb_modes, nb_head, nb_exc,
                        BLOCK)};
                    const std::complex<T> * a2{A2 + m * BLOCK};

                    Cplx<T> previous[Dim];
                    for (Index_t i{0}; i < Dim; ++i) {
                        previous[i] = load(y[prev_plane + m * Dim + i]);
                    }

                    Cplx<T> tmp[Dim];
                    for (Index_t i{0}; i < Dim; ++i) {
                        Cplx<T> acc{};
                        for (Index_t j{0}; j < Dim; ++j) {
                            fma(acc, load(dinv[i * Dim + j]), previous[j]);
                        }
                        tmp[i] = acc;
                    }

                    for (Index_t i{0}; i < Dim; ++i) {
                        Cplx<T> acc{load(rhs[plane + m * Dim + i])};
                        for (Index_t j{0}; j < Dim; ++j) {
                            fnma(acc, load(a2[i * Dim + j]), tmp[j]);
                        }
                        store(y[plane + m * Dim + i], acc);
                    }
                }
            }

            // Backward: out_k = Dinv_k (y_k - A0 out_{k+1}).
            for (Index_t k{nz - 1}; k >= 0; --k) {
                const Index_t plane{k * nb_modes * Dim};
                const Index_t next_plane{(k + 1) * nb_modes * Dim};
                const bool has_next{k < nz - 1};
                for (Index_t m{0}; m < nb_modes; ++m) {
                    const int slot{exc_index[m]};
                    const std::complex<T> * dinv{diagonal_inverse<T>(
                        head, exc, slot, k, m, nb_modes, nb_head, nb_exc,
                        BLOCK)};
                    const std::complex<T> * a0{A0 + m * BLOCK};

                    Cplx<T> rhs_k[Dim];
                    for (Index_t i{0}; i < Dim; ++i) {
                        rhs_k[i] = load(y[plane + m * Dim + i]);
                    }
                    if (has_next) {
                        Cplx<T> next[Dim];
                        for (Index_t j{0}; j < Dim; ++j) {
                            next[j] = load(out[next_plane + m * Dim + j]);
                        }
                        for (Index_t i{0}; i < Dim; ++i) {
                            for (Index_t j{0}; j < Dim; ++j) {
                                fnma(rhs_k[i], load(a0[i * Dim + j]), next[j]);
                            }
                        }
                    }

                    for (Index_t i{0}; i < Dim; ++i) {
                        Cplx<T> acc{};
                        for (Index_t j{0}; j < Dim; ++j) {
                            fma(acc, load(dinv[i * Dim + j]), rhs_k[j]);
                        }
                        store(out[plane + m * Dim + i], acc);
                    }
                }
            }
        }

        // Explicit instantiations: 2D and 3D, double and single precision.
        template void sweep<2, Real>(
            const std::complex<Real> *, const std::complex<Real> *,
            const std::complex<Real> *, const int *,
            const std::complex<Real> *, const std::complex<Real> *,
            std::complex<Real> *, std::complex<Real> *, Index_t, Index_t,
            Index_t, Index_t);
        template void sweep<3, Real>(
            const std::complex<Real> *, const std::complex<Real> *,
            const std::complex<Real> *, const int *,
            const std::complex<Real> *, const std::complex<Real> *,
            std::complex<Real> *, std::complex<Real> *, Index_t, Index_t,
            Index_t, Index_t);
        template void sweep<2, Real32>(
            const std::complex<Real32> *, const std::complex<Real32> *,
            const std::complex<Real32> *, const int *,
            const std::complex<Real32> *, const std::complex<Real32> *,
            std::complex<Real32> *, std::complex<Real32> *, Index_t, Index_t,
            Index_t, Index_t);
        template void sweep<3, Real32>(
            const std::complex<Real32> *, const std::complex<Real32> *,
            const std::complex<Real32> *, const int *,
            const std::complex<Real32> *, const std::complex<Real32> *,
            std::complex<Real32> *, std::complex<Real32> *, Index_t, Index_t,
            Index_t, Index_t);

    }  // namespace block_thomas

}  // namespace muGrid
