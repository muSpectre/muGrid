/**
 * @file   pcr.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   13 Nov 2026
 *
 * @brief  Fused parallel-cyclic-reduction step of the hybrid
 *         Fourier/tridiagonal preconditioner's interface solve
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

#ifndef SRC_LIBMUGRID_LINALG_PCR_HH_
#define SRC_LIBMUGRID_LINALG_PCR_HH_

#include "core/types.hh"

#include <complex>

namespace muGrid {

    /**
     * One level of the parallel cyclic reduction (PCR) that solves the
     * interface system of `HybridFourierTridiagonalPreconditioner`, fused
     * into a single kernel.
     *
     * Per Fourier mode the rank holds `x = (L, R)`, its `2d` interface
     * unknowns (`d` = spatial dimension). A level against the ranks at
     * `±s` is
     *
     *     x' = D^-1 x - Ã R_{p-s} - C̃ L_{p+s},
     *
     * where `Ã = D^-1 A[:, d:]` and `C̃ = D^-1 C[:, :d]` are folded once at
     * setup. Only those column blocks of `A` and `C` are nonzero, at every
     * level, so each rank needs only the `R` half of its lower and the `L`
     * half of its upper partner: `d` numbers per mode in each direction.
     *
     * The final variant additionally applies the closing `2d x 2d` inverse
     * through the two `d x 2d` operators that produce what the neighbours
     * need, `to_right = A2 [0 I] F x'` and `to_left = A0 [I 0] F x'`, so the
     * last level writes those two `d`-vectors instead of `x'`.
     *
     * One thread per mode; everything a mode needs is read once and its
     * vectors stay in registers, which also makes `x_out == x` (in place)
     * safe.
     *
     * Layouts (device addresses, structure-of-arrays, `M = nb_modes`,
     * `m = 2d`): vector component `i` of mode `k` at `v[i * M + k]`;
     * an `r x c` block entry `(i, j)` at `b[(i * c + j) * M + k]`. So
     * `dinv` is `(m*m, M)`, `a` and `c` `(m*d, M)`, `x`/`x_out` `(m, M)`,
     * `from_low`/`from_high`/`to_right`/`to_left` `(d, M)`, and
     * `op_right`/`op_left` `(d*m, M)`.
     */
    namespace pcr {

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        //! One PCR level; `dim` is 2 or 3.
        template <typename T>
        void step_gpu(Index_t dim, Index_t nb_modes,
                      const std::complex<T> * dinv,
                      const std::complex<T> * a, const std::complex<T> * c,
                      const std::complex<T> * x,
                      const std::complex<T> * from_low,
                      const std::complex<T> * from_high,
                      std::complex<T> * x_out);

        //! The last PCR level fused with the closing solve and the
        //! neighbour products; `dim` is 2 or 3.
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
                            std::complex<T> * to_left);

        //! Only the neighbour products of the final variant, for when the
        //! last level ran on a subset of the modes (see
        //! HybridFourierTridiagonalPreconditioner._truncate_pcr):
        //! `to_right = op_right x`, `to_left = op_left x` per mode.
        template <typename T>
        void products_gpu(Index_t dim, Index_t nb_modes,
                          const std::complex<T> * op_right,
                          const std::complex<T> * op_left,
                          const std::complex<T> * x,
                          std::complex<T> * to_right,
                          std::complex<T> * to_left);
#endif

    }  // namespace pcr

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_LINALG_PCR_HH_
