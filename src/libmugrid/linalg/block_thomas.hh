/**
 * @file   block_thomas.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   19 Sep 2026
 *
 * @brief  Fused block-Thomas sweep along one axis, batched over Fourier modes
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

#ifndef SRC_LIBMUGRID_LINALG_BLOCK_THOMAS_HH_
#define SRC_LIBMUGRID_LINALG_BLOCK_THOMAS_HH_

#include "core/types.hh"

#include <complex>

namespace muGrid {

    namespace block_thomas {

        /**
         * @brief Block-Thomas solve of a block-tridiagonal system along one
         *        axis, batched over independent Fourier modes.
         *
         * Solves `T x = rhs` where, for each mode, `T` is block-tridiagonal
         * with constant coupling blocks `A0` (super) and `A2` (sub) and
         * pre-factorised diagonal inverses. This is the host counterpart of the
         * fused device kernel, and exists because the elementwise alternative
         * issues four batched matrix products per plane: each of those streams
         * the whole mode array, so a solve costs `4 * nz` passes over memory
         * where this costs two.
         *
         * **Loop order differs from the device deliberately.** The device
         * assigns one thread per mode and marches the axis in registers, so its
         * wavefront wants consecutive modes at a fixed plane. A single host
         * thread walking one mode at a time would instead stride by
         * `nb_modes * Dim` between plane steps. This implementation therefore
         * runs plane-outer and mode-inner, which makes every array access
         * contiguous; the recurrence's carried state is the previous plane of
         * `y`, which is already materialised and stays in cache.
         *
         * @tparam Dim block size (2 or 3); compile-time so the inner products
         *         unroll
         * @tparam T  real scalar type of the complex entries
         *
         * @param rhs       `(nz, nb_modes, Dim)` right-hand side
         * @param head      `(nb_head, nb_modes, Dim, Dim)` leading diagonal
         *                  inverses; planes beyond `nb_head` reuse the last,
         *                  which is exact once the factor recurrence has
         *                  reached its fixed point
         * @param exc       `(nz, nb_exc, Dim, Dim)` full lines for the modes
         *                  that converge too slowly to be compressed
         * @param exc_index `(nb_modes,)` slot into `exc`, or negative to use
         *                  `head`
         * @param A0, A2    `(nb_modes, Dim, Dim)` constant coupling blocks
         * @param y         `(nz, nb_modes, Dim)` scratch for the forward sweep
         * @param out       `(nz, nb_modes, Dim)` solution
         */
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
                   Index_t nb_exc);

    }  // namespace block_thomas

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_LINALG_BLOCK_THOMAS_HH_
