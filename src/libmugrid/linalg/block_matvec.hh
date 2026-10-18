/**
 * @file   block_matvec.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   18 Oct 2026
 *
 * @brief  Per-mode n x n block product of a Fourier field, in place
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

#ifndef SRC_LIBMUGRID_LINALG_BLOCK_MATVEC_HH_
#define SRC_LIBMUGRID_LINALG_BLOCK_MATVEC_HH_

#include "core/types.hh"

#include <complex>

namespace muGrid {

    /**
     * Per-mode block product of an `N`-component Fourier field,
     *
     *     z_i(q) = sum_j K_ij(q) s_j(q),     i, j = 0 .. N-1,
     *
     * computed in place (`z` overwrites `s`). This is the per-mode step of
     * the block spectral (Green-function) preconditioner, whose blocks
     * `K(q)` are the inverse reference-stiffness symbol.
     *
     * One thread per mode reads the `N` input components and the block once,
     * forms all `N` outputs in registers and writes them back: one pass over
     * memory, `N^2` complex multiply-adds per mode. Every output depends on
     * every input, so the outputs cannot be written before all of them are
     * formed; holding them in registers is what makes the in-place update
     * possible without a temporary field.
     *
     * Layouts (device addresses, element strides; `M = nb_modes`):
     * - `field`: component `i`, mode `m` at `field[i * component_stride + m]`
     *   (structure-of-arrays, the device layout of muGrid fields);
     * - Hermitian blocks: real diagonal `diag[i * M + m]` and the strict
     *   upper triangle `upper[p * M + m]`, with `p` enumerating `(i, j)`,
     *   `i < j`, row by row; `K_ji = conj(K_ij)`. `N^2` reals per mode.
     * - general blocks: `blocks[(i * N + j) * M + m]`, `2 N^2` reals per mode.
     *
     * Templated on `N` so the component vectors live in registers; `N` from 1
     * to `MAX_COMPONENTS` is instantiated.
     */
    namespace block_matvec {

        //! Largest component count with a compiled kernel. The vector fields
        //! of muGrid's elasticity preconditioners have N = d = 2 or 3.
        constexpr int MAX_COMPONENTS{4};

        //! Number of stored upper-triangle entries of an `n x n` block.
        constexpr Index_t nb_upper(Index_t n) { return n * (n - 1) / 2; }

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        //! In-place product with Hermitian blocks; `n` in 1 .. MAX_COMPONENTS.
        template <typename T>
        void hermitian_gpu(Index_t n, const T * diag,
                           const std::complex<T> * upper,
                           std::complex<T> * field, Index_t component_stride,
                           Index_t nb_modes);

        //! In-place product with general blocks; `n` in 1 .. MAX_COMPONENTS.
        template <typename T>
        void dense_gpu(Index_t n, const std::complex<T> * blocks,
                       std::complex<T> * field, Index_t component_stride,
                       Index_t nb_modes);
#endif

    }  // namespace block_matvec

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_LINALG_BLOCK_MATVEC_HH_
