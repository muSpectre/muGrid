/**
 * @file   linalg/green_symbol.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   24 Sep 2026
 *
 * @brief  Per-mode application of a reference operator's inverse symbol
 *
 * A uniform (translation-invariant) operator is a `3^Dim` stencil, and its
 * Fourier symbol is the closed-form sum `K(q) = sum_d S[d] exp(-2 pi i q.d)`
 * over that stencil's offsets. The reference-material preconditioner is
 * `K(q)^-1` applied mode by mode, so it can be *evaluated* per mode instead of
 * assembled and stored: the stencil is 243 numbers in 3D, where the stored
 * symbol is `n^2` values per Fourier point -- 2.3 GB at 512^3 in single
 * precision, and it grows with the grid.
 *
 * This kernel does that evaluation. It builds the symbol for one mode, inverts
 * the `Dim x Dim` Hermitian block in registers, and multiplies the mode's
 * component vector in place. Nothing per-mode is ever written to memory.
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

#ifndef SRC_LIBMUGRID_LINALG_GREEN_SYMBOL_HH_
#define SRC_LIBMUGRID_LINALG_GREEN_SYMBOL_HH_

#include "core/types.hh"

#include <complex>

namespace muGrid {

    namespace green_symbol {

        /**
         * @brief Apply the inverse symbol of a uniform stencil operator to a
         *        Fourier field, in place, evaluating the symbol per mode.
         *
         * For every mode this forms `K(q) = sum_d S[d] exp(-2 pi i q.d)` over
         * the stencil's `3^Dim` offsets, inverts the `Dim x Dim` block, scales
         * by `normalisation` and replaces the mode's component vector by
         * `K(q)^-1 v`. The `q = 0` block is singular -- the rigid-body
         * translations are the operator's null space -- and is projected out by
         * setting that mode to zero, matching the pseudo-inverse the assembled
         * path stores.
         *
         * The symbol is Hermitian by construction: the stencil of a
         * self-adjoint operator satisfies `S[d] = S[-d]^T` exactly, so only its
         * upper triangle is formed.
         *
         * **Frequencies are passed as per-axis tables, not per mode.** A
         * `(Dim, nb_modes)` array of frequencies would be 809 MB at 512^3 in
         * single precision, which would give back much of what evaluating the
         * symbol saves. The tables are `nb_fourier_grid_pts[axis]` entries
         * each, a few kilobytes, and the kernel decomposes the flat mode index
         * against `nb_fourier_grid_pts` to look up each axis. That
         * decomposition assumes muGrid's Fourier layout, axis 0 fastest.
         *
         * @tparam Dim number of components, and the block size; compile-time so
         *         the inversion and the inner products unroll
         * @tparam T  real scalar type of the complex entries
         *
         * @param field        Fourier field, modified in place
         * @param stride_component  element stride between components (1 for
         *                     the host's array-of-structures layout,
         *                     `nb_modes` for the device's
         *                     structure-of-arrays)
         * @param stride_mode  element stride between consecutive modes (`Dim`
         *                     for array-of-structures, 1 for
         *                     structure-of-arrays)
         * @param stencil      `3^Dim * Dim * Dim` real stencil, indexed
         *                     `S[d + 1]` with the offset axes slowest and the
         *                     `Dim x Dim` block fastest (C order), as
         *                     `reference_stencil` returns it
         * @param q            `Dim` pointers to per-axis frequency tables, in
         *                     cycles; entry `i` is the frequency of local mode
         *                     index `i` along that axis
         * @param nb_fourier_grid_pts  local mode count per axis; their product
         *                     is `nb_modes`
         * @param normalisation  the engine's inverse-transform normalisation,
         *                     folded in here rather than by a separate pass
         * @param nb_modes     number of local Fourier modes
         */
        template <Dim_t Dim, typename T>
        void apply_inverse(std::complex<T> * field, Index_t stride_component,
                           Index_t stride_mode, const T * stencil,
                           const T * const * q,
                           const Index_t * nb_fourier_grid_pts,
                           T normalisation, Index_t nb_modes);

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        /**
         * @brief Device counterpart of `apply_inverse`, identical semantics.
         *
         * Defined in `green_symbol_gpu.cc`. One thread per mode, with the
         * symbol, its inverse and the component vector all held in registers,
         * which is the whole point: the per-mode block never reaches memory.
         * Pointers are device pointers, except `nb_fourier_grid_pts` and the
         * table of table-pointers, which are read on the host.
         */
        template <Dim_t Dim, typename T>
        void apply_inverse_gpu(std::complex<T> * field,
                               Index_t stride_component, Index_t stride_mode,
                               const T * stencil, const T * const * q,
                               const Index_t * nb_fourier_grid_pts,
                               T normalisation, Index_t nb_modes);
#endif

    }  // namespace green_symbol

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_LINALG_GREEN_SYMBOL_HH_
