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
         * (non-periodic along the axis) with `Dim x Dim` blocks,
         *
         *     A2 x_{k-1} + B x_k + A0 x_{k+1} = rhs_k,   k = 0 .. nz-1,
         *
         * (terms with `k-1 < 0` or `k+1 >= nz` absent) with constant coupling
         * blocks `A0` (super) and `A2` (sub). The caller supplies the block LU
         * factorisation as the inverses `Dinv_k = D_k^-1` of the Schur
         * complements `D_0 = B`, `D_k = B - A2 D_{k-1}^-1 A0`. Because the
         * blocks are constant this recurrence tends to a fixed point, which
         * is what lets `head` store only its first `nb_head` planes. The sweep
         * is then
         *
         *     forward:  y_0 = rhs_0,  y_k = rhs_k - A2 Dinv_{k-1} y_{k-1}
         *     backward: x_{nz-1} = Dinv_{nz-1} y_{nz-1},
         *               x_k = Dinv_k (y_k - A0 x_{k+1}),
         *
         * i.e. block Gaussian elimination without pivoting, which is stable
         * when the factorisation it is given exists and is well conditioned
         * (e.g. for Hermitian positive definite `T`). Cost is O(nz * nb_modes
         * * Dim^2) with two passes over the `(nz, nb_modes, Dim)` arrays.
         *
         * This is the host counterpart of the fused device kernel, and exists
         * because the elementwise alternative issues four batched matrix
         * products per plane: each of those streams the whole mode array, so
         * a solve costs `4 * nz` passes over memory where this costs two.
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
         * @tparam T  real scalar type of the complex entries (Real or Real32)
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
         * @param nz        number of planes along the solved axis
         * @param nb_modes  number of independent modes per plane
         * @param nb_head   number of planes stored in `head` (>= 1)
         * @param nb_exc    number of modes stored in `exc`
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

        //! Planes per prefetch chunk of the device sweep: the loads of
        //! `DEVICE_CHUNK` planes are issued before their recurrence steps.
        //! Measured on MI300A (3D, double, 64-thread blocks, no spills), ms
        //! per sweep at 128^3 / 256^3: chunk 1 0.179 / 0.946, chunk 2
        //! 0.167 / 0.907, chunk 4 0.169 / 0.918, chunk 8 0.185 / 0.909.
        //! Past 2 the extra registers buy nothing; 2 needs about 200 per
        //! thread, within both CDNA3's and NVIDIA's 255.
        constexpr int DEVICE_CHUNK{2};

        //! Threads per block of the device sweep, also its
        //! `__launch_bounds__`. One thread per mode, and the mode count is
        //! small (8320 at 128^3), so small blocks spread the work over more
        //! compute units: on MI300A 64-thread blocks occupy 130 of 228 CUs
        //! where 256-thread blocks occupy 33 (0.307 vs 0.423 ms per sweep).
        constexpr int DEVICE_THREADS{64};

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        /**
         * @brief Device counterpart of `sweep`, with identical semantics.
         *
         * Defined in `block_thomas_gpu.cc`. The loop order differs from the
         * host: one thread per mode marching the axis in registers, which is
         * what the z-major layout is for. Pointers are device pointers.
         * Requires `nz` and `nb_modes` to fit in an `int`.
         */
        template <Dim_t Dim, typename T>
        void sweep_gpu(const std::complex<T> * rhs,
                       const std::complex<T> * head,
                       const std::complex<T> * exc, const int * exc_index,
                       const std::complex<T> * A0, const std::complex<T> * A2,
                       std::complex<T> * y, std::complex<T> * out, Index_t nz,
                       Index_t nb_modes, Index_t nb_head, Index_t nb_exc);
#endif

    }  // namespace block_thomas

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_LINALG_BLOCK_THOMAS_HH_
