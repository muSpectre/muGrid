/**
 * @file   kokkos_init.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   11 Dec 2024
 *
 * @brief  GPU runtime initialization helpers for libmuGrid
 *
 * These functions are kept for API compatibility but are now no-ops
 * since Kokkos has been removed. GPU initialization is handled
 * automatically by the CUDA/HIP runtime.
 *
 * Copyright © 2024 Lars Pastewka
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

namespace muGrid {

/**
 * Initialize GPU runtime (no-op, kept for API compatibility).
 */
void initialize_kokkos() {
    // No-op: CUDA/HIP runtime handles initialization automatically
}

/**
 * Finalize GPU runtime (no-op, kept for API compatibility).
 */
void finalize_kokkos() {
    // No-op: CUDA/HIP runtime handles cleanup automatically
}

}  // namespace muGrid
