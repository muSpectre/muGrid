/**
 * @file   kokkos_init.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   11 Dec 2024
 *
 * @brief  Kokkos initialization helpers for libmuGrid
 *
 * This file provides helper functions for Kokkos initialization/finalization
 * to be called by applications and language bindings. Library-level automatic
 * initialization is not used because it causes static initialization order
 * issues with Kokkos 5.0's internal data structures.
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

#include <Kokkos_Core.hpp>

namespace muGrid {

/**
 * Initialize Kokkos if not already initialized.
 *
 * This should be called by applications before using muGrid functionality.
 * For Python bindings, this is called automatically in the module init.
 */
void initialize_kokkos() {
  if (!Kokkos::is_initialized()) {
    Kokkos::initialize();
  }
}

/**
 * Finalize Kokkos if initialized and not already finalized.
 *
 * This should be called by applications when done using muGrid functionality.
 * For Python bindings, this is called automatically in module cleanup.
 */
void finalize_kokkos() {
  if (Kokkos::is_initialized() && !Kokkos::is_finalized()) {
    Kokkos::finalize();
  }
}

}  // namespace muGrid
