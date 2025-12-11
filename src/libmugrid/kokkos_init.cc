/**
 * @file   kokkos_init.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   11 Dec 2024
 *
 * @brief  Automatic Kokkos initialization/finalization for libmuGrid
 *
 * This file ensures Kokkos is initialized when libmuGrid is loaded and
 * finalized when it is unloaded. This is necessary because Kokkos has
 * internal static state that must be properly managed when used in a
 * shared library context.
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
namespace internal {

/**
 * RAII class for Kokkos lifetime management within the shared library.
 *
 * This ensures that Kokkos is initialized before any Kokkos::View objects
 * are created and finalized after all Views are destroyed. The static
 * instance is constructed when the library loads and destroyed when
 * the library unloads.
 */
class KokkosLifetimeManager {
 public:
  KokkosLifetimeManager() {
    if (!Kokkos::is_initialized()) {
      Kokkos::initialize();
    }
  }

  ~KokkosLifetimeManager() {
    if (Kokkos::is_initialized() && !Kokkos::is_finalized()) {
      Kokkos::finalize();
    }
  }

  // Non-copyable, non-movable
  KokkosLifetimeManager(const KokkosLifetimeManager &) = delete;
  KokkosLifetimeManager & operator=(const KokkosLifetimeManager &) = delete;
  KokkosLifetimeManager(KokkosLifetimeManager &&) = delete;
  KokkosLifetimeManager & operator=(KokkosLifetimeManager &&) = delete;
};

// Static instance ensures Kokkos is initialized on library load
// and finalized on library unload
static KokkosLifetimeManager kokkos_lifetime_manager;

}  // namespace internal
}  // namespace muGrid
