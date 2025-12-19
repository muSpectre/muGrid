/**
 * @file   kokkos_types.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   09 Dec 2024
 *
 * @brief  GPU-portable memory management type definitions
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

#ifndef SRC_LIBMUGRID_KOKKOS_TYPES_HH_
#define SRC_LIBMUGRID_KOKKOS_TYPES_HH_

// This header provides backwards compatibility after Kokkos removal.
// All memory space types are now defined in memory/memory_space.hh
// All array types are now defined in memory/device_array.hh

#include "memory/memory_space.hh"
#include "memory/device_array.hh"

namespace muGrid {

    // Re-export from memory_space.hh for backwards compatibility
    // Types available: HostSpace, CudaSpace (if CUDA), HIPSpace (if HIP)
    // Also available: DefaultDeviceSpace, is_host_space_v, is_device_space_v

    // For backwards compatibility, provide execution space types
    // (These are now just aliases since we don't use Kokkos execution model)
    using HostExecutionSpace = HostSpace;
    using DefaultExecutionSpace = DefaultDeviceSpace;

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_KOKKOS_TYPES_HH_
