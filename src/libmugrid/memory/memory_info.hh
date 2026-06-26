/**
 * @file   memory/memory_info.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   25 Jun 2026
 *
 * @brief  Best-effort queries of host and device memory capacity
 *
 * Used by the allocation profiler to report how much of a memory pool is in
 * use relative to what physically exists. On a unified-memory architecture
 * host and device draw on the same pool, so only the device query is used
 * (it already reports the shared capacity); see memory/unified_memory.hh.
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

#ifndef SRC_LIBMUGRID_MEMORY_MEMORY_INFO_HH_
#define SRC_LIBMUGRID_MEMORY_MEMORY_INFO_HH_

#include <cstddef>

namespace muGrid {

    /**
     * Capacity of a memory pool, in bytes. @c valid is false when the query
     * is not supported on this platform/build (the byte fields are then 0).
     */
    struct MemoryCapacity {
        std::size_t total{0};      //!< total physical bytes in the pool
        std::size_t available{0};  //!< currently free bytes in the pool
        bool valid{false};         //!< whether the query succeeded
    };

    /**
     * Total and available host (CPU) RAM. Best-effort: implemented for Linux
     * and macOS, returns @c {0,0,false} elsewhere.
     */
    MemoryCapacity host_memory_capacity();

    /**
     * Total and available device memory via the GPU runtime
     * (cudaMemGetInfo / hipMemGetInfo). On a build without a GPU backend, or
     * if the query fails, returns @c {0,0,false}.
     *
     * @param device_id GPU device to query, or -1 for the currently selected
     *        device. On a unified-memory device this reports the shared pool,
     *        which equals the host RAM.
     */
    MemoryCapacity device_memory_capacity(int device_id = -1);

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_MEMORY_MEMORY_INFO_HH_
