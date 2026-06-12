/**
 * @file   memory/device_alloc.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   12 Jun 2026
 *
 * @brief  Runtime device memory allocation with a pluggable allocator hook
 *
 * Unlike the compile-time-typed Array<T, MemorySpace>, these functions
 * allocate device memory based on a runtime decision (e.g. a field's
 * is_on_device flag). All device allocations in muGrid should go through
 * this interface (directly or via the Array allocators) so that an
 * externally registered allocator — e.g. cupy's memory pool when muGrid
 * is driven from Python — owns every device byte. Two independent
 * allocators on one device starve each other: a caching pool never
 * returns freed blocks to the driver, so raw cudaMalloc/hipMalloc in the
 * other allocator fails even though memory is "free".
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

#ifndef SRC_LIBMUGRID_MEMORY_DEVICE_ALLOC_HH_
#define SRC_LIBMUGRID_MEMORY_DEVICE_ALLOC_HH_

#include <cstddef>

namespace muGrid {

    //! Signature of an external device allocator: returns a device pointer
    //! to at least `bytes` bytes, or nullptr on failure.
    using DeviceAllocateFn = void * (*)(std::size_t bytes);
    //! Signature of the matching deallocator.
    using DeviceDeallocateFn = void (*)(void * ptr);

    /**
     * Register an external device allocator (e.g. one drawing from cupy's
     * memory pool). Pass nullptrs to restore the default backend
     * allocator. Pointers allocated before the switch are freed through
     * the allocator that produced them; do not unregister while such
     * allocations are alive unless the external allocator outlives them.
     */
    void set_device_allocator(DeviceAllocateFn allocate,
                              DeviceDeallocateFn deallocate);

    //! True if an external device allocator is registered.
    bool has_external_device_allocator();

    /**
     * Allocate `bytes` bytes of device memory through the registered
     * allocator (or raw cudaMalloc/hipMalloc by default). Throws
     * RuntimeError on failure or when no GPU backend is compiled in.
     */
    void * device_allocate(std::size_t bytes);

    //! Free memory obtained from device_allocate().
    void device_deallocate(void * ptr);

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_MEMORY_DEVICE_ALLOC_HH_
