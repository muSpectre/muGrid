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

    /**
     * An external device allocator (e.g. one drawing from cupy's memory pool,
     * or an Umpire pool). `ctx` is passed back to every callback, so a
     * stateful allocator needs no global state.
     *
     * Every pointer is freed through the registration that produced it, and
     * a registration stays alive while any of its pointers do: replacing or
     * clearing the allocator never frees memory still in use. Once a
     * registration is replaced or cleared *and* its last pointer is freed,
     * `release(ctx)` (if set) is called to dispose of `ctx`. muGrid's
     * allocator state is never torn down, so `release` is not called for a
     * registration still live at process exit.
     *
     * The callbacks are invoked without any muGrid lock held.
     */
    struct DeviceAllocator {
        //! Return a device pointer to at least `bytes` bytes, or nullptr on
        //! failure.
        void * (*allocate)(void * ctx, std::size_t bytes){nullptr};
        //! Free a pointer previously returned by `allocate` with this `ctx`.
        void (*deallocate)(void * ctx, void * ptr){nullptr};
        //! Optional: dispose of `ctx` once it is no longer referenced.
        void (*release)(void * ctx){nullptr};
        //! Opaque state handed to the callbacks.
        void * ctx{nullptr};
    };

    /**
     * Route all subsequent device allocations through `allocator`. Both
     * `allocate` and `deallocate` must be set. Allocations made through a
     * previously registered allocator are still freed through it.
     */
    void set_device_allocator(const DeviceAllocator & allocator);

    //! Restore the default backend allocator (raw cudaMalloc/hipMalloc) for
    //! subsequent allocations.
    void clear_device_allocator();

    //! True if an external device allocator is currently registered.
    bool device_allocator_is_external();

    /**
     * Allocate `bytes` bytes of device memory through the registered
     * allocator (or raw cudaMalloc/hipMalloc by default). Throws
     * RuntimeError on failure or when no GPU backend is compiled in.
     *
     * When @p label is non-null the allocation is reported to the
     * AllocationProfiler under that label, so library scratch buffers and
     * externally-routed allocations (e.g. cupy) become visible at this single
     * chokepoint. Field buffers pass null: they are recorded (with their field
     * name) by the owning Field's Array instead, so recording each buffer
     * exactly once. Pass a descriptive label for scratch (e.g.
     * "fft-nd-scratch").
     */
    void * device_allocate(std::size_t bytes, const char * label = nullptr);

    //! Free memory obtained from device_allocate().
    void device_deallocate(void * ptr);

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_MEMORY_DEVICE_ALLOC_HH_
