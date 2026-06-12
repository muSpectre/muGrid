/**
 * @file   memory/device_alloc.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   12 Jun 2026
 *
 * @brief  Runtime device memory allocation with a pluggable allocator hook
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

#include "memory/device_alloc.hh"

#include <mutex>
#include <string>
#include <unordered_set>

#include "core/exception.hh"
#include "memory/gpu_runtime.hh"

namespace muGrid {

    namespace {
        struct DeviceAllocatorState {
            DeviceAllocateFn allocate{nullptr};
            DeviceDeallocateFn deallocate{nullptr};
            // Pointers produced by the external allocator. A pointer must be
            // freed by the allocator that produced it, even if the hook was
            // switched in between.
            std::unordered_set<void *> external_ptrs{};
            std::mutex mutex{};
        };

        DeviceAllocatorState & allocator_state() {
            static DeviceAllocatorState state{};
            return state;
        }
    }  // namespace

    void set_device_allocator(DeviceAllocateFn allocate,
                              DeviceDeallocateFn deallocate) {
        if ((allocate == nullptr) != (deallocate == nullptr)) {
            throw RuntimeError(
                "set_device_allocator: allocate and deallocate must both be "
                "set or both be null");
        }
        auto & state{allocator_state()};
        std::lock_guard<std::mutex> lock{state.mutex};
        state.allocate = allocate;
        state.deallocate = deallocate;
    }

    void * device_allocate(std::size_t bytes) {
        if (bytes == 0) {
            return nullptr;
        }
        auto & state{allocator_state()};
        {
            std::lock_guard<std::mutex> lock{state.mutex};
            if (state.allocate != nullptr) {
                void * ptr{state.allocate(bytes)};
                if (ptr == nullptr) {
                    throw RuntimeError(
                        "External device allocator failed to allocate " +
                        std::to_string(bytes) + " bytes");
                }
                state.external_ptrs.insert(ptr);
                return ptr;
            }
        }
#if defined(MUGRID_ENABLE_CUDA)
        void * ptr{nullptr};
        cudaError_t err{cudaMalloc(&ptr, bytes)};
        if (err != cudaSuccess) {
            throw RuntimeError(std::string("CUDA allocation failed: ") +
                               cudaGetErrorString(err));
        }
        return ptr;
#elif defined(MUGRID_ENABLE_HIP)
        void * ptr{nullptr};
        hipError_t err{hipMalloc(&ptr, bytes)};
        if (err != hipSuccess) {
            throw RuntimeError(std::string("HIP allocation failed: ") +
                               hipGetErrorString(err));
        }
        return ptr;
#else
        // GCOVR_EXCL_START -- unreachable: device fields cannot be created
        // in a build without a GPU backend
        throw RuntimeError(
            "device_allocate: muGrid was compiled without GPU support");
        // GCOVR_EXCL_STOP
#endif
    }

    void device_deallocate(void * ptr) {
        if (ptr == nullptr) {
            return;
        }
        auto & state{allocator_state()};
        {
            std::lock_guard<std::mutex> lock{state.mutex};
            auto it{state.external_ptrs.find(ptr)};
            if (it != state.external_ptrs.end()) {
                state.external_ptrs.erase(it);
                if (state.deallocate != nullptr) {
                    state.deallocate(ptr);
                }
                // If the external allocator was unregistered while its
                // allocations were alive there is nothing safe to do; the
                // pointer is intentionally leaked (documented contract).
                return;
            }
        }
#if defined(MUGRID_ENABLE_CUDA)
        (void)cudaFree(ptr);
#elif defined(MUGRID_ENABLE_HIP)
        (void)hipFree(ptr);
#endif
    }

}  // namespace muGrid
