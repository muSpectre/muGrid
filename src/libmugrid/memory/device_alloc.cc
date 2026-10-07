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

#include "device.hh"
#include "memory/device_alloc.hh"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "core/exception.hh"
#include "memory/gpu_runtime.hh"
#include "memory/allocation_profiler.hh"

namespace muGrid {

    namespace {
        //! Canonical device-pool key for the active device, matching the space
        //! string a device Field uses (see Field::get_device_string), so
        //! chokepoint-recorded allocations land in the same profiler pool.
        std::string current_device_space() {
#if defined(MUGRID_ENABLE_CUDA)
            return "cuda:" + std::to_string(gpu_get_device());
#elif defined(MUGRID_ENABLE_HIP)
            return "rocm:" + std::to_string(gpu_get_device());
#else
            return "device";
#endif
        }

        //! One set_device_allocator() call. Shared by the allocator state
        //! (while current) and by every pointer it produced, so `ctx` is
        //! released only when nothing can call back into it any more.
        struct Registration {
            DeviceAllocator allocator;
            explicit Registration(const DeviceAllocator & a) : allocator{a} {}
            Registration(const Registration &) = delete;
            Registration & operator=(const Registration &) = delete;
            ~Registration() {
                if (allocator.release != nullptr) {
                    allocator.release(allocator.ctx);
                }
            }
        };

        struct DeviceAllocatorState {
            std::shared_ptr<Registration> current{};
            // Owner of each externally allocated pointer. A pointer must be
            // freed by the registration that produced it, even if the
            // allocator was replaced or cleared in between.
            std::unordered_map<void *, std::shared_ptr<Registration>>
                external_ptrs{};
            std::mutex mutex{};
        };

        DeviceAllocatorState & allocator_state() {
            // Intentionally leaked: device buffers may still be freed during
            // static destruction, and no release() callback (which may call
            // into a finalized interpreter) should run at exit.
            static auto * state{new DeviceAllocatorState{}};
            return *state;
        }

        //! Swap in a new current registration. The previous one is destroyed
        //! (if unreferenced) after the lock is dropped, so its release()
        //! callback runs without muGrid's lock held.
        void replace_current(std::shared_ptr<Registration> next) {
            auto & state{allocator_state()};
            {
                std::lock_guard<std::mutex> lock{state.mutex};
                state.current.swap(next);
            }
        }
    }  // namespace

    void set_device_allocator(const DeviceAllocator & allocator) {
        if (allocator.allocate == nullptr || allocator.deallocate == nullptr) {
            throw RuntimeError(
                "set_device_allocator: allocate and deallocate must both be "
                "set; use clear_device_allocator() to restore the default");
        }
        replace_current(std::make_shared<Registration>(allocator));
    }

    void clear_device_allocator() { replace_current(nullptr); }

    bool device_allocator_is_external() {
        auto & state{allocator_state()};
        std::lock_guard<std::mutex> lock{state.mutex};
        return static_cast<bool>(state.current);
    }

    void * device_allocate(std::size_t bytes, const char * label) {
        // First GPU use: fail loudly on a header/runtime version mismatch
        // rather than far away on garbage device properties.
        check_gpu_runtime_version();
        if (bytes == 0) {
            return nullptr;
        }
        void * ptr{nullptr};
        auto & state{allocator_state()};
        std::shared_ptr<Registration> registration{};
        {
            std::lock_guard<std::mutex> lock{state.mutex};
            registration = state.current;
        }
        if (registration) {
            // Called without the lock: the callback may block (e.g. on the
            // Python GIL) or allocate re-entrantly.
            ptr = registration->allocator.allocate(registration->allocator.ctx,
                                                   bytes);
            if (ptr == nullptr) {
                throw RuntimeError(
                    "External device allocator failed to allocate " +
                    std::to_string(bytes) + " bytes");
            }
            std::lock_guard<std::mutex> lock{state.mutex};
            state.external_ptrs.emplace(ptr, std::move(registration));
        }
        if (ptr == nullptr) {
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
            ptr = gpu_malloc_checked(bytes);
#else
            // GCOVR_EXCL_START -- unreachable: device fields cannot be created
            // in a build without a GPU backend
            throw RuntimeError(
                "device_allocate: muGrid was compiled without GPU support");
            // GCOVR_EXCL_STOP
#endif
        }
        // Record labelled allocations (library scratch, externally-routed
        // memory like cupy) at this single chokepoint so the profiler sees
        // them too. Field buffers pass a null label and are recorded once, by
        // their owning Array, under the field name. The profiler lock is taken
        // here outside the allocator-state lock to avoid nesting; recording is
        // a cheap no-op when disabled.
        if (label != nullptr) {
            AllocationProfiler::instance().record_alloc(
                ptr, label, current_device_space(), bytes);
        }
        return ptr;
    }

    void device_deallocate(void * ptr) {
        if (ptr == nullptr) {
            return;
        }
        AllocationProfiler::instance().record_free(ptr);
        auto & state{allocator_state()};
        std::shared_ptr<Registration> owner{};
        {
            std::lock_guard<std::mutex> lock{state.mutex};
            auto it{state.external_ptrs.find(ptr)};
            if (it != state.external_ptrs.end()) {
                owner = std::move(it->second);
                state.external_ptrs.erase(it);
            }
        }
        if (owner) {
            // Without the lock; dropping `owner` may release its ctx.
            owner->allocator.deallocate(owner->allocator.ctx, ptr);
            return;
        }
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        GPU_FREE(ptr);
#endif
    }

}  // namespace muGrid
