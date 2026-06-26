/**
 * @file   memory/memory_info.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   25 Jun 2026
 *
 * @brief  Best-effort queries of host and device memory capacity
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

#include "memory/memory_info.hh"

#include <cstdint>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

#include "memory/gpu_runtime.hh"

namespace muGrid {

    MemoryCapacity host_memory_capacity() {
        MemoryCapacity cap{};
#if defined(__APPLE__)
        // Total physical memory from sysctl.
        std::uint64_t total{0};
        std::size_t len{sizeof(total)};
        if (sysctlbyname("hw.memsize", &total, &len, nullptr, 0) == 0) {
            cap.total = static_cast<std::size_t>(total);
        }
        // Available memory: free + inactive (reclaimable) pages.
        vm_size_t page_size{0};
        mach_port_t host{mach_host_self()};
        vm_statistics64_data_t vm{};
        mach_msg_type_number_t count{HOST_VM_INFO64_COUNT};
        if (host_page_size(host, &page_size) == KERN_SUCCESS &&
            host_statistics64(host, HOST_VM_INFO64,
                              reinterpret_cast<host_info64_t>(&vm),
                              &count) == KERN_SUCCESS) {
            const std::size_t free_bytes{
                (static_cast<std::size_t>(vm.free_count) +
                 static_cast<std::size_t>(vm.inactive_count)) *
                static_cast<std::size_t>(page_size)};
            cap.available = free_bytes;
            cap.valid = cap.total > 0;
        }
#elif defined(__linux__)
        const long page_size{sysconf(_SC_PAGESIZE)};
        const long phys_pages{sysconf(_SC_PHYS_PAGES)};
        const long avail_pages{sysconf(_SC_AVPHYS_PAGES)};
        if (page_size > 0 && phys_pages > 0) {
            cap.total = static_cast<std::size_t>(phys_pages) *
                        static_cast<std::size_t>(page_size);
            if (avail_pages > 0) {
                cap.available = static_cast<std::size_t>(avail_pages) *
                                static_cast<std::size_t>(page_size);
            }
            cap.valid = true;
        }
#endif
        return cap;
    }

    MemoryCapacity device_memory_capacity(int device_id) {
        MemoryCapacity cap{};
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        int previous{-1};
        if (device_id >= 0) {
            previous = gpu_get_device();
            gpu_set_device(device_id);
        }
        std::size_t free_bytes{0}, total_bytes{0};
        if (gpu_mem_get_info(free_bytes, total_bytes)) {
            cap.total = total_bytes;
            cap.available = free_bytes;
            cap.valid = true;
        }
        if (previous >= 0) {
            gpu_set_device(previous);
        }
#else
        (void)device_id;
#endif
        return cap;
    }

}  // namespace muGrid
