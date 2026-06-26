/**
 * @file   memory/unified_memory.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   25 Jun 2026
 *
 * @brief  Runtime detection of physically unified host/device memory
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

#include "memory/unified_memory.hh"

#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>

#if defined(MUGRID_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif
#if defined(MUGRID_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace muGrid {

    namespace {

        //! Parse MUGRID_UNIFIED_MEMORY. Returns -1 if unset, 0/1 otherwise.
        int env_override() {
            const char * env{std::getenv("MUGRID_UNIFIED_MEMORY")};
            if (env == nullptr || std::strlen(env) == 0) {
                return -1;
            }
            const char c{env[0]};
            return (c == '1' || c == 'y' || c == 'Y' || c == 't' || c == 'T')
                       ? 1
                       : 0;
        }

        //! Query the GPU runtime for the "integrated" (shared physical
        //! memory) attribute of a device. -1 selects the current device.
        bool query_integrated(int device_id) {
#if defined(MUGRID_ENABLE_CUDA)
            int dev{device_id};
            if (dev < 0 && cudaGetDevice(&dev) != cudaSuccess) {
                return false;
            }
            int integrated{0};
            if (cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated,
                                       dev) != cudaSuccess) {
                return false;
            }
            return integrated != 0;
#elif defined(MUGRID_ENABLE_HIP)
            int dev{device_id};
            if (dev < 0 && hipGetDevice(&dev) != hipSuccess) {
                return false;
            }
            int integrated{0};
            if (hipDeviceGetAttribute(&integrated, hipDeviceAttributeIntegrated,
                                      dev) != hipSuccess) {
                return false;
            }
            return integrated != 0;
#else
            // No GPU backend: there is no device, hence nothing unified.
            (void)device_id;
            return false;
#endif
        }

        bool detect_unified_memory(int device_id) {
            const int override{env_override()};
            if (override >= 0) {
                return override == 1;
            }
            return query_integrated(device_id);
        }

    }  // namespace

    bool device_has_unified_memory(int device_id) {
        // Cache per device id. The environment override is global but cheap
        // to re-read; detection (the expensive part) still runs at most once
        // per device id.
        static std::mutex mutex{};
        static std::map<int, bool> cache{};
        std::lock_guard<std::mutex> lock{mutex};
        auto it{cache.find(device_id)};
        if (it != cache.end()) {
            return it->second;
        }
        const bool result{detect_unified_memory(device_id)};
        cache.emplace(device_id, result);
        return result;
    }

}  // namespace muGrid
