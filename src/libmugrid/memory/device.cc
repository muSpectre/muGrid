/**
 * @file   device.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   29 Dec 2025
 *
 * @brief  Device abstraction layer implementation
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

#include "device.hh"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>

#if defined(MUGRID_ENABLE_CUDA)
#include <cuda_runtime.h>
#elif defined(MUGRID_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace muGrid {

namespace {
    //! Parse MUGRID_UNIFIED_MEMORY. Returns -1 if unset, 0/1 otherwise. The
    //! variable forces the integrated/host-accessible answer for platforms
    //! where the runtime probe is unreliable (e.g. APUs needing a particular
    //! unified/XNACK mode) or to force the conservative host-staging path.
    int unified_memory_env_override() {
        const char * env{std::getenv("MUGRID_UNIFIED_MEMORY")};
        if (env == nullptr || std::strlen(env) == 0) {
            return -1;
        }
        const char c{env[0]};
        return (c == '1' || c == 'y' || c == 'Y' || c == 't' || c == 'T') ? 1
                                                                          : 0;
    }
}  // namespace

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
namespace {
    //! Query (once per device id, then cache) whether the GPU is an
    //! integrated / unified-memory device whose allocations are host-coherent.
    bool gpu_is_integrated(int device_id) {
        static std::map<int, bool> cache{};
        auto it{cache.find(device_id)};
        if (it != cache.end()) {
            return it->second;
        }
        bool integrated{false};
        const int override{unified_memory_env_override()};
        if (override >= 0) {
            integrated = (override == 1);
        } else {
#if defined(MUGRID_ENABLE_CUDA)
            cudaDeviceProp prop{};
            if (cudaGetDeviceProperties(&prop, device_id) == cudaSuccess) {
                integrated = (prop.integrated != 0);
            }
#else   // MUGRID_ENABLE_HIP
            hipDeviceProp_t prop{};
            if (hipGetDeviceProperties(&prop, device_id) == hipSuccess) {
                integrated = (prop.integrated != 0);
            }
#endif
        }
        cache[device_id] = integrated;
        return integrated;
    }
}  // namespace
#endif  // MUGRID_ENABLE_CUDA || MUGRID_ENABLE_HIP

Device Device::current_gpu() {
#if defined(MUGRID_ENABLE_CUDA)
    int id{0};
    if (cudaGetDevice(&id) != cudaSuccess) {
        id = 0;
    }
    return Device{DeviceType::CUDA, id};
#elif defined(MUGRID_ENABLE_HIP)
    int id{0};
    if (hipGetDevice(&id) != hipSuccess) {
        id = 0;
    }
    return Device{DeviceType::ROCm, id};
#else
    return Device::cpu();
#endif
}

bool Device::is_host_accessible() const {
    switch (this->device_type) {
        case DeviceType::CPU:
        case DeviceType::CUDAHost:  // pinned host memory
        case DeviceType::ROCmHost:  // pinned host memory
            return true;
        case DeviceType::CUDA:
        case DeviceType::ROCm:
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
            return gpu_is_integrated(this->device_id);
#else
            return false;
#endif
        default:
            return false;
    }
}

std::string Device::get_device_string() const {
    switch (this->device_type) {
        case DeviceType::CPU:
            return "cpu";
        case DeviceType::CUDA:
        case DeviceType::CUDAHost:
            return "cuda:" + std::to_string(this->device_id);
        case DeviceType::ROCm:
        case DeviceType::ROCmHost:
            return "rocm:" + std::to_string(this->device_id);
        default:
            return "unknown";
    }
}

Device Device::from_string(const std::string & spec) {
    // Lowercase copy so the parse is case-insensitive (get_device_string emits
    // lowercase; users may type "ROCm:1").
    std::string s{spec};
    for (char & c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    const auto colon{s.find(':')};
    const std::string kind{s.substr(0, colon)};
    int id{0};
    if (colon != std::string::npos) {
        const std::string id_str{s.substr(colon + 1)};
        // Require a non-empty, purely numeric id: std::stoi would otherwise
        // silently accept "1abc" and reject an empty id with an opaque throw.
        if (id_str.empty() ||
            id_str.find_first_not_of("0123456789") != std::string::npos) {
            throw std::invalid_argument(
                "Device::from_string: invalid device id in '" + spec + "'");
        }
        try {
            id = std::stoi(id_str);
        } catch (const std::out_of_range &) {
            throw std::invalid_argument(
                "Device::from_string: device id out of range in '" + spec +
                "'");
        }
    }

    if (kind == "cpu") {
        if (colon != std::string::npos) {
            throw std::invalid_argument(
                "Device::from_string: 'cpu' takes no device id in '" + spec +
                "'");
        }
        return Device::cpu();
    }
    if (kind == "cuda") {
        return Device::cuda(id);
    }
    if (kind == "rocm") {
        return Device::rocm(id);
    }
    if (kind == "gpu") {
        return Device::gpu(id);
    }
    throw std::invalid_argument("Device::from_string: unknown device '" + spec +
                                "'");
}

const char * Device::get_type_name() const {
    switch (this->device_type) {
        case DeviceType::CPU:
            return "CPU";
        case DeviceType::CUDA:
            return "CUDA";
        case DeviceType::CUDAHost:
            return "CUDAHost";
        case DeviceType::ROCm:
            return "ROCm";
        case DeviceType::ROCmHost:
            return "ROCmHost";
        default:
            return "Unknown";
    }
}

}  // namespace muGrid
