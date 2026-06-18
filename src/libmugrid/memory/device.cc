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

#include <map>

#if defined(MUGRID_ENABLE_CUDA)
#include <cuda_runtime.h>
#elif defined(MUGRID_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace muGrid {

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
        cache[device_id] = integrated;
        return integrated;
    }
}  // namespace
#endif  // MUGRID_ENABLE_CUDA || MUGRID_ENABLE_HIP

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
