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

namespace muGrid {

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
