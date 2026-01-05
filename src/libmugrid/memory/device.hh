/**
 * @file   device.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   29 Dec 2024
 *
 * @brief  Device abstraction layer for runtime device representation
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

#ifndef SRC_LIBMUGRID_MEMORY_DEVICE_HH_
#define SRC_LIBMUGRID_MEMORY_DEVICE_HH_

#include <cstdint>
#include <string>

namespace muGrid {

/**
 * Device type enumeration following DLPack conventions.
 * Values match DLPack's DLDeviceType for interoperability.
 */
enum class DeviceType : std::int8_t {
    CPU = 1,       //!< kDLCPU - CPU device
    CUDA = 2,      //!< kDLCUDA - CUDA GPU
    CUDAHost = 3,  //!< kDLCUDAHost - CUDA pinned memory
    ROCm = 10,     //!< kDLROCm - ROCm/HIP GPU
    ROCmHost = 11  //!< kDLROCMHost - ROCm pinned memory
};

/**
 * Runtime device abstraction that encapsulates device information.
 *
 * This class provides a unified way to represent and query device
 * information, independent of the Field or FieldCollection that
 * may reside on that device.
 *
 * Device is a value type that can be:
 * - Compared for equality
 * - Passed as function arguments
 * - Stored and returned from functions
 * - Constructed from memory spaces at compile time
 *
 * The Device class complements (does not replace) the compile-time
 * MemorySpace system:
 * - MemorySpace: Compile-time template parameter determining code generation
 * - Device: Runtime value for queries, comparison, and multi-GPU support
 */
class Device {
   public:
    /**
     * Default constructor creates a CPU device (device_id = 0).
     */
    constexpr Device() : device_type{DeviceType::CPU}, device_id{0} {}

    /**
     * Construct device with explicit type and ID.
     * @param type The device type (CPU, CUDA, ROCm, etc.)
     * @param id Device ID for multi-GPU systems (default 0)
     */
    constexpr Device(DeviceType type, int id = 0)
        : device_type{type}, device_id{id} {}

    //! Check if this is a device (GPU) memory location
    constexpr bool is_device() const {
        return this->device_type == DeviceType::CUDA ||
               this->device_type == DeviceType::ROCm;
    }

    //! Check if this is a host (CPU) memory location
    constexpr bool is_host() const {
        return this->device_type == DeviceType::CPU;
    }

    //! Get the device type
    constexpr DeviceType get_type() const { return this->device_type; }

    //! Get DLPack device type (for Python/DLPack interoperability)
    constexpr int get_dlpack_device_type() const {
        return static_cast<int>(this->device_type);
    }

    //! Get device ID for multi-GPU systems
    constexpr int get_device_id() const { return this->device_id; }

    //! Get device string for Python interoperability ("cpu", "cuda:0", "rocm:0")
    std::string get_device_string() const;

    //! Get device type name ("CPU", "CUDA", "ROCm", etc.)
    const char * get_type_name() const;

    //! Equality comparison
    constexpr bool operator==(const Device & other) const {
        return this->device_type == other.device_type &&
               this->device_id == other.device_id;
    }

    //! Inequality comparison
    constexpr bool operator!=(const Device & other) const {
        return !(*this == other);
    }

    //! Static factory for CPU device
    static constexpr Device cpu() { return Device{DeviceType::CPU, 0}; }

    //! Static factory for CUDA device
    static constexpr Device cuda(int id = 0) {
        return Device{DeviceType::CUDA, id};
    }

    //! Static factory for ROCm device
    static constexpr Device rocm(int id = 0) {
        return Device{DeviceType::ROCm, id};
    }

    /**
     * Static factory for default GPU device.
     *
     * Returns the default GPU device based on compile-time configuration:
     * - If CUDA is enabled, returns Device::cuda(0)
     * - If HIP/ROCm is enabled (and CUDA is not), returns Device::rocm(0)
     * - If no GPU backend is available, returns Device::cpu() as fallback
     *
     * This provides a portable way to request "any available GPU" without
     * knowing which backend is compiled in.
     */
    static constexpr Device gpu(int id = 0) {
#if defined(MUGRID_ENABLE_CUDA)
        return Device{DeviceType::CUDA, id};
#elif defined(MUGRID_ENABLE_HIP)
        return Device{DeviceType::ROCm, id};
#else
        // Fallback to CPU if no GPU backend is available
        // This allows code to compile without #ifdefs everywhere
        (void)id;  // Suppress unused parameter warning
        return Device{DeviceType::CPU, 0};
#endif
    }

   protected:
    DeviceType device_type;  //!< Type of device (CPU, CUDA, ROCm, etc.)
    int device_id;           //!< Device ID for multi-GPU systems
};

// Forward declarations of memory space tags
struct HostSpace;
struct CUDASpace;
struct ROCmSpace;

/**
 * Compile-time conversion from memory space tag to Device.
 * @tparam MemorySpace The memory space tag type
 * @return Device corresponding to the memory space
 */
template <typename MemorySpace>
constexpr Device memory_space_to_device();

// Specialization for HostSpace
template <>
constexpr Device memory_space_to_device<HostSpace>() {
    return Device::cpu();
}

#if defined(MUGRID_ENABLE_CUDA)
// Specialization for CUDASpace
// Note: Returns device 0 because this is a constexpr function evaluated at
// compile time. The actual device ID for multi-GPU is a runtime concern
// that must be handled when creating FieldCollections with specific devices.
template <>
constexpr Device memory_space_to_device<CUDASpace>() {
    return Device::cuda(0);
}
#endif

#if defined(MUGRID_ENABLE_HIP)
// Specialization for ROCmSpace
// Note: Returns device 0 because this is a constexpr function evaluated at
// compile time. The actual device ID for multi-GPU is a runtime concern
// that must be handled when creating FieldCollections with specific devices.
template <>
constexpr Device memory_space_to_device<ROCmSpace>() {
    return Device::rocm(0);
}
#endif

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_MEMORY_DEVICE_HH_
