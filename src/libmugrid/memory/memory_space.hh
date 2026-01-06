/**
 * @file   memory_space.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   19 Dec 2024
 *
 * @brief  Memory space definitions for GPU-portable memory management
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

#ifndef SRC_LIBMUGRID_MEMORY_SPACE_HH_
#define SRC_LIBMUGRID_MEMORY_SPACE_HH_

#include <type_traits>

#include "core/enums.hh"

namespace muGrid {

    /**
     * Tag type for host (CPU) memory space.
     *
     * Uses ArrayOfStructures storage order for optimal CPU cache
     * locality when iterating over components within a pixel.
     */
    struct HostSpace {
        static constexpr const char * name = "Host";
        static constexpr StorageOrder storage_order =
            StorageOrder::ArrayOfStructures;
    };

#if defined(MUGRID_ENABLE_CUDA)
    /**
     * Tag type for CUDA device memory space.
     *
     * Uses StructureOfArrays storage order for optimal GPU memory
     * coalescence when threads in a warp access adjacent pixels.
     */
    struct CUDASpace {
        static constexpr const char * name = "CUDA";
        static constexpr StorageOrder storage_order =
            StorageOrder::StructureOfArrays;
    };
#endif

#if defined(MUGRID_ENABLE_HIP)
    /**
     * Tag type for ROCm device memory space.
     *
     * Uses StructureOfArrays storage order for optimal GPU memory
     * coalescence when threads in a warp access adjacent pixels.
     */
    struct ROCmSpace {
        static constexpr const char * name = "ROCm";
        static constexpr StorageOrder storage_order =
            StorageOrder::StructureOfArrays;
    };
#endif

    // Default device space based on backend
#if defined(MUGRID_ENABLE_CUDA)
    using DefaultDeviceSpace = CUDASpace;
#elif defined(MUGRID_ENABLE_HIP)
    using DefaultDeviceSpace = ROCmSpace;
#else
    using DefaultDeviceSpace = HostSpace;
#endif

    /**
     * Type trait to check if a memory space is host-accessible.
     */
    template <typename MemorySpace>
    struct is_host_space : std::is_same<MemorySpace, HostSpace> {};

    template <typename MemorySpace>
    inline constexpr bool is_host_space_v = is_host_space<MemorySpace>::value;

    /**
     * Type trait to check if a memory space is device (GPU) memory.
     */
    template <typename MemorySpace>
    struct is_device_space : std::false_type {};

#if defined(MUGRID_ENABLE_CUDA)
    template <>
    struct is_device_space<CUDASpace> : std::true_type {};
#endif

#if defined(MUGRID_ENABLE_HIP)
    template <>
    struct is_device_space<ROCmSpace> : std::true_type {};
#endif

    template <typename MemorySpace>
    inline constexpr bool is_device_space_v =
        is_device_space<MemorySpace>::value;

    /**
     * DLPack device type constants (from dlpack/dlpack.h)
     */
    namespace DLPackDeviceType {
        constexpr int CPU = 1;        // kDLCPU
        constexpr int CUDA = 2;       // kDLCUDA
        constexpr int CUDAHost = 3;   // kDLCUDAHost (pinned memory)
        constexpr int ROCm = 10;      // kDLROCm
        constexpr int ROCmHost = 11;  // kDLROCMHost
    }  // namespace DLPackDeviceType

    /**
     * Type trait mapping memory spaces to DLPack device types.
     */
    template <typename MemorySpace>
    struct dlpack_device_type {
        static constexpr int value = DLPackDeviceType::CPU;
    };

#if defined(MUGRID_ENABLE_CUDA)
    template <>
    struct dlpack_device_type<CUDASpace> {
        static constexpr int value = DLPackDeviceType::CUDA;
    };
#endif

#if defined(MUGRID_ENABLE_HIP)
    template <>
    struct dlpack_device_type<ROCmSpace> {
        static constexpr int value = DLPackDeviceType::ROCm;
    };
#endif

    template <typename MemorySpace>
    inline constexpr int dlpack_device_type_v =
        dlpack_device_type<MemorySpace>::value;

    /**
     * Get device name string for a memory space.
     */
    template <typename MemorySpace>
    constexpr const char * device_name() {
        if constexpr (is_host_space_v<MemorySpace>) {
            return "cpu";
        }
#if defined(MUGRID_ENABLE_CUDA)
        else if constexpr (std::is_same_v<MemorySpace, CUDASpace>) {
            return "cuda";
        }
#endif
#if defined(MUGRID_ENABLE_HIP)
        else if constexpr (std::is_same_v<MemorySpace, ROCmSpace>) {
            return "rocm";
        }
#endif
        else {
            return "unknown";
        }
    }

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_MEMORY_SPACE_HH_
