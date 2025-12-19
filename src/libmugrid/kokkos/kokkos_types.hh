/**
 * @file   kokkos_types.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   09 Dec 2024
 *
 * @brief  Kokkos type definitions for GPU-portable memory management
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

#ifndef SRC_LIBMUGRID_KOKKOS_TYPES_HH_
#define SRC_LIBMUGRID_KOKKOS_TYPES_HH_

#include <Kokkos_Core.hpp>

namespace muGrid {

    // Default execution/memory spaces based on backend
#if defined(KOKKOS_ENABLE_CUDA)
    using DefaultDeviceSpace = Kokkos::CudaSpace;
    using DefaultExecutionSpace = Kokkos::Cuda;
#elif defined(KOKKOS_ENABLE_HIP)
    using DefaultDeviceSpace = Kokkos::HIPSpace;
    using DefaultExecutionSpace = Kokkos::HIP;
#else
    using DefaultDeviceSpace = Kokkos::HostSpace;
    using DefaultExecutionSpace = Kokkos::DefaultExecutionSpace;
#endif

    using HostSpace = Kokkos::HostSpace;
    using HostExecutionSpace = Kokkos::DefaultHostExecutionSpace;

    /**
     * Standard View type for field storage: 1D, column-major (LayoutLeft),
     * with configurable memory space. LayoutLeft is used for NumPy
     * compatibility (Fortran-style column-major ordering).
     */
    template<typename T, typename MemorySpace = HostSpace>
    using FieldView = Kokkos::View<T*, Kokkos::LayoutLeft, MemorySpace>;

    /**
     * 2D View type for multi-dimensional access patterns.
     */
    template<typename T, typename MemorySpace = HostSpace>
    using FieldView2D = Kokkos::View<T**, Kokkos::LayoutLeft, MemorySpace>;

    /**
     * Type trait to check if a memory space is host-accessible.
     */
    template<typename MemorySpace>
    struct is_host_space
        : std::is_same<MemorySpace, HostSpace> {};

    template<typename MemorySpace>
    inline constexpr bool is_host_space_v = is_host_space<MemorySpace>::value;

    /**
     * Type trait to check if a memory space is device (GPU) memory.
     * This is used to determine whether fields need device-specific handling.
     */
    template<typename MemorySpace>
    struct is_device_space : std::false_type {};

#if defined(KOKKOS_ENABLE_CUDA)
    template<>
    struct is_device_space<Kokkos::CudaSpace> : std::true_type {};
    template<>
    struct is_device_space<Kokkos::CudaUVMSpace> : std::true_type {};
#endif

#if defined(KOKKOS_ENABLE_HIP)
    template<>
    struct is_device_space<Kokkos::HIPSpace> : std::true_type {};
    template<>
    struct is_device_space<Kokkos::HIPManagedSpace> : std::true_type {};
#endif

    template<typename MemorySpace>
    inline constexpr bool is_device_space_v = is_device_space<MemorySpace>::value;

    /**
     * DLPack device type constants (from dlpack/dlpack.h)
     */
    namespace DLPackDeviceType {
        constexpr int CPU = 1;           // kDLCPU
        constexpr int CUDA = 2;          // kDLCUDA
        constexpr int CUDAHost = 3;      // kDLCUDAHost (pinned memory)
        constexpr int ROCm = 10;         // kDLROCm
        constexpr int ROCmHost = 11;     // kDLROCMHost
        constexpr int CUDAManaged = 13;  // kDLCUDAManaged (UVM)
    }

    /**
     * Type trait mapping Kokkos memory spaces to DLPack device types.
     * This enables compile-time selection of the correct DLPack device type
     * for a given memory space.
     */
    template<typename MemorySpace>
    struct dlpack_device_type {
        static constexpr int value = DLPackDeviceType::CPU;
    };

#if defined(KOKKOS_ENABLE_CUDA)
    template<>
    struct dlpack_device_type<Kokkos::CudaSpace> {
        static constexpr int value = DLPackDeviceType::CUDA;
    };
    template<>
    struct dlpack_device_type<Kokkos::CudaUVMSpace> {
        static constexpr int value = DLPackDeviceType::CUDAManaged;
    };
#endif

#if defined(KOKKOS_ENABLE_HIP)
    template<>
    struct dlpack_device_type<Kokkos::HIPSpace> {
        static constexpr int value = DLPackDeviceType::ROCm;
    };
    template<>
    struct dlpack_device_type<Kokkos::HIPManagedSpace> {
        static constexpr int value = DLPackDeviceType::ROCm;
    };
#endif

    template<typename MemorySpace>
    inline constexpr int dlpack_device_type_v = dlpack_device_type<MemorySpace>::value;

    /**
     * Get device name string for a memory space.
     * Returns "cpu", "cuda", or "rocm" based on the memory space.
     */
    template<typename MemorySpace>
    constexpr const char* device_name() {
        if constexpr (is_host_space_v<MemorySpace>) {
            return "cpu";
        }
#if defined(KOKKOS_ENABLE_CUDA)
        else if constexpr (std::is_same_v<MemorySpace, Kokkos::CudaSpace> ||
                          std::is_same_v<MemorySpace, Kokkos::CudaUVMSpace>) {
            return "cuda";
        }
#endif
#if defined(KOKKOS_ENABLE_HIP)
        else if constexpr (std::is_same_v<MemorySpace, Kokkos::HIPSpace> ||
                          std::is_same_v<MemorySpace, Kokkos::HIPManagedSpace>) {
            return "rocm";
        }
#endif
        else {
            return "unknown";
        }
    }

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_KOKKOS_TYPES_HH_
