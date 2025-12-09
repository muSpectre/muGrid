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

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_KOKKOS_TYPES_HH_
