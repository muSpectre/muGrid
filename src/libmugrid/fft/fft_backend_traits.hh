/**
 * @file   fft/fft_backend_traits.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   21 Dec 2024
 *
 * @brief  Compile-time FFT backend selection based on memory space
 *
 * Copyright (c) 2024 Lars Pastewka
 *
 * muGrid is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * muGrid is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with muGrid; see the file COPYING. If not, write to the
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

#ifndef SRC_LIBMUGRID_FFT_FFT_BACKEND_TRAITS_HH_
#define SRC_LIBMUGRID_FFT_FFT_BACKEND_TRAITS_HH_

#include "memory/memory_space.hh"
#include "memory/device.hh"
#include "fft_1d_backend.hh"
#include "pocketfft_backend.hh"

#if defined(MUGRID_ENABLE_CUDA)
#include "cufft_backend.hh"
#endif

#if defined(MUGRID_ENABLE_HIP)
#include "rocfft_backend.hh"
#endif

#include <memory>

namespace muGrid {

/**
 * Traits class that maps MemorySpace to the appropriate FFT backend type.
 *
 * This enables compile-time selection of the FFT backend based on the
 * memory space template parameter.
 */
template <typename MemorySpace>
struct FFTBackendSelector;

// Host space always uses PocketFFT
template <>
struct FFTBackendSelector<HostSpace> {
  using type = PocketFFTBackend;

  static std::unique_ptr<FFT1DBackend> create() {
    return std::make_unique<PocketFFTBackend>();
  }

  static constexpr const char * name() { return "PocketFFT"; }
};

#if defined(MUGRID_ENABLE_CUDA)
// CUDA space uses cuFFT
template <>
struct FFTBackendSelector<CUDASpace> {
  using type = cuFFTBackend;

  static std::unique_ptr<FFT1DBackend> create() {
    return std::make_unique<cuFFTBackend>();
  }

  static constexpr const char * name() { return "cuFFT"; }
};
#endif

#if defined(MUGRID_ENABLE_HIP)
// ROCm space uses native rocFFT (not hipFFT) for better stride support
template <>
struct FFTBackendSelector<ROCmSpace> {
  using type = rocFFTBackend;

  static std::unique_ptr<FFT1DBackend> create() {
    return std::make_unique<rocFFTBackend>();
  }

  static constexpr const char * name() { return "rocFFT"; }
};
#endif

/**
 * Helper alias for the backend type for a given memory space.
 */
template <typename MemorySpace>
using FFTBackend_t = typename FFTBackendSelector<MemorySpace>::type;

/**
 * Create the appropriate FFT backend for a memory space.
 */
template <typename MemorySpace>
std::unique_ptr<FFT1DBackend> create_fft_backend() {
  return FFTBackendSelector<MemorySpace>::create();
}

/**
 * Get the backend name for a memory space.
 */
template <typename MemorySpace>
constexpr const char * fft_backend_name() {
  return FFTBackendSelector<MemorySpace>::name();
}

/**
 * Convert MemorySpace to Device.
 */
template <typename MemorySpace>
constexpr Device memory_space_to_device() {
  if constexpr (is_host_space_v<MemorySpace>) {
    return Device::cpu();
  } else {
    return Device::cuda();  // or Device::rocm() depending on build config
  }
}

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_FFT_FFT_BACKEND_TRAITS_HH_
