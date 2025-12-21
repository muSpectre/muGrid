/**
 * @file   fft/fft_work_buffer.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   21 Dec 2024
 *
 * @brief  Runtime work buffer and memory operations for FFT engine
 *
 * This file provides utilities for managing work buffers that can be
 * allocated on either host or device memory, with the location determined
 * at runtime. This is essential for FFT operations that need temporary
 * storage matching the input/output field memory locations.
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

#ifndef SRC_LIBMUGRID_FFT_FFT_WORK_BUFFER_HH_
#define SRC_LIBMUGRID_FFT_FFT_WORK_BUFFER_HH_

#include "core/types.hh"

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>

#if defined(MUGRID_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

#if defined(MUGRID_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace muGrid {

/**
 * @brief Copy memory between buffers with runtime memory location detection.
 *
 * This function handles all combinations of host/device memory copies:
 * - Host to Host: uses std::memcpy
 * - Device to Device: uses cudaMemcpy/hipMemcpy with DeviceToDevice
 * - Host to Device: uses cudaMemcpy/hipMemcpy with HostToDevice
 * - Device to Host: uses cudaMemcpy/hipMemcpy with DeviceToHost
 *
 * @tparam T Element type
 * @param dst Destination pointer
 * @param src Source pointer
 * @param count Number of elements to copy
 * @param dst_on_device True if destination is on GPU device
 * @param src_on_device True if source is on GPU device
 */
template <typename T>
void runtime_memcpy(T * dst, const T * src, std::size_t count,
                    bool dst_on_device, bool src_on_device) {
  if (count == 0) return;

  std::size_t bytes = count * sizeof(T);

  if (!src_on_device && !dst_on_device) {
    // Host to Host
    std::memcpy(dst, src, bytes);
  }
#if defined(MUGRID_ENABLE_CUDA)
  else if (src_on_device && dst_on_device) {
    // Device to Device
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("CUDA memcpy D2D failed: ") +
                               cudaGetErrorString(err));
    }
  } else if (!src_on_device && dst_on_device) {
    // Host to Device
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("CUDA memcpy H2D failed: ") +
                               cudaGetErrorString(err));
    }
  } else {
    // Device to Host
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("CUDA memcpy D2H failed: ") +
                               cudaGetErrorString(err));
    }
  }
#elif defined(MUGRID_ENABLE_HIP)
  else if (src_on_device && dst_on_device) {
    // Device to Device
    hipError_t err = hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice);
    if (err != hipSuccess) {
      throw std::runtime_error(std::string("HIP memcpy D2D failed: ") +
                               hipGetErrorString(err));
    }
  } else if (!src_on_device && dst_on_device) {
    // Host to Device
    hipError_t err = hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
      throw std::runtime_error(std::string("HIP memcpy H2D failed: ") +
                               hipGetErrorString(err));
    }
  } else {
    // Device to Host
    hipError_t err = hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
      throw std::runtime_error(std::string("HIP memcpy D2H failed: ") +
                               hipGetErrorString(err));
    }
  }
#else
  else {
    // No GPU support compiled in, but device memory requested
    throw std::runtime_error(
        "GPU memory copy requested but no GPU backend compiled");
  }
#endif
}

/**
 * @brief Work buffer that allocates on host or device based on runtime flag.
 *
 * This class provides a simple RAII wrapper for temporary buffers used in
 * FFT operations. The memory location (host or device) is determined at
 * construction time based on a runtime flag.
 *
 * @tparam T Element type (typically Complex)
 */
template <typename T>
class RuntimeWorkBuffer {
 public:
  /**
   * @brief Construct a work buffer.
   *
   * @param size Number of elements to allocate
   * @param on_device If true, allocate on GPU device; otherwise on host
   */
  RuntimeWorkBuffer(std::size_t size, bool on_device)
      : size_(size), on_device_(on_device), data_(nullptr) {
    if (size == 0) return;

    if (!on_device) {
      // Host allocation
      data_ = new T[size];
    }
#if defined(MUGRID_ENABLE_CUDA)
    else {
      cudaError_t err = cudaMalloc(&data_, size * sizeof(T));
      if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA allocation failed: ") +
                                 cudaGetErrorString(err));
      }
    }
#elif defined(MUGRID_ENABLE_HIP)
    else {
      hipError_t err = hipMalloc(&data_, size * sizeof(T));
      if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP allocation failed: ") +
                                 hipGetErrorString(err));
      }
    }
#else
    else {
      throw std::runtime_error(
          "GPU allocation requested but no GPU backend compiled");
    }
#endif
  }

  ~RuntimeWorkBuffer() {
    if (data_ == nullptr) return;

    if (!on_device_) {
      delete[] data_;
    }
#if defined(MUGRID_ENABLE_CUDA)
    else {
      (void)cudaFree(data_);
    }
#elif defined(MUGRID_ENABLE_HIP)
    else {
      (void)hipFree(data_);
    }
#endif
  }

  // Non-copyable
  RuntimeWorkBuffer(const RuntimeWorkBuffer &) = delete;
  RuntimeWorkBuffer & operator=(const RuntimeWorkBuffer &) = delete;

  // Movable
  RuntimeWorkBuffer(RuntimeWorkBuffer && other) noexcept
      : size_(other.size_), on_device_(other.on_device_), data_(other.data_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }

  RuntimeWorkBuffer & operator=(RuntimeWorkBuffer && other) noexcept {
    if (this != &other) {
      // Free existing data
      if (data_ != nullptr) {
        if (!on_device_) {
          delete[] data_;
        }
#if defined(MUGRID_ENABLE_CUDA)
        else {
          (void)cudaFree(data_);
        }
#elif defined(MUGRID_ENABLE_HIP)
        else {
          (void)hipFree(data_);
        }
#endif
      }
      // Take ownership
      data_ = other.data_;
      size_ = other.size_;
      on_device_ = other.on_device_;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  /**
   * @brief Get raw pointer to data.
   */
  T * data() { return data_; }
  const T * data() const { return data_; }

  /**
   * @brief Get number of elements.
   */
  std::size_t size() const { return size_; }

  /**
   * @brief Check if buffer is on device.
   */
  bool is_on_device() const { return on_device_; }

  /**
   * @brief Copy data from another buffer.
   *
   * @param src Source pointer
   * @param count Number of elements to copy
   * @param src_on_device True if source is on device
   */
  void copy_from(const T * src, std::size_t count, bool src_on_device) {
    runtime_memcpy(data_, src, count, on_device_, src_on_device);
  }

  /**
   * @brief Copy data to another buffer.
   *
   * @param dst Destination pointer
   * @param count Number of elements to copy
   * @param dst_on_device True if destination is on device
   */
  void copy_to(T * dst, std::size_t count, bool dst_on_device) const {
    runtime_memcpy(dst, data_, count, dst_on_device, on_device_);
  }

 private:
  std::size_t size_;
  bool on_device_;
  T * data_;
};

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_FFT_FFT_WORK_BUFFER_HH_
