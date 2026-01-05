/**
 * @file   array.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   19 Dec 2024
 *
 * @brief  One-dimensional array class for portable memory management on CPUs
 *         and GPUs
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

#ifndef SRC_LIBMUGRID_DEVICE_ARRAY_HH_
#define SRC_LIBMUGRID_DEVICE_ARRAY_HH_

#include <cstddef>
#include <cstring>
#include <string>
#include <stdexcept>

#include "memory_space.hh"

#if defined(MUGRID_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

#if defined(MUGRID_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace muGrid {

    // Forward declaration
    template <typename T, typename MemorySpace>
    class Array;

    namespace detail {

        /**
         * Memory allocator for host space - uses standard allocation
         */
        template <typename T>
        struct HostAllocator {
            static T * allocate(std::size_t n) {
                if (n == 0)
                    return nullptr;
                return new T[n];
            }

            static void deallocate(T * ptr) { delete[] ptr; }

            static void memset(T * ptr, int value, std::size_t n) {
                std::memset(ptr, value, n * sizeof(T));
            }
        };

#if defined(MUGRID_ENABLE_CUDA)
        /**
         * Memory allocator for CUDA device space
         */
        template <typename T>
        struct CudaAllocator {
            static T * allocate(std::size_t n) {
                if (n == 0)
                    return nullptr;
                T * ptr = nullptr;
                cudaError_t err = cudaMalloc(&ptr, n * sizeof(T));
                if (err != cudaSuccess) {
                    throw std::runtime_error(
                        std::string("CUDA allocation failed: ") +
                        cudaGetErrorString(err));
                }
                return ptr;
            }

            static void deallocate(T * ptr) {
                if (ptr)
                    (void)cudaFree(ptr);
            }

            static void memset(T * ptr, int value, std::size_t n) {
                (void)cudaMemset(ptr, value, n * sizeof(T));
            }
        };

#endif  // MUGRID_ENABLE_CUDA

#if defined(MUGRID_ENABLE_HIP)
        /**
         * Memory allocator for HIP device space
         */
        template <typename T>
        struct HIPAllocator {
            static T * allocate(std::size_t n) {
                if (n == 0)
                    return nullptr;
                T * ptr = nullptr;
                hipError_t err = hipMalloc(&ptr, n * sizeof(T));
                if (err != hipSuccess) {
                    throw std::runtime_error(
                        std::string("HIP allocation failed: ") +
                        hipGetErrorString(err));
                }
                return ptr;
            }

            static void deallocate(T * ptr) {
                if (ptr)
                    (void)hipFree(ptr);
            }

            static void memset(T * ptr, int value, std::size_t n) {
                (void)hipMemset(ptr, value, n * sizeof(T));
            }
        };

#endif  // MUGRID_ENABLE_HIP

        /**
         * Type trait to select the correct allocator for a memory space
         */
        template <typename T, typename MemorySpace>
        struct AllocatorSelector {
            using type = HostAllocator<T>;
        };

#if defined(MUGRID_ENABLE_CUDA)
        template <typename T>
        struct AllocatorSelector<T, CUDASpace> {
            using type = CudaAllocator<T>;
        };
#endif

#if defined(MUGRID_ENABLE_HIP)
        template <typename T>
        struct AllocatorSelector<T, ROCmSpace> {
            using type = HIPAllocator<T>;
        };
#endif

        template <typename T, typename MemorySpace>
        using Allocator = typename AllocatorSelector<T, MemorySpace>::type;

    }  // namespace detail

    /**
     * @brief GPU-portable 1D array class
     *
     * This class provides a simple interface for managing arrays in different
     * memory spaces (host, CUDA, HIP).
     *
     * @tparam T Element type
     * @tparam MemorySpace Memory space tag (HostSpace, CUDASpace, ROCmSpace,
     * etc.)
     */
    template <typename T, typename MemorySpace = HostSpace>
    class Array {
       public:
        using value_type = T;
        using memory_space = MemorySpace;
        using allocator_type = detail::Allocator<T, MemorySpace>;

        /**
         * Default constructor - creates empty array
         */
        Array() : data_(nullptr), size_(0) {}

        /**
         * Constructor with label (for debugging) - creates empty array
         */
        explicit Array(const std::string & /* label */)
            : data_(nullptr), size_(0) {}

        /**
         * Constructor that allocates n elements
         */
        explicit Array(std::size_t n)
            : data_(allocator_type::allocate(n)), size_(n) {}

        /**
         * Constructor with label and size
         */
        Array(const std::string & /* label */, std::size_t n)
            : data_(allocator_type::allocate(n)), size_(n) {}

        /**
         * Destructor - frees memory
         */
        ~Array() {
            if (data_) {
                allocator_type::deallocate(data_);
            }
        }

        // Non-copyable (to avoid accidental expensive copies)
        Array(const Array &) = delete;
        Array & operator=(const Array &) = delete;

        // Movable
        Array(Array && other) noexcept
            : data_(other.data_), size_(other.size_) {
            other.data_ = nullptr;
            other.size_ = 0;
        }

        Array & operator=(Array && other) noexcept {
            if (this != &other) {
                if (data_) {
                    allocator_type::deallocate(data_);
                }
                data_ = other.data_;
                size_ = other.size_;
                other.data_ = nullptr;
                other.size_ = 0;
            }
            return *this;
        }

        /**
         * Get raw pointer to data
         */
        T * data() { return data_; }
        const T * data() const { return data_; }

        /**
         * Get number of elements
         */
        std::size_t size() const { return size_; }

        /**
         * Check if array is empty
         */
        bool empty() const { return size_ == 0; }

        /**
         * Resize the array, reallocating if necessary.
         * Note: Does NOT preserve existing data (unlike std::vector).
         */
        void resize(std::size_t new_size) {
            if (new_size == size_)
                return;

            if (data_) {
                allocator_type::deallocate(data_);
            }
            data_ = allocator_type::allocate(new_size);
            size_ = new_size;
        }

        /**
         * Set all bytes to zero
         */
        void fill_zero() {
            if (data_ && size_ > 0) {
                allocator_type::memset(data_, 0, size_);
            }
        }

        /**
         * Element access (host-space only)
         */
        template <typename M = MemorySpace>
        std::enable_if_t<is_host_space_v<M>, T &> operator[](std::size_t i) {
            return data_[i];
        }

        template <typename M = MemorySpace>
        std::enable_if_t<is_host_space_v<M>, const T &>
        operator[](std::size_t i) const {
            return data_[i];
        }

       private:
        T * data_;
        std::size_t size_;
    };

    /**
     * @brief Resize a Array (free function for compatibility)
     */
    template <typename T, typename MemorySpace>
    void resize(Array<T, MemorySpace> & arr, std::size_t new_size) {
        arr.resize(new_size);
    }

    /**
     * @brief Deep copy between arrays, potentially in different memory spaces
     */
    template <typename T, typename DstSpace, typename SrcSpace>
    void deep_copy(Array<T, DstSpace> & dst, const Array<T, SrcSpace> & src) {
        if (dst.size() != src.size()) {
            throw std::runtime_error(
                "deep_copy: destination and source sizes must match");
        }
        if (src.size() == 0)
            return;

        // Host to Host
        if constexpr (is_host_space_v<DstSpace> && is_host_space_v<SrcSpace>) {
            std::memcpy(dst.data(), src.data(), src.size() * sizeof(T));
        }
#if defined(MUGRID_ENABLE_CUDA)
        // Host to CUDA
        else if constexpr (is_host_space_v<SrcSpace> &&
                           std::is_same_v<DstSpace, CUDASpace>) {
            (void)cudaMemcpy(dst.data(), src.data(), src.size() * sizeof(T),
                             cudaMemcpyHostToDevice);
        }
        // CUDA to Host
        else if constexpr (std::is_same_v<SrcSpace, CUDASpace> &&
                           is_host_space_v<DstSpace>) {
            (void)cudaMemcpy(dst.data(), src.data(), src.size() * sizeof(T),
                             cudaMemcpyDeviceToHost);
        }
        // CUDA to CUDA
        else if constexpr (std::is_same_v<SrcSpace, CUDASpace> &&
                           std::is_same_v<DstSpace, CUDASpace>) {
            (void)cudaMemcpy(dst.data(), src.data(), src.size() * sizeof(T),
                             cudaMemcpyDeviceToDevice);
        }
#endif
#if defined(MUGRID_ENABLE_HIP)
        // Host to ROCm
        else if constexpr (is_host_space_v<SrcSpace> &&
                           std::is_same_v<DstSpace, ROCmSpace>) {
            (void)hipMemcpy(dst.data(), src.data(), src.size() * sizeof(T),
                            hipMemcpyHostToDevice);
        }
        // ROCm to Host
        else if constexpr (std::is_same_v<SrcSpace, ROCmSpace> &&
                           is_host_space_v<DstSpace>) {
            (void)hipMemcpy(dst.data(), src.data(), src.size() * sizeof(T),
                            hipMemcpyDeviceToHost);
        }
        // ROCm to ROCm
        else if constexpr (std::is_same_v<SrcSpace, ROCmSpace> &&
                           std::is_same_v<DstSpace, ROCmSpace>) {
            (void)hipMemcpy(dst.data(), src.data(), src.size() * sizeof(T),
                            hipMemcpyDeviceToDevice);
        }
#endif
        else {
            static_assert(is_host_space_v<DstSpace> ||
                              is_host_space_v<SrcSpace>,
                          "Unsupported memory space combination for deep_copy");
        }
    }

    /**
     * @brief Fill array with a scalar value
     */
    template <typename T, typename MemorySpace>
    void deep_copy(Array<T, MemorySpace> & dst, const T & value) {
        if constexpr (is_host_space_v<MemorySpace>) {
            // Host: simple loop
            for (std::size_t i = 0; i < dst.size(); ++i) {
                dst[i] = value;
            }
        }
#if defined(MUGRID_ENABLE_CUDA)
        else if constexpr (std::is_same_v<MemorySpace, CUDASpace>) {
            // For CUDA, we need a kernel (or use thrust)
            // For now, copy via host for scalar fill
            if (dst.size() > 0) {
                Array<T, HostSpace> tmp(dst.size());
                for (std::size_t i = 0; i < tmp.size(); ++i) {
                    tmp[i] = value;
                }
                deep_copy(dst, tmp);
            }
        }
#endif
#if defined(MUGRID_ENABLE_HIP)
        else if constexpr (std::is_same_v<MemorySpace, ROCmSpace>) {
            // Same approach for ROCm
            if (dst.size() > 0) {
                Array<T, HostSpace> tmp(dst.size());
                for (std::size_t i = 0; i < tmp.size(); ++i) {
                    tmp[i] = value;
                }
                deep_copy(dst, tmp);
            }
        }
#endif
    }

    /**
     * @brief Deep copy between raw pointers in different memory spaces.
     *
     * This overload allows copying between raw pointers when the memory
     * spaces are known at compile time. Useful for FFT work buffers.
     *
     * @tparam T Element type
     * @tparam DstSpace Destination memory space
     * @tparam SrcSpace Source memory space
     * @param dst Destination pointer
     * @param src Source pointer
     * @param count Number of elements to copy
     */
    template <typename T, typename DstSpace, typename SrcSpace>
    void deep_copy(T * dst, const T * src, std::size_t count) {
        if (count == 0)
            return;

        // Host to Host
        if constexpr (is_host_space_v<DstSpace> && is_host_space_v<SrcSpace>) {
            std::memcpy(dst, src, count * sizeof(T));
        }
#if defined(MUGRID_ENABLE_CUDA)
        // Host to CUDA
        else if constexpr (is_host_space_v<SrcSpace> &&
                           std::is_same_v<DstSpace, CUDASpace>) {
            cudaError_t err =
                cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("CUDA memcpy H2D failed: ") +
                    cudaGetErrorString(err));
            }
        }
        // CUDA to Host
        else if constexpr (std::is_same_v<SrcSpace, CUDASpace> &&
                           is_host_space_v<DstSpace>) {
            cudaError_t err =
                cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("CUDA memcpy D2H failed: ") +
                    cudaGetErrorString(err));
            }
        }
        // CUDA to CUDA
        else if constexpr (std::is_same_v<SrcSpace, CUDASpace> &&
                           std::is_same_v<DstSpace, CUDASpace>) {
            cudaError_t err = cudaMemcpy(dst, src, count * sizeof(T),
                                         cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("CUDA memcpy D2D failed: ") +
                    cudaGetErrorString(err));
            }
        }
#endif
#if defined(MUGRID_ENABLE_HIP)
        // Host to ROCm
        else if constexpr (is_host_space_v<SrcSpace> &&
                           std::is_same_v<DstSpace, ROCmSpace>) {
            hipError_t err =
                hipMemcpy(dst, src, count * sizeof(T), hipMemcpyHostToDevice);
            if (err != hipSuccess) {
                throw std::runtime_error(
                    std::string("ROCm memcpy H2D failed: ") +
                    hipGetErrorString(err));
            }
        }
        // ROCm to Host
        else if constexpr (std::is_same_v<SrcSpace, ROCmSpace> &&
                           is_host_space_v<DstSpace>) {
            hipError_t err =
                hipMemcpy(dst, src, count * sizeof(T), hipMemcpyDeviceToHost);
            if (err != hipSuccess) {
                throw std::runtime_error(
                    std::string("ROCm memcpy D2H failed: ") +
                    hipGetErrorString(err));
            }
        }
        // ROCm to ROCm
        else if constexpr (std::is_same_v<SrcSpace, ROCmSpace> &&
                           std::is_same_v<DstSpace, ROCmSpace>) {
            hipError_t err =
                hipMemcpy(dst, src, count * sizeof(T), hipMemcpyDeviceToDevice);
            if (err != hipSuccess) {
                throw std::runtime_error(
                    std::string("ROCm memcpy D2D failed: ") +
                    hipGetErrorString(err));
            }
        }
#endif
        else {
            static_assert(is_host_space_v<DstSpace> ||
                              is_host_space_v<SrcSpace>,
                          "Unsupported memory space combination for deep_copy");
        }
    }

    /**
     * @brief Deep copy within the same memory space (raw pointers).
     *
     * Convenience overload when source and destination are in the same space.
     */
    template <typename T, typename MemorySpace>
    void deep_copy(T * dst, const T * src, std::size_t count) {
        deep_copy<T, MemorySpace, MemorySpace>(dst, src, count);
    }

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_DEVICE_ARRAY_HH_
