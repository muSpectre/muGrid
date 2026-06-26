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

#include "device_alloc.hh"
#include "memory_space.hh"
#include "allocation_profiler.hh"
#include "gpu_runtime.hh"

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

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        /**
         * Memory allocator for device (CUDA/HIP) space. The CUDA and HIP
         * variants were byte-identical apart from the memset spelling, so they
         * share a single implementation that routes through the gpu_runtime.hh
         * shim.
         */
        template <typename T>
        struct DeviceAllocator {
            // Allocation goes through device_allocate() so that an
            // externally registered allocator (e.g. cupy's memory pool)
            // owns every device byte; see memory/device_alloc.hh.
            static T * allocate(std::size_t n) {
                if (n == 0)
                    return nullptr;
                return static_cast<T *>(device_allocate(n * sizeof(T)));
            }

            static void deallocate(T * ptr) {
                if (ptr)
                    device_deallocate(ptr);
            }

            static void memset(T * ptr, int value, std::size_t n) {
                GPU_MEMSET(ptr, value, n * sizeof(T));
            }
        };

#endif  // MUGRID_ENABLE_CUDA || MUGRID_ENABLE_HIP

        /**
         * Type trait to select the correct allocator for a memory space
         */
        template <typename T, typename MemorySpace>
        struct AllocatorSelector {
            using type = HostAllocator<T>;
        };

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        // DefaultDeviceSpace is the active device space (CUDASpace under CUDA,
        // ROCmSpace under HIP), so this one specialisation serves both backends.
        template <typename T>
        struct AllocatorSelector<T, DefaultDeviceSpace> {
            using type = DeviceAllocator<T>;
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
        explicit Array(const std::string & label)
            : data_(nullptr), size_(0), label_(label) {}

        /**
         * Constructor that allocates n elements
         */
        explicit Array(std::size_t n)
            : data_(allocator_type::allocate(n)), size_(n) {
            this->prof_record_alloc(data_, size_);
        }

        /**
         * Constructor with label and size
         */
        Array(const std::string & label, std::size_t n)
            : data_(allocator_type::allocate(n)), size_(n), label_(label) {
            this->prof_record_alloc(data_, size_);
        }

        /**
         * Destructor - frees memory
         */
        ~Array() {
            if (data_) {
                this->prof_record_free(data_);
                allocator_type::deallocate(data_);
            }
        }

        // Non-copyable (to avoid accidental expensive copies)
        Array(const Array &) = delete;
        Array & operator=(const Array &) = delete;

        // Movable. A move transfers ownership of the same buffer pointer, so
        // it is transparent to the allocation profiler (which tracks live
        // buffers by pointer); only the label/space metadata travels along.
        Array(Array && other) noexcept
            : data_(other.data_), size_(other.size_),
              label_(std::move(other.label_)),
              space_(std::move(other.space_)) {
            other.data_ = nullptr;
            other.size_ = 0;
        }

        Array & operator=(Array && other) noexcept {
            if (this != &other) {
                if (data_) {
                    this->prof_record_free(data_);
                    allocator_type::deallocate(data_);
                }
                data_ = other.data_;
                size_ = other.size_;
                label_ = std::move(other.label_);
                space_ = std::move(other.space_);
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
                this->prof_record_free(data_);
                allocator_type::deallocate(data_);
            }
            data_ = allocator_type::allocate(new_size);
            size_ = new_size;
            this->prof_record_alloc(data_, size_);
        }

        /**
         * Set the buffer's profiling label and requesting memory space (e.g.
         * "cuda:0"). Called by the owning Field before (re)allocation so the
         * allocation profiler can attribute the buffer to a named field. A
         * no-op unless built with MUGRID_PROFILE_ALLOCATIONS.
         */
        void set_label(const std::string & label, const std::string & space) {
            label_ = label;
            space_ = space;
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
        //! Report an allocation of @p n elements at @p p to the profiler.
        //! Cheap when profiling is disabled (one relaxed atomic load).
        void prof_record_alloc(T * p, std::size_t n) {
            if (p != nullptr && n > 0) {
                AllocationProfiler::instance().record_alloc(
                    p, label_, space_, n * sizeof(T));
            }
        }
        //! Report the deallocation of @p p to the profiler.
        static void prof_record_free(T * p) {
            if (p != nullptr) {
                AllocationProfiler::instance().record_free(p);
            }
        }

        T * data_;
        std::size_t size_;
        std::string label_{"<unnamed>"};
        std::string space_{device_name<MemorySpace>()};
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
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        // Host to device
        else if constexpr (is_host_space_v<SrcSpace> &&
                           std::is_same_v<DstSpace, DefaultDeviceSpace>) {
            GPU_MEMCPY_H2D(dst.data(), src.data(), src.size() * sizeof(T));
        }
        // Device to host
        else if constexpr (std::is_same_v<SrcSpace, DefaultDeviceSpace> &&
                           is_host_space_v<DstSpace>) {
            GPU_MEMCPY_D2H(dst.data(), src.data(), src.size() * sizeof(T));
        }
        // Device to device
        else if constexpr (std::is_same_v<SrcSpace, DefaultDeviceSpace> &&
                           std::is_same_v<DstSpace, DefaultDeviceSpace>) {
            GPU_MEMCPY_D2D(dst.data(), src.data(), src.size() * sizeof(T));
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
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        else if constexpr (std::is_same_v<MemorySpace, DefaultDeviceSpace>) {
            // No device-side scalar fill yet; stage the value on the host and
            // copy across.
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
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        // Host to device
        else if constexpr (is_host_space_v<SrcSpace> &&
                           std::is_same_v<DstSpace, DefaultDeviceSpace>) {
            GPU_MEMCPY_H2D(dst, src, count * sizeof(T));
            if (const char * err{gpu_last_error()}; err != nullptr) {
                throw std::runtime_error(
                    std::string("GPU memcpy H2D failed: ") + err);
            }
        }
        // Device to host
        else if constexpr (std::is_same_v<SrcSpace, DefaultDeviceSpace> &&
                           is_host_space_v<DstSpace>) {
            GPU_MEMCPY_D2H(dst, src, count * sizeof(T));
            if (const char * err{gpu_last_error()}; err != nullptr) {
                throw std::runtime_error(
                    std::string("GPU memcpy D2H failed: ") + err);
            }
        }
        // Device to device
        else if constexpr (std::is_same_v<SrcSpace, DefaultDeviceSpace> &&
                           std::is_same_v<DstSpace, DefaultDeviceSpace>) {
            GPU_MEMCPY_D2D(dst, src, count * sizeof(T));
            if (const char * err{gpu_last_error()}; err != nullptr) {
                throw std::runtime_error(
                    std::string("GPU memcpy D2D failed: ") + err);
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
