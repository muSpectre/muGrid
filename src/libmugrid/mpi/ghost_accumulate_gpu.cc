/**
 * @file   mpi/ghost_accumulate_gpu.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   13 Jun 2026
 *
 * @brief  Native device accumulation for ghost reduction (CUDA/HIP)
 *
 * Copyright © 2026 Lars Pastewka
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
 */

#include "mpi/ghost_accumulate_gpu.hh"

#include "core/types.hh"
#include "core/exception.hh"
#include "memory/gpu_runtime.hh"
#include "memory/device_alloc.hh"

namespace muGrid {

    namespace {

        constexpr int BLOCK_SIZE = 256;

        //! Block-strided accumulate: for block b in [0, nb_blocks) and element
        //! j in [0, block_len), dst[b*dst_stride + j] += src[b*src_stride + j].
        //! A single launch covers the whole halo region (one sync), which is
        //! what makes this fast -- a per-block launch/sync is dominated by
        //! launch overhead.
        template <typename T>
        __global__ void accumulate_strided_kernel(
            T * dst, const T * src, std::size_t nb_blocks, std::size_t block_len,
            std::size_t dst_stride, std::size_t src_stride) {
            std::size_t g{blockIdx.x * blockDim.x + threadIdx.x};
            std::size_t total{nb_blocks * block_len};
            if (g < total) {
                std::size_t b{g / block_len};
                std::size_t j{g % block_len};
                dst[b * dst_stride + j] += src[b * src_stride + j];
            }
        }

        // Cached device scratch holding host-resident src data while the
        // kernel accumulates it. Grows on demand; intentionally leaked at
        // process exit (like the other muGrid device caches). Not thread-safe
        // (reduce_ghosts runs in single-threaded MPI code).
        void * scratch_ptr{nullptr};
        std::size_t scratch_bytes{0};

        void * get_scratch(std::size_t bytes) {
            if (scratch_bytes < bytes) {
                if (scratch_ptr != nullptr) {
                    device_deallocate(scratch_ptr);
                }
                scratch_ptr = device_allocate(bytes);
                scratch_bytes = bytes;
            }
            return scratch_ptr;
        }

        template <typename T>
        void launch_accumulate(T * dst, const T * src, std::size_t nb_blocks,
                               std::size_t block_len, std::size_t dst_stride,
                               std::size_t src_stride, bool src_on_device) {
            std::size_t total{nb_blocks * block_len};
            if (total == 0) {
                return;
            }
            const T * device_src{src};
            if (!src_on_device) {
                // Stage the whole (contiguous host) src region to the device
                // once, then accumulate on the device: the destination never
                // leaves the device and there is a single host->device copy
                // for the entire halo rather than one per block.
                std::size_t span{(nb_blocks - 1) * src_stride + block_len};
                T * scratch{static_cast<T *>(get_scratch(span * sizeof(T)))};
                GPU_MEMCPY_H2D(scratch, src, span * sizeof(T));
                device_src = scratch;
            }
            int nb_threads{static_cast<int>((total + BLOCK_SIZE - 1) /
                                            BLOCK_SIZE)};
            GPU_LAUNCH_KERNEL(accumulate_strided_kernel<T>, nb_threads,
                              BLOCK_SIZE, dst, device_src, nb_blocks, block_len,
                              dst_stride, src_stride);
            GPU_DEVICE_SYNCHRONIZE();
        }

    }  // namespace

    void device_accumulate(void * dst, const void * src, std::size_t nb_blocks,
                           std::size_t block_len, std::size_t dst_block_stride,
                           std::size_t src_block_stride, TypeDescriptor type,
                           bool src_on_device) {
        switch (type) {
        case TypeDescriptor::Real:
            launch_accumulate(static_cast<Real *>(dst),
                              static_cast<const Real *>(src), nb_blocks,
                              block_len, dst_block_stride, src_block_stride,
                              src_on_device);
            break;
        case TypeDescriptor::Complex:
            // Accumulate the underlying reals (complex addition is
            // component-wise): two reals per complex element, so element
            // counts and strides double.
            launch_accumulate(reinterpret_cast<Real *>(dst),
                              reinterpret_cast<const Real *>(src), nb_blocks,
                              2 * block_len, 2 * dst_block_stride,
                              2 * src_block_stride, src_on_device);
            break;
        case TypeDescriptor::Int:
            launch_accumulate(static_cast<Int *>(dst),
                              static_cast<const Int *>(src), nb_blocks,
                              block_len, dst_block_stride, src_block_stride,
                              src_on_device);
            break;
        case TypeDescriptor::Uint:
            launch_accumulate(static_cast<Uint *>(dst),
                              static_cast<const Uint *>(src), nb_blocks,
                              block_len, dst_block_stride, src_block_stride,
                              src_on_device);
            break;
        case TypeDescriptor::Index:
            launch_accumulate(static_cast<Index_t *>(dst),
                              static_cast<const Index_t *>(src), nb_blocks,
                              block_len, dst_block_stride, src_block_stride,
                              src_on_device);
            break;
        default:
            throw RuntimeError(
                "device_accumulate: unsupported type descriptor");
        }
    }

}  // namespace muGrid
