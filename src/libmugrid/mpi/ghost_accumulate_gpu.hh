/**
 * @file   mpi/ghost_accumulate_gpu.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   13 Jun 2026
 *
 * @brief  Native device accumulation for ghost reduction
 *
 * `reduce_ghosts` accumulates received ghost contributions into the
 * interior: dst[i] += src[i]. This header exposes a host-callable entry
 * point that performs that accumulation with a GPU kernel, replacing the
 * earlier copy-to-host / accumulate / copy-back round-trip. It lives in a
 * separate translation unit because the MPI communicator is host-compiled
 * (no kernel launches), whereas this file is compiled as CUDA/HIP.
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

#ifndef SRC_LIBMUGRID_MPI_GHOST_ACCUMULATE_GPU_HH_
#define SRC_LIBMUGRID_MPI_GHOST_ACCUMULATE_GPU_HH_

#include <cstddef>

#include "core/type_descriptor.hh"

namespace muGrid {

    /**
     * Block-strided accumulate into device memory, in a single kernel
     * launch: for block `b` in [0, `nb_blocks`) and element `j` in
     * [0, `block_len`),
     *
     *   dst[b * dst_block_stride + j] += src[b * src_block_stride + j]
     *
     * (`dst`/`src` already point at the first block; strides and lengths are
     * in elements of `type`). This covers a whole halo region at once -- a
     * per-block launch is dominated by kernel-launch/synchronize overhead.
     *
     * `dst` is always device memory. `src` is device memory when
     * `src_on_device` is true (read directly); otherwise it is a contiguous
     * host buffer, copied to a cached device scratch buffer once. Complex is
     * accumulated as reals (counts and strides double). The destination
     * never leaves the device.
     *
     * Only available in CUDA/HIP builds. Not thread-safe (the scratch
     * buffer is shared); `reduce_ghosts` is called from single-threaded MPI
     * code.
     */
    void device_accumulate(void * dst, const void * src, std::size_t nb_blocks,
                           std::size_t block_len, std::size_t dst_block_stride,
                           std::size_t src_block_stride, TypeDescriptor type,
                           bool src_on_device);

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_MPI_GHOST_ACCUMULATE_GPU_HH_
