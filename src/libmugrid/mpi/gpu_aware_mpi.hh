/**
 * @file   mpi/gpu_aware_mpi.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   12 Jun 2026
 *
 * @brief  Runtime detection of GPU-aware MPI
 *
 * muGrid communicates device data as contiguous staging buffers. A
 * GPU-aware MPI can take those device pointers directly; a plain MPI
 * build would dereference them on the host and crash, so the
 * communication routines bounce the staging buffers through host memory
 * instead. This header provides the runtime decision.
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

#ifndef SRC_LIBMUGRID_MPI_GPU_AWARE_MPI_HH_
#define SRC_LIBMUGRID_MPI_GPU_AWARE_MPI_HH_

namespace muGrid {

    /**
     * True if the MPI library can take device pointers directly
     * (GPU-aware MPI). The result is determined once and cached:
     *
     * 1. The environment variable ``MUGRID_GPU_AWARE_MPI`` (``1``/``0``)
     *    overrides any detection — use it for MPI stacks whose support
     *    cannot be queried, or to force the host-staging fallback.
     * 2. Open MPI is queried via ``MPIX_Query_cuda_support()`` /
     *    ``MPIX_Query_rocm_support()`` where available.
     * 3. Otherwise the conservative answer is false: device buffers are
     *    bounced through host staging, which is correct with any MPI.
     */
    bool mpi_is_gpu_aware();

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_MPI_GPU_AWARE_MPI_HH_
