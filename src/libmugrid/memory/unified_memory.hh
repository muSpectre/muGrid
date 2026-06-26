/**
 * @file   memory/unified_memory.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   25 Jun 2026
 *
 * @brief  Runtime detection of physically unified host/device memory
 *
 * Some accelerators share a single physical memory pool with the host
 * (integrated GPUs and APUs such as the AMD MI300A). On these machines a
 * plain device allocation is already addressable by the host with no copy
 * and no migration, so the host-staging buffers used for NetCDF I/O and
 * (non-GPU-aware) MPI can be skipped entirely. This header provides the
 * runtime decision.
 *
 * IMPORTANT: this detects *physically* unified memory only. It deliberately
 * does NOT report true for NVIDIA managed/UVM memory (which papers over two
 * separate physical pools with implicit page migration) nor for
 * coherent-but-separate memories such as Grace Hopper (two physical pools,
 * two capacity budgets, cache-coherent over a link). Those are not "no copy"
 * and would also misreport available memory.
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

#ifndef SRC_LIBMUGRID_MEMORY_UNIFIED_MEMORY_HH_
#define SRC_LIBMUGRID_MEMORY_UNIFIED_MEMORY_HH_

namespace muGrid {

    /**
     * True if device allocations on @p device_id reside in the *same physical
     * memory pool* as the host, so that an ordinary device pointer is directly
     * dereferenceable by the host, MPI and NetCDF without an explicit copy
     * (e.g. an AMD MI300A APU or an integrated GPU). The result is determined
     * once per device and cached:
     *
     * 1. The environment variable ``MUGRID_UNIFIED_MEMORY`` (``1``/``0``)
     *    overrides any detection — use it for platforms where auto-detection
     *    is unreliable (e.g. APUs that require a specific unified/XNACK mode)
     *    or to force the conservative host-staging path.
     * 2. Otherwise the GPU runtime is queried for the ``integrated`` device
     *    attribute (``cudaDevAttrIntegrated`` / ``hipDeviceAttributeIntegrated``),
     *    which is true exactly when the device shares physical memory with the
     *    host.
     * 3. The conservative default is false: stage through host memory, which
     *    is correct on every architecture.
     *
     * @param device_id GPU device to query, or -1 (default) for the device
     *        currently selected in the GPU runtime.
     */
    bool device_has_unified_memory(int device_id = -1);

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_MEMORY_UNIFIED_MEMORY_HH_
