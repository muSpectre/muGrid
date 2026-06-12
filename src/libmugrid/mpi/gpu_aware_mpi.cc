/**
 * @file   mpi/gpu_aware_mpi.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   12 Jun 2026
 *
 * @brief  Runtime detection of GPU-aware MPI
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

#include "mpi/gpu_aware_mpi.hh"

#include <cstdlib>
#include <cstring>

#ifdef WITH_MPI
#include <mpi.h>
#if defined(OPEN_MPI)
#include <mpi-ext.h>
#endif
#endif

namespace muGrid {

    namespace {
        bool detect_gpu_aware_mpi() {
            const char * env{std::getenv("MUGRID_GPU_AWARE_MPI")};
            if (env != nullptr && std::strlen(env) > 0) {
                return env[0] == '1' || env[0] == 'y' || env[0] == 'Y' ||
                       env[0] == 't' || env[0] == 'T';
            }
#if defined(WITH_MPI) && defined(MPIX_CUDA_AWARE_SUPPORT)
            if (MPIX_Query_cuda_support()) {
                return true;
            }
#endif
#if defined(WITH_MPI) && defined(MPIX_ROCM_AWARE_SUPPORT)
            if (MPIX_Query_rocm_support()) {
                return true;
            }
#endif
            // Unknown MPI stack (or one built without GPU support):
            // bounce device buffers through host staging, which is correct
            // with any MPI.
            return false;
        }
    }  // namespace

    bool mpi_is_gpu_aware() {
        static bool gpu_aware{detect_gpu_aware_mpi()};
        return gpu_aware;
    }

}  // namespace muGrid
