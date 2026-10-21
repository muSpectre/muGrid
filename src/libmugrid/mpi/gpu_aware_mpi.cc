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
#include "core/environment.hh"

#include <cstdlib>
#include <cstring>

#ifdef WITH_MPI
#include <mpi.h>
#if defined(MUGRID_HAVE_MPI_EXT_H)
#include <mpi-ext.h>
#endif
#endif

namespace muGrid {

    namespace {
        /**
         * Ask the MPI library whether it accepts device buffers of the GPU
         * runtime muGrid was built for. Which queries exist is determined at
         * configure time (MUGRID_HAVE_MPIX_* below), by compiling and
         * linking against the MPI in use rather than by naming MPI
         * implementations: Open MPI provides MPIX_Query_{cuda,rocm}_support
         * in mpi-ext.h, MPICH MPIX_Query_{cuda,hip}_support and
         * MPIX_GPU_query_support in mpi.h, and derivatives of either may
         * provide any subset. Any query answering yes counts.
         */
        bool mpi_reports_gpu_support() {
            bool supported{false};
#if defined(WITH_MPI) && defined(MUGRID_ENABLE_CUDA)
#if defined(MUGRID_HAVE_MPIX_QUERY_CUDA_SUPPORT)
            supported = supported || MPIX_Query_cuda_support();
#endif
#if defined(MUGRID_HAVE_MPIX_GPU_QUERY_SUPPORT)
            {
                int is_supported{0};
                if (MPIX_GPU_query_support(MPIX_GPU_SUPPORT_CUDA,
                                           &is_supported) == MPI_SUCCESS) {
                    supported = supported || is_supported;
                }
            }
#endif
#endif  // WITH_MPI && MUGRID_ENABLE_CUDA
#if defined(WITH_MPI) && defined(MUGRID_ENABLE_HIP)
#if defined(MUGRID_HAVE_MPIX_QUERY_ROCM_SUPPORT)
            supported = supported || MPIX_Query_rocm_support();
#endif
#if defined(MUGRID_HAVE_MPIX_QUERY_HIP_SUPPORT)
            supported = supported || MPIX_Query_hip_support();
#endif
#if defined(MUGRID_HAVE_MPIX_GPU_QUERY_SUPPORT)
            {
                int is_supported{0};
                if (MPIX_GPU_query_support(MPIX_GPU_SUPPORT_HIP,
                                           &is_supported) == MPI_SUCCESS) {
                    supported = supported || is_supported;
                }
            }
#endif
#endif  // WITH_MPI && MUGRID_ENABLE_HIP
            return supported;
        }

        bool detect_gpu_aware_mpi() {
            const int env{environment_flag("MUGRID_GPU_AWARE_MPI")};
            if (env >= 0) {
                return env == 1;
            }
            // An MPI without any of the queries (or one built without GPU
            // support) answers no: device buffers are then bounced through
            // host staging, which is correct with any MPI.
            return mpi_reports_gpu_support();
        }
    }  // namespace

    bool mpi_is_gpu_aware() {
#ifdef WITH_MPI
        // MPI may only be asked once initialised (MPICH aborts on a query
        // before MPI_Init). Until then only the environment variable
        // counts, and nothing is cached, so the question is put to MPI on
        // the first call after initialisation.
        int initialized{0};
        MPI_Initialized(&initialized);
        if (!initialized) {
            return environment_flag("MUGRID_GPU_AWARE_MPI") == 1;
        }
#endif
        static bool gpu_aware{detect_gpu_aware_mpi()};
        return gpu_aware;
    }

}  // namespace muGrid
