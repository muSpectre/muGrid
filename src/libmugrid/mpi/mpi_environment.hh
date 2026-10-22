/**
 * @file   mpi/mpi_environment.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   21 Oct 2026
 *
 * @brief  Starting and stopping MPI, and what the MPI library is
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
 *
 */

#ifndef SRC_LIBMUGRID_MPI_MPI_ENVIRONMENT_HH_
#define SRC_LIBMUGRID_MPI_MPI_ENVIRONMENT_HH_

#include <cstdint>
#include <mutex>
#include <string>

namespace muGrid {

    /**
     * muGrid never starts MPI on its own account: a serial run of an MPI
     * build must not pay for MPI_Init, which loads the MPI library's
     * transport and accelerator plugins (and on some systems disturbs the
     * GPU driver). MPI is started by the first Communicator::world() or
     * split(), with MPI_THREAD_SERIALIZED, and stopped by finalize_mpi(),
     * which does something only if muGrid was the one to start it. All of
     * this is a no-op when another library (mpi4py, say) started MPI first.
     *
     * Every function here is declared in builds without MPI as well and
     * then answers as for an MPI that is never initialised.
     */

    //! Whether MPI_Init has been called, by anyone
    bool mpi_initialized();

    //! Whether MPI_Finalize has been called, by anyone
    bool mpi_finalized();

    //! Whether MPI was started by ensure_mpi_initialized()
    bool mpi_initialized_by_mugrid();

    /**
     * Start MPI if nobody has yet (MPI_Init_thread, MPI_THREAD_SERIALIZED)
     * and remember that muGrid did. Throws if MPI has already been
     * finalised, or in a build without MPI.
     */
    void ensure_mpi_initialized();

    /**
     * Stop MPI if muGrid started it and nobody has stopped it yet (with
     * `force`, also if somebody else started it). Owned communicators free
     * their handles only while MPI is running, so this may be called with
     * communicators still alive.
     */
    void finalize_mpi(bool force = false);

    //! The thread level MPI granted, as MPI_Query_thread names it:
    //! "single", "funneled", "serialized" or "multiple"; "none" before
    //! MPI is initialised or in a build without MPI
    std::string mpi_thread_level();

    //! MPI_Get_library_version(), or "none" in a build without MPI
    std::string mpi_library_version();

    /**
     * The value of MPI_COMM_WORLD as this build's MPI library represents
     * it (a pointer in Open MPI, an integer in MPICH), for checking that a
     * handle from another binding of MPI -- mpi4py's -- refers to the same
     * library: two different MPI libraries never agree on it. 0 in a build
     * without MPI.
     */
    std::uintptr_t mpi_comm_world_handle();

    /**
     * Lock to hold around MPI calls made while other threads may make
     * them: muGrid asks for MPI_THREAD_SERIALIZED, under which two threads
     * must not be inside MPI at once. The C++ library does not lock
     * itself (its callers are single-threaded); the Python binding takes
     * it wherever it releases the GIL around an MPI call.
     */
    std::mutex & mpi_call_mutex();

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_MPI_MPI_ENVIRONMENT_HH_
