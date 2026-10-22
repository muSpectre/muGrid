/**
 * @file   mpi/mpi_environment.cc
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

#include "mpi/mpi_environment.hh"
#include "core/exception.hh"

#ifdef WITH_MPI
#include <mpi.h>
#endif

#include <sstream>
#include <type_traits>

namespace muGrid {

    namespace {
        //! Set by ensure_mpi_initialized() when it was the one to call
        //! MPI_Init; finalize_mpi() clears it. Guarded by mpi_call_mutex().
        bool initialized_by_mugrid{false};
    }  // namespace

    std::mutex & mpi_call_mutex() {
        static std::mutex mutex{};
        return mutex;
    }

#ifdef WITH_MPI

    namespace {
        //! MPI_Comm as an integer: a pointer in Open MPI, an int in MPICH
        //! (a template, so that only the matching branch is compiled)
        template <typename Handle>
        std::uintptr_t handle_value(Handle handle) {
            if constexpr (std::is_pointer_v<Handle>) {
                return reinterpret_cast<std::uintptr_t>(handle);
            } else {
                return static_cast<std::uintptr_t>(handle);
            }
        }
    }  // namespace

    bool mpi_initialized() {
        int flag{0};
        MPI_Initialized(&flag);
        return flag != 0;
    }

    bool mpi_finalized() {
        int flag{0};
        MPI_Finalized(&flag);
        return flag != 0;
    }

    bool mpi_initialized_by_mugrid() { return initialized_by_mugrid; }

    void ensure_mpi_initialized() {
        std::lock_guard<std::mutex> lock{mpi_call_mutex()};
        if (mpi_finalized()) {
            throw RuntimeError(
                "MPI has already been finalised; it cannot be started again "
                "in this process");
        }
        if (mpi_initialized()) {
            return;
        }
        // SERIALIZED: muGrid's callers may call from any thread (the Python
        // binding releases the GIL around MPI calls) but never two at once,
        // which the binding guarantees with mpi_call_mutex(). MULTIPLE would
        // cost locking inside the MPI library on every call.
        int provided{MPI_THREAD_SINGLE};
        const int status{MPI_Init_thread(nullptr, nullptr,
                                         MPI_THREAD_SERIALIZED, &provided)};
        if (status != MPI_SUCCESS) {
            std::stringstream error{};
            error << "MPI_Init_thread failed with " << status;
            throw RuntimeError(error.str());
        }
        initialized_by_mugrid = true;
    }

    void finalize_mpi(bool force) {
        std::lock_guard<std::mutex> lock{mpi_call_mutex()};
        if (!(initialized_by_mugrid || force) || !mpi_initialized() ||
            mpi_finalized()) {
            return;
        }
        initialized_by_mugrid = false;
        const int status{MPI_Finalize()};
        if (status != MPI_SUCCESS) {
            std::stringstream error{};
            error << "MPI_Finalize failed with " << status;
            throw RuntimeError(error.str());
        }
    }

    std::string mpi_thread_level() {
        if (!mpi_initialized() || mpi_finalized()) {
            return "none";
        }
        int level{MPI_THREAD_SINGLE};
        MPI_Query_thread(&level);
        switch (level) {
        case MPI_THREAD_SINGLE:
            return "single";
        case MPI_THREAD_FUNNELED:
            return "funneled";
        case MPI_THREAD_SERIALIZED:
            return "serialized";
        case MPI_THREAD_MULTIPLE:
            return "multiple";
        default:
            return "unknown";
        }
    }

    std::string mpi_library_version() {
        char version[MPI_MAX_LIBRARY_VERSION_STRING];
        int length{0};
        MPI_Get_library_version(version, &length);
        return std::string(version, static_cast<std::size_t>(length));
    }

    std::uintptr_t mpi_comm_world_handle() {
        // MPI_Comm is a pointer (Open MPI) or an int (MPICH); either way it
        // fits an integer the size of a pointer, which is what mpi4py's
        // _handleof returns.
        return handle_value(MPI_COMM_WORLD);
    }

#else  // WITH_MPI

    bool mpi_initialized() { return false; }
    bool mpi_finalized() { return false; }
    bool mpi_initialized_by_mugrid() { return false; }

    void ensure_mpi_initialized() {
        throw RuntimeError("this build of muGrid has no MPI support");
    }

    void finalize_mpi(bool) {}

    std::string mpi_thread_level() { return "none"; }
    std::string mpi_library_version() { return "none"; }
    std::uintptr_t mpi_comm_world_handle() { return 0; }

#endif  // WITH_MPI

}  // namespace muGrid
