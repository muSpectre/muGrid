/**
 * @file   mpi/mpi_counts.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   23 Sep 2026
 *
 * @brief  Range-checked narrowing of muGrid counts to the MPI int range
 *
 * MPI-1 through MPI-3 type every count, block length, stride and
 * displacement as `int`, while muGrid carries them as `Index_t`
 * (`ptrdiff_t`). A silent `static_cast` there is a time bomb: the sizes
 * involved grow with the cube of the resolution, so a transfer that is
 * comfortably in range at one grid size wraps to a negative count at the
 * next and corrupts data instead of failing. This header narrows with a
 * check so the failure is an exception naming the quantity.
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

#ifndef SRC_LIBMUGRID_MPI_MPI_COUNTS_HH_
#define SRC_LIBMUGRID_MPI_MPI_COUNTS_HH_

#include "core/exception.hh"
#include "core/types.hh"

#include <limits>
#include <sstream>

namespace muGrid {

    /**
     * @brief Narrow an Index_t to the int range MPI requires for counts,
     * block lengths, strides and displacements; throw instead of silently
     * truncating.
     *
     * @param value the quantity to narrow
     * @param what  what it is, for the error message (e.g. "send block
     *              length") -- the caller knows which of a dozen counts
     *              overflowed and the message should say so
     */
    inline int checked_mpi_int(Index_t value, const char * what) {
        if (value < std::numeric_limits<int>::min() ||
            value > std::numeric_limits<int>::max()) {
            std::stringstream error{};
            error << what << " (" << value
                  << ") exceeds the int range required by MPI; this transfer "
                     "is too large for a single message";
            throw RuntimeError(error.str());
        }
        return static_cast<int>(value);
    }

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_MPI_MPI_COUNTS_HH_
