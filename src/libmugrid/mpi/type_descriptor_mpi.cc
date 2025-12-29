/**
 * @file   type_descriptor_mpi.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   29 Dec 2025
 *
 * @brief  MPI type conversion implementations for TypeDescriptor
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

#include "type_descriptor_mpi.hh"

#ifdef WITH_MPI

#include "core/exception.hh"

#include <cstddef>
#include <sstream>

namespace muGrid {

MPI_Datatype descriptor_to_mpi_type(TypeDescriptor td) {
    switch (td) {
        case TypeDescriptor::Int:
            return MPI_INT;
        case TypeDescriptor::Uint:
            return MPI_UNSIGNED;
        case TypeDescriptor::Real:
            return MPI_DOUBLE;
        case TypeDescriptor::Complex:
            return MPI_DOUBLE_COMPLEX;
        case TypeDescriptor::Index:
            // Index is std::ptrdiff_t, which is platform-dependent:
            // - LP64 (Unix 64-bit): long (8 bytes) -> MPI_LONG
            // - LLP64 (Windows 64-bit): long long (8 bytes) -> MPI_LONG_LONG_INT
            if constexpr (sizeof(std::ptrdiff_t) == sizeof(long)) {
                return MPI_LONG;
            } else {
                return MPI_LONG_LONG_INT;
            }
        default: {
            std::stringstream err{};
            err << "Cannot convert TypeDescriptor '"
                << type_descriptor_name(td) << "' to MPI_Datatype";
            throw RuntimeError(err.str());
        }
    }
}

TypeDescriptor mpi_type_to_descriptor(MPI_Datatype mpi_type) {
    if (mpi_type == MPI_INT) {
        return TypeDescriptor::Int;
    } else if (mpi_type == MPI_UNSIGNED) {
        return TypeDescriptor::Uint;
    } else if (mpi_type == MPI_DOUBLE) {
        return TypeDescriptor::Real;
    } else if (mpi_type == MPI_DOUBLE_COMPLEX) {
        return TypeDescriptor::Complex;
    } else if (mpi_type == MPI_LONG || mpi_type == MPI_LONG_LONG_INT) {
        // Both map to Index (std::ptrdiff_t) which is platform-dependent
        return TypeDescriptor::Index;
    } else {
        throw RuntimeError("Unrecognized MPI_Datatype for TypeDescriptor. "
                           "Only types corresponding to Int, Uint, Real, "
                           "Complex, and Index are supported.");
    }
}

}  // namespace muGrid

#endif  // WITH_MPI
