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

// muGrid type <-> MPI_Datatype mapping, generated for both directions from one
// table. Index is handled separately because its MPI type is platform-dependent
// (LP64: MPI_LONG, LLP64: MPI_LONG_LONG_INT) and the reverse mapping must accept
// both. To support a new scalar type over MPI, add a row here (and a row to
// MUGRID_SCALAR_TYPES in core/type_descriptor.hh).
// The complex rows use the C datatypes (MPI_C_*): the Fortran
// MPI_DOUBLE_COMPLEX/MPI_COMPLEX are only guaranteed to exist when the MPI
// library was built with Fortran support.
#define MUGRID_MPI_TYPES(X)          \
    X(Int, MPI_INT)                  \
    X(Uint, MPI_UNSIGNED)            \
    X(Real, MPI_DOUBLE)              \
    X(Complex, MPI_C_DOUBLE_COMPLEX) \
    X(Real32, MPI_FLOAT)             \
    X(Complex32, MPI_C_FLOAT_COMPLEX)

MPI_Datatype descriptor_to_mpi_type(TypeDescriptor td) {
    switch (td) {
#define MUGRID_MPI_FWD_ROW(tag, mpi) \
    case TypeDescriptor::tag:        \
        return mpi;
        MUGRID_MPI_TYPES(MUGRID_MPI_FWD_ROW)
#undef MUGRID_MPI_FWD_ROW
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
#define MUGRID_MPI_REV_ROW(tag, mpi)        \
    if (mpi_type == mpi) {                  \
        return TypeDescriptor::tag;         \
    }
    MUGRID_MPI_TYPES(MUGRID_MPI_REV_ROW)
#undef MUGRID_MPI_REV_ROW
    if (mpi_type == MPI_LONG || mpi_type == MPI_LONG_LONG_INT) {
        // Both map to Index (std::ptrdiff_t) which is platform-dependent
        return TypeDescriptor::Index;
    }
    throw RuntimeError("Unrecognized MPI_Datatype for TypeDescriptor. "
                       "Only types corresponding to Int, Uint, Real, "
                       "Complex, and Index are supported.");
}

}  // namespace muGrid

#endif  // WITH_MPI
