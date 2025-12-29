/**
 * @file   type_descriptor_mpi.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   29 Dec 2024
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

#include <sstream>

namespace muGrid {

MPI_Datatype descriptor_to_mpi_type(TypeDescriptor td) {
    switch (td) {
        case TypeDescriptor::Char:
            return MPI_CHAR;
        case TypeDescriptor::SignedChar:
            return MPI_SIGNED_CHAR;
        case TypeDescriptor::UnsignedChar:
            return MPI_UNSIGNED_CHAR;
        case TypeDescriptor::Short:
            return MPI_SHORT;
        case TypeDescriptor::UnsignedShort:
            return MPI_UNSIGNED_SHORT;
        case TypeDescriptor::Int:
            return MPI_INT;
        case TypeDescriptor::UnsignedInt:
            return MPI_UNSIGNED;
        case TypeDescriptor::Long:
            return MPI_LONG;
        case TypeDescriptor::UnsignedLong:
            return MPI_UNSIGNED_LONG;
        case TypeDescriptor::LongLong:
            return MPI_LONG_LONG_INT;
        case TypeDescriptor::UnsignedLongLong:
            return MPI_UNSIGNED_LONG_LONG;
        case TypeDescriptor::Float:
            return MPI_FLOAT;
        case TypeDescriptor::Double:
            return MPI_DOUBLE;
        case TypeDescriptor::Complex:
            return MPI_DOUBLE_COMPLEX;
        default: {
            std::stringstream err{};
            err << "Cannot convert TypeDescriptor '"
                << type_descriptor_name(td) << "' to MPI_Datatype";
            throw RuntimeError(err.str());
        }
    }
}

TypeDescriptor mpi_type_to_descriptor(MPI_Datatype mpi_type) {
    if (mpi_type == MPI_CHAR) {
        return TypeDescriptor::Char;
    } else if (mpi_type == MPI_SIGNED_CHAR) {
        return TypeDescriptor::SignedChar;
    } else if (mpi_type == MPI_UNSIGNED_CHAR) {
        return TypeDescriptor::UnsignedChar;
    } else if (mpi_type == MPI_SHORT) {
        return TypeDescriptor::Short;
    } else if (mpi_type == MPI_UNSIGNED_SHORT) {
        return TypeDescriptor::UnsignedShort;
    } else if (mpi_type == MPI_INT) {
        return TypeDescriptor::Int;
    } else if (mpi_type == MPI_UNSIGNED) {
        return TypeDescriptor::UnsignedInt;
    } else if (mpi_type == MPI_LONG) {
        return TypeDescriptor::Long;
    } else if (mpi_type == MPI_UNSIGNED_LONG) {
        return TypeDescriptor::UnsignedLong;
    } else if (mpi_type == MPI_LONG_LONG_INT) {
        return TypeDescriptor::LongLong;
    } else if (mpi_type == MPI_UNSIGNED_LONG_LONG) {
        return TypeDescriptor::UnsignedLongLong;
    } else if (mpi_type == MPI_FLOAT) {
        return TypeDescriptor::Float;
    } else if (mpi_type == MPI_DOUBLE) {
        return TypeDescriptor::Double;
    } else if (mpi_type == MPI_DOUBLE_COMPLEX) {
        return TypeDescriptor::Complex;
    } else {
        throw RuntimeError("Unrecognized MPI_Datatype for TypeDescriptor");
    }
}

}  // namespace muGrid

#endif  // WITH_MPI
