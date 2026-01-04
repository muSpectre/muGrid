/**
 * @file   type_descriptor.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   29 Dec 2024
 *
 * @brief  Unified type descriptor for cross-library type mapping
 *
 * This header provides a TypeDescriptor enum that serves as a unified type
 * identifier for muGrid. It can be converted to MPI_Datatype (when MPI is
 * available) or nc_type (when NetCDF is available), enabling type-safe
 * communication without void* casts.
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

#ifndef SRC_LIBMUGRID_CORE_TYPE_DESCRIPTOR_HH_
#define SRC_LIBMUGRID_CORE_TYPE_DESCRIPTOR_HH_

#include <complex>
#include <cstddef>
#include <cstdint>
#include <typeinfo>
#include <type_traits>

namespace muGrid {

// Forward declaration of Complex type (defined in types.hh)
using Complex = std::complex<double>;

/**
 * @enum TypeDescriptor
 * @brief Unified type identifier that works across C++, MPI, and NetCDF.
 *
 * This enum provides a single, unified way to describe the muGrid scalar
 * types that can be converted to MPI_Datatype (when WITH_MPI is defined) or
 * nc_type (when NetCDF is available). It eliminates the need for void*
 * type passing in communication routines.
 *
 * The supported types correspond to the muGrid scalar types:
 * - Int: signed integer (int)
 * - Uint: unsigned integer (unsigned int)
 * - Real: double precision floating point (double)
 * - Complex: complex double (std::complex<double>)
 * - Index: signed index type (std::ptrdiff_t)
 */
enum class TypeDescriptor : std::uint8_t {
    Unknown = 0,
    Int,      // int (muGrid::Int)
    Uint,     // unsigned int (muGrid::Uint)
    Real,     // double (muGrid::Real)
    Complex,  // std::complex<double> (muGrid::Complex)
    Index     // std::ptrdiff_t (muGrid::Index_t)
};

/**
 * @brief Get the size in bytes for a TypeDescriptor.
 *
 * @param td The type descriptor
 * @return Size in bytes, or 0 for Unknown
 */
constexpr std::size_t type_descriptor_size(TypeDescriptor td) {
    switch (td) {
        case TypeDescriptor::Int:
            return sizeof(int);
        case TypeDescriptor::Uint:
            return sizeof(unsigned int);
        case TypeDescriptor::Real:
            return sizeof(double);
        case TypeDescriptor::Complex:
            return sizeof(std::complex<double>);
        case TypeDescriptor::Index:
            return sizeof(std::ptrdiff_t);
        default:
            return 0;
    }
}

/**
 * @brief Check if a TypeDescriptor represents a signed type.
 *
 * @param td The type descriptor
 * @return true if signed, false otherwise
 */
constexpr bool is_signed(TypeDescriptor td) {
    switch (td) {
        case TypeDescriptor::Int:
        case TypeDescriptor::Real:
        case TypeDescriptor::Complex:
        case TypeDescriptor::Index:
            return true;
        default:
            return false;
    }
}

/**
 * @brief Check if a TypeDescriptor represents an integer type.
 *
 * @param td The type descriptor
 * @return true if integer, false otherwise
 */
constexpr bool is_integer(TypeDescriptor td) {
    switch (td) {
        case TypeDescriptor::Int:
        case TypeDescriptor::Uint:
        case TypeDescriptor::Index:
            return true;
        default:
            return false;
    }
}

/**
 * @brief Check if a TypeDescriptor represents a floating point type.
 *
 * @param td The type descriptor
 * @return true if floating point, false otherwise
 */
constexpr bool is_floating_point(TypeDescriptor td) {
    return td == TypeDescriptor::Real;
}

/**
 * @brief Check if a TypeDescriptor represents a complex type.
 *
 * @param td The type descriptor
 * @return true if complex, false otherwise
 */
constexpr bool is_complex(TypeDescriptor td) {
    return td == TypeDescriptor::Complex;
}

/**
 * @brief Get the TypeDescriptor for a C++ type (compile-time).
 *
 * Only muGrid scalar types are supported: Int (int), Uint (unsigned int),
 * Real (double), Complex (std::complex<double>), Index_t (std::ptrdiff_t).
 *
 * @tparam T The C++ type
 * @return The corresponding TypeDescriptor
 */
template <typename T>
constexpr TypeDescriptor type_to_descriptor() {
    if constexpr (std::is_same_v<T, int>) {
        return TypeDescriptor::Int;
    } else if constexpr (std::is_same_v<T, unsigned int>) {
        return TypeDescriptor::Uint;
    } else if constexpr (std::is_same_v<T, double>) {
        return TypeDescriptor::Real;
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        return TypeDescriptor::Complex;
    } else if constexpr (std::is_same_v<T, std::ptrdiff_t>) {
        return TypeDescriptor::Index;
    } else {
        static_assert(sizeof(T) == 0, "Unsupported type for TypeDescriptor. "
                      "Only Int, Uint, Real, Complex, and Index_t are supported.");
        return TypeDescriptor::Unknown;
    }
}

/**
 * @brief Get the TypeDescriptor from std::type_info (runtime).
 *
 * This function provides runtime type mapping when the type is not
 * known at compile time (e.g., through polymorphic Field pointers).
 *
 * @param type_id The type_info from typeid()
 * @return The corresponding TypeDescriptor
 * @throws RuntimeError if the type is not recognized
 */
TypeDescriptor typeid_to_descriptor(const std::type_info & type_id);

/**
 * @brief Get human-readable name for a TypeDescriptor.
 *
 * @param td The type descriptor
 * @return A C string with the type name
 */
const char * type_descriptor_name(TypeDescriptor td);

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_CORE_TYPE_DESCRIPTOR_HH_
