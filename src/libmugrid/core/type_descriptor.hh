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
/**
 * @brief The single source of truth for muGrid's scalar types.
 *
 * Every per-type table in the code base (the enum below, the constexpr traits,
 * the runtime name lookup, and the NetCDF/MPI backend mappings in their own
 * translation units) is generated from this X-macro. To add a scalar type, add
 * ONE row here, then ONE row to each backend table that can represent it
 * (MUGRID_NC_TYPES in io/type_descriptor_netcdf.cc, MUGRID_MPI_TYPES in
 * mpi/type_descriptor_mpi.cc). Columns: enum tag, C++ type, human-readable name.
 */
#define MUGRID_SCALAR_TYPES(X)                              \
    X(Int, int, "Int")                                      \
    X(Uint, unsigned int, "Uint")                           \
    X(Real, double, "Real")                                 \
    X(Complex, std::complex<double>, "Complex")             \
    X(Real32, float, "Real32")                              \
    X(Complex32, std::complex<float>, "Complex32")          \
    X(Index, std::ptrdiff_t, "Index")

enum class TypeDescriptor : std::uint8_t {
    Unknown = 0,
#define MUGRID_TD_ENUM_ROW(tag, type, name) tag,
    MUGRID_SCALAR_TYPES(MUGRID_TD_ENUM_ROW)
#undef MUGRID_TD_ENUM_ROW
};

namespace detail {
    //! Trait identifying std::complex specialisations.
    template <typename T>
    struct is_complex_type : std::false_type {};
    template <typename U>
    struct is_complex_type<std::complex<U>> : std::true_type {};

    //! Per-type classification used to generate the constexpr predicates
    //! below. Complex is treated as signed (matching the historical
    //! behaviour); std::is_signed_v is false for std::complex.
    template <typename T>
    inline constexpr bool td_is_signed =
        std::is_signed<T>::value || is_complex_type<T>::value;
    template <typename T>
    inline constexpr bool td_is_integer = std::is_integral<T>::value;
    template <typename T>
    inline constexpr bool td_is_floating_point =
        std::is_floating_point<T>::value;
    template <typename T>
    inline constexpr bool td_is_complex = is_complex_type<T>::value;
}  // namespace detail

/**
 * @brief Get the size in bytes for a TypeDescriptor.
 *
 * @param td The type descriptor
 * @return Size in bytes, or 0 for Unknown
 */
constexpr std::size_t type_descriptor_size(TypeDescriptor td) {
    switch (td) {
#define MUGRID_TD_SIZE_ROW(tag, type, name) \
    case TypeDescriptor::tag:               \
        return sizeof(type);
        MUGRID_SCALAR_TYPES(MUGRID_TD_SIZE_ROW)
#undef MUGRID_TD_SIZE_ROW
        case TypeDescriptor::Unknown:
            return 0;
    }
    return 0;
}

/**
 * @brief Check if a TypeDescriptor represents a signed type.
 *
 * @param td The type descriptor
 * @return true if signed, false otherwise
 */
constexpr bool is_signed(TypeDescriptor td) {
    switch (td) {
#define MUGRID_TD_SIGNED_ROW(tag, type, name) \
    case TypeDescriptor::tag:                 \
        return detail::td_is_signed<type>;
        MUGRID_SCALAR_TYPES(MUGRID_TD_SIGNED_ROW)
#undef MUGRID_TD_SIGNED_ROW
        case TypeDescriptor::Unknown:
            return false;
    }
    return false;
}

/**
 * @brief Check if a TypeDescriptor represents an integer type.
 *
 * @param td The type descriptor
 * @return true if integer, false otherwise
 */
constexpr bool is_integer(TypeDescriptor td) {
    switch (td) {
#define MUGRID_TD_INTEGER_ROW(tag, type, name) \
    case TypeDescriptor::tag:                  \
        return detail::td_is_integer<type>;
        MUGRID_SCALAR_TYPES(MUGRID_TD_INTEGER_ROW)
#undef MUGRID_TD_INTEGER_ROW
        case TypeDescriptor::Unknown:
            return false;
    }
    return false;
}

/**
 * @brief Check if a TypeDescriptor represents a floating point type.
 *
 * @param td The type descriptor
 * @return true if floating point, false otherwise
 */
constexpr bool is_floating_point(TypeDescriptor td) {
    switch (td) {
#define MUGRID_TD_FLOAT_ROW(tag, type, name) \
    case TypeDescriptor::tag:                \
        return detail::td_is_floating_point<type>;
        MUGRID_SCALAR_TYPES(MUGRID_TD_FLOAT_ROW)
#undef MUGRID_TD_FLOAT_ROW
        case TypeDescriptor::Unknown:
            return false;
    }
    return false;
}

/**
 * @brief Check if a TypeDescriptor represents a complex type.
 *
 * @param td The type descriptor
 * @return true if complex, false otherwise
 */
constexpr bool is_complex(TypeDescriptor td) {
    switch (td) {
#define MUGRID_TD_COMPLEX_ROW(tag, type, name) \
    case TypeDescriptor::tag:                  \
        return detail::td_is_complex<type>;
        MUGRID_SCALAR_TYPES(MUGRID_TD_COMPLEX_ROW)
#undef MUGRID_TD_COMPLEX_ROW
        case TypeDescriptor::Unknown:
            return false;
    }
    return false;
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
#define MUGRID_TD_T2D_ROW(tag, type, name)    \
    if constexpr (std::is_same_v<T, type>) {  \
        return TypeDescriptor::tag;           \
    } else
    MUGRID_SCALAR_TYPES(MUGRID_TD_T2D_ROW)
#undef MUGRID_TD_T2D_ROW
    {
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
