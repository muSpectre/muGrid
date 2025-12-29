/**
 * @file   type_descriptor_netcdf.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   29 Dec 2024
 *
 * @brief  NetCDF type conversion implementations for TypeDescriptor
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

#include "type_descriptor_netcdf.hh"

#ifdef WITH_NETCDF_IO

#include "io/file_io_base.hh"

#include <sstream>

namespace muGrid {

nc_type descriptor_to_nc_type(TypeDescriptor td) {
    switch (td) {
        case TypeDescriptor::Char:
            return NC_CHAR;
        case TypeDescriptor::SignedChar:
            return NC_BYTE;
        case TypeDescriptor::UnsignedChar:
            return NC_UBYTE;
        case TypeDescriptor::Short:
            return NC_SHORT;
        case TypeDescriptor::UnsignedShort:
            return NC_USHORT;
        case TypeDescriptor::Int:
            // int size is platform-dependent, map based on size
            if constexpr (sizeof(int) == 4) {
                return NC_INT;
            } else if constexpr (sizeof(int) == 8) {
                return NC_INT64;
            } else {
                return NC_INT;  // fallback
            }
        case TypeDescriptor::UnsignedInt:
            if constexpr (sizeof(unsigned int) == 4) {
                return NC_UINT;
            } else if constexpr (sizeof(unsigned int) == 8) {
                return NC_UINT64;
            } else {
                return NC_UINT;  // fallback
            }
        case TypeDescriptor::Long:
            // long size is platform-dependent (4 on Windows, 8 on Unix)
            if constexpr (sizeof(long) == 4) {
                return NC_INT;
            } else if constexpr (sizeof(long) == 8) {
                return NC_INT64;
            } else {
                return NC_INT64;  // fallback
            }
        case TypeDescriptor::UnsignedLong:
            if constexpr (sizeof(unsigned long) == 4) {
                return NC_UINT;
            } else if constexpr (sizeof(unsigned long) == 8) {
                return NC_UINT64;
            } else {
                return NC_UINT64;  // fallback
            }
        case TypeDescriptor::LongLong:
            return NC_INT64;
        case TypeDescriptor::UnsignedLongLong:
            return NC_UINT64;
        case TypeDescriptor::Float:
            return NC_FLOAT;
        case TypeDescriptor::Double:
            return NC_DOUBLE;
        case TypeDescriptor::Complex:
            // NetCDF doesn't have a native complex type
            // This should be handled at a higher level by storing as
            // two doubles or a compound type
            {
                std::stringstream err{};
                err << "NetCDF does not support complex types directly. "
                    << "Use two separate real fields for real and imaginary "
                       "parts.";
                throw FileIOError(err.str());
            }
        default: {
            std::stringstream err{};
            err << "Cannot convert TypeDescriptor '"
                << type_descriptor_name(td) << "' to nc_type";
            throw FileIOError(err.str());
        }
    }
}

TypeDescriptor nc_type_to_descriptor(nc_type nc) {
    switch (nc) {
        case NC_CHAR:
            return TypeDescriptor::Char;
        case NC_BYTE:
            return TypeDescriptor::SignedChar;
        case NC_UBYTE:
            return TypeDescriptor::UnsignedChar;
        case NC_SHORT:
            return TypeDescriptor::Short;
        case NC_USHORT:
            return TypeDescriptor::UnsignedShort;
        case NC_INT:
            // Map to the type that matches int's size on this platform
            if constexpr (sizeof(int) == 4) {
                return TypeDescriptor::Int;
            } else {
                return TypeDescriptor::Long;  // unlikely but handle it
            }
        case NC_UINT:
            if constexpr (sizeof(unsigned int) == 4) {
                return TypeDescriptor::UnsignedInt;
            } else {
                return TypeDescriptor::UnsignedLong;
            }
        case NC_INT64:
            // Map to long long (always 64-bit)
            return TypeDescriptor::LongLong;
        case NC_UINT64:
            return TypeDescriptor::UnsignedLongLong;
        case NC_FLOAT:
            return TypeDescriptor::Float;
        case NC_DOUBLE:
            return TypeDescriptor::Double;
        default: {
            std::stringstream err{};
            err << "Unrecognized nc_type value: " << nc;
            throw FileIOError(err.str());
        }
    }
}

}  // namespace muGrid

#endif  // WITH_NETCDF_IO
