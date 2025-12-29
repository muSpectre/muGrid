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
        case TypeDescriptor::Int:
            // int is always 4 bytes on common platforms
            return NC_INT;
        case TypeDescriptor::Uint:
            // unsigned int is always 4 bytes on common platforms
            return NC_UINT;
        case TypeDescriptor::Real:
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
        case TypeDescriptor::Index:
            // Index is std::ptrdiff_t which is 64-bit on 64-bit platforms
            return NC_INT64;
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
        case NC_INT:
            return TypeDescriptor::Int;
        case NC_UINT:
            return TypeDescriptor::Uint;
        case NC_DOUBLE:
            return TypeDescriptor::Real;
        case NC_INT64:
            return TypeDescriptor::Index;
        default: {
            std::stringstream err{};
            err << "Unrecognized nc_type value: " << nc
                << ". Only NC_INT, NC_UINT, NC_DOUBLE, and NC_INT64 "
                << "are supported.";
            throw FileIOError(err.str());
        }
    }
}

}  // namespace muGrid

#endif  // WITH_NETCDF_IO
