/**
 * @file   type_descriptor_netcdf.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   29 Dec 2025
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

// muGrid type <-> nc_type mapping. Generated for both directions from one
// table so they can never drift. Complex is intentionally absent: NetCDF has
// no native complex type, so it is handled by the explicit throw below rather
// than mapped. To support a new scalar type in NetCDF, add a row here (and a
// row to MUGRID_SCALAR_TYPES in core/type_descriptor.hh). Types that NetCDF
// cannot represent are simply left out and rejected at runtime.
#define MUGRID_NC_TYPES(X) \
    X(Int, NC_INT)         \
    X(Uint, NC_UINT)       \
    X(Real, NC_DOUBLE)     \
    X(Real32, NC_FLOAT)    \
    X(Index, NC_INT64)

nc_type descriptor_to_nc_type(TypeDescriptor td) {
    switch (td) {
#define MUGRID_NC_FWD_ROW(tag, nc) \
    case TypeDescriptor::tag:      \
        return nc;
        MUGRID_NC_TYPES(MUGRID_NC_FWD_ROW)
#undef MUGRID_NC_FWD_ROW
        case TypeDescriptor::Complex: {
            // NetCDF doesn't have a native complex type; this is handled at a
            // higher level by storing real and imaginary parts separately.
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
#define MUGRID_NC_REV_ROW(tag, nc_const) \
    case nc_const:                       \
        return TypeDescriptor::tag;
        MUGRID_NC_TYPES(MUGRID_NC_REV_ROW)
#undef MUGRID_NC_REV_ROW
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
