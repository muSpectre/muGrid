/**
 * @file   type_descriptor_netcdf.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   29 Dec 2024
 *
 * @brief  NetCDF type conversions for TypeDescriptor
 *
 * This header provides conversion functions between TypeDescriptor and
 * nc_type. It is only available when NetCDF support is enabled (WITH_NETCDF_IO).
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

#ifndef SRC_LIBMUGRID_IO_TYPE_DESCRIPTOR_NETCDF_HH_
#define SRC_LIBMUGRID_IO_TYPE_DESCRIPTOR_NETCDF_HH_

#include "core/type_descriptor.hh"

#ifdef WITH_NETCDF_IO

#ifdef WITH_MPI
#include <pnetcdf.h>
#else
#include <netcdf.h>
#endif

namespace muGrid {

/**
 * @brief Convert TypeDescriptor to nc_type.
 *
 * This function maps TypeDescriptor values to NetCDF type constants,
 * taking into account platform-dependent type sizes (LP64, ILP64, LLP64).
 *
 * @param td The type descriptor
 * @return The corresponding nc_type
 * @throws FileIOError if the TypeDescriptor is Unknown or unsupported
 */
nc_type descriptor_to_nc_type(TypeDescriptor td);

/**
 * @brief Convert nc_type to TypeDescriptor.
 *
 * @param nc The NetCDF type constant
 * @return The corresponding TypeDescriptor
 * @throws FileIOError if the nc_type is not recognized
 */
TypeDescriptor nc_type_to_descriptor(nc_type nc);

}  // namespace muGrid

#endif  // WITH_NETCDF_IO

#endif  // SRC_LIBMUGRID_IO_TYPE_DESCRIPTOR_NETCDF_HH_
