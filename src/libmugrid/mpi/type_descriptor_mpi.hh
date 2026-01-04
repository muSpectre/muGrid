/**
 * @file   type_descriptor_mpi.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   29 Dec 2024
 *
 * @brief  MPI type conversions for TypeDescriptor
 *
 * This header provides conversion functions between TypeDescriptor and
 * MPI_Datatype. It is only available when MPI support is enabled (WITH_MPI).
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

#ifndef SRC_LIBMUGRID_MPI_TYPE_DESCRIPTOR_MPI_HH_
#define SRC_LIBMUGRID_MPI_TYPE_DESCRIPTOR_MPI_HH_

#include "core/type_descriptor.hh"

#ifdef WITH_MPI
#include <mpi.h>

namespace muGrid {

/**
 * @brief Convert TypeDescriptor to MPI_Datatype.
 *
 * @param td The type descriptor
 * @return The corresponding MPI_Datatype
 * @throws RuntimeError if the TypeDescriptor is Unknown or unsupported
 */
MPI_Datatype descriptor_to_mpi_type(TypeDescriptor td);

/**
 * @brief Convert MPI_Datatype to TypeDescriptor.
 *
 * @param mpi_type The MPI datatype
 * @return The corresponding TypeDescriptor
 * @throws RuntimeError if the MPI_Datatype is not recognized
 */
TypeDescriptor mpi_type_to_descriptor(MPI_Datatype mpi_type);

}  // namespace muGrid

#endif  // WITH_MPI

#endif  // SRC_LIBMUGRID_MPI_TYPE_DESCRIPTOR_MPI_HH_
