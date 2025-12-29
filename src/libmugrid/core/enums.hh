/**
 * @file   core/enums.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   24 Jan 2019
 *
 * @brief  Enumeration definitions for muGrid
 *
 * Copyright © 2019 Till Junge
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

#ifndef SRC_LIBMUGRID_CORE_ENUMS_HH_
#define SRC_LIBMUGRID_CORE_ENUMS_HH_

#include <ostream>
#include <string>

namespace muGrid {

    /**
     * @enum IterUnit
     * @brief An enumeration class for iteration units.
     *
     * This enumeration class is used in two contexts within the µGrid codebase.
     * Firstly, it is used in `Field`s to specify the relative storage of data
     * with respect to pixels, quadrature points, or nodal points.
     * Secondly, it is used in `FieldMap`s to specify the unit of iteration,
     * whether it be over pixels, quadrature points, or nodal points.
     *
     * @var Pixel Represents degrees of freedom (dofs) relative to a
     * pixel/voxel, with no subdivision.
     * @var SubPt Represents dofs relative to sub-points (e.g. quadrature
     * points).
     */
    enum class IterUnit {
        Pixel,  //!< dofs relative to a pixel/voxel, no subdivision
        SubPt   //!< dofs relative to sub-points (e.g. quadrature points)
    };

    /**
     * @enum StorageOrder
     * @brief An enumeration class for storage orders of field components.
     *
     * This enumeration class defines three types of storage orders:
     * ArrayOfStructures, StructureOfArrays and Automatic.
     * These storage orders can be used to determine the order in which field
     * components are stored in memory.
     *
     * @var ArrayOfStructures Represents a column-major storage order. In this
     * order, the first index changes fastest, and the last index changes
     * slowest.
     * @var StructureOfArrays Represents a structure of arrays storage order. In
     * this order, pixels are consecutive in memory.
     * @var Automatic Represents an automatic storage order. In this order, the
     * storage order is inherited from `FieldCollection`.
     */
    enum class StorageOrder {
        ArrayOfStructures,  //!< components are consecutive in memory
        StructureOfArrays,  //< pixels are consecutive in memory
        Automatic  //!< inherit storage order from `FieldCollection`
    };

    /**
     * @enum Mapping
     * @brief An enumeration class for mapping types.
     *
     * This enumeration class defines two types of mappings: Const and Mut.
     * These mappings can be used to determine the type of access (constant or
     * mutable) to the mapped field through their iterators or access operators.
     *
     * @var Const Represents a constant mapping. It is used when the mapped
     * field should not be modified.
     * @var Mut Represents a mutable mapping. It is used when the mapped field
     * can be modified.
     */
    enum class Mapping { Const, Mut };

    //! inserts `muGrid::IterUnit` into `std::ostream`s
    std::ostream & operator<<(std::ostream & os, const IterUnit & sub_division);

    //! inserts `muGrid::StorageOrder` into `std::ostream`s
    std::ostream & operator<<(std::ostream & os,
                              const StorageOrder & storage_order);

    /**
     * this tag is always defined to one in every field collection
     */
    const std::string PixelTag{"pixel"};

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_CORE_ENUMS_HH_
