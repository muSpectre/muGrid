/**
 * @file   nfield_collection_global.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   11 Aug 2019
 *
 * @brief  Global field collections
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
 * General Public License for more details.
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
 */

#ifndef SRC_LIBMUGRID_NFIELD_COLLECTION_GLOBAL_HH_
#define SRC_LIBMUGRID_NFIELD_COLLECTION_GLOBAL_HH_

#include "nfield_collection.hh"
#include "ccoord_operations.hh"

namespace muGrid {

  template <Dim_t DimS>
  class GlobalNFieldCollection : public NFieldCollection {
   public:
    using Parent = NFieldCollection;
    //! cell coordinates
    using Ccoord = Ccoord_t<DimS>;
    //! pixel iterator
    using Pixels = CcoordOps::Pixels<DimS>;

    //! Default constructor
    GlobalNFieldCollection() = delete;

    /**
     * Constructor
     * @param nb_quad_pts number of quadrature points per pixel/voxel
     */
    explicit GlobalNFieldCollection(Dim_t nb_quad_pts);

    //! Copy constructor
    GlobalNFieldCollection(const GlobalNFieldCollection & other) = delete;

    //! Move constructor
    GlobalNFieldCollection(GlobalNFieldCollection && other) = delete;

    //! Destructor
    virtual ~GlobalNFieldCollection() = default;

    //! Copy assignment operator
    GlobalNFieldCollection &
    operator=(const GlobalNFieldCollection & other) = delete;

    //! Move assignment operator
    GlobalNFieldCollection & operator=(
        GlobalNFieldCollection && other) = delete;

    //! Return the pixels class that allows to iterator over pixels
    const Pixels & get_pixels() const;

    //! Return index for a ccoord
    Dim_t get_index(const Ccoord & ccoord) const {
      return this->get_pixels().get_index(ccoord);
    }

    /**
     * freeze the problem size and allocate memory for all fields of the
     * collection. NFields added later on will have their memory allocated
     * upon construction.
     */
    void initialise(Ccoord nb_grid_pts, Ccoord locations = {});

    /**
     * freeze the problem size and allocate memory for all fields of the
     * collection. NFields added later on will have their memory allocated
     * upon construction.
     */
    void initialise(Ccoord nb_grid_pts, Ccoord locations, Ccoord strides);

   private:
    //! number of discretisation cells in each of the DimS spatial directions
    Ccoord nb_grid_pts{};
    //! subdomain locations (i.e. coordinates of hind bottom left corner of this
    //! subdomain)
    Ccoord locations{};
    Pixels pixels{};  //!< helper to iterate over the grid
  };

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_NFIELD_COLLECTION_GLOBAL_HH_
