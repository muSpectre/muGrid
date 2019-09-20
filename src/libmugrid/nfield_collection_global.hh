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

  /** `muGrid::GlobalNFieldCollection` derives from `muGrid::NFieldCollection`
   * and stores global fields that live throughout the whole computational
   * domain, i.e. are defined for every pixel/voxel.
   */
  class GlobalNFieldCollection : public NFieldCollection {
   public:
    //! alias of base class
    using Parent = NFieldCollection;
    //! pixel iterator
    using DynamicPixels = CcoordOps::DynamicPixels;

    //! Default constructor
    GlobalNFieldCollection() = delete;

    /**
     * Constructor
     * @param spatial_dimension number of spatial dimensions, must be 1, 2, 3,
     * or Unknown
     * @param nb_quad_pts number of quadrature points per pixel/voxel
     */
    GlobalNFieldCollection(Dim_t spatial_dimension, Dim_t nb_quad_pts);

    //! Copy constructor
    GlobalNFieldCollection(const GlobalNFieldCollection & other) = delete;

    //! Move constructor
    GlobalNFieldCollection(GlobalNFieldCollection && other) = default;

    //! Destructor
    virtual ~GlobalNFieldCollection() = default;

    //! Copy assignment operator
    GlobalNFieldCollection &
    operator=(const GlobalNFieldCollection & other) = delete;

    //! Move assignment operator
    GlobalNFieldCollection &
    operator=(GlobalNFieldCollection && other) = delete;

    //! Return the pixels class that allows to iterator over pixels
    const DynamicPixels & get_pixels() const;

    //! Return index for a ccoord
    template <size_t Dim>
    Dim_t get_index(const Ccoord_t<Dim> & ccoord) const {
      return this->get_pixels().get_index(ccoord);
    }

    //! return coordinates of the i-th pixel
    DynCcoord_t get_ccoord(const Dim_t & index) const {
      return CcoordOps::get_ccoord_from_strides(
          this->pixels.get_nb_grid_pts(), this->pixels.get_locations(),
          this->pixels.get_strides(), index);
    }

    /**
     * freeze the problem size and allocate memory for all fields of the
     * collection. NFields added later on will have their memory allocated
     * upon construction.
     */
    void initialise(const DynCcoord_t & nb_grid_pts,
                    const DynCcoord_t & locations = {});

    /**
     * freeze the problem size and allocate memory for all fields of the
     * collection. NFields added later on will have their memory allocated
     * upon construction.
     */
    template <size_t Dim>
    void initialise(const Ccoord_t<Dim> & nb_grid_pts,
                    const Ccoord_t<Dim> & locations = {}) {
      this->initialise(DynCcoord_t{nb_grid_pts}, DynCcoord_t{locations});
    }

    /**
     * freeze the problem size and allocate memory for all fields of the
     * collection. NFields added later on will have their memory allocated
     * upon construction.
     */
    void initialise(const DynCcoord_t & nb_grid_pts,
                    const DynCcoord_t & locations, const DynCcoord_t & strides);

    /**
     * freeze the problem size and allocate memory for all fields of the
     * collection. NFields added later on will have their memory allocated
     * upon construction.
     */
    template <size_t Dim>
    void initialise(const Ccoord_t<Dim> & nb_grid_pts,
                    const Ccoord_t<Dim> & locations,
                    const Ccoord_t<Dim> & strides) {
      this->initialise(DynCcoord_t{nb_grid_pts}, DynCcoord_t{locations},
                       DynCcoord_t{strides});
    }

    /**
     * obtain a new field collection with the same domain and pixels
     */
    GlobalNFieldCollection get_empty_clone() const;

   protected:
    DynamicPixels pixels{};  //!< helper to iterate over the grid
  };

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_NFIELD_COLLECTION_GLOBAL_HH_
