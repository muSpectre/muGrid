/**
 * @file   field_collection_global.hh
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

#ifndef SRC_LIBMUGRID_FIELD_COLLECTION_GLOBAL_HH_
#define SRC_LIBMUGRID_FIELD_COLLECTION_GLOBAL_HH_

#include "field_collection.hh"
#include "ccoord_operations.hh"

namespace muGrid {

  /** `muGrid::GlobalFieldCollection` derives from `muGrid::FieldCollection`
   * and stores global fields that live throughout the whole computational
   * domain, i.e. are defined for every pixel/voxel.
   */
  class GlobalFieldCollection : public FieldCollection {
   public:
    //! alias of base class
    using Parent = FieldCollection;
    using Parent::SubPtMap_t;
    //! pixel iterator
    using DynamicPixels = CcoordOps::DynamicPixels;

    //! Default constructor
    GlobalFieldCollection() = delete;

    /**
     * Constructor
     * @param spatial_dimension number of spatial dimensions, must be 1, 2, 3,
     * or Unknown
     * @param nb_quad_pts number of quadrature points per pixel/voxel
     */
    GlobalFieldCollection(const Index_t & spatial_dimension,
                          const SubPtMap_t & nb_sub_pts = {},
                          StorageOrder storage_order =
                              StorageOrder::ArrayOfStructures);

    /**
     * Constructor with initialization
     * @param spatial_dimension number of spatial dimensions, must be 1, 2, 3,
     * or Unknown
     * @param nb_quad_pts number of quadrature points per pixel/voxel
     * @param nb_subdomain_grid_pts number of grid points on the current MPI
     * process (subdomain)
     * @param subdomain_locations location of the current subdomain within the
     * global grid
     */
    GlobalFieldCollection(const Index_t & spatial_dimension,
                          const DynCcoord_t & nb_subdomain_grid_pts,
                          const DynCcoord_t & subdomain_locations = {},
                          const SubPtMap_t & nb_sub_pts = {},
                          StorageOrder storage_order =
                              StorageOrder::ArrayOfStructures);

    /**
     * Constructor with initialisation
     * @param spatial_dimension number of spatial dimensions, must be 1, 2, 3,
     * or Unknown
     * @param nb_subdomain_grid_pts number of grid points on the current MPI
     * process (subdomain)
     * @param subdomain_locations location of the current subdomain within the
     * global grid
     * @param pixels_strides strides specifying memory layout of the pixels
     * @param storage_order Storage order of the pixels vs subdivision portion
     *                    of the field. In a column-major storage order, the
     *                    pixel subdivision (i.e. the components of the field)
     *                    are stored next to each other in memory, file in a
     *                    row-major storage order for each component the
     *                    pixels are stored next to each other in memory.
     *                    (This is also sometimes called the array of structures
     *                    vs. structure of arrays storage order.)
     *                    Important: The pixels or subpoints have their own
     *                    storage order that is not affected by this setting.
     */
    GlobalFieldCollection(Index_t spatial_dimension,
                          const DynCcoord_t & nb_subdomain_grid_pts,
                          const DynCcoord_t & subdomain_locations,
                          const DynCcoord_t & pixels_strides,
                          const SubPtMap_t & nb_sub_pts = {},
                          StorageOrder storage_order =
                              StorageOrder::ArrayOfStructures);

    /**
     * Constructor with initialisation
     * @param spatial_dimension number of spatial dimensions, must be 1, 2, 3,
     * or Unknown
     * @param nb_subdomain_grid_pts number of grid points on the current MPI
     * process (subdomain)
     * @param subdomain_locations location of the current subdomain within the
     * global grid
     * @param pixels_storage_order Storage order of the pixels
     * @param storage_order Storage order of the pixels vs subdivision portion
     *                    of the field. In a column-major storage order, the
     *                    pixel subdivision (i.e. the components of the field)
     *                    are stored next to each other in memory, file in a
     *                    row-major storage order for each component the
     *                    pixels are stored next to each other in memory.
     *                    (This is also sometimes called the array of structures
     *                    vs. structure of arrays storage order.)
     *                    Important: The pixels or subpoints have their own
     *                    storage order that is not affected by this setting.
     */
    GlobalFieldCollection(Index_t spatial_dimension,
                          const DynCcoord_t & nb_subdomain_grid_pts,
                          const DynCcoord_t & subdomain_locations,
                          StorageOrder pixels_storage_order,
                          const SubPtMap_t & nb_sub_pts = {},
                          StorageOrder storage_order =
                              StorageOrder::ArrayOfStructures);

    //! Copy constructor
    GlobalFieldCollection(const GlobalFieldCollection & other) = delete;

    //! Move constructor
    GlobalFieldCollection(GlobalFieldCollection && other) = default;

    //! Destructor
    virtual ~GlobalFieldCollection() = default;

    //! Copy assignment operator
    GlobalFieldCollection &
    operator=(const GlobalFieldCollection & other) = delete;

    //! Move assignment operator
    GlobalFieldCollection & operator=(GlobalFieldCollection && other) = delete;

    //! Return the pixels class that allows to iterator over pixels
    const DynamicPixels & get_pixels() const;

    //! evaluate and return the linear index corresponding to dynamic `ccoord`
    Index_t get_index(const DynCcoord_t & ccoord) const {
      return this->get_pixels().get_index(ccoord);
    }

    //! evaluate and return the linear index corresponding to `ccoord`
    template <size_t Dim>
    Index_t get_index(const Ccoord_t<Dim> & ccoord) const {
      return this->pixels.get_index(ccoord);
    }

    //! return coordinates of the i-th pixel
    DynCcoord_t get_ccoord(const Index_t & index) const {
      return this->pixels.get_ccoord(index);
    }

    /**
     * freeze the problem size and allocate memory for all fields of the
     * collection. Fields added later on will have their memory allocated
     * upon construction.
     */
    void initialise(const DynCcoord_t & nb_subdomain_grid_pts,
                    const DynCcoord_t & subdomain_locations,
                    const DynCcoord_t & pixels_strides);

    /**
     * freeze the problem size and allocate memory for all fields of the
     * collection. Fields added later on will have their memory allocated
     * upon construction.
     */
    template <size_t Dim>
    void initialise(const Ccoord_t<Dim> & nb_subdomain_grid_pts,
                    const Ccoord_t<Dim> & subdomain_locations,
                    const Ccoord_t<Dim> & pixels_strides) {
      this->initialise(DynCcoord_t{nb_subdomain_grid_pts},
                       DynCcoord_t{subdomain_locations},
                       DynCcoord_t{pixels_strides});
    }

    /**
     * freeze the problem size and allocate memory for all fields of the
     * collection. Fields added later on will have their memory allocated
     * upon construction.
     */
    void initialise(const DynCcoord_t & nb_subdomain_grid_pts,
                    const DynCcoord_t & subdomain_locations = {},
                    StorageOrder pixels_storage_order =
                        StorageOrder::Automatic);

    /**
     * freeze the problem size and allocate memory for all fields of the
     * collection. Fields added later on will have their memory allocated
     * upon construction.
     */
    template <size_t Dim>
    void initialise(const Ccoord_t<Dim> & nb_subdomain_grid_pts,
                    const Ccoord_t<Dim> & subdomain_locations = {},
                    StorageOrder pixels_storage_order =
                        StorageOrder::Automatic) {
      this->initialise(DynCcoord_t{nb_subdomain_grid_pts},
                       DynCcoord_t{subdomain_locations},
                       pixels_storage_order);
    }

    /**
     * obtain a new field collection with the same domain and pixels
     */
    GlobalFieldCollection get_empty_clone() const;

    //! return shape of the pixels
    virtual Shape_t get_pixels_shape() const;

    //! return strides of the pixels
    virtual Shape_t get_pixels_strides(Index_t element_size = 1) const;

   protected:
    DynamicPixels pixels{};  //!< helper to iterate over the grid
  };

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_FIELD_COLLECTION_GLOBAL_HH_
