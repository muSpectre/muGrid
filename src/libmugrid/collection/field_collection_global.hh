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

#include "collection/field_collection.hh"
#include "grid/pixels.hh"

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
        using Pixels = CcoordOps::Pixels;

        //! Default constructor
        GlobalFieldCollection() = delete;

        /**
         * Constructor
         * @param spatial_dimension number of spatial dimensions, must be 1, 2,
         * 3, or Unknown
         * @param nb_sub_pts number of quadrature points per pixel/voxel
         * @param storage_order storage order for field components
         * @param device where to allocate field memory
         */
        GlobalFieldCollection(
            Dim_t spatial_dimension,
            const SubPtMap_t & nb_sub_pts = {},
            StorageOrder storage_order = StorageOrder::ArrayOfStructures,
            Device device = Device::cpu());

        /**
         * Constructor with initialization
         * @param nb_subdomain_grid_pts_with_ghosts number of grid points on the
         * current MPI process (subdomain)
         * @param subdomain_locations_with_ghosts location of the current
         * subdomain within the global grid
         * @param nb_sub_pts number of quadrature points per pixel/voxel
         * @param device where to allocate field memory
         */
        GlobalFieldCollection(
            const DynGridIndex & nb_domain_grid_pts,
            const DynGridIndex & nb_subdomain_grid_pts_with_ghosts = {},
            const DynGridIndex & subdomain_locations_with_ghosts = {},
            const SubPtMap_t & nb_sub_pts = {},
            StorageOrder storage_order = StorageOrder::ArrayOfStructures,
            const DynGridIndex & nb_ghosts_left = {},
            const DynGridIndex & nb_ghosts_right = {},
            Device device = Device::cpu());

        /**
         * Constructor with initialisation
         * @param nb_subdomain_grid_pts_with_ghosts number of grid points on the
         * current MPI process (subdomain)
         * @param subdomain_locations_with_ghosts location of the current
         * subdomain within the global grid
         * @param pixels_strides strides specifying memory layout of the pixels
         * @param storage_order Storage order of the pixels vs subdivision
         * portion of the field. In a column-major storage order, the pixel
         * subdivision (i.e. the components of the field) are stored next to
         * each other in memory, while in a row-major storage order, for each
         * component the pixels are stored next to each other in memory. (This
         * is also sometimes called the array of structures vs. structure of
         * arrays storage order.) Important: The pixels or subpoints have their
         * own storage order that is not affected by this setting.
         * @param device where to allocate field memory
         */
        GlobalFieldCollection(
            const DynGridIndex & nb_domain_grid_pts,
            const DynGridIndex & nb_subdomain_grid_pts_with_ghosts,
            const DynGridIndex & subdomain_locations_with_ghosts,
            const DynGridIndex & pixels_strides,
            const SubPtMap_t & nb_sub_pts = {},
            StorageOrder storage_order = StorageOrder::ArrayOfStructures,
            const DynGridIndex & nb_ghosts_left = {},
            const DynGridIndex & nb_ghosts_right = {},
            Device device = Device::cpu());

        /**
         * Constructor with initialisation
         * @param nb_subdomain_grid_pts_with_ghosts number of grid points on the
         * current MPI process (subdomain)
         * @param subdomain_locations_with_ghosts location of the current
         * subdomain within the global grid
         * @param pixels_storage_order Storage order of the pixels
         * @param storage_order Storage order of the pixels vs subdivision
         * portion of the field. In a column-major storage order, the pixel
         * subdivision (i.e. the components of the field) are stored next to
         * each other in memory, while in a row-major storage order for each
         * component the pixels are stored next to each other in memory. (This
         * is also sometimes called the array of structures vs. structure of
         * arrays storage order.) Important: The pixels or subpoints have their
         * own storage order that is not affected by this setting.
         * @param device where to allocate field memory
         */
        GlobalFieldCollection(
            const DynGridIndex & nb_domain_grid_pts,
            const DynGridIndex & nb_subdomain_grid_pts_with_ghosts,
            const DynGridIndex & subdomain_locations_with_ghosts,
            StorageOrder pixels_storage_order,
            const SubPtMap_t & nb_sub_pts = {},
            StorageOrder storage_order = StorageOrder::ArrayOfStructures,
            const DynGridIndex & nb_ghosts_left = {},
            const DynGridIndex & nb_ghosts_right = {},
            Device device = Device::cpu());

        //! Copy constructor
        GlobalFieldCollection(const GlobalFieldCollection & other) = delete;

        //! Move constructor
        GlobalFieldCollection(GlobalFieldCollection && other) = default;

        //! Destructor
        ~GlobalFieldCollection() override = default;

        //! Copy assignment operator
        GlobalFieldCollection &
        operator=(const GlobalFieldCollection & other) = delete;

        //! Move assignment operator
        GlobalFieldCollection &
        operator=(GlobalFieldCollection && other) = delete;

        //! Return the pixels class that allows to iterator over pixels
        const Pixels & get_pixels_with_ghosts() const;

        //! Return the pixels class that allows to iterator over pixels
        const Pixels & get_pixels_without_ghosts() const;

        //! evaluate and return the linear index corresponding to dynamic
        //! `ccoord`
        Index_t get_index(const DynGridIndex & ccoord) const {
            return this->get_pixels_with_ghosts().get_index(ccoord);
        }

        //! evaluate and return the linear index corresponding to `ccoord`
        template <size_t Dim>
        Index_t get_index(const GridIndex<Dim> & ccoord) const {
            return this->pixels_with_ghosts.get_index(ccoord);
        }

        //! return coordinates of the i-th pixel
        DynGridIndex get_coord(const Index_t & index) const {
            return this->pixels_with_ghosts.get_coord(index);
        }

        /**
         * freeze the problem size and allocate memory for all fields of the
         * collection. Fields added later on will have their memory allocated
         * upon construction.
         */
        void initialise(const DynGridIndex & nb_domain_grid_pts,
                        const DynGridIndex & nb_subdomain_grid_pts_with_ghosts,
                        const DynGridIndex & subdomain_locations_with_ghosts,
                        const DynGridIndex & pixels_strides,
                        const DynGridIndex & nb_ghosts_left = {},
                        const DynGridIndex & nb_ghosts_right = {});

        /**
         * freeze the problem size and allocate memory for all fields of the
         * collection. Fields added later on will have their memory allocated
         * upon construction.
         */
        template <size_t Dim>
        void initialise(const GridIndex<Dim> & nb_domain_grid_pts,
                        const GridIndex<Dim> & nb_subdomain_grid_pts,
                        const GridIndex<Dim> & subdomain_locations,
                        const GridIndex<Dim> & pixels_strides,
                        const GridIndex<Dim> & nb_ghosts_left = {},
                        const GridIndex<Dim> & nb_ghosts_right = {}) {
            this->initialise(
                DynGridIndex{nb_domain_grid_pts},
                DynGridIndex{nb_subdomain_grid_pts},
                DynGridIndex{subdomain_locations}, DynGridIndex{pixels_strides},
                DynGridIndex{nb_ghosts_left}, DynGridIndex{nb_ghosts_right});
        }

        /**
         * freeze the problem size and allocate memory for all fields of the
         * collection. Fields added later on will have their memory allocated
         * upon construction.
         */
        void
        initialise(const DynGridIndex & nb_domain_grid_pts,
                   const DynGridIndex & nb_subdomain_grid_pts_with_ghosts = {},
                   const DynGridIndex & subdomain_locations_with_ghosts = {},
                   StorageOrder pixels_storage_order = StorageOrder::Automatic,
                   const DynGridIndex & nb_ghosts_left = {},
                   const DynGridIndex & nb_ghosts_right = {});

        /**
         * freeze the problem size and allocate memory for all fields of the
         * collection. Fields added later on will have their memory allocated
         * upon construction.
         */
        template <size_t Dim>
        void
        initialise(const GridIndex<Dim> & nb_domain_grid_pts,
                   const GridIndex<Dim> & nb_subdomain_grid_pts = {},
                   const GridIndex<Dim> & subdomain_locations = {},
                   StorageOrder pixels_storage_order = StorageOrder::Automatic,
                   const GridIndex<Dim> & nb_ghosts_left = {},
                   const GridIndex<Dim> & nb_ghosts_right = {}) {
            this->initialise(DynGridIndex{nb_domain_grid_pts},
                             DynGridIndex{nb_subdomain_grid_pts},
                             DynGridIndex{subdomain_locations},
                             pixels_storage_order, DynGridIndex{nb_ghosts_left},
                             DynGridIndex{nb_ghosts_right});
        }

        /**
         * obtain a new field collection with the same domain and pixels
         */
        GlobalFieldCollection get_empty_clone() const;

        //! return shape of the pixels
        Shape_t get_pixels_shape() const override;

        //! return shape of the pixels without ghosts
        Shape_t get_pixels_shape_without_ghosts() const override;

        //! return the number of pixels without ghosts
        Index_t get_nb_pixels_without_ghosts() const override;

        //! return the offset of the pixels in the storage without ghosts
        Shape_t get_pixels_offset_without_ghosts() const override;

        //! return strides of the pixels
        Shape_t get_pixels_strides(Index_t element_size = 1) const override;

        //! returns the global (domain) number of grid points in each direction
        const DynGridIndex & get_nb_domain_grid_pts() const {
            return this->nb_domain_grid_pts;
        }

        //! returns the process-local (subdomain) number of grid points in each
        //! direction including the ghost cells
        const DynGridIndex & get_nb_subdomain_grid_pts_with_ghosts() const {
            return this->get_pixels_with_ghosts().get_nb_subdomain_grid_pts();
        }

        //! returns the process-local (subdomain) number of grid points in each
        //! directionl, but without the ghost cells
        DynGridIndex get_nb_subdomain_grid_pts_without_ghosts() const {
            return this->get_pixels_with_ghosts().get_nb_subdomain_grid_pts() -
                   this->nb_ghosts_left - this->nb_ghosts_right;
        }

        //! returns the process-local (subdomain) locations of subdomain grid
        //! including the ghost cells
        const DynGridIndex & get_subdomain_locations_with_ghosts() const {
            return this->get_pixels_with_ghosts().get_subdomain_locations();
        }

        //! returns the process-local (subdomain) locations of subdomain grid,
        //! but without the ghost cells
        DynGridIndex get_subdomain_locations_without_ghosts() const {
            return this->get_pixels_without_ghosts().get_subdomain_locations();
        }

        /**
         * @brief Returns the number of ghost cells on the left side of the
         * subdomain.
         *
         * @return const reference to a `DynGridIndex` object containing the
         * number of left ghost cells.
         */
        const DynGridIndex & get_nb_ghosts_left() const {
            return this->nb_ghosts_left;
        }

        /**
         * @brief Returns the number of ghost cells on the right side of the
         * subdomain.
         *
         * @return const reference to a `DynGridIndex` object containing the
         * number of right ghost cells.
         */
        const DynGridIndex & get_nb_ghosts_right() const {
            return this->nb_ghosts_right;
        }

        /**
        * @brief Returns the index difference between iterators of pixels with
        * and without ghosts
         *
         * @return an Index_t instance.
         */
        Index_t get_pixels_index_diff() const {
            return this->pixels_with_ghosts.get_index(
                this->pixels_without_ghosts.get_coord(0));
        }

       protected:
        Pixels pixels_with_ghosts{};  //!< helper to iterate over the grid
        Pixels pixels_without_ghosts{};  //!< helper to iterate over the grid
        DynGridIndex
            nb_domain_grid_pts{};  // number of domain (global) grid points
        DynGridIndex nb_ghosts_left{};
        DynGridIndex nb_ghosts_right{};
    };
}  // namespace muGrid

#endif  // SRC_LIBMUGRID_FIELD_COLLECTION_GLOBAL_HH_
