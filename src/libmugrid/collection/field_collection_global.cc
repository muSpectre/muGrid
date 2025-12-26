/**
 * @file   field_collection_global.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   11 Aug 2019
 *
 * @brief  Implementation of GlobalFieldCollection
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

#include "collection/field_collection_global.hh"
#include "util/math.hh"

namespace muGrid {
    /* ---------------------------------------------------------------------- */
    GlobalFieldCollection::GlobalFieldCollection(
        const Index_t &spatial_dimension, const SubPtMap_t &nb_sub_pts,
        StorageOrder storage_order, MemoryLocation memory_location)
        : Parent{
              ValidityDomain::Global, spatial_dimension, nb_sub_pts,
              storage_order, memory_location
          },
          nb_ghosts_left{DynGridIndex(spatial_dimension)},
          nb_ghosts_right{DynGridIndex(spatial_dimension)} {
    }

    /* ---------------------------------------------------------------------- */
    GlobalFieldCollection::GlobalFieldCollection(
        const DynGridIndex &nb_domain_grid_pts,
        const DynGridIndex &nb_subdomain_grid_pts_with_ghosts,
        const DynGridIndex &subdomain_locations_with_ghosts,
        const SubPtMap_t &nb_sub_pts, StorageOrder storage_order,
        const DynGridIndex &nb_ghosts_left, const DynGridIndex &nb_ghosts_right,
        MemoryLocation memory_location)
        : Parent{
              ValidityDomain::Global, nb_domain_grid_pts.get_dim(),
              nb_sub_pts, storage_order, memory_location
          } {
        this->initialise(nb_domain_grid_pts, nb_subdomain_grid_pts_with_ghosts,
                         subdomain_locations_with_ghosts, storage_order,
                         nb_ghosts_left, nb_ghosts_right);
    }

    /* ---------------------------------------------------------------------- */
    GlobalFieldCollection::GlobalFieldCollection(
        const DynGridIndex &nb_domain_grid_pts,
        const DynGridIndex &nb_subdomain_grid_pts_with_ghosts,
        const DynGridIndex &subdomain_locations_with_ghosts,
        const DynGridIndex &pixels_strides, const SubPtMap_t &nb_sub_pts,
        StorageOrder storage_order, const DynGridIndex &nb_ghosts_left,
        const DynGridIndex &nb_ghosts_right, MemoryLocation memory_location)
        : Parent{
              ValidityDomain::Global, nb_domain_grid_pts.get_dim(),
              nb_sub_pts, storage_order, memory_location
          } {
        this->initialise(nb_domain_grid_pts, nb_subdomain_grid_pts_with_ghosts,
                         subdomain_locations_with_ghosts, pixels_strides,
                         nb_ghosts_left, nb_ghosts_right);
    }

    /* ---------------------------------------------------------------------- */
    GlobalFieldCollection::GlobalFieldCollection(
        const DynGridIndex &nb_domain_grid_pts,
        const DynGridIndex &nb_subdomain_grid_pts_with_ghosts,
        const DynGridIndex &subdomain_locations_with_ghosts,
        StorageOrder pixels_storage_order, const SubPtMap_t &nb_sub_pts,
        StorageOrder storage_order, const DynGridIndex &nb_ghosts_left,
        const DynGridIndex &nb_ghosts_right, MemoryLocation memory_location)
        : Parent{
            ValidityDomain::Global, nb_domain_grid_pts.get_dim(),
            nb_sub_pts, storage_order, memory_location
        } {
        this->initialise(nb_domain_grid_pts, nb_subdomain_grid_pts_with_ghosts,
                         subdomain_locations_with_ghosts, pixels_storage_order,
                         nb_ghosts_left, nb_ghosts_right);
    }

    /* ---------------------------------------------------------------------- */
    void GlobalFieldCollection::initialise(
        const DynGridIndex &nb_domain_grid_pts,
        const DynGridIndex &nb_subdomain_grid_pts_with_ghosts,
        const DynGridIndex &subdomain_locations_with_ghosts,
        const DynGridIndex &pixels_strides,
        const DynGridIndex &nb_ghosts_left,
        const DynGridIndex &nb_ghosts_right) {
        if (this->initialised) {
            throw FieldCollectionError("double initialisation");
        }

        // sanity check 1 - domain grid points
        auto nb_domain_grid_pts_total{get_nb_from_shape(nb_domain_grid_pts)};
        if (nb_domain_grid_pts_total <= 0) {
            std::stringstream s;
            s << "Invalid nb_domain_grid_pts " << nb_domain_grid_pts << " ("
                    << nb_domain_grid_pts_total
                    << " total grid points) passed during "
                    << "initialisation.";
            throw FieldCollectionError(s.str());
        }
        // sanity check 2 - the subdomain and / or ghosts
        auto _nb_ghosts_left{nb_ghosts_left.get_dim() == 0
                                   ? DynGridIndex(nb_domain_grid_pts.get_dim())
                                   : nb_ghosts_left};
        auto total_nb_ghosts_left{
            get_nb_from_shape(_nb_ghosts_left)
        };
        if (total_nb_ghosts_left < 0) {
            std::stringstream s;
            s << "Invalid nb_ghosts_left " << _nb_ghosts_left
                    << " (" << total_nb_ghosts_left
                    << " total ghosts on left) passed during "
                    << "initialisation.";
            throw FieldCollectionError(s.str());
        }

        auto _nb_ghosts_right{nb_ghosts_right.get_dim() == 0
                                    ? DynGridIndex(nb_domain_grid_pts.get_dim())
                                    : nb_ghosts_right};
        auto total_nb_ghosts_right{
            get_nb_from_shape(_nb_ghosts_right)
        };
        if (total_nb_ghosts_right < 0) {
            std::stringstream s;
            s << "Invalid nb_ghosts_right " << _nb_ghosts_right
                    << " (" << total_nb_ghosts_right
                    << " total ghosts on right) passed during "
                    << "initialisation.";
            throw FieldCollectionError(s.str());
        }

        auto _nb_subdomain_grid_pts{
            nb_subdomain_grid_pts_with_ghosts.get_dim() == 0
                ? nb_domain_grid_pts + _nb_ghosts_left + _nb_ghosts_right
                : nb_subdomain_grid_pts_with_ghosts
        };
        auto nb_subdomain_grid_pts_total{
            get_nb_from_shape(_nb_subdomain_grid_pts)
        };
        if (nb_subdomain_grid_pts_total < 0) {
            std::stringstream s;
            s << "Invalid nb_subdomain_grid_pts " << _nb_subdomain_grid_pts
                    << " (" << nb_subdomain_grid_pts_total
                    << " total grid points) passed during "
                    << "initialisation.";
            throw FieldCollectionError(s.str());
        }

        this->nb_domain_grid_pts = nb_domain_grid_pts;

        // Set ghost buffer sizes
        this->nb_ghosts_left = _nb_ghosts_left;
        this->nb_ghosts_right = _nb_ghosts_right;

        // Set subdomain pixel iterators
        auto _subdomain_locations_with_ghosts{
            subdomain_locations_with_ghosts.get_dim() == 0
                ? DynGridIndex(nb_domain_grid_pts.get_dim()) - _nb_ghosts_left
                : subdomain_locations_with_ghosts};
        this->pixels_with_ghosts =
            CcoordOps::Pixels(_nb_subdomain_grid_pts,
                              _subdomain_locations_with_ghosts, pixels_strides);
        this->pixels_without_ghosts = CcoordOps::Pixels(
            _nb_subdomain_grid_pts - _nb_ghosts_left - _nb_ghosts_right,
            _subdomain_locations_with_ghosts + _nb_ghosts_left,
            pixels_strides);

        this->nb_pixels = CcoordOps::get_size(_nb_subdomain_grid_pts);
        this->nb_buffer_pixels =
            CcoordOps::get_buffer_size(_nb_subdomain_grid_pts, pixels_strides);
        this->allocate_fields();
        this->pixel_indices.resize(this->nb_pixels);
        for (int i{0}; i < this->nb_pixels; ++i) {
            this->pixel_indices[i] = i;
        }
        // needs to be here, or initialise_maps will fail (by design)
        this->initialised = true;
        this->initialise_maps();
    }

    /* ---------------------------------------------------------------------- */
    void
    GlobalFieldCollection::initialise(const DynGridIndex &nb_domain_grid_pts,
                                      const DynGridIndex &nb_subdomain_grid_pts_with_ghosts,
                                      const DynGridIndex &subdomain_locations_with_ghosts,
                                      StorageOrder pixels_storage_order,
                                      const DynGridIndex &nb_ghosts_left,
                                      const DynGridIndex &nb_ghosts_right) {
        if (pixels_storage_order == StorageOrder::Automatic) {
            pixels_storage_order = this->get_storage_order();
        }
        // The domain and / or ghosts may be empty
        auto _nb_ghosts_left{nb_ghosts_left.get_dim() == 0
                                   ? DynGridIndex(nb_domain_grid_pts.get_dim())
                                   : nb_ghosts_left};
        auto _nb_ghosts_right{nb_ghosts_right.get_dim() == 0
                                    ? DynGridIndex(nb_domain_grid_pts.get_dim())
                                    : nb_ghosts_right};
        auto _nb_subdomain_grid_pts{
            nb_subdomain_grid_pts_with_ghosts.get_dim() == 0
                ? nb_domain_grid_pts + _nb_ghosts_left + _nb_ghosts_right
                : nb_subdomain_grid_pts_with_ghosts
        };
        // Compute pixel strides
        auto pixel_strides{
            pixels_storage_order == StorageOrder::ColMajor
                ? CcoordOps::get_col_major_strides(_nb_subdomain_grid_pts)
                : CcoordOps::get_row_major_strides(_nb_subdomain_grid_pts)};
        // Calls the other initialise with pixel strides
        this->initialise(nb_domain_grid_pts, _nb_subdomain_grid_pts,
                         subdomain_locations_with_ghosts, pixel_strides,
                         nb_ghosts_left, nb_ghosts_right);
    }

    /* ---------------------------------------------------------------------- */
    const typename GlobalFieldCollection::Pixels &
    GlobalFieldCollection::get_pixels_with_ghosts() const {
        if (not(this->initialised)) {
            throw FieldCollectionError(
                "Can't iterate over the collection before it is initialised.");
        }
        return this->pixels_with_ghosts;
    }

    /* ---------------------------------------------------------------------- */
    const typename GlobalFieldCollection::Pixels &
    GlobalFieldCollection::get_pixels_without_ghosts() const {
        if (not(this->initialised)) {
            throw FieldCollectionError(
                "Can't iterate over the collection before it is initialised.");
        }
        return this->pixels_without_ghosts;
    }

    /* ---------------------------------------------------------------------- */
    GlobalFieldCollection GlobalFieldCollection::get_empty_clone() const {
        GlobalFieldCollection ret_val{
            this->get_spatial_dim(),
            this->nb_sub_pts,
            this->get_storage_order(),
            this->get_memory_location()
        };
        ret_val.initialise(this->nb_domain_grid_pts,
                           this->pixels_with_ghosts.get_nb_subdomain_grid_pts(),
                           this->pixels_with_ghosts.get_subdomain_locations());
        return ret_val;
    }

    /* ---------------------------------------------------------------------- */
    Shape_t GlobalFieldCollection::get_pixels_shape() const {
        return static_cast<Shape_t>(this->pixels_with_ghosts.get_nb_subdomain_grid_pts());
    }

    /* ---------------------------------------------------------------------- */
    Shape_t GlobalFieldCollection::get_pixels_shape_without_ghosts() const {
        return static_cast<Shape_t>(
            this->pixels_without_ghosts.get_nb_subdomain_grid_pts());
    }

    /* ---------------------------------------------------------------------- */
    Index_t GlobalFieldCollection::get_nb_pixels_without_ghosts() const {
        auto shape_without_ghosts{this->get_pixels_shape_without_ghosts()};
        return get_nb_from_shape(shape_without_ghosts);
    }

    /* ---------------------------------------------------------------------- */
    Shape_t GlobalFieldCollection::get_pixels_offset_without_ghosts() const {
        return Shape_t{this->nb_ghosts_left};
    }

    /* ---------------------------------------------------------------------- */
    Shape_t
    GlobalFieldCollection::get_pixels_strides(Index_t element_size) const {
        Shape_t strides{this->pixels_with_ghosts.get_strides()};
        for (auto &&s: strides) {
            s *= element_size;
        }
        return strides;
    }

} // namespace muGrid
