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

#include "field_collection_global.hh"

namespace muGrid {

  /* ---------------------------------------------------------------------- */
  GlobalFieldCollection::GlobalFieldCollection(
      const Index_t & spatial_dimension,
      const SubPtMap_t & nb_sub_pts,
      StorageOrder storage_order)
      : Parent{ValidityDomain::Global, spatial_dimension, nb_sub_pts,
               storage_order} {}

  /* ---------------------------------------------------------------------- */
  GlobalFieldCollection::GlobalFieldCollection(
      const Index_t & spatial_dimension,
      const DynCcoord_t & nb_subdomain_grid_pts,
      const DynCcoord_t & subdomain_locations,
      const SubPtMap_t & nb_sub_pts,
      StorageOrder storage_order)
      : Parent{ValidityDomain::Global, spatial_dimension, nb_sub_pts,
               storage_order} {
    this->initialise(nb_subdomain_grid_pts, subdomain_locations);
  }

  /* ---------------------------------------------------------------------- */
  GlobalFieldCollection::GlobalFieldCollection(
      Index_t spatial_dimension,
      const DynCcoord_t & nb_subdomain_grid_pts,
      const DynCcoord_t & subdomain_locations,
      const DynCcoord_t & pixels_strides,
      const SubPtMap_t & nb_sub_pts,
      StorageOrder storage_order)
      : Parent{ValidityDomain::Global, spatial_dimension, nb_sub_pts,
               storage_order} {
    this->initialise(nb_subdomain_grid_pts, subdomain_locations,
                     pixels_strides);
  }

  /* ---------------------------------------------------------------------- */
  GlobalFieldCollection::GlobalFieldCollection(
      Index_t spatial_dimension,
      const DynCcoord_t & nb_subdomain_grid_pts,
      const DynCcoord_t & subdomain_locations,
      StorageOrder pixels_storage_order,
      const SubPtMap_t & nb_sub_pts,
      StorageOrder storage_order)
      : Parent{ValidityDomain::Global, spatial_dimension, nb_sub_pts,
               storage_order} {
    this->initialise(nb_subdomain_grid_pts, subdomain_locations,
                     pixels_storage_order);
  }

  /* ---------------------------------------------------------------------- */
  void
  GlobalFieldCollection::initialise(const DynCcoord_t & nb_subdomain_grid_pts,
                                    const DynCcoord_t & subdomain_locations,
                                    const DynCcoord_t & pixels_strides) {
    if (this->initialised) {
      throw FieldCollectionError("double initialisation");
    }

    this->pixels = CcoordOps::DynamicPixels(nb_subdomain_grid_pts,
                                            subdomain_locations,
                                            pixels_strides);
    this->nb_pixels = CcoordOps::get_size(nb_subdomain_grid_pts);
    this->nb_buffer_pixels = CcoordOps::get_buffer_size(nb_subdomain_grid_pts,
                                                        pixels_strides);
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
  GlobalFieldCollection::initialise(const DynCcoord_t & nb_subdomain_grid_pts,
                                    const DynCcoord_t & subdomain_locations,
                                    StorageOrder pixels_storage_order) {
    if (pixels_storage_order == StorageOrder::Automatic) {
      pixels_storage_order = this->get_storage_order();
    }
    this->initialise(
        nb_subdomain_grid_pts, subdomain_locations,
        pixels_storage_order == StorageOrder::ColMajor
             ? CcoordOps::get_col_major_strides(nb_subdomain_grid_pts)
             : CcoordOps::get_row_major_strides(nb_subdomain_grid_pts));
  }

  /* ---------------------------------------------------------------------- */
  const typename GlobalFieldCollection::DynamicPixels &
  GlobalFieldCollection::get_pixels() const {
    if (not(this->initialised)) {
      throw FieldCollectionError(
          "Can't iterate over the collection before it is initialised.");
    }
    return this->pixels;
  }

  /* ---------------------------------------------------------------------- */
  GlobalFieldCollection GlobalFieldCollection::get_empty_clone() const {
    GlobalFieldCollection ret_val{this->get_spatial_dim(), this->nb_sub_pts};
    ret_val.initialise(this->pixels.get_nb_subdomain_grid_pts(),
                       this->pixels.get_subdomain_locations());
    return ret_val;
  }

  /* ---------------------------------------------------------------------- */
  Shape_t GlobalFieldCollection::get_pixels_shape() const {
    return static_cast<Shape_t>(this->pixels.get_nb_subdomain_grid_pts());
  }

  /* ---------------------------------------------------------------------- */
  Shape_t
  GlobalFieldCollection::get_pixels_strides(Index_t element_size) const {
    Shape_t strides{this->pixels.get_strides()};
    for (auto && s : strides) {
      s *= element_size;
    }
    return strides;
  }

}  // namespace muGrid
