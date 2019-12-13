/**
 * @file   nfield_collection_global.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   11 Aug 2019
 *
 * @brief  Implementation of GlobalNFieldCollection
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

#include "nfield_collection_global.hh"
#include <iostream>
namespace muGrid {

  /* ---------------------------------------------------------------------- */
  GlobalNFieldCollection::GlobalNFieldCollection(Dim_t spatial_dimension,
                                                 Dim_t nb_quad_pts)
      : Parent{ValidityDomain::Global, spatial_dimension, nb_quad_pts} {}

  /* ---------------------------------------------------------------------- */
  GlobalNFieldCollection::GlobalNFieldCollection(
      Dim_t spatial_dimension,
      Dim_t nb_quad_pts,
      const DynCcoord_t & nb_grid_pts,
      const DynCcoord_t & locations)
      : Parent{ValidityDomain::Global, spatial_dimension, nb_quad_pts} {
    this->initialise(nb_grid_pts, locations);
  }

  /* ---------------------------------------------------------------------- */
  void GlobalNFieldCollection::initialise(const DynCcoord_t & nb_grid_pts,
                                          const DynCcoord_t & locations,
                                          const DynCcoord_t & strides) {
    if (this->initialised) {
      throw NFieldCollectionError("double initialisation");
    } else if (not this->has_nb_quad()) {
      throw NFieldCollectionError(
          "The number of quadrature points has not been set.");
    }

    this->pixels = CcoordOps::DynamicPixels(nb_grid_pts, locations, strides);
    this->nb_entries = CcoordOps::get_size(nb_grid_pts) * this->nb_quad_pts;
    this->allocate_fields();
    this->pixel_indices.resize(this->nb_entries);
    for (int i{0}; i < this->nb_entries; ++i) {
      this->pixel_indices[i] = i;
    }
    this->initialised = true;
    this->initialise_maps(); // yes, this has to be after the previous line
  }

  /* ---------------------------------------------------------------------- */
  void GlobalNFieldCollection::initialise(const DynCcoord_t & nb_grid_pts,
                                          const DynCcoord_t & locations) {
    this->initialise(nb_grid_pts,
                     ((locations.get_dim() == 0)
                          ? DynCcoord_t(nb_grid_pts.get_dim())
                          : locations),
                     muGrid::CcoordOps::get_default_strides(nb_grid_pts));
  }

  /* ---------------------------------------------------------------------- */
  const typename GlobalNFieldCollection::DynamicPixels &
  GlobalNFieldCollection::get_pixels() const {
    if (not(this->initialised)) {
      throw NFieldCollectionError(
          "Can't iterate over the collection before it is initialised.");
    }
    return this->pixels;
  }

  /* ---------------------------------------------------------------------- */
  GlobalNFieldCollection GlobalNFieldCollection::get_empty_clone() const {
    GlobalNFieldCollection ret_val{this->get_spatial_dim(),
                                   this->get_nb_quad()};
    ret_val.initialise(this->pixels.get_nb_grid_pts(),
                       this->pixels.get_locations());
    return ret_val;
  }

}  // namespace muGrid
