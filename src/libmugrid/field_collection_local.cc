/**
 * @file   field_collection_local.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   12 Aug 2019
 *
 * @brief  implementation of local field collection
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

#include "field_collection_local.hh"

namespace muGrid {

  /* ---------------------------------------------------------------------- */
  LocalFieldCollection::LocalFieldCollection(Dim_t spatial_dimension,
                                             Dim_t nb_quad_pts)
      : Parent{ValidityDomain::Local, spatial_dimension, nb_quad_pts} {}

  /* ---------------------------------------------------------------------- */
  void LocalFieldCollection::add_pixel(const size_t & global_index) {
    if (this->initialised) {
      throw FieldCollectionError(
          "Cannot add pixels once the collection has been initialised (because "
          "the fields all have been allocated)");
    }
    global_to_local_index_map.insert(
        std::make_pair(global_index, pixel_indices.size()));
    this->pixel_indices.emplace_back(global_index);
  }

  /* ---------------------------------------------------------------------- */
  void LocalFieldCollection::initialise() {
    if (this->initialised) {
      throw FieldCollectionError("double initialisation");
    } else if (not this->has_nb_quad_pts()) {
      throw FieldCollectionError(
          "The number of quadrature points has not been set.");
    }
    this->nb_entries = this->pixel_indices.size() * this->nb_quad_pts;
    this->allocate_fields();
    this->initialised = true;
    this->initialise_maps();  // yes, this has to be after the previous line
  }

  /* ---------------------------------------------------------------------- */
  LocalFieldCollection LocalFieldCollection::get_empty_clone() const {
    LocalFieldCollection ret_val{this->get_spatial_dim(),
                                     this->get_nb_quad_pts()};
    for (const auto & pixel_id : this->get_pixel_indices_fast()) {
      ret_val.add_pixel(pixel_id);
    }
    return ret_val;
  }
}  // namespace muGrid
