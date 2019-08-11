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

#include "nfield_collection_global.hh"

namespace muGrid {

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS>
  GlobalNFieldCollection<DimS>::GlobalNFieldCollection(Dim_t nb_quad_pts)
      : Parent{Domain::Global, DimS, nb_quad_pts} {}

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS>
  void GlobalNFieldCollection<DimS>::initialise(
      Ccoord nb_grid_pts, Ccoord locations, Ccoord strides) {
    if (this->initialised) {
      throw NFieldCollectionError("double initialisation");
    } else if (not this->has_nb_quad()) {
      throw NFieldCollectionError(
          "The number of quadrature points has not been set.");
    }

    this->pixels = CcoordOps::Pixels<DimS>{
      nb_grid_pts, locations, strides};
    this->nb_entries = CcoordOps::get_size(nb_grid_pts) * this->nb_quad_pts;
    this->nb_grid_pts = nb_grid_pts;
    this->locations = locations;
    this->allocate_fields();
    this->indices.resize(this->nb_entries);
    for (int i{0}; i < this->nb_entries; ++i) {
      indices[i] = i;
    }
    this->initialised = true;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS>
  void GlobalNFieldCollection<DimS>::initialise(
      Ccoord nb_grid_pts, Ccoord locations) {
    this->initialise(nb_grid_pts, locations,
                     muGrid::CcoordOps::get_default_strides(nb_grid_pts));
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS>
  const typename GlobalNFieldCollection<DimS>::Pixels &
  GlobalNFieldCollection<DimS>::get_pixels() const {
    if (not(this->initialised)) {
      throw NFieldCollectionError(
          "Can't iterate over the collection before it is initialised.");
    }
    return this->pixels;
  }

  template class GlobalNFieldCollection<oneD>;
  template class GlobalNFieldCollection<twoD>;
  template class GlobalNFieldCollection<threeD>;
}  // namespace muGrid
