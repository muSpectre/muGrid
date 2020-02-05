/**
 * @file   field.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   11 Aug 2019
 *
 * @brief  implementation of Field
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

#include "field.hh"
#include "field_collection.hh"
#include "field_collection_global.hh"

namespace muGrid {

  /* ---------------------------------------------------------------------- */
  Field::Field(const std::string & unique_name, FieldCollection & collection,
               Dim_t nb_dof_per_quad_pt)
      : name{unique_name}, collection{collection}, nb_dof_per_quad_pt{
          nb_dof_per_quad_pt} {}
  /* ---------------------------------------------------------------------- */
  const std::string & Field::get_name() const { return this->name; }

  /* ---------------------------------------------------------------------- */
  FieldCollection & Field::get_collection() const { return this->collection; }

  /* ---------------------------------------------------------------------- */
  size_t Field::size() const { return this->current_size; }

  /* ---------------------------------------------------------------------- */
  const Dim_t & Field::get_nb_dof_per_quad_pt() const {
    return this->nb_dof_per_quad_pt;
  }

  /* ---------------------------------------------------------------------- */
  std::vector<Dim_t> Field::get_shape(Iteration iter_type) const {
    std::vector<Dim_t> shape;

    if (collection.get_nb_quad_pts() == 1) { iter_type = Iteration::Pixel; }

    for (auto && n : this->get_components_shape(iter_type)) {
      shape.push_back(n);
    }
    for (auto && n : this->get_pixels_shape()) {
      shape.push_back(n);
    }
    return shape;
  }

  /* ---------------------------------------------------------------------- */
  std::vector<Dim_t> Field::get_pixels_shape() const {
    std::vector<Dim_t> shape;
    if (this->is_global()) {
      auto & coll = dynamic_cast<const GlobalFieldCollection &>(
          this->collection);
      for (auto && n : coll.get_pixels().get_nb_subdomain_grid_pts()) {
        shape.push_back(n);
      }
    } else {
      shape.push_back(this->collection.get_nb_pixels());
    }
    return shape;
  }

  /* ---------------------------------------------------------------------- */
  std::vector<Dim_t> Field::get_components_shape(Iteration iter_type) const {
    std::vector<Dim_t> shape;

    if (collection.get_nb_quad_pts() == 1 || iter_type == Iteration::Pixel) {
      shape.push_back(
          this->nb_dof_per_quad_pt * this->collection.get_nb_quad_pts());
    } else {
      shape.push_back(this->nb_dof_per_quad_pt);
      shape.push_back(this->collection.get_nb_quad_pts());
    }
    return shape;
  }

  /* ---------------------------------------------------------------------- */
  Dim_t Field::get_stride(Iteration iter_type) const {
    return (iter_type == Iteration::QuadPt)
               ? this->nb_dof_per_quad_pt
               : this->nb_dof_per_quad_pt * this->collection.get_nb_quad_pts();
  }

  /* ---------------------------------------------------------------------- */
  const size_t & Field::get_pad_size() const { return this->pad_size; }

  /* ---------------------------------------------------------------------- */
  bool Field::is_global() const {
    return this->collection.get_domain() ==
           FieldCollection::ValidityDomain::Global;
  }
}  // namespace muGrid
