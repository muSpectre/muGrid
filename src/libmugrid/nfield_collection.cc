/**
 * @file   nfield_collection.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   11 Aug 2019
 *
 * @brief  Implementations for field collections
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

#include "nfield_collection.hh"
#include "nfield.hh"
#include "state_nfield.hh"

namespace muGrid {

  /* ---------------------------------------------------------------------- */
  template <class DefaultDestroyable>
  void NFieldDestructor<DefaultDestroyable>::
  operator()(DefaultDestroyable * field) {
    delete field;
  }

  /* ---------------------------------------------------------------------- */
  template struct NFieldDestructor<NField>;
  template struct NFieldDestructor<StateNField>;

  /* ---------------------------------------------------------------------- */
  NFieldCollection::NFieldCollection(Domain domain, Dim_t spatial_dimension,
                                     Dim_t nb_quad_pts)
      : domain{domain}, spatial_dim{spatial_dimension}, nb_quad_pts{
                                                            nb_quad_pts} {}

  /* ---------------------------------------------------------------------- */
  bool NFieldCollection::field_exists(const std::string & unique_name) const {
    return this->fields.find(unique_name) != this->fields.end();
  }

  /* ---------------------------------------------------------------------- */
  bool NFieldCollection::state_field_exists(
      const std::string & unique_prefix) const {
    return this->state_fields.find(unique_prefix) != this->state_fields.end();
  }

  /* ---------------------------------------------------------------------- */
  const Dim_t & NFieldCollection::size() const { return this->nb_entries; }

  /* ---------------------------------------------------------------------- */
  size_t NFieldCollection::get_nb_pixels() const {
    return this->nb_entries / this->nb_quad_pts;
  }

  /* ---------------------------------------------------------------------- */
  bool NFieldCollection::has_nb_quad() const {
    return not(this->nb_quad_pts == Unknown);
  }

  /* ---------------------------------------------------------------------- */
  void NFieldCollection::set_nb_quad(const Dim_t & nb_quad_pts_per_pixel) {
    if (this->has_nb_quad() and (this->nb_quad_pts != nb_quad_pts_per_pixel)) {
      std::stringstream error{};
      error << "The number of quadrature points per pixel has already been set "
               "to "
            << this->nb_quad_pts << " and cannot be changed";
      throw NFieldCollectionError(error.str());
    }
    if (nb_quad_pts_per_pixel < 1) {
      std::stringstream error{};
      error << "The number of quadrature points per pixel must be positive. "
            << "You chose " << nb_quad_pts_per_pixel;
      throw NFieldCollectionError(error.str());
    }
    this->nb_quad_pts = nb_quad_pts_per_pixel;
  }

  /* ---------------------------------------------------------------------- */
  const Dim_t & NFieldCollection::get_nb_quad() const {
    return this->nb_quad_pts;
  }

  /* ---------------------------------------------------------------------- */
  auto NFieldCollection::get_domain() const -> const Domain & {
    return this->domain;
  }

  /* ---------------------------------------------------------------------- */
  bool NFieldCollection::is_initialised() const { return this->initialised; }

  /* ---------------------------------------------------------------------- */
  auto NFieldCollection::begin() const -> iterator {
    if (not this->is_initialised()) {
      throw NFieldCollectionError(
          "Can't iterate over a collection before it's initialised");
    }
    return this->indices.begin();
  }

  /* ---------------------------------------------------------------------- */
  auto NFieldCollection::end() const -> iterator { return this->indices.end(); }

  /* ---------------------------------------------------------------------- */
  void NFieldCollection::allocate_fields() {
    for (auto && item : this->fields) {
      auto && field{*item.second};
      const auto field_size = field.size();
      if ((field_size != 0) and (field_size != size_t(this->size()))) {
        std::stringstream err_stream;
        err_stream << "NField '" << field.get_name() << "' contains "
                   << field_size << " entries, but the field collection "
                   << "has " << this->size() << " pixels";
        throw NFieldCollectionError(err_stream.str());
      }
      // resize is being called unconditionally, because it alone guarantees
      // the exactness of the field's `data_ptr`
      field.resize(this->size());
    }
  }

  /* ---------------------------------------------------------------------- */
  NField & NFieldCollection::get_field(const std::string & unique_name) {
    if (not this->field_exists(unique_name)) {
      std::stringstream err_stream{};
      err_stream << "The field '" << unique_name << "' does not exist";
      throw NFieldCollectionError(err_stream.str());
    }
    return *this->fields[unique_name];
  }

  /* ---------------------------------------------------------------------- */
  StateNField &
  NFieldCollection::get_state_field(const std::string & unique_prefix) {
    if (not this->state_field_exists(unique_prefix)) {
      std::stringstream err_stream{};
      err_stream << "The state field '" << unique_prefix << "' does not exist";
      throw NFieldCollectionError(err_stream.str());
    }
    return *this->state_fields[unique_prefix];
  }
}  // namespace muGrid
