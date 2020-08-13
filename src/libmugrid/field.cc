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
               const Index_t & nb_components,
               const std::string & sub_division_tag, const Unit & unit)
      : name{unique_name}, collection{collection}, nb_components{nb_components},
        components_shape{nb_components},
        nb_sub_pts{collection.get_nb_sub_pts(sub_division_tag)},
        sub_division_tag{sub_division_tag}, unit{unit} {}

  /* ---------------------------------------------------------------------- */
  Field::Field(const std::string & unique_name, FieldCollection & collection,
               const Shape_t & components_shape,
               const std::string & sub_division_tag, const Unit & unit)
      : name{unique_name}, collection{collection},
        nb_components{std::accumulate(components_shape.begin(),
                                      components_shape.end(),
                                      1, std::multiplies<Index_t>())},
        components_shape{components_shape},
        nb_sub_pts{collection.get_nb_sub_pts(sub_division_tag)},
        sub_division_tag{sub_division_tag}, unit{unit} {}

  /* ---------------------------------------------------------------------- */
  const std::string & Field::get_name() const { return this->name; }

  /* ---------------------------------------------------------------------- */
  FieldCollection & Field::get_collection() const { return this->collection; }

  /* ---------------------------------------------------------------------- */
  Index_t Field::get_current_nb_entries() const {
    return this->current_nb_entries;
  }

  /* ---------------------------------------------------------------------- */
  const Index_t & Field::get_nb_components() const {
    return this->nb_components;
  }

  /* ---------------------------------------------------------------------- */
  const Index_t & Field::get_nb_sub_pts() const { return this->nb_sub_pts; }

  /* ---------------------------------------------------------------------- */
  Index_t Field::get_nb_dof_per_pixel() const {
    return this->get_nb_components() * this->get_nb_sub_pts();
  }

  /* ---------------------------------------------------------------------- */
  Index_t Field::get_nb_pixels() const {
    return this->collection.get_nb_pixels();
  }

  /* ---------------------------------------------------------------------- */
  Index_t Field::get_nb_buffer_pixels() const {
    return this->collection.get_nb_buffer_pixels();
  }

  /* ---------------------------------------------------------------------- */
  Index_t Field::get_nb_entries() const {
    if (not this->has_nb_sub_pts()) {
      return Unknown;
    }
    return this->nb_sub_pts * this->get_nb_pixels();
  }

  /* ---------------------------------------------------------------------- */
  Index_t Field::get_nb_buffer_entries() const {
    if (not this->has_nb_sub_pts()) {
      return Unknown;
    }
    return this->nb_sub_pts * this->get_nb_buffer_pixels();
  }

  /* ---------------------------------------------------------------------- */
  Shape_t Field::get_components_shape() const {
    return this->components_shape;
  }

  /* ---------------------------------------------------------------------- */
  void Field::reshape(const Shape_t & new_components_shape) {
    if (std::accumulate(new_components_shape.begin(),
                        new_components_shape.end(),
                        1, std::multiplies<Index_t>()) !=
        this->nb_components) {
      std::stringstream message{};
      message << "This field was set up for " << this->get_nb_components()
              << " components. Setting the component shape to "
              << new_components_shape << " is not supported because it would "
              << "change the total number of components.";
      throw FieldError(message.str());
    }
    this->components_shape = new_components_shape;
    this->nb_components = std::accumulate(this->components_shape.begin(),
                                          this->components_shape.end(), 1,
                                          std::multiplies<Index_t>());
  }

  /* ---------------------------------------------------------------------- */
  void Field::reshape(const Shape_t & new_components_shape,
                      const std::string & sub_div_tag) {
    auto new_nb_sub_pts{collection.get_nb_sub_pts(sub_div_tag)};
    if (std::accumulate(new_components_shape.begin(),
                        new_components_shape.end(),
                        1, std::multiplies<Index_t>()) * new_nb_sub_pts !=
        this->get_nb_dof_per_pixel()) {
      std::stringstream message{};
      message << "This field was set up for " << this->get_nb_components()
              << " components and " << this->nb_sub_pts << " sub-points. "
              << "Setting the component shape to " << new_components_shape
              << " and the number of sub-points to " << new_nb_sub_pts
              << " (sub-point tag '" << sub_div_tag << "') is not supported "
              << "because it would change the total number of degrees of "
              << "freedom per pixel.";
      throw FieldError(message.str());
    }
    this->components_shape = new_components_shape;
    this->nb_components = std::accumulate(this->components_shape.begin(),
                                          this->components_shape.end(), 1,
                                          std::multiplies<Index_t>());
    this->nb_sub_pts = new_nb_sub_pts;
    this->sub_division_tag = sub_div_tag;
  }

  /* ---------------------------------------------------------------------- */
  Shape_t Field::get_sub_pt_shape(const IterUnit & iter_type) const {
    Shape_t shape{};
    for (auto && n : this->get_components_shape()) {
      shape.push_back(n);
    }
    if (iter_type == IterUnit::Pixel) {
      shape[shape.size() - 1] *= this->get_nb_sub_pts();
    } else {
      shape.push_back(this->get_nb_sub_pts());
    }
    return shape;
  }

  /* ---------------------------------------------------------------------- */
  Shape_t Field::get_pixels_shape() const {
    return this->collection.get_pixels_shape();
  }

  /* ---------------------------------------------------------------------- */
  Shape_t Field::get_shape(const IterUnit & iter_type) const {
    Shape_t shape;

    if (this->get_nb_dof_per_pixel() > 1) {
      for (auto && n : this->get_sub_pt_shape(iter_type)) {
        shape.push_back(n);
      }
    }
    for (auto && n : this->get_pixels_shape()) {
      shape.push_back(n);
    }
    return shape;
  }

  /* ---------------------------------------------------------------------- */
  Shape_t Field::get_components_strides(Index_t element_size) const {
    if (this->get_storage_order() != StorageOrder::ColMajor and
        this->get_storage_order() != StorageOrder::RowMajor) {
      std::stringstream s;
      s << "Don't know how to construct strides for storage order "
        << this->get_storage_order();
      throw FieldError(s.str());
    }
    Shape_t strides{};
    Index_t accumulator{element_size};
    auto components_shape{this->get_components_shape()};
    if (this->get_storage_order() == StorageOrder::RowMajor) {
      std::reverse(components_shape.begin(), components_shape.end());
    }
    for (auto && n : components_shape) {
      strides.push_back(accumulator);
      accumulator *= n;
    }
    if (this->get_storage_order() == StorageOrder::RowMajor) {
      std::reverse(strides.begin(), strides.end());
    }
    return strides;
  }

  /* ---------------------------------------------------------------------- */
  Shape_t Field::get_sub_pt_strides(const IterUnit & iter_type,
                                    Index_t element_size) const {
    if (this->get_storage_order() != StorageOrder::ColMajor and
        this->get_storage_order() != StorageOrder::RowMajor) {
      std::stringstream s;
      s << "Don't know how to construct strides for storage order "
        << this->get_storage_order();
      throw FieldError(s.str());
    }
    Shape_t strides{get_components_strides(element_size)};
    if (iter_type != IterUnit::Pixel) {
      if (this->get_storage_order() == StorageOrder::RowMajor) {
        for (auto && s : strides) {
          s *= this->get_nb_sub_pts();
        }
        strides.push_back(element_size);
      } else {
        strides.push_back(this->get_nb_components() * element_size);
      }
    }
    return strides;
  }

  /* ---------------------------------------------------------------------- */
  Shape_t Field::get_pixels_strides(Index_t element_size) const {
    return this->collection.get_pixels_strides(element_size);
  }

  /* ---------------------------------------------------------------------- */
  Shape_t Field::get_strides(const IterUnit & iter_type,
                             Index_t element_size) const {
    Shape_t strides{this->get_sub_pt_strides(iter_type, element_size)};
    if (this->get_nb_dof_per_pixel() <= 1) {
      strides.clear();
    }
    if (this->get_storage_order() == StorageOrder::ColMajor) {
      for (auto && n : this->get_pixels_strides(element_size)) {
        strides.push_back(n * this->get_nb_dof_per_pixel());
      }
    } else if (this->get_storage_order() == StorageOrder::RowMajor) {
      // storage order is RowMajor, which means all pixels for each dof
      for (auto && s : strides) {
        s *= this->collection.get_nb_buffer_pixels();
      }
      for (auto && n : this->get_pixels_strides(element_size)) {
        strides.push_back(n);
      }
    } else {
      std::stringstream s;
      s << "Don't know how to construct strides for storage order "
        << this->collection.get_storage_order();
      throw FieldError(s.str());
    }
    return strides;
  }

  /* ---------------------------------------------------------------------- */
  StorageOrder Field::get_storage_order() const {
    return this->collection.get_storage_order();
  }

  /* ---------------------------------------------------------------------- */
  Index_t Field::get_stride(const IterUnit & iter_type) const {
    if (iter_type == IterUnit::Pixel) {
      if (not this->get_collection().has_nb_sub_pts(
              this->get_sub_division_tag())) {
        std::stringstream message{};
        message
            << "You are trying to map a pixel map onto the '"
            << this->get_sub_division_tag() << "' field '" << this->get_name()
            << "', but the number of sub points is unknown to the "
               "field collection. Please use FieldCollection::set_nb_sub_pts(\""
            << this->get_sub_division_tag()
            << "\") before this call to fix the situation.";
        throw FieldError(message.str());
      }
      return this->get_nb_components() * this->get_nb_sub_pts();
    } else {
      return this->get_nb_components();
    }
  }

  /* ---------------------------------------------------------------------- */
  bool Field::has_same_memory_layout(const Field & other) const {
    return
      this->get_collection().has_same_memory_layout(other.get_collection()) &&
      this->get_nb_sub_pts() == other.get_nb_sub_pts() &&
      this->get_components_strides() == other.get_components_strides();
  }

  /* ---------------------------------------------------------------------- */
  Index_t Field::get_default_nb_rows(const IterUnit & /*iter_type*/) const {
    return this->get_nb_components();
  }

  /* ---------------------------------------------------------------------- */
  Index_t Field::get_default_nb_cols(const IterUnit & iter_type) const {
    return (iter_type == IterUnit::Pixel ? this->get_nb_sub_pts() : 1);
  }
  /* ---------------------------------------------------------------------- */
  const size_t & Field::get_pad_size() const { return this->pad_size; }

  /* ---------------------------------------------------------------------- */
  bool Field::is_global() const {
    return this->collection.get_domain() ==
           FieldCollection::ValidityDomain::Global;
  }

  /* ---------------------------------------------------------------------- */
  bool Field::has_nb_sub_pts() const { return this->nb_sub_pts != Unknown; }

  /* ---------------------------------------------------------------------- */
  const std::string & Field::get_sub_division_tag() const {
    return this->sub_division_tag;
  }

  /* ---------------------------------------------------------------------- */
  void Field::set_nb_sub_pts(const Index_t & nb_sub_pts_per_pixel) {
    this->nb_sub_pts = nb_sub_pts_per_pixel;
  }

}  // namespace muGrid
