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
               const Index_t & nb_dof_per_sub_pt,
               const std::string & sub_division_tag, const Unit & unit)
      : name{unique_name}, collection{collection},
        nb_dof_per_sub_pt{nb_dof_per_sub_pt},
        nb_sub_pts{collection.get_nb_sub_pts(sub_division_tag)},
        sub_division_tag{sub_division_tag}, unit{unit} {}
  /* ---------------------------------------------------------------------- */
  const std::string & Field::get_name() const { return this->name; }

  /* ---------------------------------------------------------------------- */
  FieldCollection & Field::get_collection() const { return this->collection; }

  /* ---------------------------------------------------------------------- */
  size_t Field::size() const { return this->current_size; }

  /* ---------------------------------------------------------------------- */
  const Index_t & Field::get_nb_dof_per_sub_pt() const {
    return this->nb_dof_per_sub_pt;
  }

  /* ---------------------------------------------------------------------- */
  const Index_t & Field::get_nb_sub_pts() const { return this->nb_sub_pts; }

  /* ---------------------------------------------------------------------- */
  Index_t Field::get_nb_dof_per_pixel() const {
    return this->get_nb_dof_per_sub_pt() * this->get_nb_sub_pts();
  }

  /* ---------------------------------------------------------------------- */
  Index_t Field::get_nb_pixels() const {
    return this->collection.get_nb_pixels();
  }

  /* ---------------------------------------------------------------------- */
  Index_t Field::get_nb_entries() const {
    if (not this->has_nb_sub_pts()) {
      return Unknown;
    }
    return this->nb_sub_pts * this->get_nb_pixels();
  }

  /* ---------------------------------------------------------------------- */
  std::vector<Index_t> Field::get_shape(const IterUnit & iter_type) const {
    std::vector<Index_t> shape;

    auto && use_iter_type{this->get_nb_sub_pts() == 1 ? IterUnit::Pixel
                                                      : iter_type};
    for (auto && n : this->get_components_shape(use_iter_type)) {
      shape.push_back(n);
    }
    for (auto && n : this->get_pixels_shape()) {
      shape.push_back(n);
    }
    return shape;
  }

  /* ---------------------------------------------------------------------- */
  std::vector<Index_t> Field::get_strides(const IterUnit & iter_type,
                                          const Index_t & multiplier) const {
    std::vector<Index_t> strides;

    auto && use_iter_type{this->get_nb_sub_pts() == 1 ? IterUnit::Pixel
                                                      : iter_type};
    for (auto && n : this->get_components_strides(use_iter_type)) {
      strides.push_back(n * multiplier);
    }
    for (auto && n : this->get_pixels_strides()) {
      strides.push_back(n * this->get_nb_dof_per_pixel() * multiplier);
    }
    return strides;
  }

  /* ---------------------------------------------------------------------- */
  std::vector<Index_t>
  Field::get_strides(const std::vector<Index_t> & custom_component_shape,
                     const Index_t & multiplier) const {
    std::vector<Index_t> strides;

    //! check compatibility of component_shape
    auto && nb_components{std::accumulate(custom_component_shape.begin(),
                                          custom_component_shape.end(), 1,
                                          std::multiplies<Index_t>())};
    if (nb_components != this->get_nb_dof_per_pixel()) {
      std::stringstream message{};
      message << "The component shape " << custom_component_shape << " has "
              << nb_components << " entries, but this field has "
              << this->get_nb_dof_per_pixel()
              << " degrees of freedom per pixel.";
      throw FieldError{message.str()};
    }
    Index_t accumulator{multiplier};
    for (auto && n : custom_component_shape) {
      strides.push_back(accumulator);
      accumulator *= n;
    }
    for (auto && n : this->get_pixels_strides()) {
      strides.push_back(n * this->get_nb_dof_per_pixel() * multiplier);
    }
    return strides;
  }

  /* ---------------------------------------------------------------------- */
  std::vector<Index_t> Field::get_pixels_shape() const {
    std::vector<Index_t> shape;
    if (this->is_global()) {
      auto & coll =
          dynamic_cast<const GlobalFieldCollection &>(this->collection);
      for (auto && n : coll.get_pixels().get_nb_subdomain_grid_pts()) {
        shape.push_back(n);
      }
    } else {
      shape.push_back(this->collection.get_nb_pixels());
    }
    return shape;
  }

  /* ---------------------------------------------------------------------- */
  std::vector<Index_t> Field::get_pixels_strides() const {
    std::vector<Index_t> strides;
    if (this->is_global()) {
      auto & coll{
          dynamic_cast<const GlobalFieldCollection &>(this->collection)};
      for (auto && s : coll.get_pixels().get_strides()) {
        strides.push_back(s);
      }
    } else {
      strides.push_back(1);
    }
    return strides;
  }

  /* ---------------------------------------------------------------------- */
  std::vector<Index_t>
  Field::get_components_strides(const IterUnit & iter_type) const {
    std::vector<Index_t> strides{};
    Index_t accumulator{1};
    for (auto && n : this->get_components_shape(iter_type)) {
      strides.push_back(accumulator);
      accumulator *= n;
    }
    return strides;
  }

  /* ---------------------------------------------------------------------- */
  std::vector<Index_t>
  Field::get_components_shape(const IterUnit & iter_type) const {
    std::vector<Index_t> shape;

    if (this->get_nb_sub_pts() == 1 || iter_type == IterUnit::Pixel) {
      shape.push_back(this->nb_dof_per_sub_pt * this->get_nb_sub_pts());
    } else {
      shape.push_back(this->nb_dof_per_sub_pt);
      shape.push_back(this->get_nb_sub_pts());
    }
    return shape;
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
      return this->get_nb_dof_per_sub_pt() * this->get_nb_sub_pts();
    } else {
      return this->get_nb_dof_per_sub_pt();
    }
  }

  /* ---------------------------------------------------------------------- */
  Index_t Field::get_default_nb_rows(const IterUnit & /*iter_type*/) const {
    return this->get_nb_dof_per_sub_pt();
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
