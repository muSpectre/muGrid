/**
 * @file   field_map.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   15 Aug 2019
 *
 * @brief  Implementation for basic FieldMap
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

#include "field_map.hh"
#include "field_typed.hh"
#include "field_collection.hh"
#include "iterators.hh"

#include <sstream>
#include <iostream>

namespace muGrid {

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  FieldMap<T, Mutability>::FieldMap(
      Field_t & field, const IterUnit & iter_type)
      : field{field}, iteration{iter_type}, stride{this->field.get_stride(
                                                iter_type)},
        nb_rows{this->field.get_default_nb_rows(iter_type)},
        nb_cols{this->field.get_default_nb_cols(iter_type)} {
    if (field.get_storage_order() != StorageOrder::ColMajor) {
      std::stringstream s;
      s << "FieldMap requires column-major storage order, but storage order is "
        << field.get_storage_order();
      throw RuntimeError(s.str());
    }
    auto & collection{this->field.get_collection()};
    if (collection.is_initialised()) {
      this->set_data_ptr();
    } else {
      this->callback = std::make_shared<std::function<void()>>(
          [this]() { this->set_data_ptr(); });
      collection.preregister_map(this->callback);
    }
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  FieldMap<T, Mutability>::FieldMap(Field_t & field, Index_t nb_rows_,
                                    const IterUnit & iter_type)
      : field{field}, iteration{iter_type}, stride{this->field.get_stride(
                                                iter_type)},
        nb_rows{nb_rows_}, nb_cols{this->stride / nb_rows_} {
    if (field.get_storage_order() != StorageOrder::ColMajor) {
      std::stringstream s;
      s << "FieldMap requires column-major storage order, but storage order is "
        << field.get_storage_order();
      throw RuntimeError(s.str());
    }
    auto & collection{this->field.get_collection()};
    if (collection.is_initialised()) {
      this->set_data_ptr();
    } else {
      this->callback = std::make_shared<std::function<void()>>(
          [this]() { this->set_data_ptr(); });
      collection.preregister_map(this->callback);
    }
    if (this->nb_rows * this->nb_cols != this->stride) {
      std::stringstream error{};
      error << "You chose an iterate with " << this->nb_rows
            << " rows, but it is not a divisor of the number of scalars stored "
               "in this field per iteration ("
            << this->stride << ")";
      throw FieldMapError(error.str());
    }
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  FieldMap<T, Mutability>::FieldMap(FieldMap && other)
      : field{other.field}, iteration{other.iteration}, stride{other.stride},
        nb_rows{other.nb_rows}, nb_cols{other.nb_cols},
        data_ptr{other.data_ptr}, is_initialised{other.is_initialised} {
    if (field.get_storage_order() != StorageOrder::ColMajor) {
      std::stringstream s;
      s << "FieldMap requires column-major storage order, but storage order is "
        << field.get_storage_order();
      throw RuntimeError(s.str());
    }
    auto & collection{this->field.get_collection()};
    if (not collection.is_initialised()) {
      this->callback = std::make_shared<std::function<void()>>(
          [this]() { this->set_data_ptr(); });
      collection.preregister_map(this->callback);
    }
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  auto FieldMap<T, Mutability>::begin() -> iterator {
    if (not this->is_initialised) {
      std::stringstream error{};
      error << "This map on field " << this->field.get_name()
            << " cannot yet be iterated over, as the collection is not "
               "initialised";
      throw FieldMapError(error.str());
    }
    return iterator{*this, false};
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  auto FieldMap<T, Mutability>::end() -> iterator {
    return iterator{*this, true};
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  auto FieldMap<T, Mutability>::cbegin() -> const_iterator {
    if (not this->is_initialised) {
      std::stringstream error{};
      error << "This map on field " << this->field.get_name()
            << " cannot yet be iterated over, as the collection is not "
               "initialised";
      throw FieldMapError(error.str());
    }
    return const_iterator{*this, false};
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  auto FieldMap<T, Mutability>::cend() -> const_iterator {
    return const_iterator{*this, true};
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  auto FieldMap<T, Mutability>::begin() const -> const_iterator {
    if (not this->is_initialised) {
      throw FieldMapError("Needs to be initialised");
    }
    return const_iterator{*this, false};
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  auto FieldMap<T, Mutability>::end() const -> const_iterator {
    return const_iterator{*this, true};
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  size_t FieldMap<T, Mutability>::size() const {
    return (this->iteration == IterUnit::SubPt)
               ? this->field.get_current_nb_entries()
               : this->field.get_collection().get_nb_pixels();
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  void FieldMap<T, Mutability>::set_data_ptr() {
    if (not(this->field.get_collection().is_initialised())) {
      throw FieldMapError("Can't initialise map before the field collection "
                          "has been initialised");
    }
    this->data_ptr = this->field.data();
    this->is_initialised = true;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  auto FieldMap<T, Mutability>::enumerate_pixel_indices_fast()
      -> PixelEnumeration_t {
    if (this->iteration != IterUnit::Pixel) {
      throw FieldMapError("Cannot enumerate pixels unless the iteration mode "
                          "of this map is Iteration::Pixels.");
    }
    return akantu::zip(this->field.get_collection().get_pixel_indices_fast(),
                       *this);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  auto FieldMap<T, Mutability>::enumerate_indices() -> Enumeration_t {
    return akantu::zip(this->field.get_collection().get_sub_pt_indices(
                           this->field.get_sub_division_tag()),
                       *this);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  auto FieldMap<T, Mutability>::mean() const -> PlainType {
    PlainType mean{PlainType::Zero(this->nb_rows, this->nb_cols)};
    for (auto && val : *this) {
      mean += val;
    }
    mean *= 1. / Real(this->size());
    return mean;
  }

  /* ---------------------------------------------------------------------- */
  template class FieldMap<Real, Mapping::Const>;
  template class FieldMap<Real, Mapping::Mut>;
  template class FieldMap<Complex, Mapping::Const>;
  template class FieldMap<Complex, Mapping::Mut>;
  template class FieldMap<Int, Mapping::Const>;
  template class FieldMap<Int, Mapping::Mut>;
  template class FieldMap<Uint, Mapping::Const>;
  template class FieldMap<Uint, Mapping::Mut>;
}  // namespace muGrid
