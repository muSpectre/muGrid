/**
 * @file   nfield_map.cc
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

#include "nfield_map.hh"
#include "nfield_typed.hh"
#include "nfield_collection.hh"
#include "iterators.hh"

#include "sstream"

namespace muGrid {

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  NFieldMap<T, Mutability>::NFieldMap(NField_t & field, Iteration iter_type)
      : field{field}, iteration{iter_type}, stride{this->field.get_stride(
                                                iter_type)},
        nb_rows{this->stride}, nb_cols{1} {
    if (this->field.get_collection().is_initialised()) {
      this->initialise();
    }
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  NFieldMap<T, Mutability>::NFieldMap(NField_t & field, Dim_t nb_rows_,
                                      Iteration iter_type)
      : field{field}, iteration{iter_type}, stride{this->field.get_stride(
                                                iter_type)},
        nb_rows{nb_rows_}, nb_cols{this->stride / nb_rows_} {
    if (this->field.get_collection().is_initialised()) {
      this->initialise();
    }
    if (this->nb_rows * this->nb_cols != this->stride) {
      std::stringstream error{};
      error << "You chose an iterate with " << this->nb_rows
            << " rows, but it is not a divisor of the number of scalars stored "
               "in this field per iteration ("
            << this->stride << ")";
      throw NFieldMapError(error.str());
    }
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  auto NFieldMap<T, Mutability>::begin() -> iterator {
    if (not this->is_initialised) {
      this->initialise();
    }
    return iterator{*this, false};
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  auto NFieldMap<T, Mutability>::end() -> iterator {
    return iterator{*this, true};
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  auto NFieldMap<T, Mutability>::cbegin() -> const_iterator {
    if (not this->is_initialised) {
      this->initialise();
    }
    return const_iterator{*this, false};
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  auto NFieldMap<T, Mutability>::cend() -> const_iterator {
    return const_iterator{*this, true};
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  auto NFieldMap<T, Mutability>::begin() const -> const_iterator {
    if (not this->is_initialised) {
      throw NFieldMapError("Needs to be initialised");
    }
    return const_iterator{*this, false};
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  auto NFieldMap<T, Mutability>::end() const -> const_iterator {
    return const_iterator{*this, true};
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  size_t NFieldMap<T, Mutability>::size() const {
    return (this->iteration == Iteration::QuadPt)
               ? this->field.size()
               : this->field.get_collection().get_nb_pixels();
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  void NFieldMap<T, Mutability>::initialise() {
    if (not(this->field.get_collection().is_initialised())) {
      throw NFieldMapError("Can't initialise map before the field collection "
                           "has been initialised");
    }
    this->data_ptr = this->field.data();
    this->is_initialised = true;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  auto NFieldMap<T, Mutability>::enumerate_pixel_indices_fast()
      -> PixelEnumeration_t {
    if (this->iteration != Iteration::Pixel) {
      throw NFieldMapError("Cannot enumerate pixels unless the iteration mode "
                           "of this map is Iteration::Pixels.");
    }
    return akantu::zip(this->field.get_collection().get_pixel_indices_fast(),
                       *this);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  auto NFieldMap<T, Mutability>::enumerate_indices() -> Enumeration_t {
    return akantu::zip(this->field.get_collection().get_quad_pt_indices(),
                       *this);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  auto NFieldMap<T, Mutability>::mean() const -> PlainType {
    PlainType mean{PlainType::Zero(this->nb_rows, this->nb_cols)};
    for (auto && val : *this) {
      mean += val;
    }
    mean *= 1. / Real(this->size());
    return mean;
  }

  /* ---------------------------------------------------------------------- */
  template class NFieldMap<Real, Mapping::Const>;
  template class NFieldMap<Real, Mapping::Mut>;
  template class NFieldMap<Complex, Mapping::Const>;
  template class NFieldMap<Complex, Mapping::Mut>;
  template class NFieldMap<Int, Mapping::Const>;
  template class NFieldMap<Int, Mapping::Mut>;
  template class NFieldMap<Uint, Mapping::Const>;
  template class NFieldMap<Uint, Mapping::Mut>;
}  // namespace muGrid
