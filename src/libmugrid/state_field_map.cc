/**
 * @file   state_field_map.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   22 Aug 2019
 *
 * @brief  implementation of state field maps
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

#include "state_field_map.hh"
#include "state_field.hh"
#include "field_map.hh"
#include "field_typed.hh"
#include "field_collection.hh"
#include "field.hh"

namespace muGrid {

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  auto StateFieldMap<T, Mutability>::make_maps(RefVector<Field> & fields)
      -> std::vector<FieldMap_t> {
    std::vector<FieldMap_t> retval{};
    for (auto && field : fields) {
      retval.emplace_back(static_cast<TypedField<T> &>(field), this->nb_rows,
                          this->iteration);
    }
    return retval;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  auto StateFieldMap<T, Mutability>::make_cmaps(RefVector<Field> & fields)
      -> std::vector<CFieldMap_t> {
    std::vector<CFieldMap_t> retval{};
    for (auto && field : fields) {
      retval.emplace_back(static_cast<TypedField<T> &>(field), this->nb_rows,
                          this->iteration);
    }
    return retval;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  StateFieldMap<T, Mutability>::StateFieldMap(TypedStateField<T> & state_field,
                                              IterUnit iter_type)
      : state_field{state_field}, iteration{iter_type},
        nb_rows{iter_type == IterUnit::Pixel
                    ? state_field.current().get_nb_components() *
                          state_field.current().get_nb_sub_pts()
                    : state_field.current().get_nb_components()},
        maps(this->make_maps(state_field.get_fields())),
        cmaps(this->make_cmaps(state_field.get_fields())) {}

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  StateFieldMap<T, Mutability>::StateFieldMap(
      TypedStateField<T> & state_field, Index_t nb_rows, IterUnit iter_type)
      : state_field{state_field}, iteration{iter_type}, nb_rows{nb_rows},
        maps(this->make_maps(state_field.get_fields())),
        cmaps(this->make_cmaps(state_field.get_fields())) {
    // check whether nb_rows is compatible with the underlying fields
    const auto & field{state_field.current()};
    const auto stride{field.get_stride(iter_type)};
    if (stride % this->nb_rows != 0) {
      std::stringstream error{};
      error << "You chose an iterate with " << this->nb_rows
            << " rows, but it is not a divisor of the number of scalars stored "
               "in this field per iteration ("
            << stride << ")";
      throw FieldMapError(error.str());
    }
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  auto StateFieldMap<T, Mutability>::begin() -> iterator {
    return iterator{*this, 0};
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  auto StateFieldMap<T, Mutability>::end() -> iterator {
    return iterator{*this, this->maps.front().size()};
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  auto StateFieldMap<T, Mutability>::get_state_field() const
      -> const TypedStateField<T> & {
    return this->state_field;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  const Index_t & StateFieldMap<T, Mutability>::get_nb_rows() const {
    return this->nb_rows;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  size_t StateFieldMap<T, Mutability>::size() const {
    const auto & field{this->state_field.current()};
    return (this->iteration == IterUnit::SubPt)
               ? field.get_nb_entries()
               : field.get_collection().get_nb_pixels();
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  auto StateFieldMap<T, Mutability>::get_current() -> FieldMap_t & {
    return this->maps[this->state_field.get_indices()[0]];
  }
  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  auto StateFieldMap<T, Mutability>::get_current() const
      -> const FieldMap_t & {
    return this->maps[this->state_field.get_indices()[0]];
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  auto StateFieldMap<T, Mutability>::get_old(size_t nb_steps_ago) const
      -> const CFieldMap_t & {
    return this->cmaps[this->state_field.get_indices()[nb_steps_ago]];
  }
  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  auto StateFieldMap<T, Mutability>::get_fields() -> RefVector<Field> & {
    return this->state_field.get_fields();
  }


  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  template <Mapping MutIter>
  StateFieldMap<T, Mutability>::Iterator<MutIter>::Iterator(
      StateFieldMap_t & state_field_map, size_t index)
      : state_field_map{state_field_map}, index{index} {}

  /* ---------------------------------------------------------------------- */
  template class StateFieldMap<Real, Mapping::Const>;
  template class StateFieldMap<Real, Mapping::Mut>;
  template class StateFieldMap<Complex, Mapping::Const>;
  template class StateFieldMap<Complex, Mapping::Mut>;
  template class StateFieldMap<Int, Mapping::Const>;
  template class StateFieldMap<Int, Mapping::Mut>;
  template class StateFieldMap<Uint, Mapping::Const>;
  template class StateFieldMap<Uint, Mapping::Mut>;

}  // namespace muGrid
