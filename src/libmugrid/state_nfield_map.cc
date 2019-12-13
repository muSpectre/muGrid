/**
 * @file   state_nfield_map.cc
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

#include "state_nfield_map.hh"
#include "state_nfield.hh"
#include "nfield_map.hh"
#include "nfield_typed.hh"
#include "nfield_collection.hh"
#include "nfield.hh"

namespace muGrid {

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  auto StateNFieldMap<T, Mutability>::make_maps(RefVector<NField> & fields)
      -> std::vector<NFieldMap_t> {
    std::vector<NFieldMap_t> retval{};
    for (auto && field : fields) {
      retval.emplace_back(static_cast<TypedNField<T> &>(field), this->nb_rows,
                          this->iteration);
    }
    return retval;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  auto StateNFieldMap<T, Mutability>::make_cmaps(RefVector<NField> & fields)
      -> std::vector<CNFieldMap_t> {
    std::vector<CNFieldMap_t> retval{};
    for (auto && field : fields) {
      retval.emplace_back(static_cast<TypedNField<T> &>(field), this->nb_rows,
                          this->iteration);
    }
    return retval;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  StateNFieldMap<T, Mutability>::StateNFieldMap(
      TypedStateNField<T> & state_field, Iteration iter_type)
      : state_field{state_field}, iteration{iter_type},
        nb_rows{(iter_type == Iteration::QuadPt)
                    ? state_field.current().get_nb_components()
                    : state_field.current().get_nb_components() *
                          state_field.current().get_collection().get_nb_quad()},
        maps(this->make_maps(state_field.get_fields())),
        cmaps(this->make_cmaps(state_field.get_fields())) {}

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  StateNFieldMap<T, Mutability>::StateNFieldMap(
      TypedStateNField<T> & state_field, Dim_t nb_rows, Iteration iter_type)
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
      throw NFieldMapError(error.str());
    }
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  auto StateNFieldMap<T, Mutability>::begin() -> iterator {
    return iterator{*this, 0};
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  auto StateNFieldMap<T, Mutability>::end() -> iterator {
    return iterator{*this, this->maps.front().size()};
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  void StateNFieldMap<T, Mutability>::initialise() {
    // TODO: comment
    // for (auto && map : this->maps) {
    //   map.initialise();
    // }
    // for (auto && map : this->cmaps) {
    //   map.initialise();
    // }
  }

  /* ---------------------------------------------------------------------- */
  // template <typename T, Mapping Mutability>
  // auto StateNFieldMap<T, Mutability>::begin() const -> const_iterator {
  //   return const_iterator{0, this->get_current_it(false),
  //                         this->get_old_its(false)};
  // }

  /* ---------------------------------------------------------------------- */
  // template <typename T, Mapping Mutability>
  // auto StateNFieldMap<T, Mutability>::end() const -> const_iterator {
  //   return const_iterator{this->current->size(), this->get_current_it(true),
  //                         this->get_old_its(true)};
  // }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  auto StateNFieldMap<T, Mutability>::get_state_field() const
      -> const TypedStateNField<T> & {
    return this->state_field;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  const Dim_t & StateNFieldMap<T, Mutability>::get_nb_rows() const {
    return this->nb_rows;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  size_t StateNFieldMap<T, Mutability>::size() const {
    const auto & field{this->state_field.current()};
    return (this->iteration == Iteration::QuadPt)
               ? field.size()
               : field.get_collection().get_nb_pixels();
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  auto StateNFieldMap<T, Mutability>::get_current() -> NFieldMap_t & {
    return this->maps[this->state_field.get_indices()[0]];
  }
  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  auto StateNFieldMap<T, Mutability>::get_current() const
      -> const NFieldMap_t & {
    return this->maps[this->state_field.get_indices()[0]];
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  auto StateNFieldMap<T, Mutability>::get_old(size_t nb_steps_ago) const
      -> const CNFieldMap_t & {
    return this->cmaps[this->state_field.get_indices()[nb_steps_ago]];
  }
  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  auto StateNFieldMap<T, Mutability>::get_fields() -> RefVector<NField> & {
    return this->state_field.get_fields();
  }

  // /*
  // ----------------------------------------------------------------------
  // */ template <typename T, Mapping Mutability> auto StateNFieldMap<T,
  // Mutability>::get_current() const
  //     -> std::unique_ptr<NFieldMap_t> {
  //   return std::make_unique<NFieldMap_t>(this->state_field.current(),
  //                                        this->nb_rows, this->iteration);
  // }

  // /*
  // ----------------------------------------------------------------------
  // */ template <typename T, Mapping Mutability> auto StateNFieldMap<T,
  // Mutability>::get_olds() const
  //     -> std::vector<CNFieldMap_t> {
  //   std::vector<NFieldMap<T, true>> ret_val{};
  //   const Dim_t nb_memory{this->state_field.get_nb_memory()};

  //   ret_val.reserve(nb_memory);

  //   for (Dim_t nb_steps_ago{1}; nb_steps_ago < nb_memory + 1;
  //   ++nb_steps_ago)
  //   {
  //     ret_val.emplace_back(this->state_field.old(nb_steps_ago),
  //     this->nb_rows);
  //   }
  //   return ret_val;
  // }

  // /*
  // ----------------------------------------------------------------------
  // */ template <typename T, Mapping Mutability> auto StateNFieldMap<T,
  // Mutability>::get_current_it(bool end)
  //     -> CurrentIteratort {
  //   this->current = std::move(this->get_current());
  //   return end ? this->current->end() : this->current->begin();
  // }

  // /*
  // ----------------------------------------------------------------------
  // */ template <typename T, Mapping Mutability> auto StateNFieldMap<T,
  // Mutability>::get_old_its(bool end)
  //     -> std::vector<OldIteratort> {
  //   this->olds = this->get_olds();
  //   std::vector<OldIteratort> ret_val{};
  //   ret_val.reserve(this->state_field.get_nb_memory());
  //   for (auto && old_field_map : this->olds) {
  //     ret_val.push_back(end ? old_field_map.end() :
  //     old_field_map.begin());
  //   }
  //   return ret_val;
  // }

  /* ----------------------------------------------------------------------
   */
  template <typename T, Mapping Mutability>
  template <Mapping MutIter>
  StateNFieldMap<T, Mutability>::Iterator<MutIter>::Iterator(
      StateNFieldMap_t & state_field_map, size_t index)
      : state_field_map{state_field_map}, index{index} {}

  /* ---------------------------------------------------------------------- */
  template class StateNFieldMap<Real, Mapping::Const>;
  template class StateNFieldMap<Real, Mapping::Mut>;
  template class StateNFieldMap<Complex, Mapping::Const>;
  template class StateNFieldMap<Complex, Mapping::Mut>;
  template class StateNFieldMap<Int, Mapping::Const>;
  template class StateNFieldMap<Int, Mapping::Mut>;
  template class StateNFieldMap<Uint, Mapping::Const>;
  template class StateNFieldMap<Uint, Mapping::Mut>;

}  // namespace muGrid
