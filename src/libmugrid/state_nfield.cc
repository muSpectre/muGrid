/**
 * @file   state_nfield.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   20 Aug 2019
 *
 * @brief  implementation for state fields
 *
 * Copyright © 2019 Till Junge
 *
 * µGrid is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µGrid is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with µGrid; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#include "state_nfield.hh"
#include "nfield.hh"
#include "nfield_typed.hh"
#include "nfield_collection.hh"

#include <sstream>

namespace muGrid {

  /* ---------------------------------------------------------------------- */
  StateNField::StateNField(const std::string & unique_prefix,
                           NFieldCollection & collection, Dim_t nb_memory)
      : prefix{unique_prefix}, collection{collection}, nb_memory{nb_memory} {
    if (nb_memory < 1) {
      throw NFieldError("State fields must have a memory size of at least 1.");
    }
    this->indices.reserve(nb_memory + 1);
    this->fields.reserve(nb_memory + 1);

    for (Dim_t i{0}; i < nb_memory + 1; ++i) {
      indices.push_back((nb_memory + 1 - i) & nb_memory);
    }
  }

  /* ---------------------------------------------------------------------- */
  const Dim_t & StateNField::get_nb_memory() const { return this->nb_memory; }

  /* ---------------------------------------------------------------------- */
  void StateNField::cycle() {
    for (auto & val : this->indices) {
      val = (val + 1) % (this->nb_memory + 1);
    }
  }

  /* ---------------------------------------------------------------------- */
  NField & StateNField::current() { return this->fields[this->indices[0]]; }

  /* ---------------------------------------------------------------------- */
  const NField & StateNField::current() const {
    return this->fields[this->indices[0]];
  }

  /* ---------------------------------------------------------------------- */
  const NField & StateNField::old(size_t nb_steps_ago) const {
    return this->fields[this->indices.at(nb_steps_ago)];
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  TypedStateNField<T>::TypedStateNField(const std::string & unique_prefix,
                                        NFieldCollection & collection,
                                        Dim_t nb_memory, Dim_t nb_components)
      : Parent{unique_prefix, collection, nb_memory} {
    for (Dim_t i{0}; i < nb_memory + 1; ++i) {
      std::stringstream unique_name_stream{};
      unique_name_stream << this->prefix << ", sub_field index " << i;
      this->fields.push_back(
          this->collection.template register_field<T>(
              unique_name_stream.str(), nb_components));
    }
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  const std::type_info & TypedStateNField<T>::get_stored_typeid() const {
    return typeid(T);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  TypedNField<T> & TypedStateNField<T>::current() {
    /*
     * note: this is a downcast, and should usially be done safely with
     * dynamic_cast(). Since the underlying fields have been created by the
     * constructor of this class, we know the theoretically unsafe static_cast
     * to always be valid.
     */
    return static_cast<TypedNField<T> &>(Parent::current());
  }

  template <typename T>
  const TypedNField<T> & TypedStateNField<T>::current() const {
    /*
     * note: this is a downcast, and should usially be done safely with
     * dynamic_cast(). Since the underlying fields have been created by the
     * constructor of this class, we know the theoretically unsafe static_cast
     * to always be valid.
     */
    return static_cast<const TypedNField<T> &>(Parent::current());
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  const TypedNField<T> & TypedStateNField<T>::old(size_t nb_steps_ago) const {
    /*
     * note: this is a downcast, and should usially be done safely with
     * dynamic_cast(). Since the underlying fields have been created by the
     * constructor of this class, we know the theoretically unsafe static_cast
     * to always be valid.
     */
    return static_cast<const TypedNField<T> &>(Parent::old(nb_steps_ago));
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedStateNField<T>::get_fields() -> RefVector<NField> & {
    return this->fields;
  }

  template class TypedStateNField<Real>;
  template class TypedStateNField<Complex>;
  template class TypedStateNField<Int>;
  template class TypedStateNField<Uint>;
}  // namespace muGrid
