/**
 * @file   state_field.cc
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

#include "state_field.hh"
#include "field.hh"
#include "field_typed.hh"
#include "field_collection.hh"

#include <sstream>

namespace muGrid {

  /* ---------------------------------------------------------------------- */
  StateField::StateField(const std::string & unique_prefix,
                         FieldCollection & collection,
                         const Index_t & nb_memory,
                         const Index_t & nb_components,
                         const std::string & sub_division_tag,
                         const Unit & unit)
      : prefix{unique_prefix}, collection{collection}, nb_memory{nb_memory},
        nb_components{nb_components},
        sub_division_tag{sub_division_tag}, unit{unit},
        nb_sub_pts{collection.get_nb_sub_pts(sub_division_tag)} {
    if (nb_memory < 1) {
      throw FieldError("State fields must have a memory size of at least 1.");
    }
    this->indices.reserve(nb_memory + 1);
    this->fields.reserve(nb_memory + 1);

    for (Index_t i{0}; i < nb_memory + 1; ++i) {
      indices.push_back((nb_memory + 1 - i) & nb_memory);
    }
  }

  /* ---------------------------------------------------------------------- */
  const Index_t & StateField::get_nb_memory() const { return this->nb_memory; }

  /* ---------------------------------------------------------------------- */
  void StateField::cycle() {
    for (auto & val : this->indices) {
      val = (val + 1) % (this->nb_memory + 1);
    }
  }

  /* ---------------------------------------------------------------------- */
  Field & StateField::current() { return this->fields[this->indices[0]]; }

  /* ---------------------------------------------------------------------- */
  const Field & StateField::current() const {
    return this->fields[this->indices[0]];
  }

  /* ---------------------------------------------------------------------- */
  const Field & StateField::old(const size_t & nb_steps_ago) const {
    return this->fields[this->indices.at(nb_steps_ago)];
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  TypedStateField<T>::TypedStateField(const std::string & unique_prefix,
                                      FieldCollection & collection,
                                      const Index_t & nb_memory,
                                      const Index_t & nb_components,
                                      const std::string & sub_division_tag,
                                      const Unit & unit)
      : Parent{unique_prefix, collection,       nb_memory,
               nb_components, sub_division_tag, unit} {
    for (Index_t i{0}; i < nb_memory + 1; ++i) {
      std::stringstream unique_name_stream{};
      unique_name_stream << this->prefix << ", sub_field index " << i;
      this->fields.push_back(this->collection.template register_field<T>(
          unique_name_stream.str(), nb_components, sub_division_tag, unit));
    }
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  const std::type_info & TypedStateField<T>::get_stored_typeid() const {
    return typeid(T);
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  TypedField<T> & TypedStateField<T>::current() {
    /*
     * note: this is a downcast, and should usially be done safely with
     * dynamic_cast(). Since the underlying fields have been created by the
     * constructor of this class, we know the theoretically unsafe static_cast
     * to always be valid.
     */
    return static_cast<TypedField<T> &>(Parent::current());
  }

  template <typename T>
  const TypedField<T> & TypedStateField<T>::current() const {
    /*
     * note: this is a downcast, and should usially be done safely with
     * dynamic_cast(). Since the underlying fields have been created by the
     * constructor of this class, we know the theoretically unsafe static_cast
     * to always be valid.
     */
    return static_cast<const TypedField<T> &>(Parent::current());
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  const TypedField<T> & TypedStateField<T>::old(size_t nb_steps_ago) const {
    /*
     * note: this is a downcast, and should usially be done safely with
     * dynamic_cast(). Since the underlying fields have been created by the
     * constructor of this class, we know the theoretically unsafe static_cast
     * to always be valid.
     */
    return static_cast<const TypedField<T> &>(Parent::old(nb_steps_ago));
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  auto TypedStateField<T>::get_fields() -> RefVector<Field> & {
    return this->fields;
  }

  template class TypedStateField<Real>;
  template class TypedStateField<Complex>;
  template class TypedStateField<Int>;
  template class TypedStateField<Uint>;
}  // namespace muGrid
