/**
 * @file   field_collection_base.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   05 Nov 2017
 *
 * @brief  Base class for field collections
 *
 * Copyright © 2017 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Emacs; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#ifndef FIELD_COLLECTION_BASE_H
#define FIELD_COLLECTION_BASE_H

#include "common/common.hh"
#include "common/field.hh"
#include "common/statefield.hh"

#include <map>
#include <vector>

namespace muSpectre {
  /* ---------------------------------------------------------------------- */
  /** `FieldCollectionBase` is the base class for collections of fields. All
    * fields in a field collection have the same number of pixels. The field
    * collection is templated with @a DimS is the spatial dimension (i.e.
    * whether the simulation domain is one, two or three-dimensional).
    * All fields within a field collection have a unique string identifier.
    * A `FieldCollectionBase` is therefore comparable to a dictionary of fields
    * that live on the same grid.
    * `FieldCollectionBase` has the specialisations `GlobalFieldCollection` and
    * `LocalFieldCollection`.
    */
  template <Dim_t DimS, class FieldCollectionDerived>
  class FieldCollectionBase
  {
  public:
    //! polymorphic base type to store
    using Field_t = internal::FieldBase<FieldCollectionDerived>;
    template<typename T>
    using TypedField_t = TypedField<FieldCollectionDerived, T>;

    using Field_p = std::unique_ptr<Field_t>; //!< stored type
    using StateField_t = StateFieldBase<FieldCollectionDerived>;
    template<typename T>
    using TypedStateField_t = TypedStateField<FieldCollectionDerived, T>;

    using StateField_p = std::unique_ptr<StateField_t>;
    using Ccoord = Ccoord_t<DimS>; //!< cell coordinates type

    //! Default constructor
    FieldCollectionBase();

    //! Copy constructor
    FieldCollectionBase(const FieldCollectionBase &other) = delete;

    //! Move constructor
    FieldCollectionBase(FieldCollectionBase &&other) = delete;

    //! Destructor
    virtual ~FieldCollectionBase() = default;

    //! Copy assignment operator
    FieldCollectionBase& operator=(const FieldCollectionBase &other) = delete;

    //! Move assignment operator
    FieldCollectionBase& operator=(FieldCollectionBase &&other) = delete;

    //! Register a new field (fields need to be in heap, so I want to keep them
    //! as shared pointers
    void register_field(Field_p&& field);

    //! Register a new field (fields need to be in heap, so I want to keep them
    //! as shared pointers
    void register_statefield(StateField_p&& field);

    //! for return values of iterators
    constexpr inline static Dim_t spatial_dim();

    //! for return values of iterators
    inline Dim_t get_spatial_dim() const;

    //! return names of all stored fields
    std::vector<std::string> get_field_names() const {
      std::vector<std::string> names{};
      for (auto & tup: this->fields) {
        names.push_back(std::get<0>(tup));
      }
      return names;
    }

    //! return names of all state fields
    std::vector<std::string> get_statefield_names() const {
      std::vector<std::string> names{};
      for (auto & tup: this->statefields) {
        names.push_back(std::get<0>(tup));
      }
      return names;
    }

    //! retrieve field by unique_name
    inline Field_t& operator[](std::string unique_name);

    //! retrieve field by unique_name with bounds checking
    inline Field_t& at(std::string unique_name);

    //! retrieve typed field by unique_name
    template <typename T>
    inline TypedField_t<T> & get_typed_field(std::string unique_name);

    //! retrieve state field by unique_prefix with bounds checking
    template <typename T>
    inline TypedStateField_t<T>& get_typed_statefield(std::string unique_prefix);

    //! retrieve state field by unique_prefix with bounds checking
    inline StateField_t& get_statefield(std::string unique_prefix) {
      return *(this->statefields.at(unique_prefix));
    }

    //! retrieve state field by unique_prefix with bounds checking
    inline const StateField_t& get_statefield(std::string unique_prefix) const {
      return *(this->statefields.at(unique_prefix));
    }

    /**
     * retrieve current value of typed state field by unique_prefix with
     * bounds checking
     */
    template <typename T>
    inline TypedField_t<T>& get_current(std::string unique_prefix);

    /**
     * retrieve old value of typed state field by unique_prefix with
     * bounds checking
     */
    template<typename T>
    inline const TypedField_t<T> & get_old(std::string unique_prefix,
                                           size_t nb_steps_ago = 1) const;

    //! returns size of collection, this refers to the number of pixels handled
    //! by the collection, not the number of fields
    inline size_t size() const {return this->size_;}

    //! check whether a field is present
    bool check_field_exists(const std::string & unique_name);

    //! check whether the collection is initialised
    bool initialised() const {return this->is_initialised;}


  protected:
    std::map<const std::string, Field_p> fields{}; //!< contains the field ptrs
    //! contains ptrs to state fields
    std::map<const std::string, StateField_p> statefields{};
    bool is_initialised{false}; //!< to handle double initialisation correctly
    const Uint id; //!< unique identifier
    static Uint counter; //!< used to assign unique identifiers
    size_t size_{0}; //!< holds the number of pixels after initialisation
  private:
  };

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, class FieldCollectionDerived>
  Uint FieldCollectionBase<DimS, FieldCollectionDerived>::counter{0};

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, class FieldCollectionDerived>
  FieldCollectionBase<DimS, FieldCollectionDerived>::FieldCollectionBase()
    :id(counter++){}


  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, class FieldCollectionDerived>
  void FieldCollectionBase<DimS, FieldCollectionDerived>::
  register_field(Field_p &&field) {
    auto&& search_it = this->fields.find(field->get_name());
    auto&& does_exist = search_it != this->fields.end();
    if (does_exist) {
      std::stringstream err_str;
      err_str << "a field named '" << field->get_name()
              << "' is already registered in this field collection. "
              << "Currently registered fields: ";
      std::string prelude{""};
      for (const auto& name_field_pair: this->fields) {
        err_str << prelude << '\'' << name_field_pair.first << '\'';
        prelude = ", ";
      }
      throw FieldCollectionError(err_str.str());
    }
    if (this->is_initialised) {
      field->resize(this->size());
    }
    this->fields[field->get_name()] = std::move(field);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, class FieldCollectionDerived>
  void FieldCollectionBase<DimS, FieldCollectionDerived>::
  register_statefield(StateField_p&& field) {
    auto&& search_it = this->statefields.find(field->get_prefix());
    auto&& does_exist = search_it != this->statefields.end();
    if (does_exist) {
      std::stringstream err_str;
      err_str << "a state field named '" << field->get_prefix()
              << "' is already registered in this field collection. "
              << "Currently registered fields: ";
      std::string prelude{""};
      for (const auto& name_field_pair: this->statefields) {
        err_str << prelude << '\'' << name_field_pair.first << '\'';
        prelude = ", ";
      }
      throw FieldCollectionError(err_str.str());
    }
    this->statefields[field->get_prefix()] = std::move(field);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, class FieldCollectionDerived>
  constexpr Dim_t FieldCollectionBase<DimS, FieldCollectionDerived>::
  spatial_dim() {
    return DimS;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, class FieldCollectionDerived>
  Dim_t FieldCollectionBase<DimS, FieldCollectionDerived>::
  get_spatial_dim() const {
    return DimS;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, class FieldCollectionDerived>
  auto
  FieldCollectionBase<DimS, FieldCollectionDerived>::
  operator[](std::string unique_name) -> Field_t & {
    return *(this->fields[unique_name]);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, class FieldCollectionDerived>
  auto
  FieldCollectionBase<DimS, FieldCollectionDerived>::
  at(std::string unique_name) -> Field_t & {
    return *(this->fields.at(unique_name));
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, class FieldCollectionDerived>
  bool
  FieldCollectionBase<DimS, FieldCollectionDerived>::
  check_field_exists(const std::string & unique_name) {
    return this->fields.find(unique_name) != this->fields.end();
  }

  //! retrieve typed field by unique_name
  template <Dim_t DimS, class FieldCollectionDerived>
  template <typename T>
  auto FieldCollectionBase<DimS, FieldCollectionDerived>::
  get_typed_field(std::string unique_name) -> TypedField_t<T> &  {
    auto & unqualified_field{this->at(unique_name)};
    if (unqualified_field.get_stored_typeid().hash_code() !=
        typeid(T).hash_code()) {
      std::stringstream err{};
      err << "Field '" << unique_name << "' is of type "
          << unqualified_field.get_stored_typeid().name()
          << ", but should be of type " << typeid(T).name() << std::endl;
      throw FieldCollectionError(err.str());
    }
    return static_cast<TypedField_t<T> &>(unqualified_field);
  }

  /* ---------------------------------------------------------------------- */
  //! retrieve state field by unique_prefix with bounds checking
  template <Dim_t DimS, class FieldCollectionDerived>
  template <typename T>
  auto FieldCollectionBase<DimS, FieldCollectionDerived>::
  get_typed_statefield(std::string unique_prefix)
    -> TypedStateField_t<T> & {
    auto & unqualified_statefield{this->get_statefield(unique_prefix)};
    if (unqualified_statefield.get_stored_typeid().hash_code() !=
        typeid(T).hash_code()) {
      std::stringstream err{};
      err << "Statefield '" << unique_prefix << "' is of type "
          << unqualified_statefield.get_stored_typeid().name()
          << ", but should be of type " << typeid(T).name() << std::endl;
      throw FieldCollectionError(err.str());
    }
    return static_cast<TypedStateField_t<T> &>(unqualified_statefield);
  }


  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, class FieldCollectionDerived>
  template <typename T>
  auto
  FieldCollectionBase<DimS, FieldCollectionDerived>::
  get_current(std::string unique_prefix) -> TypedField_t<T> & {
    auto & unqualified_statefield = this->get_statefield(unique_prefix);

    //! check for correct underlying fundamental type
    if (unqualified_statefield.get_stored_typeid().hash_code() !=
        typeid(T).hash_code()) {
      std::stringstream err{};
      err << "StateField '" << unique_prefix << "' is of type "
          << unqualified_statefield.get_stored_typeid().name()
          << ", but should be of type " << typeid(T).name() << std::endl;
      throw FieldCollectionError(err.str());
    }

    using Typed_t = TypedStateField<FieldCollectionDerived, T>;
    auto & typed_field{static_cast<Typed_t&>(unqualified_statefield)};
    return typed_field.get_current_field();
  }

  /* ---------------------------------------------------------------------- */
  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, class FieldCollectionDerived>
  template <typename T>
  auto
  FieldCollectionBase<DimS, FieldCollectionDerived>::
  get_old(std::string unique_prefix, size_t nb_steps_ago) const
    -> const TypedField_t<T> & {
    auto & unqualified_statefield = this->get_statefield(unique_prefix);

    //! check for correct underlying fundamental type
    if (unqualified_statefield.get_stored_typeid().hash_code() !=
        typeid(T).hash_code()) {
      std::stringstream err{};
      err << "StateField '" << unique_prefix << "' is of type "
          << unqualified_statefield.get_stored_typeid().name()
          << ", but should be of type " << typeid(T).name() << std::endl;
      throw FieldCollectionError(err.str());
    }

    using Typed_t = TypedStateField<FieldCollectionDerived, T>;
    auto & typed_field{static_cast<const Typed_t&>(unqualified_statefield)};
    return typed_field.get_old_field(nb_steps_ago);
  }

}  // muSpectre

#endif /* FIELD_COLLECTION_BASE_H */
