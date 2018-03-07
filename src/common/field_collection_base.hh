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

#include <map>
#include <vector>

namespace muSpectre {
  /* ---------------------------------------------------------------------- */
  /** `FieldCollectionBase` is the base class for collections of fields. All
    * fields in a field collection have the same number of pixels. The field
    * collection is templated with @a DimS is the spatial dimension (i.e.
    * whether the simulation domain is one, two or three-dimensional) and
    * @a DimM is the material dimension (i.e., the dimension of constitutive
    * law; even for e.g. two-dimensional problems the constitutive law could
    * live in three-dimensional space for e.g. plane strain or stress problems).
    * All fields within a field collection have a unique string identifier.
    * A `FieldCollectionBase` is therefore comparable to a dictionary of fields
    * that live on the same grid.
    * `FieldCollectionBase` has the specialisations `GlobalFieldCollection` and
    * `LocalFieldCollection`.
    */
  template <Dim_t DimS, Dim_t DimM, class FieldCollectionDerived>
  class FieldCollectionBase
  {
  public:
    //! polymorphic base type to store
    using Field = internal::FieldBase<FieldCollectionDerived>;
    using Field_p = std::unique_ptr<Field>; //!< stored type
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

    //! for return values of iterators
    constexpr inline static Dim_t spatial_dim();

    //! for return values of iterators
    inline Dim_t get_spatial_dim() const;

    //! retrieve field by unique_name
    inline Field& operator[](std::string unique_name);

    //! retrieve field by unique_name with bounds checking
    inline Field& at(std::string unique_name);

    //! returns size of collection, this refers to the number of pixels handled
    //! by the collection, not the number of fields
    inline size_t size() const {return this->size_;}

    //! check whether a field is present
    bool check_field_exists(std::string unique_name);


  protected:
    std::map<const std::string, Field_p> fields{}; //!< contains the field ptrs
    bool is_initialised{false}; //!< to handle double initialisation correctly
    const Uint id; //!< unique identifier
    static Uint counter; //!< used to assign unique identifiers
    size_t size_{0}; //!< holds the number of pixels after initialisation
  private:
  };

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, class FieldCollectionDerived>
  Uint FieldCollectionBase<DimS, DimM, FieldCollectionDerived>::counter{0};

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, class FieldCollectionDerived>
  FieldCollectionBase<DimS, DimM, FieldCollectionDerived>::FieldCollectionBase()
    :id(counter++){}


  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, class FieldCollectionDerived>
  void FieldCollectionBase<DimS, DimM, FieldCollectionDerived>::register_field(Field_p &&field) {
    auto&& search_it = this->fields.find(field->get_name());
    auto&& does_exist = search_it != this->fields.end();
    if (does_exist) {
      std::stringstream err_str;
      err_str << "a field named " << field->get_name()
              << "is already registered in this field collection. "
              << "Currently registered fields: ";
      for (const auto& name_field_pair: this->fields) {
        err_str << ", " << name_field_pair.first;
      }
      throw FieldCollectionError(err_str.str());
    }
    if (this->is_initialised) {
      field->resize(this->size());
    }
    this->fields[field->get_name()] = std::move(field);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, class FieldCollectionDerived>
  constexpr Dim_t FieldCollectionBase<DimS, DimM, FieldCollectionDerived>::
  spatial_dim() {
    return DimS;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, class FieldCollectionDerived>
  Dim_t FieldCollectionBase<DimS, DimM, FieldCollectionDerived>::
  get_spatial_dim() const {
    return DimS;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, class FieldCollectionDerived>
  typename FieldCollectionBase<DimS, DimM, FieldCollectionDerived>::Field&
  FieldCollectionBase<DimS, DimM, FieldCollectionDerived>::
  operator[](std::string unique_name) {
    return *(this->fields[unique_name]);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, class FieldCollectionDerived>
  typename FieldCollectionBase<DimS, DimM, FieldCollectionDerived>::Field&
  FieldCollectionBase<DimS, DimM, FieldCollectionDerived>::
  at(std::string unique_name) {
    return *(this->fields.at(unique_name));
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, class FieldCollectionDerived>
  bool
  FieldCollectionBase<DimS, DimM, FieldCollectionDerived>::
  check_field_exists(std::string unique_name) {
    return this->fields.find(unique_name) != this->fields.end();
  }


}  // muSpectre

#endif /* FIELD_COLLECTION_BASE_H */
