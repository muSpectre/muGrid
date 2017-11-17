/**
 * file   field_collection_base.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   05 Nov 2017
 *
 * @brief  Base class for field collections
 *
 * @section LICENCE
 *
 * Copyright (C) 2017 Till Junge
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

#include <map>
#include <vector>
#include "common/common.hh"
#include "common/field.hh"


namespace muSpectre {
  /* ---------------------------------------------------------------------- */
  class FieldCollectionError: public std::runtime_error {
  public:
    explicit FieldCollectionError(const std::string& what)
      :std::runtime_error(what){}
    explicit FieldCollectionError(const char * what)
      :std::runtime_error(what){}
  };

  class FieldError: public FieldCollectionError {
    using Parent = FieldCollectionError;
  public:
    explicit FieldError(const std::string& what)
      :Parent(what){}
    explicit FieldError(const char * what)
      :Parent(what){}
  };
  class FieldInterpretationError: public FieldError
  {
  public:
    explicit FieldInterpretationError(const std::string & what)
      :FieldError(what){}
    explicit FieldInterpretationError(const char * what)
      :FieldError(what){}
  };

  /* ---------------------------------------------------------------------- */
  //! DimS spatial dimension (dimension of problem)
  //! DimM material_dimension (dimension of constitutive law)
  template <Dim_t DimS, Dim_t DimM, class FieldCollectionDerived>
  class FieldCollectionBase
  {
  public:
    using Field = internal::FieldBase<FieldCollectionDerived>;
    using Field_p = std::unique_ptr<Field>;
    using Ccoord = Ccoord_t<DimS>;

    //! Default constructor
    FieldCollectionBase();

    //! Copy constructor
    FieldCollectionBase(const FieldCollectionBase &other) = delete;

    //! Move constructor
    FieldCollectionBase(FieldCollectionBase &&other) noexcept = delete;

    //! Destructor
    virtual ~FieldCollectionBase() noexcept = default;

    //! Copy assignment operator
    FieldCollectionBase& operator=(const FieldCollectionBase &other) = delete;

    //! Move assignment operator
    FieldCollectionBase& operator=(FieldCollectionBase &&other) noexcept = delete;

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

  protected:
    std::map<const std::string, Field_p> fields;
    bool is_initialised = false;
    const Uint id;
    static Uint counter;
    size_t size_ = 0;
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


}  // muSpectre

#endif /* FIELD_COLLECTION_BASE_H */
