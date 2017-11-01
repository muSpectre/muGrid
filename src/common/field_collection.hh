/**
 * file   field_collection.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   07 Sep 2017
 *
 * @brief  Provides pixel-iterable containers for scalar and tensorial fields,
 *         addressable by field name
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

#ifndef FIELD_COLLECTION_H
#define FIELD_COLLECTION_H

#include "common/common.hh"
#include "common/ccoord_operations.hh"
#include "common/field.hh"
#include <memory>
#include <exception>
#include <string>
#include <sstream>
#include <map>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace muSpectre {

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
  //! DimS spatial dimension (dimension of problem
  //! DimM material_dimension (dimension of constitutive law)
  //! Global determines whether this field is present everywhere or per material
  template <Dim_t DimS, Dim_t DimM, bool Global = true>
  class FieldCollection
  {
  public:
    using Field = internal::FieldBase<FieldCollection>;
    using Field_p = std::unique_ptr<Field>;
    using Ccoord = Ccoord_t<DimS>;

    //! Default constructor
    FieldCollection();

    //! Copy constructor
    FieldCollection(const FieldCollection &other) = delete;

    //! Move constructor
    FieldCollection(FieldCollection &&other) noexcept = delete;

    //! Destructor
    virtual ~FieldCollection() noexcept = default;

    //! Copy assignment operator
    FieldCollection& operator=(const FieldCollection &other) = delete;

    //! Move assignment operator
    FieldCollection& operator=(FieldCollection &&other) noexcept = delete;

    //! add a pixel/voxel to the field collection
    template <bool NotGlobal = !Global>
    inline std::enable_if_t<NotGlobal> add_pixel(const Ccoord & local_ccoord);

    //! allocate memory, etc
    template <bool NotGlobal = !Global>
    inline std::enable_if_t<NotGlobal> initialise();
    template <bool isGlobal = Global>
    inline std::enable_if_t<isGlobal> initialise(Ccoord sizes);

    //! Register a new field (fields need to be in heap, so I want to keep them
    //! yas shared pointers
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

    //! return the pixel sizes
    template <bool isGlobal = Global>
    inline const std::enable_if_t<isGlobal, Ccoord> & get_sizes() const;

    //! returns the linear index corresponding to cell coordinates
    inline size_t get_index(Ccoord && ccoord) const;
    //! returns the cell coordinates corresponding to a linear index
    inline Ccoord get_ccoord(size_t index) const;

  protected:
    std::map<const std::string, Field_p> fields;
    bool is_initialised = false;
    const Uint id;
    static Uint counter;
    size_t size_ = 0;
    Ccoord sizes;
    //! container of pixel coords for non-global collections
    std::vector<Ccoord> ccoords;
    //! container of indices for non-global collections (slow!)
    std::map<Ccoord, size_t> indices;
  private:
  };

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, bool Global>
  Uint FieldCollection<DimS, DimM, Global>::counter{0};

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, bool Global>
  FieldCollection<DimS, DimM, Global>::FieldCollection()
    :id(counter++){}

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, bool Global>
  template <bool NotGlobal>
  std::enable_if_t<NotGlobal> FieldCollection<DimS, DimM, Global>::
  add_pixel(const Ccoord & local_ccoord) {
    this->indices[local_ccoord] = this->ccoords.size();
    this->ccoords.push_back(local_ccoord);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, bool Global>
  template <bool NotGlobal>
  std::enable_if_t<NotGlobal> FieldCollection<DimS, DimM, Global>::
  initialise() {
    this->size_ = this->ccoords.size();
    std::for_each(std::begin(this->fields), std::end(this->fields),
                  [this](auto && item) {
                    item.second->initialise(this->size());
                  });
    this->is_initialised = true;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, bool Global>
  template <bool isGlobal>
  std::enable_if_t<isGlobal> FieldCollection<DimS, DimM, Global>::
  initialise(Ccoord sizes) {
    this->size_ = std::accumulate(sizes.begin(), sizes.end(), 1,
                                   std::multiplies<Dim_t>());
    this->sizes = sizes;
    std::for_each(std::begin(this->fields), std::end(this->fields),
                  [this](auto && item) {
                    item.second->initialise(this->size());
                  });
    this->is_initialised = true;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, bool Global>
  void FieldCollection<DimS, DimM, Global>::register_field(Field_p &&field) {
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
  template <Dim_t DimS, Dim_t DimM, bool Global>
  constexpr Dim_t FieldCollection<DimS, DimM, Global>::spatial_dim() {
    return DimS;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, bool Global>
  Dim_t FieldCollection<DimS, DimM, Global>::get_spatial_dim() const {
    return DimS;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, bool Global>
  typename FieldCollection<DimS, DimM, Global>::Field&
  FieldCollection<DimS, DimM, Global>::operator[](std::string unique_name) {
    return *(this->fields[unique_name]);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM, bool Global>
  typename FieldCollection<DimS, DimM, Global>::Field&
  FieldCollection<DimS, DimM, Global>::at(std::string unique_name) {
    return *(this->fields.at(unique_name));
  }

  //----------------------------------------------------------------------------//
  //! return the pixel sizes
  template <Dim_t DimS, Dim_t DimM, bool Global>
  template <bool isGlobal>
  const std::enable_if_t<isGlobal, Ccoord_t<DimS>>&
  FieldCollection<DimS, DimM, Global>::get_sizes() const {
    return this->sizes;
  }

  //----------------------------------------------------------------------------//
  //! returns the linear index corresponding to cell coordinates
  template <Dim_t DimS, Dim_t DimM, bool Global>
  size_t
  FieldCollection<DimS, DimM, Global>::get_index(Ccoord && ccoord) const {
    return CcoordOps::get_index(this->sizes, std::move(ccoord));
  }

  //----------------------------------------------------------------------------//
  //! returns the cell coordinates corresponding to a linear index
  template <Dim_t DimS, Dim_t DimM, bool Global>
  typename FieldCollection<DimS, DimM, Global>::Ccoord
  FieldCollection<DimS, DimM, Global>::get_ccoord(size_t index) const {
    return CcoordOps::get_ccoord(this->sizes, std::move(index));
  }


}  // muSpectre

#endif /* FIELD_COLLECTION_H */
