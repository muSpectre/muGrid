/**
 * file   field.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   07 Sep 2017
 *
 * @brief  header-only implementation of a field for field collections
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


#ifndef FIELD_H
#define FIELD_H

#include <string>
#include <utility>
#include <typeinfo>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <type_traits>

namespace muSpectre {

  namespace internal {

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection>
    class FieldBase
    {

    protected:
      //! constructor
      //! unique name (whithin Collection)
      //! number of components
      //! collection to which this field belongs (eg, material, system)
      FieldBase(std::string unique_name,
                size_t nb_components,
                FieldCollection & collection);

    public:

      //! Copy constructor
      FieldBase(const FieldBase &other) = delete;

      //! Move constructor
      FieldBase(FieldBase &&other) noexcept = delete;

      //! Destructor
      virtual ~FieldBase() noexcept = default;

      //! Copy assignment operator
      FieldBase& operator=(const FieldBase &other) = delete;

      //! Move assignment operator
      FieldBase& operator=(FieldBase &&other) noexcept = delete;

      /* ---------------------------------------------------------------------- */
      //!Identifying accessors
      //! return field name
      inline const std::string & get_name() const;
      //! return field type
      //inline const Field_t & get_type() const;
      //! return my collection (for iterating)
      inline const FieldCollection & get_collection() const;
      //! return my collection (for iterating)
      inline const size_t & get_nb_components() const;
      //! return type_id of stored type
      virtual const std::type_info & get_stored_typeid() const = 0;

      virtual size_t size() const = 0;

      //! initialise field to zero (do more complicated initialisations through
      //! fully typed maps)
      virtual void set_zero() = 0;

      //! give access to collections
      friend FieldCollection;

    protected:
      /* ---------------------------------------------------------------------- */
      //! allocate memory etc
      virtual void resize(size_t size) = 0;
      const std::string name;
      const size_t nb_components;
      const FieldCollection & collection;
    private:
    };


    /* ---------------------------------------------------------------------- */
    //! declaraton for friending
    template <class FieldCollection, typename T, Dim_t NbComponents>
    class FieldMap;

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents>
    class TypedFieldBase: public FieldBase<FieldCollection>
    {
      friend class FieldMap<FieldCollection, T, NbComponents>;
    public:
      using Parent = FieldBase<FieldCollection>;
      using Base = Parent;
      //using storage_type = Eigen::Array<T, Eigen::Dynamic, NbComponents>;
      using StoredType = Eigen::Array<T, NbComponents, 1>;
      using StorageType = std::vector<StoredType,
                                       Eigen::aligned_allocator<StoredType>>;
      TypedFieldBase(std::string unique_name,
                     FieldCollection& collection);
      virtual ~TypedFieldBase() = default;
      //! return type_id of stored type
      virtual const std::type_info & get_stored_typeid() const;

      //! initialise field to zero (do more complicated initialisations through
      //! fully typed maps)
      inline void set_zero() override final;
      inline void push_back(const StoredType & value);
      template< class... Args>
      inline void emplace_back(Args&&... args);
      size_t size() const override final;
    protected:
      inline T* get_ptr_to_entry(const size_t&& index);
      inline T& get_ref_to_entry(const size_t&& index);
      inline virtual void resize(size_t size) override final;
      StorageType array;
    };

  }  // internal

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T, Dim_t order, Dim_t dim>
  class TensorField: public internal::TypedFieldBase<FieldCollection,
                                                     T,
                                                     ipow(dim,order)>
  {
  public:
    using Parent = internal::TypedFieldBase<FieldCollection,
                                            T,
                                            ipow(dim,order)>;
    using Base = typename Parent::Base;
    using Field_p = typename FieldCollection::Field_p;
    using component_type = T;
    //! Copy constructor
    TensorField(const TensorField &other) = delete;

    //! Move constructor
    TensorField(TensorField &&other) noexcept = delete;

    //! Destructor
    virtual ~TensorField() noexcept = default;

    //! Copy assignment operator
    TensorField& operator=(const TensorField &other) = delete;

    //! Move assignment operator
    TensorField& operator=(TensorField &&other) noexcept = delete;

    //! accessors
    inline Dim_t get_order() const;
    inline Dim_t get_dim() const;


    //! factory function
    template<class FieldType, class CollectionType, typename... Args>
    friend typename FieldType::Base&  make_field(std::string unique_name,
                             CollectionType & collection,
                             Args&&... args);

  protected:
    //! constructor protected!
    TensorField(std::string unique_name, FieldCollection & collection);

  private:
  };

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T, Dim_t NbRow, Dim_t NbCol=NbRow>
  class MatrixField: public internal::TypedFieldBase<FieldCollection,
                                                     T,
                                                     NbRow*NbCol>
  {
  public:
    using Parent = internal::TypedFieldBase<FieldCollection,
                                            T,
                                            NbRow*NbCol>;
    using Base = typename Parent::Base;
    using Field_p = std::unique_ptr<internal::FieldBase<FieldCollection>>;
    using component_type = T;
    //! Copy constructor
    MatrixField(const MatrixField &other) = delete;

    //! Move constructor
    MatrixField(MatrixField &&other) noexcept = delete;

    //! Destructor
    virtual ~MatrixField() noexcept = default;

    //! Copy assignment operator
    MatrixField& operator=(const MatrixField &other) = delete;

    //! Move assignment operator
    MatrixField& operator=(MatrixField &&other) noexcept = delete;

    //! accessors
    inline Dim_t get_nb_row() const;
    inline Dim_t get_nb_col() const;


    //! factory function
    template<class FieldType, class CollectionType, typename... Args>
    friend typename FieldType::Base&  make_field(std::string unique_name,
                             CollectionType & collection,
                             Args&&... args);

  protected:
    //! constructor protected!
    MatrixField(std::string unique_name, FieldCollection & collection);

  private:
  };

  /* ---------------------------------------------------------------------- */
  //! convenience alias
  template <class FieldCollection, typename T>
  using ScalarField = TensorField<FieldCollection, T, 1, 1>;
  /* ---------------------------------------------------------------------- */
  // Implementations
  namespace internal {
    /* ---------------------------------------------------------------------- */
    template <class FieldCollection>
    FieldBase<FieldCollection>::FieldBase(std::string unique_name,
                                          size_t nb_components_,
                                          FieldCollection & collection_)
      :name(unique_name), nb_components(nb_components_),
       collection(collection_) {}

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection>
    inline const std::string & FieldBase<FieldCollection>::get_name() const {
      return this->name;
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection>
    inline const FieldCollection & FieldBase<FieldCollection>::
    get_collection() const {
      return this->collection;
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection>
    inline const size_t & FieldBase<FieldCollection>::
    get_nb_components() const {
      return this->nb_components;
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents>
    TypedFieldBase<FieldCollection, T, NbComponents>::
    TypedFieldBase(std::string unique_name, FieldCollection & collection)
      :FieldBase<FieldCollection>(unique_name, NbComponents, collection){
      static_assert
        ((std::is_arithmetic<T>::value ||
          std::is_same<T, Complex>::value),
         "Use TypedFieldBase for integer, real or complex scalars for T");
      static_assert(NbComponents > 0,
                    "Only fields with more than 0 components");
    }

    /* ---------------------------------------------------------------------- */
    //! return type_id of stored type
    template <class FieldCollection, typename T, Dim_t NbComponents>
    const std::type_info & TypedFieldBase<FieldCollection, T, NbComponents>::
    get_stored_typeid() const {
      return typeid(T);
    }

    /* ---------------------------------------------------------------------- */
    template<class FieldCollection, typename T, Dim_t NbComponents>
    void
    TypedFieldBase<FieldCollection, T, NbComponents>::
    set_zero() {
      std::fill(this->array.begin(), this->array.end(), T{});
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents>
    size_t TypedFieldBase<FieldCollection, T, NbComponents>::
    size() const {
      return this->array.size();
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents>
    T* TypedFieldBase<FieldCollection, T, NbComponents>::
    get_ptr_to_entry(const size_t&& index) {
      return &this->array[std::move(index)](0, 0);
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents>
    T& TypedFieldBase<FieldCollection, T, NbComponents>::
    get_ref_to_entry(const size_t && index) {
      return this->array[std::move(index)](0, 0);
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents>
    void TypedFieldBase<FieldCollection, T, NbComponents>::
    resize(size_t size) {
      this->array.resize(size);
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents>
    void TypedFieldBase<FieldCollection, T, NbComponents>::
    push_back(const StoredType & value) {
      this->array.push_back(value);
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents>
    template <class... Args>
    void TypedFieldBase<FieldCollection, T, NbComponents>::
    emplace_back(Args&&... args) {
      this->array.emplace_back(std::move(args...));
    }

  }  // internal

  /* ---------------------------------------------------------------------- */
  //! Factory function, guarantees that only fields get created that are
  //! properly registered and linked to a collection.
  template<class FieldType, class FieldCollection, typename... Args>
  typename FieldType::Base &
  make_field(std::string unique_name,
             FieldCollection & collection,
             Args&&... args) {
    auto && ptr = std::unique_ptr<FieldType>{
      new FieldType(unique_name, collection, args...)};
    auto& retref = *ptr;
    collection.register_field(std::move(ptr));
    return retref;
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T, Dim_t order, Dim_t dim>
  TensorField<FieldCollection, T, order, dim>::
  TensorField(std::string unique_name, FieldCollection & collection)
    :Parent(unique_name, collection) {}

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T, Dim_t order, Dim_t dim>
  Dim_t TensorField<FieldCollection, T, order, dim>::
  get_order() const {
    return order;
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T, Dim_t order, Dim_t dim>
  Dim_t TensorField<FieldCollection, T, order, dim>::
  get_dim() const {
    return dim;
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T, Dim_t NbRow, Dim_t NbCol>
  MatrixField<FieldCollection, T, NbRow, NbCol>::
  MatrixField(std::string unique_name, FieldCollection & collection)
    :Parent(unique_name, collection) {}

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T, Dim_t NbRow, Dim_t NbCol>
  Dim_t MatrixField<FieldCollection, T, NbRow, NbCol>::
  get_nb_col() const {
    return NbCol;
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T, Dim_t NbRow, Dim_t NbCol>
  Dim_t MatrixField<FieldCollection, T, NbRow, NbCol>::
  get_nb_row() const {
    return NbRow;
  }

}  // muSpectre
#endif /* FIELD_H */
