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
#include <sstream>
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
      using collection_t = FieldCollection;

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
    template <class FieldCollection, typename T, Dim_t NbComponents, bool isConst>
    class FieldMap;

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents,
              bool ArrayStore=false>
    class TypedFieldBase: public FieldBase<FieldCollection>
    {
      friend class FieldMap<FieldCollection, T, NbComponents, true>;
      friend class FieldMap<FieldCollection, T, NbComponents, false>;
    public:
      constexpr static auto nb_components{NbComponents};
      using Parent = FieldBase<FieldCollection>;
      using Parent::collection_t;
      using Scalar = T;
      using Base = Parent;
      //using storage_type = Eigen::Array<T, Eigen::Dynamic, NbComponents>;
      using StoredType = Eigen::Array<T, NbComponents, 1>;
      using StorageType = std::conditional_t
        <ArrayStore,
         std::vector<StoredType,
                     Eigen::aligned_allocator<StoredType>>,
         std::vector<T>>;
      TypedFieldBase(std::string unique_name,
                     FieldCollection& collection);
      virtual ~TypedFieldBase() = default;
      //! return type_id of stored type
      virtual const std::type_info & get_stored_typeid() const;

      //! initialise field to zero (do more complicated initialisations through
      //! fully typed maps)
      inline void set_zero() override final;
      template <bool isArrayStore = ArrayStore>
      inline void push_back(const
                            std::enable_if_t<isArrayStore, StoredType> &
                            value);
      template <bool componentStore = !ArrayStore>
      inline std::enable_if_t<componentStore> push_back(const StoredType & value);
      size_t size() const override final;

      static TypedFieldBase & check_ref(Parent & other);
      static const TypedFieldBase & check_ref(const Base & parent);
    protected:

      template <bool isArray=ArrayStore>
      inline std::enable_if_t<isArray, T*> get_ptr_to_entry(const size_t&& index);

      template <bool isArray=ArrayStore>
      inline std::enable_if_t<isArray, const T*>
      get_ptr_to_entry(const size_t&& index) const;

      template <bool noArray = !ArrayStore>
      inline T* get_ptr_to_entry(std::enable_if_t<noArray, const size_t&&> index);

      template <bool noArray = !ArrayStore>
      inline const T*
      get_ptr_to_entry(std::enable_if_t<noArray, const size_t&&> index) const;

      inline virtual void resize(size_t size) override final;
      StorageType array{};
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
    friend FieldType& make_field(std::string unique_name,
                             CollectionType & collection,
                             Args&&... args);

    static TensorField & check_ref(Base & other) {
      return static_cast<TensorField &>(Parent::check_ref(other));}
    static const TensorField & check_ref(const Base & other) {
      return static_cast<const TensorField &>(Parent::check_ref(other));}

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
    friend FieldType&  make_field(std::string unique_name,
                             CollectionType & collection,
                             Args&&... args);

    static MatrixField & check_ref(Base & other) {
      return static_cast<MatrixField &>(Parent::check_ref(other));}
    static const MatrixField & check_ref(const Base & other) {
      return static_cast<const MatrixField &>(Parent::check_ref(other));}

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
    template <class FieldCollection, typename T, Dim_t NbComponents, bool ArrayStore>
    TypedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
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
    template <class FieldCollection, typename T, Dim_t NbComponents, bool ArrayStore>
    const std::type_info & TypedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
    get_stored_typeid() const {
      return typeid(T);
    }

    /* ---------------------------------------------------------------------- */
    template<class FieldCollection, typename T, Dim_t NbComponents, bool ArrayStore>
    void
    TypedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
    set_zero() {
      std::fill(this->array.begin(), this->array.end(), T{});
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents, bool ArrayStore>
    size_t TypedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
    size() const {
      if (ArrayStore) {
        return this->array.size();
      } else  {
        return this->array.size()/NbComponents;
      }
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents, bool ArrayStore>
    TypedFieldBase<FieldCollection, T, NbComponents, ArrayStore> &
    TypedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::check_ref(Base & other) {
      if (typeid(T).hash_code() != other.get_stored_typeid().hash_code()) {
        std::string err ="Cannot create a Reference of requested type " +(
           "for field '" + other.get_name() + "' of type '" +
           other.get_stored_typeid().name() + "'");
        throw std::runtime_error
          (err);
      }
      //check size compatibility
      if (NbComponents != other.get_nb_components()) {
        throw std::runtime_error
          ("Cannot create a Reference to a field with " +
           std::to_string(NbComponents) + " components " +
           "for field '" + other.get_name() + "' with " +
           std::to_string(other.get_nb_components()) + " components");
      }
      return static_cast<TypedFieldBase&>(other);
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents, bool ArrayStore>
    const TypedFieldBase<FieldCollection, T, NbComponents, ArrayStore> &
    TypedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
    check_ref(const Base & other) {
      if (typeid(T).hash_code() != other.get_stored_typeid().hash_code()) {
        std::stringstream err_str{};
        err_str << "Cannot create a Reference of requested type "
                << "for field '"  << other.get_name() << "' of type '"
                << other.get_stored_typeid().name() << "'";
        throw std::runtime_error
          (err_str.str());
      }
      //check size compatibility
      if (NbComponents != other.get_nb_components()) {
        throw std::runtime_error
          ("Cannot create a Reference to a field with " +
           std::to_string(NbComponents) + " components " +
           "for field '" + other.get_name() + "' with " +
           std::to_string(other.get_nb_components()) + " components");
      }
      return static_cast<const TypedFieldBase&>(other);
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents, bool ArrayStore>
    template <bool isArray>
    std::enable_if_t<isArray, T*>
    TypedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
    get_ptr_to_entry(const size_t&& index) {
      static_assert (isArray == ArrayStore, "SFINAE");
      return &this->array[std::move(index)](0, 0);
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents, bool ArrayStore>
    template <bool noArray>
    T* TypedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
    get_ptr_to_entry(std::enable_if_t<noArray, const size_t&&> index) {
      static_assert (noArray != ArrayStore, "SFINAE");
      return &this->array[NbComponents*std::move(index)];
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents, bool ArrayStore>
    template <bool isArray>
    std::enable_if_t<isArray, const T*>
    TypedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
    get_ptr_to_entry(const size_t&& index) const {
      static_assert (isArray == ArrayStore, "SFINAE");
      return &this->array[std::move(index)](0, 0);
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents, bool ArrayStore>
    template <bool noArray>
    const T* TypedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
    get_ptr_to_entry(std::enable_if_t<noArray, const size_t&&> index) const {
      static_assert (noArray != ArrayStore, "SFINAE");
      return &this->array[NbComponents*std::move(index)];
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents, bool ArrayStore>
    void TypedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
    resize(size_t size) {
      if (ArrayStore) {
        this->array.resize(size);
      } else {
        this->array.resize(size*NbComponents);
      }
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents, bool ArrayStore>
    template <bool isArrayStore>
    void
    TypedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
    push_back(const std::enable_if_t<isArrayStore,StoredType> & value) {
      static_assert(isArrayStore == ArrayStore, "SFINAE");
      this->array.push_back(value);
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents, bool ArrayStore>
    template <bool componentStore>
    std::enable_if_t<componentStore>
    TypedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
    push_back(const StoredType & value) {
      static_assert(componentStore != ArrayStore, "SFINAE");
      for (Dim_t i = 0; i < NbComponents; ++i) {
        this->array.push_back(value(i));
      }
    }

  }  // internal

  /* ---------------------------------------------------------------------- */
  //! Factory function, guarantees that only fields get created that are
  //! properly registered and linked to a collection.
  template<class FieldType, class FieldCollection, typename... Args>
  FieldType &
  make_field(std::string unique_name,
             FieldCollection & collection,
             Args&&... args) {
    auto ptr = std::unique_ptr<FieldType>{
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
