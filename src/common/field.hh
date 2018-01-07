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

#include "common/T4_map_proxy.hh"

#include <Eigen/Dense>

#include <string>
#include <sstream>
#include <utility>
#include <typeinfo>
#include <vector>
#include <algorithm>
#include <cmath>
#include <memory>
#include <type_traits>

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
      friend typename FieldCollection::Parent;

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
      using collection_t = typename Parent::collection_t;
      using Scalar = T;
      using Base = Parent;
      //using storage_type = Eigen::Array<T, Eigen::Dynamic, NbComponents>;
      using StoredType = Eigen::Array<T, NbComponents, 1>;
      using StorageType = std::conditional_t
        <ArrayStore,
         std::vector<StoredType,
                     Eigen::aligned_allocator<StoredType>>,
         std::vector<T,Eigen::aligned_allocator<T>>>;

      using EigenRep = Eigen::Array<T, NbComponents, Eigen::Dynamic>;
      using EigenMap = std::conditional_t<
        ArrayStore,
        Eigen::Map<EigenRep, Eigen::Aligned,
                   Eigen::OuterStride<sizeof(StoredType)/sizeof(T)>>,
        Eigen::Map<EigenRep>>;

      using ConstEigenMap = std::conditional_t<
        ArrayStore,
        Eigen::Map<const EigenRep, Eigen::Aligned,
                   Eigen::OuterStride<sizeof(StoredType)/sizeof(T)>>,
        Eigen::Map<const EigenRep>>;

      TypedFieldBase(std::string unique_name,
                     FieldCollection& collection);
      virtual ~TypedFieldBase() = default;
      //! return type_id of stored type
      virtual const std::type_info & get_stored_typeid() const override final;

      //! initialise field to zero (do more complicated initialisations through
      //! fully typed maps)
      inline void set_zero() override final;
      template <bool isArrayStore = ArrayStore>
      inline void push_back(const
                            std::enable_if_t<isArrayStore, StoredType> &
                            value);
      template <bool componentStore = !ArrayStore>
      inline std::enable_if_t<componentStore> push_back(const StoredType & value);

      //! Number of stored arrays (i.e. total number of stored
      //! scalars/NbComponents)
      size_t size() const override final;

      static TypedFieldBase & check_ref(Parent & other);
      static const TypedFieldBase & check_ref(const Base & parent);

      inline T* data() {return this->get_ptr_to_entry(0);}
      inline const T* data() const {return this->get_ptr_to_entry(0);}

      inline EigenMap eigen();
      inline ConstEigenMap eigen() const;

      template<typename otherT>
      inline Real inner_product(const TypedFieldBase<FieldCollection, otherT, NbComponents,
                                ArrayStore> & other) const;
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
      StorageType values{};
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
    using Scalar = typename Parent::Scalar;
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

    /**
     * Pure convenience functions to get a MatrixFieldMap of
     * appropriate dimensions mapped to this field. You can also
     * create other types of maps, as long as they have the right
     * fundamental type (T) and the correct size (nbComponents).
     */
    decltype(auto) get_map();
    decltype(auto) get_map() const;

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

    decltype(auto) get_map();
    decltype(auto) get_map() const;


protected:
    //! constructor protected!
    MatrixField(std::string unique_name, FieldCollection & collection);

  private:
  };

  /* ---------------------------------------------------------------------- */
  //! convenience alias (
  template <class FieldCollection, typename T>
  using ScalarField = MatrixField<FieldCollection, T, 1, 1>;
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
      std::fill(this->values.begin(), this->values.end(), T{});
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents, bool ArrayStore>
    size_t TypedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
    size() const {
      if (ArrayStore) {
        return this->values.size();
      } else  {
        return this->values.size()/NbComponents;
      }
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents, bool ArrayStore>
    TypedFieldBase<FieldCollection, T, NbComponents, ArrayStore> &
    TypedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
    check_ref(Base & other) {
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
    template <class FieldCollection, typename T, Dim_t NbComponents,
              bool ArrayStore>
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
    template <class FieldCollection, typename T, Dim_t NbComponents,
              bool ArrayStore>
    typename TypedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::EigenMap
    TypedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
    eigen() {
      return EigenMap(this->data(), NbComponents, this->size());
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents,
              bool ArrayStore>
    typename TypedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::ConstEigenMap
    TypedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
    eigen() const {
      return ConstEigenMap(this->data(), NbComponents, this->size());
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents,
              bool ArrayStore>
    template<typename otherT>
    Real
    TypedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
    inner_product(const TypedFieldBase<FieldCollection, otherT, NbComponents,
                  ArrayStore> & other) const {
      return (this->eigen() * other.eigen()).sum();
    }


    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents, bool ArrayStore>
    template <bool isArray>
    std::enable_if_t<isArray, T*>
    TypedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
    get_ptr_to_entry(const size_t&& index) {
      static_assert (isArray == ArrayStore, "SFINAE");
      return &this->values[std::move(index)](0, 0);
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents, bool ArrayStore>
    template <bool noArray>
    T* TypedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
    get_ptr_to_entry(std::enable_if_t<noArray, const size_t&&> index) {
      static_assert (noArray != ArrayStore, "SFINAE");
      return &this->values[NbComponents*std::move(index)];
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents, bool ArrayStore>
    template <bool isArray>
    std::enable_if_t<isArray, const T*>
    TypedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
    get_ptr_to_entry(const size_t&& index) const {
      static_assert (isArray == ArrayStore, "SFINAE");
      return &this->values[std::move(index)](0, 0);
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents, bool ArrayStore>
    template <bool noArray>
    const T* TypedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
    get_ptr_to_entry(std::enable_if_t<noArray, const size_t&&> index) const {
      static_assert (noArray != ArrayStore, "SFINAE");
      return &this->values[NbComponents*std::move(index)];
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents, bool ArrayStore>
    void TypedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
    resize(size_t size) {
      if (ArrayStore) {
        this->values.resize(size);
      } else {
        this->values.resize(size*NbComponents);
      }
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents, bool ArrayStore>
    template <bool isArrayStore>
    void
    TypedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
    push_back(const std::enable_if_t<isArrayStore,StoredType> & value) {
      static_assert(isArrayStore == ArrayStore, "SFINAE");
      this->values.push_back(value);
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents, bool ArrayStore>
    template <bool componentStore>
    std::enable_if_t<componentStore>
    TypedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
    push_back(const StoredType & value) {
      static_assert(componentStore != ArrayStore, "SFINAE");
      for (Dim_t i = 0; i < NbComponents; ++i) {
        this->values.push_back(value(i));
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

#include "common/field_map.hh"

namespace muSpectre {

  namespace internal {

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, size_t order, Dim_t dim,
              bool ConstMap>
    struct tensor_map_type {
    };

    template <class FieldCollection, typename T, Dim_t dim, bool ConstMap>
    struct tensor_map_type<FieldCollection, T, firstOrder, dim, ConstMap> {
      using type = MatrixFieldMap<FieldCollection, T, dim, 1, ConstMap>;
    };

    template <class FieldCollection, typename T, Dim_t dim, bool ConstMap>
    struct tensor_map_type<FieldCollection, T, secondOrder, dim, ConstMap> {
      using type = MatrixFieldMap<FieldCollection, T, dim, dim, ConstMap>;
    };

    template <class FieldCollection, typename T, Dim_t dim, bool ConstMap>
    struct tensor_map_type<FieldCollection, T, fourthOrder, dim, ConstMap> {
      using type = T4MatrixFieldMap<FieldCollection, T, dim, ConstMap>;
    };

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbRow, Dim_t NbCol,
              bool ConstMap>
    struct matrix_map_type {
      using type =
        MatrixFieldMap<FieldCollection, T, NbRow, NbCol, ConstMap>;
    };

    template <class FieldCollection, typename T, bool ConstMap>
    struct matrix_map_type<FieldCollection, T, oneD, oneD, ConstMap> {
      using type = ScalarFieldMap<FieldCollection, T, ConstMap>;
    };

  }  // internal

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T, Dim_t order, Dim_t dim>
  decltype(auto) TensorField<FieldCollection, T, order, dim>::
  get_map() {
    constexpr bool map_constness{false};
    using RawMap_t =
      typename internal::tensor_map_type<FieldCollection, T, order, dim,
                                         map_constness>::type;
    return RawMap_t(*this);
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T, Dim_t order, Dim_t dim>
  decltype(auto) TensorField<FieldCollection, T, order, dim>::
  get_map() const {
    constexpr bool map_constness{true};
    using RawMap_t =
      typename internal::tensor_map_type<FieldCollection, T, order, dim,
                                         map_constness>::type;
    return RawMap_t(*this);
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T, Dim_t NbRow, Dim_t NbCol>
  decltype(auto) MatrixField<FieldCollection, T, NbRow, NbCol>::
  get_map() {
    constexpr bool map_constness{false};
    using RawMap_t =
      typename internal::matrix_map_type<FieldCollection, T, NbRow, NbCol,
                                         map_constness>::type;
    return RawMap_t(*this);
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T, Dim_t NbRow, Dim_t NbCol>
  decltype(auto) MatrixField<FieldCollection, T, NbRow, NbCol>::
  get_map() const {
    constexpr bool map_constness{true};
    using RawMap_t =
      typename internal::matrix_map_type<FieldCollection, T, NbRow, NbCol,
                                         map_constness>::type;
    return RawMap_t(*this);
  }


}  // muSpectre

#endif /* FIELD_H */
