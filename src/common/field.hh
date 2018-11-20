/**
 * @file   field.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   07 Sep 2017
 *
 * @brief  header-only implementation of a field for field collections
 *
 * Copyright © 2017 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 */


#ifndef FIELD_H
#define FIELD_H

#include "common/T4_map_proxy.hh"
#include "common/field_typed.hh"

#include <Eigen/Dense>

#include <string>
#include <sstream>
#include <utility>
#include <typeinfo>
#include <vector>
#include <algorithm>
#include <cmath>
#include <type_traits>

namespace muSpectre {

  namespace internal {

    /* ---------------------------------------------------------------------- */
    //! declaraton for friending
    template <class FieldCollection, typename T, Dim_t NbComponents, bool isConst>
    class FieldMap;

    /* ---------------------------------------------------------------------- */
    /**
     * A `TypedSizedFieldBase` is the base class for fields that contain a
     * statically known number of scalars of a statically known type per pixel
     * in a `FieldCollection`. The actual data for all pixels is
     * stored in `TypedSizeFieldBase::values`.
     * `TypedSizedFieldBase` is the base class for `MatrixField` and
     * `TensorField`.
     */
    template <class FieldCollection, typename T, Dim_t NbComponents>
    class TypedSizedFieldBase: public TypedField<FieldCollection, T>
    {
      friend class FieldMap<FieldCollection, T, NbComponents, true>;
      friend class FieldMap<FieldCollection, T, NbComponents, false>;
    public:
      //! for compatibility checks
      constexpr static auto nb_components{NbComponents};
      using Parent = TypedField<FieldCollection, T>; //!< base class
      using Scalar = T; //!< for type checking
      using Base = typename Parent::Base; //!< root base class

      //! storage container
      using Storage_t = typename Parent::Storage_t;

      //! Plain type that is being mapped (Eigen lingo)
      using EigenRep_t = Eigen::Array<T, NbComponents, Eigen::Dynamic>;
      //! maps returned when iterating over field
      using EigenMap_t = Eigen::Map<EigenRep_t>;

      //! maps returned when iterating over field
      using ConstEigenMap_t = Eigen::Map<const EigenRep_t>;

      //! constructor
      TypedSizedFieldBase(std::string unique_name,
                     FieldCollection& collection);
      virtual ~TypedSizedFieldBase() = default;

      //! add a new value at the end of the field
      template <class Derived>
      inline void push_back(const Eigen::DenseBase<Derived> & value);

      //! add a new scalar value at the end of the field
      template <bool scalar_store = NbComponents==1>
      inline std::enable_if_t<scalar_store>
      push_back(const T & value);

      /**
       * returns an upcasted reference to a field, or throws an
       * exception if the field is incompatible
       */
      static TypedSizedFieldBase & check_ref(Base & other);
      /**
       * returns an upcasted reference to a field, or throws an
       * exception if the field is incompatible
       */
      static const TypedSizedFieldBase & check_ref(const Base & other);

      //! return a map representing the entire field as a single `Eigen::Array`
      inline EigenMap_t eigen();
      //! return a map representing the entire field as a single `Eigen::Array`
      inline ConstEigenMap_t eigen() const;
      /**
       * return a map representing the entire field as a single
       * dynamically sized `Eigen::Array` (for python bindings)
       */
      inline typename Parent::EigenMap_t dyn_eigen() {return Parent::eigen();}

      //! inner product between compatible fields
      template <typename T2>
      inline Real inner_product(const TypedSizedFieldBase<
                                FieldCollection, T2, NbComponents> & other) const;

    protected:
      //! returns a raw pointer to the entry, for `Eigen::Map`
      inline T* get_ptr_to_entry(const size_t&& index);

      //! returns a raw pointer to the entry, for `Eigen::Map`
      inline const T*
      get_ptr_to_entry(const size_t&& index) const;

    };

  }  // internal


  /* ---------------------------------------------------------------------- */
  /**
   * The `TensorField` is a subclass of `muSpectre::internal::TypedSizedFieldBase`
   * that represents tensorial fields, i.e. arbitrary-dimensional arrays with
   * identical number of rows/columns (that typically correspond to the spatial
   * cartesian dimensions). It is defined by the stored scalar type @a T, the
   * tensorial order @a order (often also called degree or rank) and the
   * number of spatial dimensions @a dim.
   */
  template <class FieldCollection, typename T, Dim_t order, Dim_t dim>
  class TensorField: public internal::TypedSizedFieldBase<FieldCollection,
                                                     T,
                                                     ipow(dim,order)>  {
  public:
    //! base class
    using Parent = internal::TypedSizedFieldBase<FieldCollection,
                                            T,
                                            ipow(dim,order)>;
    using Base = typename Parent::Base; //!< root base class
    //! polymorphic base class
    using Field_p = typename FieldCollection::Field_p;
    using Scalar = typename Parent::Scalar; //!< for type checking
    //! Copy constructor
    TensorField(const TensorField &other) = delete;

    //! Move constructor
    TensorField(TensorField &&other) = delete;

    //! Destructor
    virtual ~TensorField() = default;

    //! Copy assignment operator
    TensorField& operator=(const TensorField &other) = delete;

    //! Move assignment operator
    TensorField& operator=(TensorField &&other) = delete;

    //! return the order of the stored tensor
    inline Dim_t get_order() const;
    //! return the dimension of the stored tensor
    inline Dim_t get_dim() const;


    //! factory function
    template <class FieldType, class CollectionType, typename... Args>
    friend FieldType& make_field(std::string unique_name,
                                 CollectionType & collection,
                                 Args&&... args);

    //! return a reference or throw an exception if `other` is incompatible
    static TensorField & check_ref(Base & other) {
      return static_cast<TensorField &>(Parent::check_ref(other));}
    //! return a reference or throw an exception if `other` is incompatible
    static const TensorField & check_ref(const Base & other) {
      return static_cast<const TensorField &>(Parent::check_ref(other));}

    /**
     * Convenience functions to return a map onto this field. A map allows
     * iteration over all pixels. The map's iterator returns an object that
     * represents the underlying mathematical structure of the field and
     * implements common linear algebra operations on it.
     * Specifically, this function returns
     * - A `MatrixFieldMap` with @a dim rows and one column if the tensorial
     * order @a order is unity.
     * - A `MatrixFieldMap` with @a dim rows and @a dim columns if the tensorial
     * order @a order is 2.
     * - A `T4MatrixFieldMap` if the tensorial order is 4.
     */
    inline decltype(auto) get_map();
    /**
     * Convenience functions to return a map onto this field. A map allows
     * iteration over all pixels. The map's iterator returns an object that
     * represents the underlying mathematical structure of the field and
     * implements common linear algebra operations on it.
     * Specifically, this function returns
     * - A `MatrixFieldMap` with @a dim rows and one column if the tensorial
     * order @a order is unity.
     * - A `MatrixFieldMap` with @a dim rows and @a dim columns if the tensorial
     * order @a order is 2.
     * - A `T4MatrixFieldMap` if the tensorial order is 4.
     */
    inline decltype(auto) get_const_map();
    /**
     * Convenience functions to return a map onto this field. A map allows
     * iteration over all pixels. The map's iterator returns an object that
     * represents the underlying mathematical structure of the field and
     * implements common linear algebra operations on it.
     * Specifically, this function returns
     * - A `MatrixFieldMap` with @a dim rows and one column if the tensorial
     * order @a order is unity.
     * - A `MatrixFieldMap` with @a dim rows and @a dim columns if the tensorial
     * order @a order is 2.
     * - A `T4MatrixFieldMap` if the tensorial order is 4.
     */
    inline decltype(auto) get_map() const;

    /**
     * creates a `TensorField` same size and type as this, but all
     * entries are zero. Convenience function
     */
    inline TensorField & get_zeros_like(std::string unique_name) const;

  protected:
    //! constructor protected!
    TensorField(std::string unique_name, FieldCollection & collection);

  private:
  };

  /* ---------------------------------------------------------------------- */
  /**
   * The `MatrixField` is subclass of `muSpectre::internal::TypedSizedFieldBase`
   * that represents matrix fields, i.e. a two dimensional arrays, defined by
   * the stored scalar type @a T and the number of rows @a NbRow and columns
   * @a NbCol of the matrix.
   */
  template <class FieldCollection, typename T, Dim_t NbRow, Dim_t NbCol=NbRow>
  class MatrixField: public internal::TypedSizedFieldBase<FieldCollection,
                                                     T,
                                                     NbRow*NbCol>
  {
  public:
    //! base class
    using Parent = internal::TypedSizedFieldBase<FieldCollection,
                                            T,
                                            NbRow*NbCol>;
    using Base = typename Parent::Base; //!< root base class
    //! polymorphic base field ptr to store
    using Field_p = std::unique_ptr<internal::FieldBase<FieldCollection>>;
    //! Copy constructor
    MatrixField(const MatrixField &other) = delete;

    //! Move constructor
    MatrixField(MatrixField &&other) = delete;

    //! Destructor
    virtual ~MatrixField() = default;

    //! Copy assignment operator
    MatrixField& operator=(const MatrixField &other) = delete;

    //! Move assignment operator
    MatrixField& operator=(MatrixField &&other) = delete;

    //! returns the number of rows
    inline Dim_t get_nb_row() const;
    //! returns the number of columns
    inline Dim_t get_nb_col() const;


    //! factory function
    template <class FieldType, class CollectionType, typename... Args>
    friend FieldType&  make_field(std::string unique_name,
                                  CollectionType & collection,
                                  Args&&... args);

    //! returns a `MatrixField` reference if `other` is a compatible field
    static MatrixField & check_ref(Base & other) {
      return static_cast<MatrixField &>(Parent::check_ref(other));}
    //! returns a `MatrixField` reference if `other` is a compatible field
    static const MatrixField & check_ref(const Base & other) {
      return static_cast<const MatrixField &>(Parent::check_ref(other));}

    /**
      * Convenience functions to return a map onto this field. A map allows
      * iteration over all pixels. The map's iterator returns an object that
      * represents the underlying mathematical structure of the field and
      * implements common linear algebra operations on it.
      * Specifically, this function returns
      * - A `ScalarFieldMap` if @a NbRows and @a NbCols are unity.
      * - A `MatrixFieldMap` with @a NbRows rows and @a NbCols columns
      * otherwise.
      */
    inline decltype(auto) get_map();
    /**
     * Convenience functions to return a map onto this field. A map allows
     * iteration over all pixels. The map's iterator returns an object that
     * represents the underlying mathematical structure of the field and
     * implements common linear algebra operations on it.
     * Specifically, this function returns
     * - A `ScalarFieldMap` if @a NbRows and @a NbCols are unity.
     * - A `MatrixFieldMap` with @a NbRows rows and @a NbCols columns
     * otherwise.
     */
    inline decltype(auto) get_const_map();
    /**
     * Convenience functions to return a map onto this field. A map allows
     * iteration over all pixels. The map's iterator returns an object that
     * represents the underlying mathematical structure of the field and
     * implements common linear algebra operations on it.
     * Specifically, this function returns
     * - A `ScalarFieldMap` if @a NbRows and @a NbCols are unity.
     * - A `MatrixFieldMap` with @a NbRows rows and @a NbCols columns
     * otherwise.
     */
    inline decltype(auto) get_map() const;

    /**
     * creates a `MatrixField` same size and type as this, but all
     * entries are zero. Convenience function
     */
    inline MatrixField & get_zeros_like(std::string unique_name) const;

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
  /* ---------------------------------------------------------------------- */
  namespace internal {

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents>
    TypedSizedFieldBase<FieldCollection, T, NbComponents>::
    TypedSizedFieldBase(std::string unique_name, FieldCollection & collection)
      :Parent(unique_name, collection, NbComponents){
      static_assert
        ((std::is_arithmetic<T>::value ||
          std::is_same<T, Complex>::value),
         "Use TypedSizedFieldBase for integer, real or complex scalars for T");
      static_assert(NbComponents > 0,
                    "Only fields with more than 0 components");
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents>
    TypedSizedFieldBase<FieldCollection, T, NbComponents> &
    TypedSizedFieldBase<FieldCollection, T, NbComponents>::
    check_ref(Base & other) {
      if (typeid(T).hash_code() != other.get_stored_typeid().hash_code()) {
        std::stringstream err_str{};
        err_str << "Cannot create a reference of type '" << typeid(T).name()
                << "' for field '" << other.get_name() << "' of type '"
                << other.get_stored_typeid().name() << "'";
      throw std::runtime_error(err_str.str());
      }
      //check size compatibility
      if (NbComponents != other.get_nb_components()) {
        std::stringstream err_str{};
        err_str << "Cannot create a reference to a field with "
                << NbComponents << " components "
                << "for field '" << other.get_name() << "' with "
                << other.get_nb_components() << " components";
        throw std::runtime_error{err_str.str()};
      }
      return static_cast<TypedSizedFieldBase&>(other);
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents>
    const TypedSizedFieldBase<FieldCollection, T, NbComponents> &
    TypedSizedFieldBase<FieldCollection, T, NbComponents>::
    check_ref(const Base & other) {
      if (typeid(T).hash_code() != other.get_stored_typeid().hash_code()) {
        std::stringstream err_str{};
        err_str << "Cannot create a reference of type '" << typeid(T).name()
                << "' for field '"  << other.get_name() << "' of type '"
                << other.get_stored_typeid().name() << "'";
        throw std::runtime_error
          (err_str.str());
      }
      //check size compatibility
      if (NbComponents != other.get_nb_components()) {
        std::stringstream err_str{};
        err_str << "Cannot create a reference toy a field with "
                << NbComponents << " components "
                << "for field '" << other.get_name() << "' with "
                << other.get_nb_components() << " components";
        throw std::runtime_error{err_str.str()};
      }
      return static_cast<const TypedSizedFieldBase&>(other);
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents>
    auto TypedSizedFieldBase<FieldCollection, T, NbComponents>::
    eigen() -> EigenMap_t{
      return EigenMap_t(this->data(), NbComponents, this->size());
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents>
    auto TypedSizedFieldBase<FieldCollection, T, NbComponents>::
    eigen() const -> ConstEigenMap_t{
      return ConstEigenMap_t(this->data(), NbComponents, this->size());
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents>
    template <typename otherT>
    Real
    TypedSizedFieldBase<FieldCollection, T, NbComponents>::
    inner_product(const TypedSizedFieldBase<FieldCollection, otherT, NbComponents> & other) const {
      return (this->eigen() * other.eigen()).sum();
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents>
    T* TypedSizedFieldBase<FieldCollection, T, NbComponents>::
    get_ptr_to_entry(const size_t&& index) {
      return this->data_ptr + NbComponents*std::move(index);
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents>
    const T* TypedSizedFieldBase<FieldCollection, T, NbComponents>::
    get_ptr_to_entry(const size_t&& index) const {
      return this->data_ptr + NbComponents*std::move(index);
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents>
    template <class Derived>
    void TypedSizedFieldBase<FieldCollection, T, NbComponents>::
    push_back(const Eigen::DenseBase<Derived> & value) {
      static_assert(Derived::SizeAtCompileTime == NbComponents,
                    "You provided an array with the wrong number of entries.");
      static_assert((Derived::RowsAtCompileTime == 1) or
                    (Derived::ColsAtCompileTime == 1),
                    "You have not provided a column or row vector.");
      static_assert (not FieldCollection::Global,
                     "You can only push_back data into local field "
                     "collections.");
      for (Dim_t i = 0; i < NbComponents; ++i) {
        this->values.push_back(value(i));
      }
      ++this->current_size;
      this->data_ptr = &this->values.front();
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents>
    template <bool scalar_store>
    std::enable_if_t<scalar_store>
    TypedSizedFieldBase<FieldCollection, T, NbComponents>::
    push_back(const T & value) {
      static_assert(scalar_store, "SFINAE");
      this->values.push_back(value);
      ++this->current_size;
      this->data_ptr = &this->values.front();
    }

  }  // internal

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
    /**
     * defines the default mapped type obtained when calling
     * `muSpectre::TensorField::get_map()`
     */
    template <class FieldCollection, typename T, size_t order, Dim_t dim,
              bool ConstMap>
    struct tensor_map_type {
    };

    /// specialisation for vectors
    template <class FieldCollection, typename T, Dim_t dim, bool ConstMap>
    struct tensor_map_type<FieldCollection, T, firstOrder, dim, ConstMap> {
      //! use this type
      using type = MatrixFieldMap<FieldCollection, T, dim, 1, ConstMap>;
    };

    /// specialisation to second-order tensors (matrices)
    template <class FieldCollection, typename T, Dim_t dim, bool ConstMap>
    struct tensor_map_type<FieldCollection, T, secondOrder, dim, ConstMap> {
      //! use this type
      using type = MatrixFieldMap<FieldCollection, T, dim, dim, ConstMap>;
    };

    /// specialisation to fourth-order tensors
    template <class FieldCollection, typename T, Dim_t dim, bool ConstMap>
    struct tensor_map_type<FieldCollection, T, fourthOrder, dim, ConstMap> {
      //! use this type
      using type = T4MatrixFieldMap<FieldCollection, T, dim, ConstMap>;
    };

    /* ---------------------------------------------------------------------- */
    /**
     * defines the default mapped type obtained when calling
     * `muSpectre::MatrixField::get_map()`
     */
    template <class FieldCollection, typename T, Dim_t NbRow, Dim_t NbCol,
              bool ConstMap>
    struct matrix_map_type {
      //! mapping type
      using type =
        MatrixFieldMap<FieldCollection, T, NbRow, NbCol, ConstMap>;
    };

    //! specialisation to scalar fields
    template <class FieldCollection, typename T, bool ConstMap>
    struct matrix_map_type<FieldCollection, T, oneD, oneD, ConstMap> {
      //! mapping type
      using type = ScalarFieldMap<FieldCollection, T, ConstMap>;
    };

  }  // internal

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T, Dim_t order, Dim_t dim>
  auto TensorField<FieldCollection, T, order, dim>::
  get_map() -> decltype(auto) {
    constexpr bool map_constness{false};
    using RawMap_t =
      typename internal::tensor_map_type<FieldCollection, T, order, dim,
                                         map_constness>::type;
    return RawMap_t(*this);
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T, Dim_t order, Dim_t dim>
  auto TensorField<FieldCollection, T, order, dim>::
  get_const_map() -> decltype(auto) {
    constexpr bool map_constness{true};
    using RawMap_t =
      typename internal::tensor_map_type<FieldCollection, T, order, dim,
                                         map_constness>::type;
    return RawMap_t(*this);
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T, Dim_t order, Dim_t dim>
  auto TensorField<FieldCollection, T, order, dim>::
  get_map() const -> decltype(auto) {
    constexpr bool map_constness{true};
    using RawMap_t =
      typename internal::tensor_map_type<FieldCollection, T, order, dim,
                                         map_constness>::type;
    return RawMap_t(*this);
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T, Dim_t order, Dim_t dim>
  auto TensorField<FieldCollection, T, order, dim>::
  get_zeros_like(std::string unique_name) const -> TensorField & {
    return make_field<TensorField>(unique_name,
                                   this->collection);
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T, Dim_t NbRow, Dim_t NbCol>
  auto MatrixField<FieldCollection, T, NbRow, NbCol>::
  get_map() -> decltype(auto) {
    constexpr bool map_constness{false};
    using RawMap_t =
      typename internal::matrix_map_type<FieldCollection, T, NbRow, NbCol,
                                         map_constness>::type;
    return RawMap_t(*this);
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T, Dim_t NbRow, Dim_t NbCol>
  auto MatrixField<FieldCollection, T, NbRow, NbCol>::
  get_const_map() -> decltype(auto) {
    constexpr bool map_constness{true};
    using RawMap_t =
      typename internal::matrix_map_type<FieldCollection, T, NbRow, NbCol,
                                         map_constness>::type;
    return RawMap_t(*this);
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T, Dim_t NbRow, Dim_t NbCol>
  auto MatrixField<FieldCollection, T, NbRow, NbCol>::
  get_map() const -> decltype(auto) {
    constexpr bool map_constness{true};
    using RawMap_t =
      typename internal::matrix_map_type<FieldCollection, T, NbRow, NbCol,
                                         map_constness>::type;
    return RawMap_t(*this);
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T, Dim_t order, Dim_t dim>
  auto MatrixField<FieldCollection, T, order, dim>::
  get_zeros_like(std::string unique_name) const -> MatrixField & {
    return make_field<MatrixField>(unique_name,
                                   this->collection);
  }



}  // muSpectre

#endif /* FIELD_H */
