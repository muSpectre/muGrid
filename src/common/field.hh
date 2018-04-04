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
  /**
   * base class for field collection-related exceptions
   */
  class FieldCollectionError: public std::runtime_error {
  public:
    //! constructor
    explicit FieldCollectionError(const std::string& what)
      :std::runtime_error(what){}
    //! constructor
    explicit FieldCollectionError(const char * what)
      :std::runtime_error(what){}
  };

  /// base class for field-related exceptions
  class FieldError: public FieldCollectionError {
    using Parent = FieldCollectionError;
  public:
    //! constructor
    explicit FieldError(const std::string& what)
      :Parent(what){}
    //! constructor
    explicit FieldError(const char * what)
      :Parent(what){}
  };

  /**
   * Thrown when a associating a field map to and incompatible field
   * is attempted
   */
  class FieldInterpretationError: public FieldError
  {
  public:
    //! constructor
    explicit FieldInterpretationError(const std::string & what)
      :FieldError(what){}
    //! constructor
    explicit FieldInterpretationError(const char * what)
      :FieldError(what){}
  };


  namespace internal {

    /* ---------------------------------------------------------------------- */
    /**
     * Virtual base class for all fields. A field represents
     * meta-information for the per-pixel storage for a scalar, vector
     * or tensor quantity and is therefore the abstract class defining
     * the field. It is used for type and size checking at runtime and
     * for storage of polymorphic pointers to fully typed and sized
     * fields. `FieldBase` (and its children) are templated with a
     * specific `FieldCollection` (derived from
     * `muSpectre::FieldCollectionBase`). A `FieldCollection` stores
     * multiple fields that all apply to the same set of
     * pixels. Addressing and managing the data for all pixels is
     * handled by the `FieldCollection`.  Note that `FieldBase` does
     * not know anything about about mathematical operations on the
     * data or how to iterate over all pixels. Mapping the raw data
     * onto for instance Eigen maps and iterating over those is
     * handled by the `FieldMap`.
     */
    template <class FieldCollection>
    class FieldBase
    {

    protected:
      //! constructor
      //! unique name (whithin Collection)
      //! number of components
      //! collection to which this field belongs (eg, material, cell)
      FieldBase(std::string unique_name,
                size_t nb_components,
                FieldCollection & collection);

    public:
      using collection_t = FieldCollection; //!< for type checks

      //! Copy constructor
      FieldBase(const FieldBase &other) = delete;

      //! Move constructor
      FieldBase(FieldBase &&other) = delete;

      //! Destructor
      virtual ~FieldBase() = default;

      //! Copy assignment operator
      FieldBase& operator=(const FieldBase &other) = delete;

      //! Move assignment operator
      FieldBase& operator=(FieldBase &&other) = delete;

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

      //! number of pixels in the field
      virtual size_t size() const = 0;
      
      //! add a pad region to the end of the field buffer; required for
      //! using this as e.g. an FFT workspace
      virtual void set_pad_size(size_t pad_size_) = 0;
      
      //! pad region size
      virtual size_t get_pad_size() const {return this->pad_size;};

      //! initialise field to zero (do more complicated initialisations through
      //! fully typed maps)
      virtual void set_zero() = 0;

      //! give access to collections
      friend FieldCollection;
      //! give access to collection's base class
      friend typename FieldCollection::Parent;

    protected:
      /* ---------------------------------------------------------------------- */
      //! allocate memory etc
      virtual void resize(size_t size) = 0;
      const std::string name; //!< the field's unique name
      const size_t nb_components; //!< number of components per entry
      //! reference to the collection this field belongs to
      const FieldCollection & collection;
      size_t pad_size; //!< size of padding region at end of buffer
    private:
    };


    /**
     * Dummy intermediate class to provide a run-time polymorphic
     * typed field. Mainly for binding Python. TypedFieldBase specifies methods
     * that return typed Eigen maps and vectors in addition to pointers to the
     * raw data.
     */
    template <class FieldCollection, typename T>
    class TypedFieldBase: public FieldBase<FieldCollection>
    {
    public:
      using Parent = FieldBase<FieldCollection>; //!< base class
      //! for type checks when mapping this field
      using collection_t = typename Parent::collection_t;
      using Scalar = T; //!< for type checks
      using Base = Parent; //!< for uniformity of interface
      //! Plain Eigen type to map
      using EigenRep = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>;
      //! map returned when iterating over field
      using EigenMap = Eigen::Map<EigenRep>;
      //! Plain eigen vector to map
      using EigenVec = Eigen::Map<Eigen::VectorXd>;
      //! vector map returned when iterating over field
      using EigenVecConst = Eigen::Map<const Eigen::VectorXd>;
      //! Default constructor
      TypedFieldBase() = delete;

      //! constructor
      TypedFieldBase(std::string unique_name,
                     size_t nb_components,
                     FieldCollection& collection);

      //! Copy constructor
      TypedFieldBase(const TypedFieldBase &other) = delete;

      //! Move constructor
      TypedFieldBase(TypedFieldBase &&other) = delete;

      //! Destructor
      virtual ~TypedFieldBase() = default;

      //! Copy assignment operator
      TypedFieldBase& operator=(const TypedFieldBase &other) = delete;

      //! Move assignment operator
      TypedFieldBase& operator=(TypedFieldBase &&other) = delete;

      //! return type_id of stored type
      virtual const std::type_info & get_stored_typeid() const override final;

      virtual size_t size() const override = 0;

      //! initialise field to zero (do more complicated initialisations through
      //! fully typed maps)
      virtual void set_zero() override = 0;

      //! raw pointer to content (e.g., for Eigen maps)
      virtual T* data() = 0;
      //! raw pointer to content (e.g., for Eigen maps)
      virtual const T* data() const = 0;

      //! return a map representing the entire field as a single `Eigen::Array`
      EigenMap eigen();
      //! return a map representing the entire field as a single Eigen vector
      EigenVec eigenvec();
      //! return a map representing the entire field as a single Eigen vector
      EigenVecConst eigenvec() const;

    protected:
    private:
    };

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
    template <class FieldCollection, typename T, Dim_t NbComponents,
              bool ArrayStore=false>
    class TypedSizedFieldBase: public TypedFieldBase<FieldCollection, T>
    {
      friend class FieldMap<FieldCollection, T, NbComponents, true>;
      friend class FieldMap<FieldCollection, T, NbComponents, false>;
    public:
      //! for compatibility checks
      constexpr static auto nb_components{NbComponents};
      using Parent = TypedFieldBase<FieldCollection, T>; //!< base class
      using Scalar = T; //!< for type checking
      using Base = typename Parent::Base; //!< root base class

      //! type stored if ArrayStore is true
      using StoredType = Eigen::Array<T, NbComponents, 1>;
      //! storage container
      using StorageType = std::conditional_t
        <ArrayStore,
         std::vector<StoredType,
                     Eigen::aligned_allocator<StoredType>>,
         std::vector<T,Eigen::aligned_allocator<T>>>;

      //! Plain type that is being mapped (Eigen lingo)
      using EigenRep = Eigen::Array<T, NbComponents, Eigen::Dynamic>;
      //! maps returned when iterating over field
      using EigenMap = std::conditional_t<
        ArrayStore,
        Eigen::Map<EigenRep, Eigen::Aligned,
                   Eigen::OuterStride<sizeof(StoredType)/sizeof(T)>>,
        Eigen::Map<EigenRep>>;

      //! maps returned when iterating over field
      using ConstEigenMap = std::conditional_t<
        ArrayStore,
        Eigen::Map<const EigenRep, Eigen::Aligned,
                   Eigen::OuterStride<sizeof(StoredType)/sizeof(T)>>,
        Eigen::Map<const EigenRep>>;

      //! constructor
      TypedSizedFieldBase(std::string unique_name,
                     FieldCollection& collection);
      virtual ~TypedSizedFieldBase() = default;

      //! initialise field to zero (do more complicated initialisations through
      //! fully typed maps)
      inline void set_zero() override final;

      //! add a new value at the end of the field
      template <bool isArrayStore = ArrayStore>
      inline void push_back(const
                            std::enable_if_t<isArrayStore, StoredType> &
                            value);

      //! add a new value at the end of the field
      template <bool componentStore = !ArrayStore>
      inline std::enable_if_t<componentStore> push_back(const StoredType & value);

      //! add a new scalar value at the end of the field
      template <bool scalar_store = NbComponents==1>
      inline std::enable_if_t<scalar_store>
      push_back(const T & value);

      //! Number of stored arrays (i.e. total number of stored
      //! scalars/NbComponents)
      size_t size() const override final;

      //! add a pad region to the end of the field buffer; required for
      //! using this as e.g. an FFT workspace
      void set_pad_size(size_t pad_size_) override final;

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

      //! return raw pointer to stored data (necessary for Eigen maps)
      inline T* data() override final {return this->get_ptr_to_entry(0);}
      //! return raw pointer to stored data (necessary for Eigen maps)
      inline const T* data() const override final {return this->get_ptr_to_entry(0);}

      //! return a map representing the entire field as a single `Eigen::Array`
      inline EigenMap eigen();
      //! return a map representing the entire field as a single `Eigen::Array`
      inline ConstEigenMap eigen() const;
      /**
       * return a map representing the entire field as a single
       * dynamically sized `Eigen::Array` (for python bindings)
       */
      inline typename Parent::EigenMap dyn_eigen() {return Parent::eigen();}

      //! inner product between compatible fields
      template<typename otherT>
      inline Real inner_product(const TypedSizedFieldBase<FieldCollection, otherT, NbComponents,
                                ArrayStore> & other) const;
    protected:

      //! returns a raw pointer to the entry, for `Eigen::Map`
      template <bool isArray=ArrayStore>
      inline std::enable_if_t<isArray, T*> get_ptr_to_entry(const size_t&& index);

      //! returns a raw pointer to the entry, for `Eigen::Map`
      template <bool isArray=ArrayStore>
      inline std::enable_if_t<isArray, const T*>
      get_ptr_to_entry(const size_t&& index) const;

      //! returns a raw pointer to the entry, for `Eigen::Map`
      template <bool noArray = !ArrayStore>
      inline T* get_ptr_to_entry(std::enable_if_t<noArray, const size_t&&> index);

      //! returns a raw pointer to the entry, for `Eigen::Map`
      template <bool noArray = !ArrayStore>
      inline const T*
      get_ptr_to_entry(std::enable_if_t<noArray, const size_t&&> index) const;

      //! set the storage size of this field
      inline virtual void resize(size_t size) override final;

      //! The actual storage container
      StorageType values{};
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
                                                     ipow(dim,order)>
  {
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
    template<class FieldType, class CollectionType, typename... Args>
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
    decltype(auto) get_map();
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
    decltype(auto) get_const_map();
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
    decltype(auto) get_map() const;

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
    template<class FieldType, class CollectionType, typename... Args>
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
    decltype(auto) get_map();
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
    decltype(auto) get_const_map();
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
    collection(collection_), pad_size{0} {}

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
    template <class FieldCollection, typename T>
    TypedFieldBase<FieldCollection, T>::
    TypedFieldBase(std::string unique_name, size_t nb_components,
                   FieldCollection & collection)
      :Parent(unique_name, nb_components, collection)
    {}

    /* ---------------------------------------------------------------------- */
    //! return type_id of stored type
    template <class FieldCollection, typename T>
    const std::type_info & TypedFieldBase<FieldCollection, T>::
    get_stored_typeid() const {
      return typeid(T);
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T>
    typename TypedFieldBase<FieldCollection, T>::EigenMap
    TypedFieldBase<FieldCollection, T>::
    eigen() {
      return EigenMap(this->data(), this->get_nb_components(), this->size());
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T>
    typename TypedFieldBase<FieldCollection, T>::EigenVec
    TypedFieldBase<FieldCollection, T>::
    eigenvec() {
      return EigenVec(this->data(), this->get_nb_components() * this->size());
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T>
    typename TypedFieldBase<FieldCollection, T>::EigenVecConst
    TypedFieldBase<FieldCollection, T>::
    eigenvec() const{
      return EigenVecConst(this->data(), this->get_nb_components() * this->size());
    }


    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents, bool ArrayStore>
    TypedSizedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
    TypedSizedFieldBase(std::string unique_name, FieldCollection & collection)
      :Parent(unique_name, NbComponents, collection){
      static_assert
        ((std::is_arithmetic<T>::value ||
          std::is_same<T, Complex>::value),
         "Use TypedSizedFieldBase for integer, real or complex scalars for T");
      static_assert(NbComponents > 0,
                    "Only fields with more than 0 components");
    }

    /* ---------------------------------------------------------------------- */
    template<class FieldCollection, typename T, Dim_t NbComponents, bool ArrayStore>
    void
    TypedSizedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
    set_zero() {
      std::fill(this->values.begin(), this->values.end(), T{});
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents, bool ArrayStore>
    size_t TypedSizedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
    size() const {
      if (ArrayStore) {
        return this->values.size() - this->pad_size;
      } else  {
        return (this->values.size() - this->pad_size)/NbComponents;
      }
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents, bool ArrayStore>
    TypedSizedFieldBase<FieldCollection, T, NbComponents, ArrayStore> &
    TypedSizedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
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
      return static_cast<TypedSizedFieldBase&>(other);
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents,
              bool ArrayStore>
    const TypedSizedFieldBase<FieldCollection, T, NbComponents, ArrayStore> &
    TypedSizedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
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
      return static_cast<const TypedSizedFieldBase&>(other);
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents,
              bool ArrayStore>
    typename TypedSizedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::EigenMap
    TypedSizedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
    eigen() {
      return EigenMap(this->data(), NbComponents, this->size());
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents,
              bool ArrayStore>
    typename TypedSizedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::ConstEigenMap
    TypedSizedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
    eigen() const {
      return ConstEigenMap(this->data(), NbComponents, this->size());
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents,
              bool ArrayStore>
    template<typename otherT>
    Real
    TypedSizedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
    inner_product(const TypedSizedFieldBase<FieldCollection, otherT, NbComponents,
                  ArrayStore> & other) const {
      return (this->eigen() * other.eigen()).sum();
    }


    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents, bool ArrayStore>
    template <bool isArray>
    std::enable_if_t<isArray, T*>
    TypedSizedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
    get_ptr_to_entry(const size_t&& index) {
      static_assert (isArray == ArrayStore, "SFINAE");
      return &this->values[std::move(index)](0, 0);
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents, bool ArrayStore>
    template <bool noArray>
    T* TypedSizedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
    get_ptr_to_entry(std::enable_if_t<noArray, const size_t&&> index) {
      static_assert (noArray != ArrayStore, "SFINAE");
      return &this->values[NbComponents*std::move(index)];
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents, bool ArrayStore>
    template <bool isArray>
    std::enable_if_t<isArray, const T*>
    TypedSizedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
    get_ptr_to_entry(const size_t&& index) const {
      static_assert (isArray == ArrayStore, "SFINAE");
      return &this->values[std::move(index)](0, 0);
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents, bool ArrayStore>
    template <bool noArray>
    const T* TypedSizedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
    get_ptr_to_entry(std::enable_if_t<noArray, const size_t&&> index) const {
      static_assert (noArray != ArrayStore, "SFINAE");
      return &this->values[NbComponents*std::move(index)];
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents, bool ArrayStore>
    void TypedSizedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
    set_pad_size(size_t pad_size) {
      if (ArrayStore) {
        this->values.resize(this->size() + pad_size);
      } else {
        this->values.resize(this->size()*NbComponents + pad_size);
      }
      this->pad_size = pad_size;
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents, bool ArrayStore>
    void TypedSizedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
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
    TypedSizedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
    push_back(const std::enable_if_t<isArrayStore,StoredType> & value) {
      static_assert(isArrayStore == ArrayStore, "SFINAE");
      this->values.push_back(value);
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents, bool ArrayStore>
    template <bool componentStore>
    std::enable_if_t<componentStore>
    TypedSizedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
    push_back(const StoredType & value) {
      static_assert(componentStore != ArrayStore, "SFINAE");
      for (Dim_t i = 0; i < NbComponents; ++i) {
        this->values.push_back(value(i));
      }
    }

    /* ---------------------------------------------------------------------- */
    template <class FieldCollection, typename T, Dim_t NbComponents, bool ArrayStore>
    template <bool scalar_store>
    std::enable_if_t<scalar_store>
    TypedSizedFieldBase<FieldCollection, T, NbComponents, ArrayStore>::
    push_back(const T & value) {
      static_assert(scalar_store, "SFINAE");
      this->values.push_back(value);
    }

  }  // internal

  /* ---------------------------------------------------------------------- */
  //! Factory function, guarantees that only fields get created that are
  //! properly registered and linked to a collection.
  template<class FieldType, class FieldCollection, typename... Args>
  inline FieldType &
  make_field(std::string unique_name,
             FieldCollection & collection,
             Args&&... args) {
    std::unique_ptr<FieldType> ptr{
      new FieldType(unique_name, collection, args...)};
    auto& retref{*ptr};
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
  get_const_map() {
    constexpr bool map_constness{true};
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
  get_const_map() {
    constexpr bool map_constness{true};
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
