/**
 * file   field_typed.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   10 Apr 2018
 *
 * @brief  Typed Field for dynamically sized fields and base class for fields
 *         of tensors, matrices, etc
 *
 * Copyright © 2018 Till Junge
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

#ifndef FIELD_TYPED_H
#define FIELD_TYPED_H

#include "common/field_base.hh"
#include "common/field_helpers.hh"

#include <sstream>

namespace muSpectre {

  /**
   * forward-declaration
   */
  template <class FieldCollection, typename T, bool ConstField>
  class TypedFieldMap;

  namespace internal {

    /* ---------------------------------------------------------------------- */
    //! declaraton for friending
    template <class FieldCollection, typename T, Dim_t NbComponents, bool isConst>
    class FieldMap;

  }  // internal


  /**
   * Dummy intermediate class to provide a run-time polymorphic
   * typed field. Mainly for binding Python. TypedField specifies methods
   * that return typed Eigen maps and vectors in addition to pointers to the
   * raw data.
   */
  template <class FieldCollection, typename T>
  class TypedField: public internal::FieldBase<FieldCollection>
  {
    friend class internal::FieldMap<FieldCollection, T, Eigen::Dynamic, true>;
    friend class internal::FieldMap<FieldCollection, T, Eigen::Dynamic, false>;

    static constexpr bool Global{FieldCollection::is_global()};

  public:
    using Parent = internal::FieldBase<FieldCollection>; //!< base class
    //! for type checks when mapping this field
    using collection_t = typename Parent::collection_t;

    //! for filling global fields from local fields and vice-versa
    using LocalField_t =
      std::conditional_t<Global,
                         TypedField<typename
                                    FieldCollection::LocalFieldCollection_t, T>,
                         TypedField>;
    //! for filling global fields from local fields and vice-versa
    using GlobalField_t =
      std::conditional_t<Global,
                         TypedField,
                         TypedField<typename
                                    FieldCollection::GlobalFieldCollection_t, T>>;

    using Scalar = T; //!< for type checks
    using Base = Parent; //!< for uniformity of interface
    //! Plain Eigen type to map
    using EigenRep_t = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>;
    using EigenVecRep_t = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    //! map returned when accessing entire field
    using EigenMap_t = Eigen::Map<EigenRep_t>;
    //! map returned when accessing entire const field
    using EigenMapConst_t = Eigen::Map<const EigenRep_t>;
    //! Plain eigen vector to map
    using EigenVec_t = Eigen::Map<EigenVecRep_t>;
    //! vector map returned when accessing entire field
    using EigenVecConst_t = Eigen::Map<const EigenVecRep_t>;
    //! associated non-const field map
    using FieldMap_t = TypedFieldMap<FieldCollection, T, false>;
    //! associated const field map
    using ConstFieldMap_t = TypedFieldMap<FieldCollection, T, true>;

    /**
     * type stored (unfortunately, we can't statically size the second
     * dimension due to an Eigen bug, i.e., creating a row vector
     * reference to a column vector does not raise an error :(
     */
    using Stored_t = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>;
    //! storage container
    using Storage_t = std::vector<T,Eigen::aligned_allocator<T>>;

    //! Default constructor
    TypedField() = delete;

    //! constructor
    TypedField(std::string unique_name,
               FieldCollection& collection,
               size_t nb_components);

    /**
     * constructor for field proxies which piggy-back on existing
     * memory. These cannot be registered in field collections and
     * should only be used for transient temporaries
     */
    TypedField(std::string unique_name,
               FieldCollection& collection,
               Eigen::Ref<Eigen::Matrix<Real, Eigen::Dynamic, 1>> vec,
               size_t nb_components);

    //! Copy constructor
    TypedField(const TypedField &other) = delete;

    //! Move constructor
    TypedField(TypedField &&other) = default;

    //! Destructor
    virtual ~TypedField() = default;

    //! Copy assignment operator
    TypedField& operator=(const TypedField &other) = delete;

    //! Move assignment operator
    TypedField& operator=(TypedField &&other) = delete;

    //! return type_id of stored type
    virtual const std::type_info & get_stored_typeid() const override final;

    //! safe reference cast
    static TypedField & check_ref(Base & other);
    //! safe reference cast
    static const TypedField & check_ref(const Base & other);

    virtual size_t size() const override final;

    //! add a pad region to the end of the field buffer; required for
    //! using this as e.g. an FFT workspace
    void set_pad_size(size_t pad_size_) override final;

    //! initialise field to zero (do more complicated initialisations through
    //! fully typed maps)
    virtual void set_zero() override final;

    //! add a new value at the end of the field
    template <class Derived>
    inline void push_back(const Eigen::DenseBase<Derived> & value);


    //! raw pointer to content (e.g., for Eigen maps)
    virtual T* data() {return this->get_ptr_to_entry(0);}
    //! raw pointer to content (e.g., for Eigen maps)
    virtual const T* data() const {return this->get_ptr_to_entry(0);}

    //! return a map representing the entire field as a single `Eigen::Array`
    EigenMap_t eigen();
    //! return a map representing the entire field as a single `Eigen::Array`
    EigenMapConst_t eigen() const ;
    //! return a map representing the entire field as a single Eigen vector
    EigenVec_t eigenvec();
    //! return a map representing the entire field as a single Eigen vector
    EigenVecConst_t eigenvec() const;
    //! return a map representing the entire field as a single Eigen vector
    EigenVecConst_t const_eigenvec() const;

    /**
     * Convenience function to return a map onto this field. A map
     * allows iteration over all pixels. The map's iterator returns a
     * dynamically sized `Eigen::Map` the data associated with a
     * pixel.
     */
    inline FieldMap_t get_map();

    /**
     * Convenience function to return a map onto this field. A map
     * allows iteration over all pixels. The map's iterator returns a
     * dynamically sized `Eigen::Map` the data associated with a
     * pixel.
     */
    inline ConstFieldMap_t get_map() const;

    /**
     * Convenience function to return a map onto this field. A map
     * allows iteration over all pixels. The map's iterator returns a
     * dynamically sized `Eigen::Map` the data associated with a
     * pixel.
     */
    inline ConstFieldMap_t get_const_map() const;


    /**
     * creates a `TypedField` same size and type as this, but all
     * entries are zero. Convenience function
     */
    inline TypedField & get_zeros_like(std::string unique_name) const;

    /**
     * Fill the content of the local field into the global field
     * (obviously only for pixels that actually are present in the
     * local field)
     */
    template <bool IsGlobal = Global>
    inline std::enable_if_t<IsGlobal>
    fill_from_local(const LocalField_t & local);

    /**
     * For pixels that are present in the local field, fill them with
     * the content of the global field at those pixels
     */
    template <bool IsLocal = not Global>
    inline std::enable_if_t<IsLocal>
    fill_from_global(const GlobalField_t & global);

  protected:
    //! returns a raw pointer to the entry, for `Eigen::Map`
    inline T* get_ptr_to_entry(const size_t&& index);

    //! returns a raw pointer to the entry, for `Eigen::Map`
    inline const T*
    get_ptr_to_entry(const size_t&& index) const;

    //! set the storage size of this field
    inline virtual void resize(size_t size) override final;

    //! The actual storage container
    Storage_t values{};
    /**
     * an unregistered typed field can be mapped onto an array of
     * existing values
     */
    optional<Eigen::Ref<Eigen::Matrix<Real, Eigen::Dynamic, 1>>> alt_values{};

    /**
     * maintains a tally of the current size, as it cannot be reliably
     * determined from either `values` or `alt_values` alone.
     */
    size_t current_size;

    /**
     * in order to accomodate both registered fields (who own and
     * manage their data) and unregistered temporary field proxies
     * (piggy-backing on a chunk of existing memory as e.g., a numpy
     * array) *efficiently*, the `get_ptr_to_entry` methods need to be
     * branchless. this means that we cannot decide on the fly whether
     * to return pointers pointing into values or into alt_values, we
     * need to maintain an (shudder) raw data pointer that is set
     * either at construction (for unregistered fields) or at any
     * resize event (which may invalidate existing pointers). For the
     * coder, this means that they need to be absolutely vigilant that
     * *any* operation on the values vector that invalidates iterators
     * needs to be followed by an update of data_ptr, or we will get
     * super annoying memory bugs.
     */
    T* data_ptr{};

  private:
  };
}  // muSpectre

#include "common/field_map_dynamic.hh"

namespace muSpectre {


  /* ---------------------------------------------------------------------- */
  /* Implementations                                                        */
  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T>
  TypedField<FieldCollection, T>::
  TypedField(std::string unique_name, FieldCollection & collection,
             size_t nb_components):
    Parent(unique_name, nb_components, collection), current_size{0}
  {}

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T>
  TypedField<FieldCollection, T>::
  TypedField(std::string unique_name, FieldCollection & collection,
             Eigen::Ref<Eigen::Matrix<Real, Eigen::Dynamic, 1>> vec,
             size_t nb_components):
    Parent(unique_name, nb_components, collection),
    alt_values{vec}, current_size{vec.size()/nb_components},
    data_ptr{vec.data()}
  {
    if (vec.size()%nb_components) {
      std::stringstream err{};
      err << "The vector you supplied has a size of " << vec.size()
          << ", which is not a multiple of the number of components ("
          << nb_components << ")";
      throw FieldError(err.str());
    }
    if (current_size != collection.size()) {
      std::stringstream err{};
      err << "The vector you supplied has the size for " << current_size
          << " pixels with " << nb_components << "components each, but the "
          << "field collection has " << collection.size() << " pixels.";
      throw FieldError(err.str());
    }
  }

  /* ---------------------------------------------------------------------- */
  //! return type_id of stored type
  template <class FieldCollection, typename T>
  const std::type_info & TypedField<FieldCollection, T>::
  get_stored_typeid() const {
    return typeid(T);
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T>
  auto TypedField<FieldCollection, T>::eigen() -> EigenMap_t {
    return EigenMap_t(this->data(), this->get_nb_components(), this->size());
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T>
  auto TypedField<FieldCollection, T>::eigen() const -> EigenMapConst_t {
    return EigenMapConst_t(this->data(), this->get_nb_components(), this->size());
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T>
  auto TypedField<FieldCollection, T>::eigenvec() -> EigenVec_t {
    return EigenVec_t(this->data(),
                      this->get_nb_components() * this->size(),
                      1);
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T>
  auto TypedField<FieldCollection, T>:: eigenvec() const -> EigenVecConst_t {
    return EigenVecConst_t(this->data(),
                           this->get_nb_components() * this->size(),
                           1);
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T>
  auto TypedField<FieldCollection, T>:: const_eigenvec() const -> EigenVecConst_t {
    return EigenVecConst_t(this->data(), this->get_nb_components() * this->size());
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T>
  auto TypedField<FieldCollection, T>::get_map() -> FieldMap_t {
    return FieldMap_t(*this);
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T>
  auto TypedField<FieldCollection, T>::get_map() const -> ConstFieldMap_t {
    return ConstFieldMap_t(*this);
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T>
  auto TypedField<FieldCollection, T>::get_const_map() const -> ConstFieldMap_t {
    return ConstFieldMap_t(*this);
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T>
  auto TypedField<FieldCollection, T>::
  get_zeros_like(std::string unique_name) const -> TypedField& {
    return make_field<TypedField>(unique_name,
                                  this->collection,
                                  this->nb_components);
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T>
  template <bool IsGlobal>
  std::enable_if_t<IsGlobal> TypedField<FieldCollection, T>::
  fill_from_local(const LocalField_t & local) {
    static_assert(IsGlobal == Global, "SFINAE parameter, do not touch");
    if (not (local.get_nb_components() == this->get_nb_components())) {
      std::stringstream err_str{};
      err_str << "Fields not compatible: You are trying to write a local "
              << local.get_nb_components() << "-component field into a global "
              << this->get_nb_components() << "-component field.";
      throw std::runtime_error(err_str.str());
    }
    auto this_map{this->get_map()};
    for (const auto && key_val: local.get_map().enumerate()) {
      const auto & key{std::get<0>(key_val)};
      const auto & value{std::get<1>(key_val)};
      this_map[key] = value;
    }
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T>
  template <bool IsLocal>
  std::enable_if_t<IsLocal> TypedField<FieldCollection, T>::
  fill_from_global(const GlobalField_t & global) {
    static_assert(IsLocal == not Global, "SFINAE parameter, do not touch");
    if (not (global.get_nb_components() == this->get_nb_components())) {
      std::stringstream err_str{};
      err_str << "Fields not compatible: You are trying to write a global "
              << global.get_nb_components() << "-component field into a local "
              << this->get_nb_components() << "-component field.";
      throw std::runtime_error(err_str.str());
    }

    auto global_map{global.get_map()};

    for (auto && key_val: this->get_map().enumerate()) {
      const auto & key{std::get<0>(key_val)};
      auto & value{std::get<1>(key_val)};
      value = global_map[key];
    }
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T>
  void TypedField<FieldCollection, T>::resize(size_t size) {
    if (this->alt_values) {
      throw FieldError("Field proxies can't resize.");
    }
    this->current_size = size;
    this->values.resize(size*this->get_nb_components() + this->pad_size);
    this->data_ptr = &this->values.front();
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T>
  void TypedField<FieldCollection, T>::set_zero() {
    std::fill(this->values.begin(), this->values.end(), T{});
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T>
  auto TypedField<FieldCollection, T>::check_ref(Base & other) -> TypedField & {
    if (typeid(T).hash_code() != other.get_stored_typeid().hash_code()) {
        std::string err ="Cannot create a Reference of requested type " +(
           "for field '" + other.get_name() + "' of type '" +
           other.get_stored_typeid().name() + "'");
        throw std::runtime_error
          (err);
      }
    return static_cast<TypedField&>(other);
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T>
  auto TypedField<FieldCollection, T>::
  check_ref(const Base & other) -> const TypedField & {
    if (typeid(T).hash_code() != other.get_stored_typeid().hash_code()) {
        std::string err ="Cannot create a Reference of requested type " +(
           "for field '" + other.get_name() + "' of type '" +
           other.get_stored_typeid().name() + "'");
        throw std::runtime_error
          (err);
      }
    return static_cast<const TypedField&>(other);
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T>
  size_t TypedField<FieldCollection, T>::
  size() const {
    return this->current_size;
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T>
  void TypedField<FieldCollection, T>::
  set_pad_size(size_t pad_size) {
    if (this->alt_values) {
      throw FieldError("You can't set the pad size of a field proxy.");
    }
    this->pad_size = pad_size;
    this->resize(this->size());
    this->data_ptr = &this->values.front();
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T>
  T* TypedField<FieldCollection, T>::
  get_ptr_to_entry(const size_t&& index) {
    return this->data_ptr + this->get_nb_components()*std::move(index);
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T>
  const T* TypedField<FieldCollection, T>::
  get_ptr_to_entry(const size_t&& index) const {
    return this->data_ptr+this->get_nb_components()*std::move(index);
  }

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T>
  template <class Derived>
  void TypedField<FieldCollection, T>::
  push_back(const Eigen::DenseBase<Derived> & value) {
    static_assert (not FieldCollection::Global,
                   "You can only push_back data into local field collections");
    if (value.cols() != 1) {
      std::stringstream err{};
      err << "Expected a column vector, but received and array with "
          << value.cols() <<" colums.";
      throw FieldError(err.str());
    }
    if (value.rows() != static_cast<Int>(this->get_nb_components())) {
      std::stringstream err{};
      err << "Expected a column vector of length " << this->get_nb_components()
          << ", but received one of length " << value.rows() <<".";
      throw FieldError(err.str());
    }
    for (size_t i = 0; i < this->get_nb_components(); ++i) {
      this->values.push_back(value(i));
    }
    ++this->current_size;
    this->data_ptr = &this->values.front();
  }

}  // muSpectre

#endif /* FIELD_TYPED_H */
