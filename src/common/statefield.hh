/**
 * file   statefield.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   28 Feb 2018
 *
 * @brief A state field is an abstraction of a field that can hold
 * current, as well as a chosen number of previous values. This is
 * useful for instance for internal state variables in plastic laws,
 * where a current, new, or trial state is computed based on its
 * previous state, and at convergence, this new state gets cycled into
 * the old, the old into the old-1 etc. The state field abstraction
 * helps doing this safely (i.e. only const references to the old
 * states are available, while the current state can be assigned
 * to/modified), and efficiently (i.e., no need to copy values from
 * new to old, we just cycle the labels). This file implements the
 * state field as well as state maps using the Field, FieldCollection
 * and FieldMap abstractions of µSpectre
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

#ifndef STATEFIELD_H
#define STATEFIELD_H

#include "common/field.hh"
#include "common/utilities.hh"

#include <array>
#include <string>
#include <sstream>

namespace muSpectre {

  /**
   * Base class for state fields, useful for storing polymorphic references
   */
  template <class FieldCollection>
  class StateFieldBase {
  public:
    //! get naming prefix
    const std::string & get_prefix() const {return this->prefix;}

    //! get a ref to the `StateField` 's field collection
    const FieldCollection & get_collection() const {
      return this->collection;}

    virtual ~StateFieldBase() = default;

    /**
     * returns number of old states that are stored
     */
    size_t get_nb_memory() const {return this->nb_memory;}

    //! return type_id of stored type
    virtual const std::type_info & get_stored_typeid() const = 0;

    /**
     * cycle the fields (current becomes old, old becomes older,
     * oldest becomes current)
     */
    virtual void cycle() = 0;
  protected:
    //! constructor
    StateFieldBase(std::string unique_prefix,
                   const FieldCollection & collection,
                   size_t nb_memory=1):
      prefix{unique_prefix},
      nb_memory{nb_memory},
      collection{collection} {}

    /**
     * the unique prefix is used as the first part of the unique name
     * of the subfields belonging to this state field
     */
    std::string prefix;
    /**
     * number of old states to store, defaults to 1
     */
    const size_t nb_memory;
    //! reference to the collection this statefield belongs to
    const FieldCollection & collection;
  };

  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, typename T>
  class TypedStateField:
    public StateFieldBase<FieldCollection> {
  public:
    //! Parent class
    using Parent = StateFieldBase<FieldCollection>;
    //! Typed field
    using TypedField_t = TypedField<FieldCollection, T>;

    //! returns a TypedField ref to the current value of this state field
    virtual TypedField_t & get_current_field() = 0;

    //! returns a const TypedField ref to an old value of this state field
    virtual const TypedField_t & get_old_field(size_t nb_steps_ago=1) const = 0;

    //! return type_id of stored type
    const std::type_info & get_stored_typeid() const override final {
      return typeid(T);
    };

    virtual ~TypedStateField() = default;

  protected:
    //! constructor
    TypedStateField(const std::string & unique_prefix,
                    const FieldCollection & collection,
                    size_t nb_memory):
      Parent{unique_prefix, collection, nb_memory}
    {}
  };



  /* ---------------------------------------------------------------------- */
  template <class FieldCollection, size_t nb_memory, typename T>
  class TypedSizedStateField: public TypedStateField<FieldCollection, T> {
  public:
    //! Parent class
    using Parent = TypedStateField<FieldCollection, T>;
    //! the current (historically accurate) ordering of the fields
    using index_t = std::array<size_t, nb_memory+1>;
    //! get the current ordering of the fields
    inline const index_t & get_indices() const {return this->indices;}
    //! destructor
    virtual ~TypedSizedStateField() = default;

  protected:
    //! constructor
    TypedSizedStateField(std::string unique_prefix,
                         const FieldCollection& collection,
                         index_t indices):
      Parent{unique_prefix, collection, nb_memory},
      indices{indices}{};
    index_t indices; ///< these are cycled through
  };

  //! early declaration
  template <class FieldMap, size_t nb_memory>
  class StateFieldMap;

  namespace internal {

    template <class Field, size_t size, size_t... I>
    inline decltype(auto)
    build_fields_helper(std::string prefix,
                        typename Field::Base::collection_t & collection,
                        std::index_sequence<I...>) {
      auto get_field{[&prefix, &collection](size_t i) -> Field&
          {
            std::stringstream name_stream{};
            name_stream << prefix << ", sub_field index " << i;
            return make_field<Field>(name_stream.str(), collection);
          }};
      return std::tie(get_field(I)...);
    }

    /* ---------------------------------------------------------------------- */
    template <size_t size, size_t... I>
    inline decltype(auto) build_indices(std::index_sequence<I...>) {
      return std::array<size_t, size>{(size-I)%size...};
    }

  }  // internal

  /**
   * A statefield is an abstraction around a Field that can hold a
   * current and `nb_memory` previous values. There are useful for
   * history variables, for instance.
   */
  template <class Field_t, size_t nb_memory=1>
  class StateField:
    public TypedSizedStateField<typename Field_t::Base::collection_t,
                                nb_memory, typename Field_t::Scalar>
  {
  public:
    //! the underlying field's collection type
    using FieldCollection_t = typename Field_t::Base::collection_t;
    //! base type for fields
    using Scalar = typename Field_t::Scalar;
    //! Base class for all state fields of same memory
    using Base = TypedSizedStateField<FieldCollection_t, nb_memory, Scalar>;
    /**
     * storage of field refs (can't be a `std::array`, because arrays
     * of refs are explicitely forbidden
     */
    using Fields_t = tuple_array<Field_t&, nb_memory+1>;
    //! Typed field
    using TypedField_t = TypedField<FieldCollection_t, Scalar>;

    //! Default constructor
    StateField() = delete;

    //! Copy constructor
    StateField(const StateField &other) = delete;

    //! Move constructor
    StateField(StateField &&other) = delete;

    //! Destructor
    virtual ~StateField() = default;

    //! Copy assignment operator
    StateField& operator=(const StateField &other) = delete;

    //! Move assignment operator
    StateField& operator=(StateField &&other) = delete;

    //! get (modifiable) current field
    inline Field_t& current() {
      return this->fields[this->indices[0]];
    }

    //! get (constant) previous field
    template <size_t nb_steps_ago=1>
    inline const Field_t& old() {
      static_assert(nb_steps_ago <= nb_memory,
                    "you can't go that far inte the past");
      static_assert(nb_steps_ago > 0,
                    "Did you mean to call current()?");
      return this->fields[this->indices.at(nb_steps_ago)];
    }

    //! returns a TypedField ref to the current value of this state field
    TypedField_t & get_current_field() override final {
      return this->current();
    }

    //! returns a const TypedField ref to an old value of this state field
    const TypedField_t &
    get_old_field(size_t nb_steps_ago=1) const override final {
      return this->fields[this->indices.at(nb_steps_ago)];
    }

   //! factory function
   template<class StateFieldType, class CollectionType>
   friend StateFieldType& make_statefield(const std::string & unique_prefix,
                                          CollectionType & collection);

    //! returns a `StateField` reference if `other is a compatible state field
    inline static StateField& check_ref(Base& other) {
      // the following triggers and exception if the fields are incompatible
      Field_t::check_ref(other.fields[0]);
      return static_cast<StateField&> (other);
    }

    //! returns a const `StateField` reference if `other` is a compatible state field
    inline static const StateField& check_ref(const Base& other) {
      // the following triggers and exception if the fields are incompatible
      Field_t::check_ref(other.fields[0]);
      return static_cast<const StateField&> (other);
    }

    //! get a ref to the `StateField` 's fields
     Fields_t & get_fields() {
       return this->fields;
     }

    /**
     * Pure convenience functions to get a MatrixFieldMap of
     * appropriate dimensions mapped to this field. You can also
     * create other types of maps, as long as they have the right
     * fundamental type (T), the correct size (nbComponents), and
     * memory (nb_memory).
     */
    inline decltype(auto) get_map() {
      using FieldMap = decltype(std::get<0>(this->fields).get_map());
      return StateFieldMap<FieldMap, nb_memory>(*this);
    }

    /**
     * Pure convenience functions to get a MatrixFieldMap of
     * appropriate dimensions mapped to this field. You can also
     * create other types of maps, as long as they have the right
     * fundamental type (T), the correct size (nbComponents), and
     * memory (nb_memory).
     */
    inline decltype(auto) get_const_map() {
      using FieldMap = decltype(std::get<0>(this->fields).get_const_map());
      return StateFieldMap<FieldMap, nb_memory>(*this);
    }

    /**
     * cycle the fields (current becomes old, old becomes older,
     * oldest becomes current)
     */
    inline void cycle() override final {
      for (auto & val: this->indices) {
        val = (val+1)%(nb_memory+1);
      }
    }

  protected:
    /**
     * Constructor.  @param unique_prefix is used to create the names
     * of the fields that this abstraction creates in the background
     * @param collection is the field collection in which the
     * subfields will be stored
     */
    inline StateField(const std::string & unique_prefix,
                      FieldCollection_t & collection)
      : Base{unique_prefix, collection,
        internal::build_indices<nb_memory+1>
        (std::make_index_sequence<nb_memory+1>{})},
        fields{internal::build_fields_helper<Field_t, nb_memory+1>
        (unique_prefix, collection, std::make_index_sequence<nb_memory+1>{})}
    {}

    Fields_t fields; //!< container for the states
  private:
  };


  namespace internal {

    template <class FieldMap, size_t size, class Fields, size_t... I>
    inline decltype(auto) build_maps_helper(Fields & fields,
                                            std::index_sequence<I...>) {
      return std::array<FieldMap, size>{FieldMap(std::get<I>(fields))...};
    }

  }  // internal

  /* ---------------------------------------------------------------------- */
  template <class StateFieldType, class CollectionType>
  inline StateFieldType &
  make_statefield(const std::string & unique_prefix,
                  CollectionType & collection) {
    std::unique_ptr<StateFieldType> ptr {
      new StateFieldType(unique_prefix, collection)};
      auto & retref{*ptr};
      collection.register_statefield(std::move(ptr));
      return retref;
  }

  /**
   * extends the StateField <-> Field equivalence to StateFieldMap <-> FieldMap
   */
  template <class FieldMap, size_t nb_memory=1>
  class StateFieldMap
  {
  public:
    /**
     * iterates over all pixels in the `muSpectre::FieldCollection` and
     * dereferences to a proxy giving access to the appropriate iterates
     * of the underlying `FieldMap` type.
     */
    class iterator;

    //! stl conformance
    using reference = typename iterator::reference;
    //! stl conformance
    using value_type = typename iterator::value_type;
    //! stl conformance
    using size_type = typename iterator::size_type;

    //! field collection type where this state field can be stored
    using FieldCollection_t= typename FieldMap::Field::collection_t;

    //! Fundamental type stored
    using Scalar = typename FieldMap::Scalar;
    //! base class (must be at least sized)
    using TypedSizedStateField_t = TypedSizedStateField<FieldCollection_t,
                                                        nb_memory,
                                                        Scalar>;
    //! for traits access
    using FieldMap_t = FieldMap;
    //! for traits access
    using ConstFieldMap_t = typename FieldMap::ConstMap;

    //! Default constructor
    StateFieldMap() = delete;

    //! constructor using a StateField
    template <class StateField>
    StateFieldMap(StateField & statefield)
      :collection{statefield.get_collection()},
       statefield{statefield},
       maps{internal::build_maps_helper
           <FieldMap, nb_memory+1>(statefield.get_fields(),
                                   std::make_index_sequence<nb_memory+1>{})},
       const_maps{internal::build_maps_helper
           <ConstFieldMap_t, nb_memory+1>(statefield.get_fields(),
                                          std::make_index_sequence<nb_memory+1>{})}
    {
      static_assert(std::is_base_of<TypedSizedStateField_t, StateField>::value,
                    "Not the right type of StateField ref");
    }

    //! Copy constructor
    StateFieldMap(const StateFieldMap &other) = delete;

    //! Move constructor
    StateFieldMap(StateFieldMap &&other) = default;

    //! Destructor
    virtual ~StateFieldMap() = default;

    //! Copy assignment operator
    StateFieldMap& operator=(const StateFieldMap &other) = delete;

    //! Move assignment operator
    StateFieldMap& operator=(StateFieldMap &&other) = delete;

    //! access the wrapper to a given pixel directly
    value_type operator[](size_type index) {
      return *iterator(*this, index);
    }

    /**
     * return a ref to the current field map. useful for instance for
     * initialisations of `StateField` instances
     */
    FieldMap& current() {
      return this->maps[this->statefield.get_indices()[0]];
    }

    //! stl conformance
    iterator begin() {
      return iterator(*this, 0);}
    //! stl conformance
    iterator end() {
      return iterator(*this, this->collection.size());}

  protected:
    const FieldCollection_t & collection; //!< collection holding the field
    TypedSizedStateField_t & statefield; //!< ref to the field itself
    std::array<FieldMap, nb_memory+1> maps;//!< refs to the addressable maps;
    //! const refs to the addressable maps;
    std::array<ConstFieldMap_t, nb_memory+1> const_maps;
  private:
  };

  /**
   * Iterator class used by the `StateFieldMap`
   */
  template <class FieldMap, size_t nb_memory>
  class StateFieldMap<FieldMap, nb_memory>::iterator
  {
  public:
    class StateWrapper;

    using Ccoord = typename FieldMap::Ccoord; //!< cell coordinates type
    using value_type = StateWrapper; //!< stl conformance
    using const_value_type = value_type; //!< stl conformance
    using pointer_type = value_type*; //!< stl conformance
    using difference_type = std::ptrdiff_t; //!< stl conformance
    using size_type = size_t; //!< stl conformance
    using iterator_category = std::random_access_iterator_tag; //!< stl conformance
    using reference = StateWrapper; //!< stl conformance

    //! Default constructor
    iterator() = delete;

    //! constructor
    iterator(StateFieldMap& map, size_t index = 0)
      :index{index}, map{map}
    {};

    //! Copy constructor
    iterator(const iterator &other) = default;

    //! Move constructor
    iterator(iterator &&other) = default;

    //! Destructor
    virtual ~iterator() = default;

    //! Copy assignment operator
    iterator& operator=(const iterator &other) = default;

    //! Move assignment operator
    iterator& operator=(iterator &&other) = default;

    //! pre-increment
    inline iterator & operator++() {
      this->index++; return *this;}
    //! post-increment
    inline iterator operator++(int) {
      iterator curr{*this}; this->index++; return curr;}
    //! dereference
    inline value_type operator*() {
      return value_type(*this);}
    //! pre-decrement
    inline iterator & operator--() {
      this->index--; return *this;}
    //! post-decrement
    inline iterator operator--(int) {
      iterator curr{*this}; this->index--; return curr;}
    //! access subscripting
    inline value_type operator[](difference_type diff) {
      return value_type{iterator{this->map, this->index+diff}};}
    //! equality
    inline bool operator==(const iterator & other) const {
      return this->index == other.index;
    }
    //! inequality
    inline bool operator!=(const iterator & other) const {
      return this->index != other.index;}
    //! div. comparisons
    inline bool operator<(const iterator & other) const {
      return this->index < other.index;
    }
    //! div. comparisons
    inline bool operator<=(const iterator & other) const {
      return this->index <= other.index;
    }
    //! div. comparisons
    inline bool operator>(const iterator & other) const {
      return this->index > other.index;
    }
    //! div. comparisons
    inline bool operator>=(const iterator & other) const {
      return this->index >= other.index;
    }
    //! additions, subtractions and corresponding assignments
    inline iterator operator+(difference_type diff) const {
      return iterator{this->map, this-index + diff};
    }
    //! additions, subtractions and corresponding assignments
    inline iterator operator-(difference_type diff) const {
      return iterator{this->map, this-index - diff};}
    //! additions, subtractions and corresponding assignments
    inline iterator& operator+=(difference_type diff) {
      this->index += diff; return *this;}
    //! additions, subtractions and corresponding assignments
    inline iterator& operator-=(difference_type diff) {
      this->index -= diff; return *this;
    }

    //! get pixel coordinates
    inline Ccoord get_ccoord() const {
      return this->map.collection.get_ccoord(this->index);
    }

    //! access the index
    inline const size_t & get_index() const {return this->index;}

  protected:
    size_t index; //!< current pixel this iterator refers to
    StateFieldMap& map; //!< map over with `this` iterates
  private:
  };



  namespace internal {

    //! FieldMap is an `Eigen::Map` or `Eigen::TensorMap` here
    template <class FieldMap, size_t size, size_t... I,
              class iterator, class maps_t, class indices_t>
    inline decltype(auto)
    build_old_vals_helper(iterator& it, maps_t & maps, indices_t & indices,
                          std::index_sequence<I...>) {
      return tuple_array<FieldMap, size>(std::forward_as_tuple(maps[indices[I+1]][it.get_index()]...));
    }

    template <class FieldMap, size_t size, class iterator, class maps_t, class indices_t>
    inline decltype(auto)
    build_old_vals(iterator& it, maps_t & maps, indices_t & indices) {
      return tuple_array<FieldMap, size>{build_old_vals_helper<FieldMap, size>
          (it, maps, indices, std::make_index_sequence<size>{})};
    }

  }  // internal

  /**
   * Light-weight resource-handle representing the current and old
   * values of a field at a given pixel identified by an iterator
   * pointing to it
   */
  template <class FieldMap, size_t nb_memory>
  class StateFieldMap<FieldMap, nb_memory>::iterator::StateWrapper
  {
  public:
    //! short-hand
    using iterator = typename StateFieldMap::iterator;
    //! short-hand
    using Ccoord = typename iterator::Ccoord;
    //! short-hand
    using Map = typename FieldMap::reference;
    //! short-hand
    using ConstMap = typename FieldMap::const_reference;

    //! Default constructor
    StateWrapper() = delete;

    //! Copy constructor
    StateWrapper(const StateWrapper &other) = default;

    //! Move constructor
    StateWrapper(StateWrapper &&other) = default;

    //! construct with `StateFieldMap::iterator`
    StateWrapper(iterator & it)
      :it{it},
       current_val{it.map.maps[it.map.statefield.get_indices()[0]][it.index]},
       old_vals(internal::build_old_vals
           <ConstMap, nb_memory>(it, it.map.const_maps,
                                 it.map.statefield.get_indices()))
    {    }

    //! Destructor
    virtual ~StateWrapper() = default;

    //! Copy assignment operator
    StateWrapper& operator=(const StateWrapper &other) = default;

    //! Move assignment operator
    StateWrapper& operator=(StateWrapper &&other) = default;

    //! returns reference to the currectly mapped value
    inline Map& current() {
      return this->current_val;
    }

    //! recurnts reference the the value that was current `nb_steps_ago` ago
    template <size_t nb_steps_ago = 1>
    inline const ConstMap & old() const{
      static_assert (nb_steps_ago <= nb_memory,
                     "You have not stored that time step");
      static_assert (nb_steps_ago > 0,
                     "Did you mean to access the current value? If so, use "
                     "current()");
      return std::get<nb_steps_ago-1>(this->old_vals);
    }

    //! read the coordinates of the current pixel
    inline Ccoord get_ccoord() const {
      return this->it.get_ccoord();
    }

  protected:
    iterator& it; //!< ref to the iterator that dereferences to `this`
    Map current_val; //!< current value
    tuple_array<ConstMap, nb_memory> old_vals; //!< all stored old values
  private:
  };
}  // muSpectre

#endif /* STATEFIELD_H */
