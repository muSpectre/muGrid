/**
 * @file   state_nfield.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   20 Aug 2019
 *
 * @brief  A state field is an abstraction of a field that can hold
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
 * and FieldMap abstractions of µGrid
 *
 * Copyright © 2019 Till Junge
 *
 * µGrid is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µGrid is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with µGrid; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#ifndef SRC_LIBMUGRID_STATE_NFIELD_HH_
#define SRC_LIBMUGRID_STATE_NFIELD_HH_

#include "grid_common.hh"
#include "ref_vector.hh"
#include "state_nfield_map.hh"

#include <string>
#include <vector>
#include <utility>

namespace muGrid {

  //! forward declaration of the `muGrid::NFieldCollection`
  class NFieldCollection;
  //! forward declaration of the `muGrid::NField`
  class NField;

  //! forward declaration of the `muGrid::TypedNField`
  template <typename T>
  class TypedNField;

  /**
   * Base class for state fields, useful for storing polymorphic references
   */
  class StateNField {
   protected:
    /**
     * Protected constructor
     */
    StateNField(const std::string & unique_prefix,
                NFieldCollection & collection, Dim_t nb_memory = 1);

   public:
    //! Default constructor
    StateNField() = delete;

    //! Copy constructor
    StateNField(const StateNField & other) = delete;

    //! Move constructor
    StateNField(StateNField && other) = delete;

    //! Destructor
    virtual ~StateNField() = default;

    //! Copy assignment operator
    StateNField & operator=(const StateNField & other) = delete;

    //! Move assignment operator
    StateNField & operator=(StateNField && other) = delete;

    /**
     * returns number of old states that are stored
     */
    const Dim_t & get_nb_memory() const;

    //! return type_id of stored type
    virtual const std::type_info & get_stored_typeid() const = 0;

    /**
     * cycle the fields (current becomes old, old becomes older,
     * oldest becomes current)
     */
    void cycle();

    //! return a reference to the field holding the current values
    NField & current();

    //! return a const reference to the field holding the current values
    const NField & current() const;

    /**
     * return a reference to the field holding the values which were current
     * `nb_steps_ago` ago
     */
    const NField & old(size_t nb_steps_ago = 1) const;

    /**
     * get the current ordering of the fields (inlineable because called in hot
     * loop)
     */
    const std::vector<size_t> & get_indices() const { return this->indices; }

   protected:
    /**
     * the unique prefix is used as the first part of the unique name
     * of the subfields belonging to this state field
     */
    std::string prefix;
    //! reference to the collection this statefield belongs to
    NFieldCollection & collection;
    /**
     * number of old states to store, defaults to 1
     */
    const Dim_t nb_memory;
    //! the current (historically accurate) ordering of the fields
    std::vector<size_t> indices{};

    //! storage of references to the diverse fields
    RefVector<NField> fields{};
  };

  //! forward-declaration for friending
  template <typename T, Mapping Mutability>
  class StateNFieldMap;

  /**
   * The `TypedStateField` class is a byte compatible daughter class of the
   * `StateField` class, and it can return fully typed `Field` references.
   */
  template <typename T>
  class TypedStateNField : public StateNField {
   protected:
    /**
     * protected constructor, to avoid the creation of unregistered fields.
     * Users should create fields through the
     * `muGrid::NFieldCollection::register_real_field()` (or `int`, `uint`,
     * `compplex`) factory functions.
     */
    TypedStateNField(const std::string & unique_prefix,
                     NFieldCollection & collection, Dim_t nb_memory,
                     Dim_t nb_components);

   public:
    //! base class
    using Parent = StateNField;

    //! Deleted default constructor
    TypedStateNField() = delete;

    //! Copy constructor
    TypedStateNField(const TypedStateNField & other) = delete;

    //! Move constructor
    TypedStateNField(TypedStateNField && other) = delete;

    //! Destructor
    virtual ~TypedStateNField() = default;

    //! Copy assignment operator
    TypedStateNField & operator=(const TypedStateNField & other) = delete;

    //! Move assignment operator
    TypedStateNField & operator=(TypedStateNField && other) = delete;

    //! return type_id of stored type
    const std::type_info & get_stored_typeid() const final;

    //! return a reference to the current field
    TypedNField<T> & current();

    //! return a const reference to the current field
    const TypedNField<T> & current() const;

    /**
     * return a const reference to the field which was current `nb_steps_ago`
     * steps ago
     */
    const TypedNField<T> & old(size_t nb_steps_ago = 1) const;

   protected:
    //! give access to the protected state field constructor
    friend NFieldCollection;

    //! give access to `get_fields()`
    friend class StateNFieldMap<T, Mapping::Const>;

    //! give access to `get_fields()`
    friend class StateNFieldMap<T, Mapping::Mut>;

    //! return a reference to the storage of the constituent fields
    RefVector<NField> & get_fields();
  };

  //! Alias for real-valued state fields
  using RealStateNField = TypedStateNField<Real>;
  //! Alias for complex-valued state fields
  using ComplexStateNField = TypedStateNField<Complex>;
  //! Alias for integer-valued state fields
  using IntStateNField = TypedStateNField<Int>;
  //! Alias for unsigned integer-valued state fields
  using Uintnfield = TypedStateNField<Uint>;

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_STATE_NFIELD_HH_
