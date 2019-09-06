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

  //! forward declarations
  class NFieldCollection;
  class NField;
  template <typename T>
  class TypedNField;

  class StateNField {
   protected:
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
    const Dim_t& get_nb_memory() const;

    //! return type_id of stored type
    virtual const std::type_info & get_stored_typeid() const = 0;

    /**
     * cycle the fields (current becomes old, old becomes older,
     * oldest becomes current)
     */
    void cycle();

    NField & current();

    const NField & current() const;

    const NField & old(size_t nb_steps_ago = 1) const;

    //! get the current ordering of the fields (inlineable because called in hot
    //! loop)
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
    RefVector<NField> fields{};
  };

  template <typename T, Mapping Mutability>
  class StateNFieldMap;

  /**
   * The `TypedStateField` class is a byte compatible daughter class of the
   * `StateField` class, and it can return fully typed `Field` references.
   */
  template <typename T>
  class TypedStateNField : public StateNField {
   protected:
    TypedStateNField(const std::string & unique_prefix,
                     NFieldCollection & collection, Dim_t nb_memory,
                     Dim_t nb_components);

   public:
    using Parent = StateNField;
    //! Default constructor
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

    TypedNField<T> & current();

    const TypedNField<T> & current() const;

    const TypedNField<T> & old(size_t nb_steps_ago = 1) const;

    friend NFieldCollection;
    friend class StateNFieldMap<T, Mapping::Const>;
    friend class StateNFieldMap<T, Mapping::Mut>;
   protected:
    RefVector<NField> & get_fields();
  };

  /* ---------------------------------------------------------------------- */
  using RealStateNField = TypedStateNField<Real>;
  using ComplexStateNField = TypedStateNField<Complex>;
  using IntStateNField = TypedStateNField<Int>;
  using Uintnfield = TypedStateNField<Uint>;

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_STATE_NFIELD_HH_
