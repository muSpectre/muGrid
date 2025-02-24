/**
 * @file   state_field.hh
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
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µGrid is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µGrid; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 *
 */

#ifndef SRC_LIBMUGRID_STATE_FIELD_HH_
#define SRC_LIBMUGRID_STATE_FIELD_HH_

#include "grid_common.hh"
#include "ref_vector.hh"
#include "state_field_map.hh"

#include <string>
#include <vector>
#include <utility>

#ifdef WITH_MPI
#include "mpi.h"
#endif

namespace muGrid {

  //! forward declaration of the `muGrid::FieldCollection`
  class FieldCollection;

  //! forward declaration of the `muGrid::Field`
  class Field;

  //! forward declaration of the `muGrid::TypedField`
  template <typename T>
  class TypedField;

  /**
   * Base class for state fields, useful for storing polymorphic references
   */
  class StateField {
   protected:
    /**
     * Protected constructor
     */
    StateField(const std::string & unique_prefix, FieldCollection & collection,
               const Index_t & nb_memory, const Index_t & nb_components,
               const std::string & sub_division, const Unit & unit);

   public:
    //! Default constructor
    StateField() = delete;

    //! Copy constructor
    StateField(const StateField & other) = delete;

    //! Move constructor
    StateField(StateField && other) = delete;

    //! Destructor
    virtual ~StateField() = default;

    //! Copy assignment operator
    StateField & operator=(const StateField & other) = delete;

    //! Move assignment operator
    StateField & operator=(StateField && other) = delete;

    //! return number of old states that are stored
    const Index_t & get_nb_memory() const;

    //! return the number of components stored per sub-point point
    const Index_t & get_nb_components() const;

    //! returns a const ref to the field's pixel sub-division type
    const std::string & get_sub_division_tag() const;

    //! returns the physical unit of the values stored in the field
    const Unit & get_physical_unit() const;

    //! return type_id of stored type
    virtual const std::type_info & get_typeid() const = 0;

    //! return the size of the elementary field entry in bytes
    virtual const std::size_t get_element_size_in_bytes() const = 0;

#ifdef WITH_MPI
    //! return the MPI representation of the stored type
    virtual const MPI_Datatype get_mpi_type() const = 0;
#endif

    /**
     * assert that the stored type corresponds to the given type id
     */
    void assert_typeid(const std::type_info & type) const;

    /**
     * cycle the fields (current becomes old, old becomes older,
     * oldest becomes current)
     */
    void cycle();

    //! return a reference to the field holding the current values
    Field & current();

    //! return a const reference to the field holding the current values
    const Field & current() const;

    /**
     * return a reference to the field holding the values which were current
     * `nb_steps_ago` ago
     */
    const Field & old(const size_t & nb_steps_ago = 1) const;

    /**
     * get the current ordering of the fields (inlineable because called in hot
     * loop)
     */
    const std::vector<size_t> & get_indices() const { return this->indices; }

    //! get the field collection which holds all fields of the state field
    FieldCollection & get_collection();

    //! get the unique prefix used for the naming of the associated fields and
    //! can be used like a name for the StateField
    const std::string & get_unique_prefix() const;

    //! return a const RefVector<Field> of fields belonging to the StateField
    const RefVector<Field> & get_fields() const;

    //! return a mutable RefVector<Field> of fields belonging to the StateField
    RefVector<Field> & set_fields();

   protected:
    /**
     * the unique prefix is used as the first part of the unique name
     * of the subfields belonging to this state field
     */
    std::string prefix;
    //! reference to the collection this statefield belongs to
    FieldCollection & collection;
    /**
     * number of old states to store, defaults to 1
     */
    const Index_t nb_memory;

    /**
     * number of dof_per_sub_pt stored per sub-point (e.g., 3 for a
     * three-dimensional vector, or 9 for a three-dimensional second-rank
     * tensor)
     */
    const Index_t nb_components;

    /**
     * Pixel subdivision kind (determines how many datapoints to store per
     * pixel)
     */
    std::string sub_division_tag;

    //! Physical unit of the values stored in this field
    Unit unit;

    /**
     * number of pixel subdivisions. Will depend on sub_division
     */
    Index_t nb_sub_pts;

    //! the current (historically accurate) ordering of the fields
    std::vector<size_t> indices{};

    //! storage of references to the diverse fields
    RefVector<Field> fields{};
  };

  //! forward-declaration for friending
  template <typename T, Mapping Mutability>
  class StateFieldMap;

  /**
   * The `TypedStateField` class is a byte compatible daughter class of the
   * `StateField` class, and it can return fully typed `Field` references.
   */
  template <typename T>
  class TypedStateField : public StateField {
   protected:
    /**
     * protected constructor, to avoid the creation of unregistered fields.
     * Users should create fields through the
     * `muGrid::FieldCollection::register_real_field()` (or `int`, `uint`,
     * `complex`) factory functions.
     */
    TypedStateField(const std::string & unique_prefix,
                    FieldCollection & collection, const Index_t & nb_memory,
                    const Index_t & nb_components,
                    const std::string & sub_division, const Unit & unit);

   public:
    //! base class
    using Parent = StateField;

    //! Deleted default constructor
    TypedStateField() = delete;

    //! Copy constructor
    TypedStateField(const TypedStateField & other) = delete;

    //! Move constructor
    TypedStateField(TypedStateField && other) = delete;

    //! Destructor
    virtual ~TypedStateField() = default;

    //! Copy assignment operator
    TypedStateField & operator=(const TypedStateField & other) = delete;

    //! Move assignment operator
    TypedStateField & operator=(TypedStateField && other) = delete;

    //! return type_id of stored type
    const std::type_info & get_typeid() const final;

#ifdef WITH_MPI
    //! return the MPI representation of the stored type
    const MPI_Datatype get_mpi_type() const final;
#endif

    //! return the size of the elementary field entry in bytes
    const std::size_t get_element_size_in_bytes() const final;

    //! return a reference to the current field
    TypedField<T> & current();

    //! return a const reference to the current field
    const TypedField<T> & current() const;

    /**
     * return a const reference to the field which was current `nb_steps_ago`
     * steps ago
     */
    const TypedField<T> & old(size_t nb_steps_ago = 1) const;

   protected:
    //! give access to the protected state field constructor
    friend FieldCollection;

    //! give access to `get_fields()`
    friend class StateFieldMap<T, Mapping::Const>;

    //! give access to `get_fields()`
    friend class StateFieldMap<T, Mapping::Mut>;

    //! return a reference to the storage of the constituent fields
    RefVector<Field> & get_fields();
  };

  //! Alias for real-valued state fields
  using RealStateField = TypedStateField<Real>;
  //! Alias for complex-valued state fields
  using ComplexStateField = TypedStateField<Complex>;
  //! Alias for integer-valued state fields
  using IntStateField = TypedStateField<Int>;
  //! Alias for unsigned integer-valued state fields
  using Uintfield = TypedStateField<Uint>;

  /* ---------------------------------------------------------------------- */
  template <typename T>
  TypedStateField<T> & FieldCollection::register_state_field_helper(
      const std::string & unique_prefix, const Index_t & nb_memory,
      const Index_t & nb_components, const std::string & sub_division_tag,
      const Unit & unit, bool allow_existing) {
    static_assert(
        std::is_scalar<T>::value or std::is_same<T, Complex>::value,
        "You can only register state fields templated with one of the "
        "numeric types Real, Complex, Int, or UInt");
    if (this->state_field_exists(unique_prefix)) {
      if (allow_existing) {
        auto & field{*this->state_fields[unique_prefix]};
        field.assert_typeid(typeid(T));
        if (field.get_nb_memory() != nb_memory) {
          throw FieldCollectionError(
              "You can't change the number of memory steps of a state field "
              "by re-registering it.");
        }
        if (field.get_nb_components() != nb_components) {
          std::stringstream error{};
          error << "You can't change the number of components of a field "
                << "by re-registering it. Field '" << unique_prefix << "' has "
                << field.get_nb_components()
                << " components and you are trying to register it with "
                << nb_components << " components.";
          throw FieldCollectionError(error.str());
        }
        if (field.get_sub_division_tag() != sub_division_tag) {
          throw FieldCollectionError(
              "You can't change the sub-division tag of a state field "
              "by re-registering it.");
        }
        if (field.get_physical_unit() != unit) {
          throw FieldCollectionError(
              "You can't change the physical unit of a state field "
              "by re-registering it.");
        }
        return static_cast<TypedStateField<T> &>(field);
      } else {
        std::stringstream error{};
        error << "A StateField of name '" << unique_prefix
              << "' is already registered in this field collection. "
              << "Currently registered state fields: ";
        std::string prelude{""};
        for (const auto & name_field_pair : this->state_fields) {
          error << prelude << '\'' << name_field_pair.first << '\'';
          prelude = ", ";
        }
        throw FieldCollectionError(error.str());
      }
    }

    //! If you get a compiler warning about narrowing conversion on the
    //! following line, please check whether you are creating a TypedField
    //! with the number of components specified in 'int' rather than 'size_t'.
    TypedStateField<T> * raw_ptr{
        new TypedStateField<T>{unique_prefix, *this, nb_memory, nb_components,
                               sub_division_tag, unit}};
    TypedStateField<T> & retref{*raw_ptr};
    StateField_ptr field{raw_ptr};
    this->state_fields[unique_prefix] = std::move(field);
    return retref;
  }

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_STATE_FIELD_HH_
