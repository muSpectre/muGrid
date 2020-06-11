/**
 * @file   state_field_map.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   22 Aug 2019
 *
 * @brief  implementation of state field maps
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

#ifndef SRC_LIBMUGRID_STATE_FIELD_MAP_HH_
#define SRC_LIBMUGRID_STATE_FIELD_MAP_HH_

#include "grid_common.hh"
#include "field_map.hh"
#include "ref_vector.hh"

#include <vector>
#include <memory>

namespace muGrid {

  //! forward declaration
  template <typename T>
  class TypedStateField;
  class Field;

  /**
   * Dynamically sized map for iterating over `muGrid::StateField`s
   */
  template <typename T, Mapping Mutability>
  class StateFieldMap {
   public:
    /**
     * type for the current-values map (may be mutable, if the underlying field
     * was)
     */
    using FieldMap_t = FieldMap<T, Mutability>;

    //! type for the old-values map, non-mutable
    using CFieldMap_t = FieldMap<T, Mapping::Const>;

    //! Default constructor
    StateFieldMap() = delete;

    /** constructor from a state field. The default case is a map iterating
     * over quadrature points with a matrix of shape (nb_components × 1) per
     * field entry
     */
    StateFieldMap(TypedStateField<T> & state_field,
                  IterUnit iter_type = IterUnit::SubPt);

    /**
     * Constructor from a state field with explicitly chosen shape of iterate.
     * (the number of columns is inferred).
     */
    StateFieldMap(TypedStateField<T> & state_field, Index_t nb_rows,
                  IterUnit iter_type = IterUnit::SubPt);

    StateFieldMap(const StateFieldMap & other) = delete;

    //! Move constructor
    StateFieldMap(StateFieldMap && other) = delete;

    //! Destructor
    virtual ~StateFieldMap() = default;

    //! Copy assignment operator
    StateFieldMap & operator=(const StateFieldMap & other) = delete;

    //! Move assignment operator
    StateFieldMap & operator=(StateFieldMap && other) = delete;

    //! iterator type
    template <Mapping MutIter>
    class Iterator;

    //! stl
    using iterator =
        Iterator<(Mutability == Mapping::Mut) ? Mapping::Mut : Mapping::Const>;

    //! stl
    using const_iterator = Iterator<Mapping::Const>;

    //! stl
    iterator begin();

    //! stl
    iterator end();

    //! return a const reference to the mapped state field
    const TypedStateField<T> & get_state_field() const;

    //! return the number of rows the iterates have
    const Index_t & get_nb_rows() const;

    /**
     * returns the number of iterates produced by this map (corresponds to
     * the number of field entries if Iteration::Quadpt, or the number of
     * pixels/voxels if Iteration::Pixel);
     */
    size_t size() const;

    /**
     * The iterate needs to give access to current or previous values. This is
     * handled by the `muGrid::StateFieldMap::StateWrapper`, a light-weight
     * wrapper around the iterate's data.
     */
    template <Mapping MutWrapper>
    class StateWrapper {
     public:
      //! convenience alias
      using StateFieldMap_t =
          std::conditional_t<MutWrapper == Mapping::Const, const StateFieldMap,
                             StateFieldMap>;

      //! return value when getting current value from iterate
      using CurrentVal_t = typename FieldMap_t::template Return_t<MutWrapper>;

      //! return value when getting old value from iterate
      using OldVal_t = typename FieldMap_t::template Return_t<Mapping::Const>;

      //! constructor (should never have to be called by user)
      StateWrapper(StateFieldMap_t & state_field_map, size_t index)
          : current_val{state_field_map.get_current()[index]} {
        const Index_t nb_memory{state_field_map.state_field.get_nb_memory()};
        this->old_vals.reserve(nb_memory);
        for (Index_t i{1}; i < nb_memory + 1; ++i) {
          this->old_vals.emplace_back(
              std::move(state_field_map.get_old(i))[index]);
        }
      }
      ~StateWrapper() = default;

      //! return the current value at this iterate
      CurrentVal_t & current() { return this->current_val; }

      //! return the value at this iterate which was current `nb_steps_ago` ago
      const OldVal_t & old(size_t nb_steps_ago) const {
        return this->old_vals.at(nb_steps_ago - 1);
      }

     protected:
      //! current value at this iterate
      CurrentVal_t current_val;
      //! all old values at this iterate
      std::vector<OldVal_t> old_vals{};
    };

    //! random access operator
    StateWrapper<Mutability> operator[](size_t index) {
      return StateWrapper<Mutability>{*this, index};
    }

    //! random constaccess operator
    StateWrapper<Mapping::Const> operator[](size_t index) const {
      return StateWrapper<Mapping::Const>{*this, index};
    }

    //! returns a reference to the map over the current data
    FieldMap_t & get_current();

    //! returns a const reference to the map over the current data
    const FieldMap_t & get_current() const;
    /**
     * returns a const reference to the map over the data which was current
     * `nb_steps_ago` ago
     */
    const CFieldMap_t & get_old(size_t nb_steps_ago) const;

   protected:
    //! protected access to the constituent fields
    RefVector<Field> & get_fields();

    //! mapped state field. Needed for query at initialisations
    TypedStateField<T> & state_field;
    const IterUnit iteration;  //!< type of map iteration
    const Index_t nb_rows;        //!< number of rows of the iterate

    /**
     * maps over nb_memory + 1 possibly mutable maps. current points to one of
     * these
     */
    std::vector<FieldMap_t> maps;

    //! helper function creating the list of maps to store for current values
    std::vector<FieldMap_t> make_maps(RefVector<Field> & fields);

    /**
     * maps over nb_memory + 1 const maps. old(nb_steps_ago) points to one of
     * these
     */
    std::vector<CFieldMap_t> cmaps;

    //! helper function creating the list of maps to store for old values
    std::vector<CFieldMap_t> make_cmaps(RefVector<Field> & fields);
  };

  /**
   * Iterator class for `muGrid::StateFieldMap`
   */
  template <typename T, Mapping Mutability>
  template <Mapping MutIter>
  class StateFieldMap<T, Mutability>::Iterator {
   public:
    //! convenience alias
    using StateFieldMap_t =
        std::conditional_t<MutIter == Mapping::Const, const StateFieldMap,
                           StateFieldMap>;
    //! const-correct proxy for iterates
    using StateWrapper_t =
        typename StateFieldMap::template StateWrapper<MutIter>;
    //! Deleted default constructor
    Iterator() = delete;

    //! constructor (should never have to be called by the user)
    Iterator(StateFieldMap_t & state_field_map, size_t index);

    //! Copy constructor
    Iterator(const Iterator & other) = delete;

    //! Move constructor
    Iterator(Iterator && other) = default;

    //! destructor
    virtual ~Iterator() = default;

    //! Copy assignment operator
    Iterator & operator=(const Iterator & other) = delete;

    //! Move assignment operator
    Iterator & operator=(Iterator && other) = default;

    //! comparison
    bool operator!=(const Iterator & other) {
      return this->index != other.index;
    }

    //! pre-increment
    Iterator & operator++() {
      ++this->index;
      return *this;
    }

    //! dereference
    StateWrapper_t operator*() {
      return StateWrapper_t{this->state_field_map, this->index};
    }

   protected:
    //! reference back to the iterated map
    StateFieldMap_t & state_field_map;
    //! current iteration progress
    size_t index;
  };

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_STATE_FIELD_MAP_HH_
