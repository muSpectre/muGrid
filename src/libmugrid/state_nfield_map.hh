/**
 * @file   state_nfield_map.hh
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
 * General Public License for more details.
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
 */

#ifndef SRC_LIBMUGRID_STATE_NFIELD_MAP_HH_
#define SRC_LIBMUGRID_STATE_NFIELD_MAP_HH_

#include "grid_common.hh"
#include "nfield_map.hh"
#include "ref_vector.hh"

#include <vector>
#include <memory>

namespace muGrid {

  //! forward declaration
  template <typename T>
  class TypedStateNField;
  class NField;

  template <typename T, Mapping Mutability>
  class StateNFieldMap {
   public:
    using NFieldMap_t = NFieldMap<T, Mutability>;
    using CNFieldMap_t = NFieldMap<T, Mapping::Const>;
    using CurrentIteratort = typename NFieldMap_t::iterator;
    using OldIteratort = typename CNFieldMap_t::iterator;

    //! Default constructor
    StateNFieldMap() = delete;

    /** constructor from a state field. The default case is a map iterating
     * over quadrature points with a matrix of shape (nb_components × 1) per
     * field entry
     */
    StateNFieldMap(TypedStateNField<T> & state_field,
                   Iteration iter_type = Iteration::QuadPt);

    /**
     * Constructor from a state field with explicitly chosen shape of iterate.
     * (the number of columns is inferred).
     */
    StateNFieldMap(TypedStateNField<T> & state_field, Dim_t nb_rows,
                   Iteration iter_type = Iteration::QuadPt);

    StateNFieldMap(const StateNFieldMap & other) = delete;

    //! Move constructor
    StateNFieldMap(StateNFieldMap && other) = delete;

    //! Destructor
    virtual ~StateNFieldMap() = default;

    //! Copy assignment operator
    StateNFieldMap & operator=(const StateNFieldMap & other) = delete;

    //! Move assignment operator
    StateNFieldMap & operator=(StateNFieldMap && other) = delete;

    template <Mapping MutIter>
    class Iterator;
    using iterator =
        Iterator<(Mutability == Mapping::Mut) ? Mapping::Mut : Mapping::Const>;
    using const_iterator = Iterator<Mapping::Const>;

    iterator begin();
    iterator end();
    // const_iterator begin() const;
    // const_iterator end() const;

    const TypedStateNField<T> & get_state_field() const;

    const Dim_t & get_nb_rows() const;

    /**
     * returns the number of iterates produced by this map (corresponds to
     * the number of field entries if Iteration::Quadpt, or the number of
     * pixels/voxels if Iteration::Pixel);
     */
    size_t size() const;

    template <Mapping MutWrapper>
    class StateWrapper {
     public:
      using StateNFieldMap_t =
          std::conditional_t<MutWrapper == Mapping::Const, const StateNFieldMap,
                             StateNFieldMap>;
      using CurrentVal_t =
          typename NFieldMap_t::template value_type<MutWrapper>;
      using OldVal_t = typename NFieldMap_t::template value_type<Mapping::Const>;
      StateWrapper(StateNFieldMap_t & state_field_map, size_t index)
          : current_val{state_field_map.get_current()[index]} {
        const Dim_t nb_memory{state_field_map.state_field.get_nb_memory()};
        this->old_vals.reserve(nb_memory);
        for (Dim_t i{1}; i < nb_memory + 1 ; ++i) {
          this->old_vals.emplace_back(
              std::move(state_field_map.get_old(i))[index]);
        }
      }
      ~StateWrapper() = default;

      CurrentVal_t & current() { return this->current_val; }
      const OldVal_t & old(size_t nb_steps_ago) const {
        return this->old_vals.at(nb_steps_ago - 1);
      }

     protected:
      CurrentVal_t current_val;
      std::vector<OldVal_t> old_vals{};
    };

    StateWrapper<Mutability> operator[](size_t index) {
      return StateWrapper<Mutability>{*this, index};
    }

    StateWrapper<Mapping::Const> operator[](size_t index) const {
      return StateWrapper<Mapping::Const>{*this, index};
    }

    void initialise();

    NFieldMap_t & get_current();
    const NFieldMap_t & get_current() const;
    const CNFieldMap_t & get_old(size_t nb_steps_ago) const;

   protected:
    RefVector<NField> & get_fields();

    //< mapped state field. Needed for query at initialisations
    TypedStateNField<T> & state_field;
    const Iteration iteration;  //!< type of map iteration
    const Dim_t nb_rows;        //!< number of rows of the iterate
    std::vector<NFieldMap_t> maps;
    std::vector<NFieldMap_t> make_maps(RefVector<NField> & fields);
    std::vector<CNFieldMap_t> cmaps;
    std::vector<CNFieldMap_t> make_cmaps(RefVector<NField> & fields);
  };

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability>
  template <Mapping MutIter>
  class StateNFieldMap<T, Mutability>::Iterator {
   public:
    using StateNFieldMap_t =
        std::conditional_t<MutIter == Mapping::Const, const StateNFieldMap,
                           StateNFieldMap>;
    using StateWrapper_t =
        typename StateNFieldMap::template StateWrapper<MutIter>;
    //! Default constructor
    Iterator() = delete;

    Iterator(StateNFieldMap_t & state_field_map, size_t index);

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
    Iterator & operator++() { ++this->index; return *this;}

    StateWrapper_t operator*() {
      return StateWrapper_t{this->state_field_map, this->index};
    }

   protected:
    StateNFieldMap_t & state_field_map;
    size_t index;
  };

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_STATE_NFIELD_MAP_HH_
