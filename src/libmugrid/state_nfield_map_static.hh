/**
 * @file   state_nfield_map_static.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   27 Aug 2019
 *
 * @brief  header-only implementation of state field maps with statically known
 *         iterate sizes
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

#ifndef SRC_LIBMUGRID_STATE_NFIELD_MAP_STATIC_HH_
#define SRC_LIBMUGRID_STATE_NFIELD_MAP_STATIC_HH_

#include "state_nfield_map.hh"
#include "nfield_map_static.hh"

#include <array>
#include <sstream>
#include <utility>

namespace muGrid {

  /* ---------------------------------------------------------------------- */
  template <typename T, bool ConstField, class MapType, size_t NbMemory,
            Iteration IterationType = Iteration::QuadPt>
  class StaticStateNFieldMap : public StateNFieldMap<T, ConstField> {
    static_assert(MapType::IsValidStaticMapType(),
                  "The MapType you chose is not compatible");

   public:
    using Parent = StateNFieldMap<T, ConstField>;
    using StaticNFieldMap_t =
        StaticNFieldMap<T, ConstField, MapType, IterationType>;
    using CStaticNFieldMap_t = StaticNFieldMap<T, true, MapType, IterationType>;
    using MapArray_t = std::array<StaticNFieldMap_t, NbMemory + 1>;
    using CMapArray_t = std::array<CStaticNFieldMap_t, NbMemory + 1>;

    //! Default constructor
    StaticStateNFieldMap() = delete;

    /** constructor from a state field. The default case is a map iterating
     * over quadrature points with a matrix of shape (nb_components × 1) per
     * field entry
     */
    explicit StaticStateNFieldMap(TypedStateNField<T> & state_field)
        : Parent{state_field, IterationType}, static_maps{this->make_maps()},
          static_cmaps{this->make_cmaps()} {}

    //! Copy constructor
    StaticStateNFieldMap(const StaticStateNFieldMap & other) = delete;

    StaticStateNFieldMap(StaticStateNFieldMap && other) = default;

    //! Destructor
    virtual ~StaticStateNFieldMap() = default;

    //! Copy assignment operator
    StaticStateNFieldMap &
    operator=(const StaticStateNFieldMap & other) = delete;

    //! Move assignment operator
    StaticStateNFieldMap & operator=(StaticStateNFieldMap && other) = default;

    template <bool ConstIter>
    class Iterator;
    using iterator = Iterator<false or ConstField>;
    using const_iterator = Iterator<true>;

    iterator begin() { return iterator{*this, 0}; }
    iterator end() { return iterator{*this, this->static_maps.front().size()}; }

    /* ---------------------------------------------------------------------- */
    const CStaticNFieldMap_t & get_old_static(size_t nb_steps_ago) const {
      return this->static_cmaps[this->state_field.get_indices()[nb_steps_ago]];
    }

    StaticNFieldMap_t & get_current_static() {
      return static_maps[this->state_field.get_indices()[0]];
    }
    StaticNFieldMap_t & get_current_static() const {
      return this->static_maps[this->state_field.get_indices()[0]];
    }

    template <bool ConstWrapper>
    class StaticStateWrapper {
     public:
      using StaticStateNFieldMap_t =
          std::conditional_t<ConstWrapper, const StaticStateNFieldMap,
                             StaticStateNFieldMap>;
      using CurrentVal_t = typename MapType::template ref_type<ConstWrapper>;
      using CurrentStorage_t =
          typename MapType::template storage_type<ConstWrapper>;
      using OldVal_t =  typename MapType::template ref_type<true>;
      using OldStorage_t = typename MapType::template storage_type<true>;
      StaticStateWrapper(StaticStateNFieldMap_t & state_field_map, size_t index)
          : current_val(MapType::template to_storage<ConstWrapper>(
                state_field_map.get_current_static()[index])),
            old_vals{this->make_old_vals_static(state_field_map, index)} {}
      ~StaticStateWrapper() = default;

      CurrentVal_t & current() {
        return MapType::template provide_ref<ConstWrapper>(this->current_val);
      }

      const OldVal_t & old(size_t nb_steps_ago) const {
        return this->old_vals[nb_steps_ago - 1];
      }

     protected:
      CurrentStorage_t current_val;
      std::array<OldStorage_t, NbMemory> old_vals{};
      std::array<OldStorage_t, NbMemory>
      make_old_vals_static(StaticStateNFieldMap_t & state_field_map,
                           size_t index) {
        return this->old_vals_helper_static(
            state_field_map, index, std::make_index_sequence<NbMemory>{});
      }
      template <size_t... NbStepsAgo>
      std::array<OldStorage_t, NbMemory>
      old_vals_helper_static(StaticStateNFieldMap_t & state_field_map,
                             size_t index, std::index_sequence<NbStepsAgo...>) {
        return std::array<OldStorage_t, NbMemory>{
            MapType::template to_storage<true>(
                state_field_map.get_old_static(NbStepsAgo)[index])...};
      }
    };

    StaticStateWrapper<ConstField> operator[](size_t index) {
      return StaticStateWrapper<ConstField>{*this, index};
    }

    StaticStateWrapper<true> operator[](size_t index) const {
      return StaticStateWrapper<true>{*this, index};
    }

    void initialise() {
      for (auto && map : this->static_maps) {
        map.initialise();
      }
      for (auto && map : this->static_cmaps) {
        map.initialise();
      }
      Parent::initialise();
    }

   protected:
    /* ---------------------------------------------------------------------- */
    template <bool ConstIter>
    using HelperRet_t = std::conditional_t<ConstIter, CMapArray_t, MapArray_t>;

    /* ---------------------------------------------------------------------- */
    template <bool ConstIter, size_t... I>
    inline auto map_helper(std::index_sequence<I...>) -> HelperRet_t<ConstIter>;

    /* ---------------------------------------------------------------------- */
    inline MapArray_t make_maps();
    inline CMapArray_t make_cmaps();

    MapArray_t static_maps;
    CMapArray_t static_cmaps;
  };

  /* ---------------------------------------------------------------------- */
  template <typename T, bool ConstField, class MapType, size_t NbMemory,
            Iteration IterationType>
  template <bool ConstIter, size_t... I>
  auto
  StaticStateNFieldMap<T, ConstField, MapType, NbMemory,
                       IterationType>::map_helper(std::index_sequence<I...>)
      -> HelperRet_t<ConstIter> {
    using Array_t = std::conditional_t<ConstIter, CMapArray_t, MapArray_t>;
    using Map_t =
        std::conditional_t<ConstIter, CStaticNFieldMap_t, StaticNFieldMap_t>;
    auto & fields{this->get_fields()};
    return Array_t{Map_t(static_cast<TypedNFieldBase<T> &>(fields[I]))...};
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, bool ConstField, class MapType, size_t NbMemory,
            Iteration IterationType>
  auto StaticStateNFieldMap<T, ConstField, MapType, NbMemory,
                            IterationType>::make_maps() -> MapArray_t {
    if (this->state_field.get_nb_memory() != NbMemory) {
      std::stringstream error{};
      error << "You ar trying to map a state field with a memory size of "
            << this->state_field.get_nb_memory()
            << " using a static map with a memory size of " << NbMemory << ".";
      throw NFieldMapError{error.str()};
    }

    return this->map_helper<ConstField>(
        std::make_index_sequence<NbMemory + 1>{});
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, bool ConstField, class MapType, size_t NbMemory,
            Iteration IterationType>
  auto StaticStateNFieldMap<T, ConstField, MapType, NbMemory,
                            IterationType>::make_cmaps() -> CMapArray_t {
    return this->map_helper<true>(std::make_index_sequence<NbMemory + 1>{});
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, bool ConstField, class MapType, size_t NbMemory,
            Iteration IterationType>
  template <bool ConstIter>
  class StaticStateNFieldMap<T, ConstField, MapType, NbMemory,
                             IterationType>::Iterator {
   public:
    using StaticStateNFieldMap_t =
        std::conditional_t<ConstIter, const StaticStateNFieldMap,
                           StaticStateNFieldMap>;
    using StateWrapper_t =
        typename StaticStateNFieldMap::template StaticStateWrapper<ConstIter>;

    //! Default constructor
    Iterator() = delete;

    //! Copy constructor
    Iterator(const Iterator & other) = delete;

    Iterator(StaticStateNFieldMap_t & state_field_map, size_t index)
        : state_field_map{state_field_map}, index{index} {}

    //! Move constructor
    Iterator(Iterator && other) = default;

    //! Destructor
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

    StateWrapper_t operator*() {
      return StateWrapper_t{this->state_field_map, this->index};
    }

   protected:
    StaticStateNFieldMap_t & state_field_map;
    size_t index;
  };

  /**
   * Convenience aliases to useful state field maps
   */
  /* ---------------------------------------------------------------------- */
  template <typename T, bool ConstField, Dim_t NbRow, Dim_t NbCol,
            size_t NbMemory, Iteration IterationType = Iteration::QuadPt>
  using MatrixStateNFieldMap =
      StaticStateNFieldMap<T, ConstField, internal::MatrixMap<T, NbRow, NbCol>,
                           NbMemory, IterationType>;

  /* ---------------------------------------------------------------------- */
  template <typename T, bool ConstField, Dim_t NbRow, Dim_t NbCol,
            size_t NbMemory, Iteration IterationType = Iteration::QuadPt>
  using ArrayStateNFieldMap =
      StaticStateNFieldMap<T, ConstField, internal::ArrayMap<T, NbRow, NbCol>,
                           NbMemory, IterationType>;

  //! the following only make sense as per-quadrature-point maps
  /* ---------------------------------------------------------------------- */
  template <typename T, bool ConstField, size_t NbMemory>
  using ScalarStateNFieldMap =
      StaticStateNFieldMap<T, ConstField, internal::ScalarMap<T>, NbMemory,
                           Iteration::QuadPt>;

  /* ---------------------------------------------------------------------- */
  template <typename T, bool ConstField, Dim_t Dim, size_t NbMemory>
  using T2StateNFieldMap =
      StaticStateNFieldMap<T, ConstField, internal::MatrixMap<T, Dim, Dim>,
                           NbMemory, Iteration::QuadPt>;

  template <typename T, bool ConstField, Dim_t Dim, size_t NbMemory>
  using T4StateNFieldMap =
      StaticStateNFieldMap<T, ConstField,
                           internal::MatrixMap<T, Dim * Dim, Dim * Dim>,
                           NbMemory, Iteration::QuadPt>;
}  // namespace muGrid

#endif  // SRC_LIBMUGRID_STATE_NFIELD_MAP_STATIC_HH_
