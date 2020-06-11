/**
 * @file   state_field_map_static.hh
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

#ifndef SRC_LIBMUGRID_STATE_FIELD_MAP_STATIC_HH_
#define SRC_LIBMUGRID_STATE_FIELD_MAP_STATIC_HH_

#include "state_field_map.hh"
#include "field_map_static.hh"
#include "field_typed.hh"

#include <array>
#include <sstream>
#include <utility>

namespace muGrid {

  /**
   * statically sized version of `muGrid::TypedStateField`. Duplicates its
   * capabilities, with much more efficient statically sized iterates.
   */
  template <typename T, Mapping Mutability, class MapType, size_t NbMemory,
            IterUnit IterationType = IterUnit::SubPt>
  class StaticStateFieldMap : public StateFieldMap<T, Mutability> {
    static_assert(MapType::IsValidStaticMapType(),
                  "The MapType you chose is not compatible");

   public:
    //! stored scalar type
    using Scalar = T;

    //! base class
    using Parent = StateFieldMap<T, Mutability>;

    //! convenience alias for current map
    using StaticFieldMap_t =
        StaticFieldMap<T, Mutability, MapType, IterationType>;

    //! convenience alias for old map
    using CStaticFieldMap_t =
        StaticFieldMap<T, Mapping::Const, MapType, IterationType>;

    //! storage type for current maps
    using MapArray_t = std::array<StaticFieldMap_t, NbMemory + 1>;

    //! storage type for old maps
    using CMapArray_t = std::array<CStaticFieldMap_t, NbMemory + 1>;

    //! determine at compile time the number of old values stored
    constexpr static size_t GetNbMemory() { return NbMemory; }

    //! determine the map's mutability at compile time
    constexpr static Mapping FieldMutability() { return Mutability; }

    //! determine the map's iteration type (pixels vs quad pts) at compile time
    constexpr static IterUnit GetIterationType() { return IterationType; }

    //! Deleted default constructor
    StaticStateFieldMap() = delete;

    /** constructor from a state field. The default case is a map iterating
     * over quadrature points with a matrix of shape (nb_components × 1) per
     * field entry
     */
    explicit StaticStateFieldMap(TypedStateField<T> & state_field)
        : Parent{state_field, IterationType}, static_maps{this->make_maps()},
          static_cmaps{this->make_cmaps()} {}

    //! Deleted copy constructor
    StaticStateFieldMap(const StaticStateFieldMap & other) = delete;

    //! Move constructor
    StaticStateFieldMap(StaticStateFieldMap && other) = default;

    //! Destructor
    virtual ~StaticStateFieldMap() = default;

    //! Copy assignment operator
    StaticStateFieldMap & operator=(const StaticStateFieldMap & other) = delete;

    //! Move assignment operator
    StaticStateFieldMap & operator=(StaticStateFieldMap && other) = default;

    //! froward declaration of iterator class
    template <Mapping MutIter>
    class Iterator;

    //! stl
    using iterator =
        Iterator<(Mutability == Mapping::Mut) ? Mapping::Mut : Mapping::Const>;

    //! stl
    using const_iterator = Iterator<Mapping::Const>;

    //! stl
    iterator begin() { return iterator{*this, 0}; }

    //! stl
    iterator end() { return iterator{*this, this->static_maps.front().size()}; }

    //! return a const ref to an old value map
    const CStaticFieldMap_t & get_old_static(size_t nb_steps_ago) const {
      return this->static_cmaps[this->state_field.get_indices()[nb_steps_ago]];
    }

    //! return a ref to an the current map
    StaticFieldMap_t & get_current_static() {
      auto && indices = this->state_field.get_indices();
      return static_maps[indices[0]];
    }

    StaticFieldMap_t & get_current() { return this->get_current_static(); }

    //! return a const ref to an the current map
    StaticFieldMap_t & get_current_static() const {
      return this->static_maps[this->state_field.get_indices()[0]];
    }

    StaticFieldMap_t & get_current() const {
      return this->get_current_static();
    }

    /**
     * The iterate needs to give access to current or previous values. This is
     * handled by the `muGrid::StaticStateFieldMap::StateWrapper`, a
     * light-weight wrapper around the iterate's data.
     * @tparam MutWrapper mutability of the mapped field. It should never be
     * necessary to set this manually, rather the iterators dereference
     * operator*() should return the correct type.
     */
    template <Mapping MutWrapper>
    class StaticStateWrapper {
     public:
      //! const-correct map
      using StaticStateFieldMap_t =
          std::conditional_t<MutWrapper == Mapping::Const,
                             const StaticStateFieldMap, StaticStateFieldMap>;

      //! return type handle for current value
      using CurrentVal_t = typename MapType::template ref_type<MutWrapper>;

      //! storage type for current value handle
      using CurrentStorage_t =
          typename MapType::template storage_type<MutWrapper>;

      //! return type handle for old value
      using OldVal_t = typename MapType::template ref_type<Mapping::Const>;

      //! storage type for old value handle
      using OldStorage_t =
          typename MapType::template storage_type<Mapping::Const>;

      //! constructor with map and index, not for user to call
      StaticStateWrapper(StaticStateFieldMap_t & state_field_map, size_t index)
          : current_val(MapType::template to_storage<MutWrapper>(
                state_field_map.get_current_static()[index])),
            old_vals{this->make_old_vals_static(state_field_map, index)} {}
      ~StaticStateWrapper() = default;

      //! return the current value of the iterate
      CurrentVal_t & current() {
        return MapType::template provide_ref<MutWrapper>(this->current_val);
      }

      /**
       * return the value of the iterate which was current `nb_steps_ago` steps
       * ago. Possibly has excess runtime cost compared to the next function,
       * and has no bounds checking, unlike the next function
       */
      const OldVal_t & old(size_t nb_steps_ago) const {
        return MapType::template provide_const_ref<Mapping::Const>(
            this->old_vals[nb_steps_ago - 1]);
      }

      /**
       * return the value of the iterate which was current `NbStepsAgo` steps
       * ago
       */
      template <size_t NbStepsAgo = 1>
      const OldVal_t & old() const {
        static_assert(NbStepsAgo <= NbMemory, "NbStepsAgo out of range");
        return MapType::template provide_const_ref<Mapping::Const>(
            this->old_vals[NbStepsAgo - 1]);
      }

     protected:
      //! handle to current value
      CurrentStorage_t current_val;

      //! storage for handles to old values
      std::array<OldStorage_t, NbMemory> old_vals{};

      //! helper function to build the list of old values
      std::array<OldStorage_t, NbMemory>
      make_old_vals_static(StaticStateFieldMap_t & state_field_map,
                           size_t index) {
        return this->old_vals_helper_static(
            state_field_map, index, std::make_index_sequence<NbMemory>{});
      }

      //! helper function to build the list of old values
      template <size_t... NbStepsAgo>
      std::array<OldStorage_t, NbMemory>
      old_vals_helper_static(StaticStateFieldMap_t & state_field_map,
                             size_t index, std::index_sequence<NbStepsAgo...>) {
        // the offset "NbStepsAgo + 1" below is because any old value is
        // necessarily at least one step old
        return std::array<OldStorage_t, NbMemory>{
            MapType::template to_storage<Mapping::Const>(
                state_field_map.get_old_static(NbStepsAgo + 1)[index])...};
      }
    };

    //! random access operator
    StaticStateWrapper<Mutability> operator[](size_t index) {
      return StaticStateWrapper<Mutability>{*this, index};
    }

    //! random const access operator
    StaticStateWrapper<Mapping::Const> operator[](size_t index) const {
      return StaticStateWrapper<Mapping::Const>{*this, index};
    }

   protected:
    //! internal convenience alias
    template <Mapping MutIter>
    using HelperRet_t =
        std::conditional_t<MutIter == Mapping::Const, CMapArray_t, MapArray_t>;

    //! helper for building the maps
    template <Mapping MutIter, size_t... I>
    inline auto map_helper(std::index_sequence<I...>) -> HelperRet_t<MutIter>;

    //! build the current value maps
    inline MapArray_t make_maps();

    //! build the old value maps
    inline CMapArray_t make_cmaps();

    //! container for current maps
    MapArray_t static_maps;

    //! container for old maps
    CMapArray_t static_cmaps;
  };

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability, class MapType, size_t NbMemory,
            IterUnit IterationType>
  template <Mapping MutIter, size_t... I>
  auto StaticStateFieldMap<T, Mutability, MapType, NbMemory,
                           IterationType>::map_helper(std::index_sequence<I...>)
      -> HelperRet_t<MutIter> {
    using Array_t =
        std::conditional_t<MutIter == Mapping::Const, CMapArray_t, MapArray_t>;
    using Map_t = std::conditional_t<MutIter == Mapping::Const,
                                     CStaticFieldMap_t, StaticFieldMap_t>;
    auto & fields{this->get_fields()};
    return Array_t{Map_t(dynamic_cast<TypedFieldBase<T> &>(fields[I]))...};
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability, class MapType, size_t NbMemory,
            IterUnit IterationType>
  auto StaticStateFieldMap<T, Mutability, MapType, NbMemory,
                           IterationType>::make_maps() -> MapArray_t {
    auto && nb_memory{ this->state_field.get_nb_memory()};
    if (nb_memory != NbMemory) {
      std::stringstream error{};
      error << "You ar trying to map a state field with a memory size of "
            << this->state_field.get_nb_memory()
            << " using a static map with a memory size of " << NbMemory << ".";
      throw FieldMapError{error.str()};
    }

    return this->map_helper<Mutability>(
        std::make_index_sequence<NbMemory + 1>{});
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability, class MapType, size_t NbMemory,
            IterUnit IterationType>
  auto StaticStateFieldMap<T, Mutability, MapType, NbMemory,
                           IterationType>::make_cmaps() -> CMapArray_t {
    return this->map_helper<Mapping::Const>(
        std::make_index_sequence<NbMemory + 1>{});
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability, class MapType, size_t NbMemory,
            IterUnit IterationType>
  template <Mapping MutIter>
  class StaticStateFieldMap<T, Mutability, MapType, NbMemory,
                            IterationType>::Iterator {
   public:
    //! const correct iterated map
    using StaticStateFieldMap_t =
        std::conditional_t<MutIter == Mapping::Const, const StaticStateFieldMap,
                           StaticStateFieldMap>;
    //! convenience alias to dererencing return type
    using StateWrapper_t =
        typename StaticStateFieldMap::template StaticStateWrapper<MutIter>;

    //! Default constructor
    Iterator() = delete;

    //! Copy constructor
    Iterator(const Iterator & other) = delete;

    //! constructor with field map and index, not for user to call
    Iterator(StaticStateFieldMap_t & state_field_map, size_t index)
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
    bool operator!=(const Iterator & other) const {
      return this->index != other.index;
    }

    //! comparison (needed by akantu::iterator
    bool operator==(const Iterator & other) const {
      return not(*this != other);
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
    //! reference bap to iterated map
    StaticStateFieldMap_t & state_field_map;
    //! current progress in iteration
    size_t index;
  };

  /**
   * Alias of `muGrid::StaticStateFieldMap` you wish to iterate over pixel by
   * pixel or quadrature point by quadrature point with statically sized
   * `Eigen::Matrix` iterates
   *
   * @tparam T scalar type stored in the field, must be one of `muGrid::Real`,
   * `muGrid::Int`, `muGrid::Uint`, `muGrid::Complex`
   * @tparam Mutability whether or not the map allows to modify the content of
   * the field
   * @tparam NbRow number of rows of the iterate
   * @tparam NbCol number of columns of the iterate
   * @tparam NbMemory number of previous values to store
   * @tparam IterationType whether to iterate over pixels or quadrature points
   */
  template <typename T, Mapping Mutability, Dim_t NbRow, Dim_t NbCol,
            size_t NbMemory, IterUnit IterationType = IterUnit::SubPt>
  using MatrixStateFieldMap =
      StaticStateFieldMap<T, Mutability, internal::MatrixMap<T, NbRow, NbCol>,
                          NbMemory, IterationType>;

  /**
   * Alias of `muGrid::StaticStateFieldMap` you wish to iterate over pixel by
   * pixel or quadrature point by quadrature point with* statically sized
   * `Eigen::Array` iterates
   *
   * @tparam T scalar type stored in the field, must be one of `muGrid::Real`,
   * `muGrid::Int`, `muGrid::Uint`, `muGrid::Complex`
   * @tparam Mutability whether or not the map allows to modify the content of
   * the field
   * @tparam NbRow number of rows of the iterate
   * @tparam NbCol number of columns of the iterate
   * @tparam NbMemory number of previous values to store
   * @tparam IterationType describes the pixel-subdivision
   */
  template <typename T, Mapping Mutability, Dim_t NbRow, Dim_t NbCol,
            size_t NbMemory, IterUnit IterationType>
  using ArrayStateFieldMap =
      StaticStateFieldMap<T, Mutability, internal::ArrayMap<T, NbRow, NbCol>,
                          NbMemory, IterationType>;

  /**
   * Alias of `muGrid::StaticStateFieldMap` over a scalar field you wish to
   * iterate over quadrature point by quadrature point.
   *
   * @tparam T scalar type stored in the field, must be one of `muGrid::Real`,
   * `muGrid::Int`, `muGrid::Uint`, `muGrid::Complex`
   * @tparam Mutability whether or not the map allows to modify the content of
   * the field
   * @tparam NbMemory number of previous values to store
   * @tparam IterationType describes the pixel-subdivision
   */
  template <typename T, Mapping Mutability, size_t NbMemory,
            IterUnit IterationType>
  using ScalarStateFieldMap =
      StaticStateFieldMap<T, Mutability, internal::ScalarMap<T>, NbMemory,
                          IterationType>;

  /**
   * Alias of `muGrid::StaticStateNFieldMap` over a first-rank tensor field you
   * wish to iterate over quadrature point by quadrature point.
   *
   * @tparam T scalar type stored in the field, must be one of `muGrid::Real`,
   * `muGrid::Int`, `muGrid::Uint`, `muGrid::Complex`
   * @tparam Mutability whether or not the map allows to modify the content of
   * the field
   * @tparam Dim spatial dimension of the tensor
   * @tparam NbMemory number of previous values to store
   * @tparam IterationType describes the pixel-subdivision
   */
  template <typename T, Mapping Mutability, Dim_t Dim, size_t NbMemory,
            IterUnit IterationType>
  using T1StateFieldMap =
      StaticStateFieldMap<T, Mutability, internal::MatrixMap<T, Dim, 1>,
                          NbMemory, IterationType>;

  /**
   * Alias of `muGrid::StaticStateNFieldMap` over a second-rank tensor field you
   * wish to iterate over quadrature point by quadrature point.
   *
   * @tparam T scalar type stored in the field, must be one of `muGrid::Real`,
   * `muGrid::Int`, `muGrid::Uint`, `muGrid::Complex`
   * @tparam Mutability whether or not the map allows to modify the content of
   * the field
   * @tparam Dim spatial dimension of the tensor
   * @tparam NbMemory number of previous values to store
   * @tparam IterationType describes the pixel-subdivision
   */
  template <typename T, Mapping Mutability, Dim_t Dim, size_t NbMemory,
            IterUnit IterationType>
  using T2StateFieldMap =
      StaticStateFieldMap<T, Mutability, internal::MatrixMap<T, Dim, Dim>,
                          NbMemory, IterationType>;

  /**
   * Alias of `muGrid::StaticStateFieldMap` over a fourth-rank tensor field you
   * wish to iterate over quadrature point by quadrature point.
   *
   * @tparam T scalar type stored in the field, must be one of `muGrid::Real`,
   * `muGrid::Int`, `muGrid::Uint`, `muGrid::Complex`
   * @tparam Mutability whether or not the map allows to modify the content of
   * the field
   * @tparam Dim spatial dimension of the tensor
   * @tparam NbMemory number of previous values to store
   * @tparam IterationType describes the pixel-subdivision
   */
  template <typename T, Mapping Mutability, Dim_t Dim, size_t NbMemory,
            IterUnit IterationType>
  using T4StateFieldMap =
      StaticStateFieldMap<T, Mutability,
                          internal::MatrixMap<T, Dim * Dim, Dim * Dim>,
                          NbMemory, IterationType>;
}  // namespace muGrid

#endif  // SRC_LIBMUGRID_STATE_FIELD_MAP_STATIC_HH_
