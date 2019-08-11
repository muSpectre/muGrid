/**
 * @file   nfield_map_static.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   20 Aug 2019
 *
 * @brief  header-only implementation of field maps with statically known
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

#ifndef SRC_LIBMUGRID_NFIELD_MAP_STATIC_HH_
#define SRC_LIBMUGRID_NFIELD_MAP_STATIC_HH_

#include "nfield_map.hh"
#include "T4_map_proxy.hh"

#include <sstream>

namespace muGrid {

  /* ---------------------------------------------------------------------- */
  template <typename T, bool ConstField, class MapType,
            Iteration IterationType = Iteration::QuadPt>
  class StaticNFieldMap : public NFieldMap<T, ConstField> {
    static_assert(MapType::IsValidStaticMapType(),
                  "The MapType you chose is not compatible");

   public:
    using Parent = NFieldMap<T, ConstField>;
    using NField_t = typename Parent::NField_t;
    template <bool ConstNess>
    using return_type = typename MapType::template return_type<ConstNess>;
    using PlainType = typename MapType::PlainType;
    //! Default constructor
    StaticNFieldMap() = delete;

    explicit StaticNFieldMap(NField_t & field) : Parent{field, IterationType} {
      if (this->stride != MapType::stride()) {
        std::stringstream error{};
        error << "Incompatible number of components in the field '"
              << this->field.get_name() << "': The field map has a stride of "
              << this->stride << " but you wish an iterate with shape "
              << MapType::shape() << ", corresponding to a stride of "
              << MapType::stride() << ".";
        throw NFieldMapError(error.str());
      }
    }

    //! Copy constructor
    StaticNFieldMap(const StaticNFieldMap & other) = default;

    //! Move constructor
    StaticNFieldMap(StaticNFieldMap && other) = default;

    //! Destructor
    virtual ~StaticNFieldMap() = default;

    //! Copy assignment operator
    StaticNFieldMap & operator=(const StaticNFieldMap & other) = default;

    //! Move assignment operator
    StaticNFieldMap & operator=(StaticNFieldMap && other) = default;

    template <bool ConstIter>
    class Iterator;
    using iterator = Iterator<false or ConstField>;
    using const_iterator = Iterator<true>;


    return_type<ConstField> operator[](size_t index) {
      return MapType::template from_data_ptr<ConstField>(
          this->data_ptr + index * MapType::stride());
    }

    return_type<true> operator[](size_t index) const {
      return MapType::template from_data_ptr<true>(this->data_ptr +
                                                   index * MapType::stride());
    }

    iterator begin() {
      if (not this->is_initialised) {
        this->initialise();
      }
      return iterator{*this, false};
    }

    iterator end() { return iterator{*this, true}; }

    const_iterator begin() const {
      if (not this->is_initialised) {
        throw NFieldMapError("Const FieldMaps cannot be initialised");
      }
      return const_iterator{*this, false};
    }

    const_iterator end() const { return const_iterator{*this, true}; }

    //! evaluate the average of the field
    inline PlainType mean() const;
  };

  template <typename T, bool ConstField, class MapType, Iteration IterationType>
  template <bool ConstIter>
  class StaticNFieldMap<T, ConstField, MapType, IterationType>::Iterator {
   public:
    constexpr static size_t Stride{MapType::stride()};
    using value_type = typename MapType::template value_type<ConstIter>;
    using storage_type = typename MapType::template storage_type<ConstIter>;
    //! Default constructor
    Iterator() = delete;

    //! Constructor to beginning, or to end
    Iterator(const StaticNFieldMap & map, bool end)
        : map{map}, index{end ? map.size() : 0}, iterate{map.data_ptr} {}

    //! Copy constructor
    Iterator(const Iterator & other) = default;

    //! Move constructor
    Iterator(Iterator && other) = default;

    //! Destructor
    virtual ~Iterator() = default;

    //! Copy assignment operator
    Iterator & operator=(const Iterator & other) = default;

    //! Move assignment operator
    Iterator & operator=(Iterator && other) = default;

    //! pre-increment
    Iterator & operator++() {
      this->index++;
      new (&this->iterate)
          storage_type(this->map.data_ptr + this->index * Stride);
      return *this;
    }
    //! dereference
    inline value_type & operator*() {
      return MapType::template provide_ref<ConstIter>(this->iterate);
    }
    //! pointer to member
    inline value_type * operator->() {
      return (MapType::template provide_ptr<ConstIter>(this->iterate));
    }
    //! equality
    inline bool operator==(const Iterator & other) const {
      return this->index == other.index;
    }
    //! inequality
    inline bool operator!=(const Iterator & other) const {
      return not(*this == other);
    }

   protected:
    //! NFieldMap being iterated over
    const StaticNFieldMap & map;
    //! current iteration index
    size_t index;
    //! map which is being returned per iterate
    storage_type iterate;
  };

  namespace internal {

    template <typename T, class EigenPlain>
    struct EigenMap {
      constexpr static bool IsValidStaticMapType() { return true; }
      using PlainType = EigenPlain;
      template <bool ConstIter>
      using value_type =
          std::conditional_t<ConstIter, Eigen::Map<const PlainType>,
                             Eigen::Map<PlainType>>;
      template <bool ConstIter>
      using ref_type = value_type<ConstIter>;
      // for direct access through operator[]
      template <bool ConstIter>
      using return_type = value_type<ConstIter>;

      template <bool ConstIter>
      using storage_type = value_type<ConstIter>;

      template <bool ConstIter>
      constexpr static value_type<ConstIter> &
      provide_ref(storage_type<ConstIter> & storage) {
        return storage;
      }

      template <bool ConstIter>
      constexpr static value_type<ConstIter> *
      provide_ptr(storage_type<ConstIter> & storage) {
        return &storage;
      }

      template <bool ConstIter>
      constexpr static return_type<ConstIter>
      from_data_ptr(std::conditional_t<ConstIter, const T *, T *> data) {
        return return_type<ConstIter>(data);
      }

      template <bool ConstIter>
      constexpr static storage_type<ConstIter>
      to_storage(value_type<ConstIter> && value) {
        return std::move(value);
      }

      constexpr static Dim_t stride() { return PlainType::SizeAtCompileTime; }

      static_assert(stride() > 0,
                    "Only statically sized Eigen types allowed here");
      static std::string shape() {
        std::stringstream shape_stream{};
        shape_stream << PlainType::RowsAtCompileTime << " × "
                     << PlainType::ColsAtCompileTime;
        return shape_stream.str();
      }
    };

    template <typename T, Dim_t NbRow, Dim_t NbCol>
    using MatrixMap = EigenMap<T, Eigen::Matrix<T, NbRow, NbCol>>;

    template <typename T, Dim_t NbRow, Dim_t NbCol>
    using ArrayMap = EigenMap<T, Eigen::Array<T, NbRow, NbCol>>;

    template <typename T>
    struct ScalarMap {
      constexpr static bool IsValidStaticMapType() { return true; }
      using PlainType = T;
      template <bool ConstIter>
      using value_type = std::conditional_t<ConstIter, const T, T>;
      template <bool ConstIter>
      using ref_type = value_type<ConstIter> &;

      // for direct access through operator[]
      template <bool ConstIter>
      using return_type = value_type<ConstIter> &;

      // need to encapsulate
      template <bool ConstIter>
      using storage_type = std::conditional_t<ConstIter, const T *, T *>;

      template <bool ConstIter>
      constexpr static value_type<ConstIter> &
      provide_ref(storage_type<ConstIter> storage) {
        return *storage;
      }

      template <bool ConstIter>
      constexpr static storage_type<ConstIter>
      provide_ptr(storage_type<ConstIter> storage) {
        return storage;
      }

      template <bool ConstIter>
      constexpr static return_type<ConstIter>
      from_data_ptr(std::conditional_t<ConstIter, const T *, T *> data) {
        return *data;
      }

      template <bool ConstIter>
      constexpr static storage_type<ConstIter>
      to_storage(ref_type<ConstIter> ref) {
        return &ref;
      }

      static std::string shape() { return "scalar"; }

      constexpr static Dim_t stride() { return 1; }
    };

  }  // namespace internal

  /* ---------------------------------------------------------------------- */
  template <typename T, bool ConstField, class MapType, Iteration IterationType>
  typename StaticNFieldMap<T, ConstField, MapType, IterationType>::PlainType
  StaticNFieldMap<T, ConstField, MapType, IterationType>::mean() const {
    PlainType mean{PlainType::Zero()};
    for (auto && val : *this) {
      mean += val;
    }
    mean *= 1. / Real(this->size());
    return mean;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T, bool ConstField, Dim_t NbRow, Dim_t NbCol,
            Iteration IterationType = Iteration::QuadPt>
  using MatrixNFieldMap =
      StaticNFieldMap<T, ConstField, internal::MatrixMap<T, NbRow, NbCol>,
                      IterationType>;

  /* ---------------------------------------------------------------------- */
  template <typename T, bool ConstField, Dim_t NbRow, Dim_t NbCol,
            Iteration IterationType = Iteration::QuadPt>
  using ArrayNFieldMap =
      StaticNFieldMap<T, ConstField, internal::ArrayMap<T, NbRow, NbCol>,
                      IterationType>;

  //! the following only make sense as per-quadrature-point maps
  template <typename T, bool ConstField>
  using ScalarNFieldMap =
      StaticNFieldMap<T, ConstField, internal::ScalarMap<T>, Iteration::QuadPt>;

  /* ---------------------------------------------------------------------- */
  template <typename T, bool ConstField, Dim_t Dim>
  using T2NFieldMap =
      StaticNFieldMap<T, ConstField, internal::MatrixMap<T, Dim, Dim>,
                      Iteration::QuadPt>;

  /* ---------------------------------------------------------------------- */
  template <typename T, bool ConstField, Dim_t Dim>
  using T4NFieldMap =
      StaticNFieldMap<T, ConstField,
                      internal::MatrixMap<T, Dim * Dim, Dim * Dim>,
                      Iteration::QuadPt>;

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_NFIELD_MAP_STATIC_HH_
