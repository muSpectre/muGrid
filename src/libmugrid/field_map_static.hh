/**
 * @file   field_map_static.hh
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

#ifndef SRC_LIBMUGRID_FIELD_MAP_STATIC_HH_
#define SRC_LIBMUGRID_FIELD_MAP_STATIC_HH_

#include "field.hh"
#include "field_typed.hh"
#include "field_map.hh"
#include "T4_map_proxy.hh"

#include <sstream>

namespace muGrid {

  /**
   * Statically sized field map. Static field maps reproduce the capabilities of
   * the (dynamically sized) `muGrid::FieldMap`, but iterate much more
   * efficiently.
   */
  template <typename T, Mapping Mutability, class MapType,
            IterUnit IterationType = IterUnit::SubPt>
  class StaticFieldMap : public FieldMap<T, Mutability> {
    static_assert(MapType::IsValidStaticMapType(),
                  "The MapType you chose is not compatible");

   public:
    //! stored scalar type
    using Scalar = T;

    //! base class
    using Parent = FieldMap<T, Mutability>;

    //! convenience alias
    using Field_t = typename Parent::Field_t;

    //! return type when dereferencing iterators over this map
    template <Mapping MutType>
    using Return_t = typename MapType::template Return_t<MutType>;

    //! stl
    using reference = Return_t<Mutability>;

    //! Eigen type representing iterates of this map
    using PlainType = typename MapType::PlainType;

    /**
     * determine at compile time  whether pixels or quadrature points are
     * iterater over
     */
    constexpr static IterUnit GetIterationType() { return IterationType; }
    //! determine the number of components in the iterate at compile time
    constexpr static size_t Stride() { return MapType::stride(); }
    //! determine whether this map has statically sized iterates at compile time
    constexpr static bool IsStatic() { return true; }

    /**
     * iterable proxy type to iterate over the quad point/pixel indices and
     * stored values simultaneously
     */
    using Enumeration_t = akantu::containers::ZipContainer<
        std::conditional_t<(IterationType == IterUnit::SubPt),
                           FieldCollection::IndexIterable,
                           FieldCollection::PixelIndexIterable>,
        StaticFieldMap &>;
    //! Default constructor
    StaticFieldMap() = delete;

    /**
     * Constructor from a non-typed field ref (has more runtime cost than the
     * next constructor
     */
    explicit StaticFieldMap(Field & field)
        : StaticFieldMap(TypedField<T>::safe_cast(field)) {}

    //! Constructor from typed field ref.
    explicit StaticFieldMap(Field_t & field)
        : Parent{field, MapType::NbRow(), IterationType} {
      if (this->stride != MapType::stride()) {
        std::stringstream error{};
        error << "Incompatible number of components in the field '"
              << this->field.get_name() << "': The field map has a stride of "
              << this->stride << " but you wish an iterate with shape "
              << MapType::shape() << ", corresponding to a stride of "
              << MapType::stride() << ".";
        throw FieldMapError(error.str());
      }
    }

    //! Copy constructor
    StaticFieldMap(const StaticFieldMap & other) = delete;

    //! Move constructor
    StaticFieldMap(StaticFieldMap && other) = default;

    //! Destructor
    virtual ~StaticFieldMap() = default;

    //! Copy assignment operator
    StaticFieldMap & operator=(const StaticFieldMap & other) = delete;

    //! Move assignment operator
    StaticFieldMap & operator=(StaticFieldMap && other) = delete;

    //! Assign a matrix-like value with dynamic size to every entry
    template <bool IsMutableField = Mutability == Mapping::Mut>
    std::enable_if_t<IsMutableField, StaticFieldMap> &
    operator=(const typename Parent::EigenRef & val) {
      dynamic_cast<Parent &>(*this) = val;
      return *this;
    }

    //! Assign a matrix-like value with static size to every entry
    template <bool IsMutableField = Mutability == Mapping::Mut>
    std::enable_if_t<IsMutableField && !MapType::IsScalarMapType(),
                     StaticFieldMap<T, Mutability, MapType, IterationType>> &
    operator=(const reference & val) {
      for (auto && entry : *this) {
        entry = val;
      }
      return *this;
    }

    //! Assign a scalar value to every entry
    template <bool IsMutableField = Mutability == Mapping::Mut>
    std::enable_if_t<IsMutableField && MapType::IsScalarMapType(),
                     StaticFieldMap<T, Mutability, MapType, IterationType>> &
    operator=(const Scalar & val) {
      if (not(this->nb_rows == 1 && this->nb_cols == 1)) {
        std::stringstream error_str{};
        error_str << "Expected an array/matrix with shape (" << this->nb_rows
                  << " × " << this->nb_cols
                  << "), but received a scalar value.";
        throw FieldMapError(error_str.str());
      }
      for (auto && entry : *this) {
        entry = val;
      }
      return *this;
    }

    template <Mapping MutIter>
    class Iterator;
    //! stl
    using iterator =
        Iterator<(Mutability == Mapping::Mut) ? Mapping::Mut : Mapping::Const>;

    //! stl
    using const_iterator = Iterator<Mapping::Const>;

    //! random access operator
    Return_t<Mutability> operator[](size_t index) {
      assert(this->is_initialised);
      return MapType::template from_data_ptr<Mutability>(
          this->data_ptr + index * MapType::stride());
    }

    //! random const access operator
    Return_t<Mapping::Const> operator[](size_t index) const {
      assert(this->is_initialised);
      return MapType::template from_data_ptr<Mapping::Const>(
          this->data_ptr + index * MapType::stride());
    }

    //! evaluate the average of the field
    inline PlainType mean() const;

    //! stl
    iterator begin() { return iterator{*this, false}; }

    //! stl
    iterator end() { return iterator{*this, true}; }

    //! stl
    const_iterator begin() const {
      if (not this->is_initialised) {
        throw FieldMapError("Const FieldMaps cannot be initialised");
      }
      return const_iterator{*this, false};
    }

    //! stl
    const_iterator end() const { return const_iterator{*this, true}; }

    //! iterate over pixel/quad point indices and stored values simultaneously
    template <bool IsPixelIterable = (IterationType == IterUnit::Pixel)>
    std::enable_if_t<IsPixelIterable, Enumeration_t> enumerate_indices() {
      static_assert(IsPixelIterable == (IterationType == IterUnit::Pixel),
                    "IsPixelIterable is a SFINAE parameter, do not touch it.");
      return akantu::zip(this->field.get_collection().get_pixel_indices_fast(),
                         *this);
    }

    //! iterate over pixel/quad point indices and stored values simultaneously
    template <IterUnit Iter = IterUnit::SubPt,
              class Dummy = std::enable_if_t<IterationType == Iter, bool>>
    Enumeration_t enumerate_indices() {
      static_assert(Iter == IterUnit::SubPt,
                    "Iter is a SFINAE parameter, do not touch it.");
      static_assert(std::is_same<Dummy, bool>::value,
                    "Dummy is a SFINAE parameter, do not touch it.");
      return akantu::zip(this->field.get_collection().get_sub_pt_indices(
                             this->field.get_sub_division_tag()),
                         *this);
    }
  };

  /**
   * Iterator class for `muGrid::StaticFieldMap`
   */
  template <typename T, Mapping Mutability, class MapType,
            IterUnit IterationType>
  template <Mapping MutIter>
  class StaticFieldMap<T, Mutability, MapType, IterationType>::Iterator {
   public:
    //! type returned by iterator
    using value_type = typename MapType::template value_type<MutIter>;

    //! type stored
    using storage_type = typename MapType::template storage_type<MutIter>;

    //! Default constructor
    Iterator() = delete;

    //! Constructor to beginning, or to end
    Iterator(const StaticFieldMap & map, bool end)
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
          storage_type(this->map.data_ptr + this->index * Stride());
      return *this;
    }
    //! dereference
    inline value_type & operator*() {
      return MapType::template provide_ref<MutIter>(this->iterate);
    }
    //! pointer to member
    inline value_type * operator->() {
      return (MapType::template provide_ptr<MutIter>(this->iterate));
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
    //! FieldMap being iterated over
    const StaticFieldMap & map;
    //! current iteration index
    size_t index;
    //! map which is being returned per iterate
    storage_type iterate;
  };

  namespace internal {

    /**
     * Internal struct for handling the matrix-shaped iterates of
     * `muGrid::FieldMap`
     */
    template <typename T, class EigenPlain>
    struct EigenMap {
      /**
       * check at compile time whether the type is meant to be a map with
       * statically sized iterates.
       */
      constexpr static bool IsValidStaticMapType() { return true; }

      /**
       * check at compiler time whether this map is scalar
       */
      constexpr static bool IsScalarMapType() { return false; }

      //! Eigen type of the iterate
      using PlainType = EigenPlain;

      //! stl (const-correct)
      template <Mapping MutIter>
      using value_type = std::conditional_t<MutIter == Mapping::Const,
                                            Eigen::Map<const PlainType>,
                                            Eigen::Map<PlainType>>;

      //! stl (const-correct)
      template <Mapping MutIter>
      using ref_type = value_type<MutIter>;

      //! for direct access through operator[]
      template <Mapping MutIter>
      using Return_t = value_type<MutIter>;

      //! stored type (cannot always be same as ref_type)
      template <Mapping MutIter>
      using storage_type = value_type<MutIter>;

      //! return the return_type version of the iterate from storage_type
      template <Mapping MutIter>
      constexpr static value_type<MutIter> &
      provide_ref(storage_type<MutIter> & storage) {
        return storage;
      }

      //! return the const return_type version of the iterate from storage_type
      template <Mapping MutIter>
      constexpr static const value_type<MutIter> &
      provide_const_ref(const storage_type<MutIter> & storage) {
        return storage;
      }

      //! return a pointer to the iterate from storage_type
      template <Mapping MutIter>
      constexpr static value_type<MutIter> *
      provide_ptr(storage_type<MutIter> & storage) {
        return &storage;
      }

      //! return a return_type version of the iterate from its pointer
      template <Mapping MutIter>
      constexpr static Return_t<MutIter> from_data_ptr(
          std::conditional_t<MutIter == Mapping::Const, const T *, T *> data) {
        return Return_t<MutIter>(data);
      }

      //! return a storage_type version of the iterate from its value
      template <Mapping MutIter>
      constexpr static storage_type<MutIter>
      to_storage(value_type<MutIter> && value) {
        return std::move(value);
      }

      //! return the nb of components of the iterate (known at compile time)
      constexpr static Index_t stride() { return PlainType::SizeAtCompileTime; }

      //! return the iterate's shape as text, mostly for error messages
      static_assert(stride() > 0,
                    "Only statically sized Eigen types allowed here");
      static std::string shape() {
        std::stringstream shape_stream{};
        shape_stream << PlainType::RowsAtCompileTime << " × "
                     << PlainType::ColsAtCompileTime;
        return shape_stream.str();
      }
      constexpr static Index_t NbRow() { return PlainType::RowsAtCompileTime; }
    };

    /**
     * internal convenience alias for creating maps iterating over statically
     * sized `Eigen::Matrix`s
     */
    template <typename T, Dim_t NbRow, Dim_t NbCol>
    using MatrixMap = EigenMap<T, Eigen::Matrix<T, NbRow, NbCol>>;

    /**
     * internal convenience alias for creating maps iterating over statically
     * sized `Eigen::Array`s
     */
    template <typename T, Dim_t NbRow, Dim_t NbCol>
    using ArrayMap = EigenMap<T, Eigen::Array<T, NbRow, NbCol>>;

    /**
     * Internal struct for handling the scalar iterates of `muGrid::FieldMap`
     */
    template <typename T>
    struct ScalarMap {
      /**
       * check at compile time whether this map is suitable for statically sized
       * iterates
       */
      constexpr static bool IsValidStaticMapType() { return true; }

      /**
       * check at compiler time whether this map is scalar
       */
      constexpr static bool IsScalarMapType() { return true; }

      /**
       * Scalar maps don't have an eigen type representing the iterate, just the
       * raw stored type itsef
       */
      using PlainType = T;

      //! return type for iterates
      template <Mapping MutIter>
      using value_type =
          std::conditional_t<MutIter == Mapping::Const, const T, T>;

      //! reference type for iterates
      template <Mapping MutIter>
      using ref_type = value_type<MutIter> &;

      //! for direct access through operator[]
      template <Mapping MutIter>
      using Return_t = value_type<MutIter> &;

      //! need to encapsulate
      template <Mapping MutIter>
      using storage_type =
          std::conditional_t<MutIter == Mapping::Const, const T *, T *>;

      //! return the return_type version of the iterate from storage_type
      template <Mapping MutIter>
      constexpr static value_type<MutIter> &
      provide_ref(storage_type<MutIter> storage) {
        return *storage;
      }

      //! return the const return_type version of the iterate from storage_type
      template <Mapping MutIter>
      constexpr static const value_type<MutIter> &
      provide_const_ref(const storage_type<MutIter> storage) {
        return *storage;
      }

      //! return a pointer to the iterate from storage_type
      template <Mapping MutIter>
      constexpr static storage_type<MutIter>
      provide_ptr(storage_type<MutIter> storage) {
        return storage;
      }

      //! return a return_type version of the iterate from its pointer
      template <Mapping MutIter>
      constexpr static Return_t<MutIter> from_data_ptr(
          std::conditional_t<MutIter == Mapping::Const, const T *, T *> data) {
        return *data;
      }

      //! return a storage_type version of the iterate from its value
      template <Mapping MutIter>
      constexpr static storage_type<MutIter> to_storage(ref_type<MutIter> ref) {
        return &ref;
      }

      //! return the nb of components of the iterate (known at compile time)
      constexpr static Index_t stride() { return 1; }

      //! return the iterate's shape as text, mostly for error messages
      static std::string shape() { return "scalar"; }
      constexpr static Index_t NbRow() { return 1; }
    };

  }  // namespace internal

  /* ---------------------------------------------------------------------- */
  template <typename T, Mapping Mutability, class MapType,
            IterUnit IterationType>
  typename StaticFieldMap<T, Mutability, MapType, IterationType>::PlainType
  StaticFieldMap<T, Mutability, MapType, IterationType>::mean() const {
    PlainType mean{PlainType::Zero()};
    for (auto && val : *this) {
      mean += val;
    }
    mean *= 1. / Real(this->size());
    return mean;
  }

  /**
   * Alias of `muGrid::StaticFieldMap` you wish to iterate over pixel by pixel
   * or quadrature point by quadrature point with statically sized
   * `Eigen::Matrix` iterates
   *
   * @tparam T scalar type stored in the field, must be one of `muGrid::Real`,
   * `muGrid::Int`, `muGrid::Uint`, `muGrid::Complex`
   * @tparam Mutability whether or not the map allows to modify the content of
   * the field
   * @tparam NbRow number of rows of the iterate
   * @tparam NbCol number of columns of the iterate
   * @tparam IterationType describes the pixel-subdivision
   */
  template <typename T, Mapping Mutability, Dim_t NbRow, Dim_t NbCol,
            IterUnit IterationType>
  using MatrixFieldMap =
      StaticFieldMap<T, Mutability, internal::MatrixMap<T, NbRow, NbCol>,
                     IterationType>;

  /**
   * Alias of `muGrid::StaticFieldMap` you wish to iterate over pixel by pixel
   * or quadrature point by quadrature point with* statically sized
   * `Eigen::Array` iterates
   *
   * @tparam T scalar type stored in the field, must be one of `muGrid::Real`,
   * `muGrid::Int`, `muGrid::Uint`, `muGrid::Complex`
   * @tparam Mutability whether or not the map allows to modify the content of
   * the field
   * @tparam NbRow number of rows of the iterate
   * @tparam NbCol number of columns of the iterate
   * @tparam IterationType describes the pixel-subdivisionuadrature points
   */
  template <typename T, Mapping Mutability, Dim_t NbRow, Dim_t NbCol,
            IterUnit IterationType>
  using ArrayFieldMap =
      StaticFieldMap<T, Mutability, internal::ArrayMap<T, NbRow, NbCol>,
                     IterationType>;

  /**
   * Alias of `muGrid::StaticFieldMap` over a scalar field you wish to iterate
   * over quadrature point by quadrature point.
   *
   * @tparam T scalar type stored in the field, must be one of `muGrid::Real`,
   * `muGrid::Int`, `muGrid::Uint`, `muGrid::Complex`
   * @tparam Mutability whether or not the map allows to modify the content of
   * the field
   * @tparam IterationType describes the pixel-subdivision
   */
  template <typename T, Mapping Mutability, IterUnit IterationType>
  using ScalarFieldMap =
      StaticFieldMap<T, Mutability, internal::ScalarMap<T>, IterationType>;

  /**
   * Alias of `muGrid::StaticNFieldMap` over a first-rank tensor field you wish
   * to iterate over quadrature point by quadrature point.
   *
   * @tparam T scalar type stored in the field, must be one of `muGrid::Real`,
   * `muGrid::Int`, `muGrid::Uint`, `muGrid::Complex`
   * @tparam Mutability whether or not the map allows to modify the content of
   * the field
   * @tparam Dim spatial dimension of the tensor
   * @tparam IterationType describes the pixel-subdivision
   */
  template <typename T, Mapping Mutability, Dim_t Dim,
            IterUnit IterationType>
  using T1NFieldMap =
      StaticFieldMap<T, Mutability, internal::MatrixMap<T, Dim, 1>,
                     IterationType>;

  /**
   * Alias of `muGrid::StaticFieldMap` over a second-rank tensor field you wish
   * to iterate over quadrature point by quadrature point.
   *
   * @tparam T scalar type stored in the field, must be one of `muGrid::Real`,
   * `muGrid::Int`, `muGrid::Uint`, `muGrid::Complex`
   * @tparam Mutability whether or not the map allows to modify the content of
   * the field
   * @tparam Dim spatial dimension of the tensor
   * @tparam IterationType describes the pixel-subdivision
   */
  template <typename T, Mapping Mutability, Dim_t Dim,
            IterUnit IterationType>
  using T1FieldMap =
      StaticFieldMap<T, Mutability, internal::MatrixMap<T, Dim, 1>,
                     IterationType>;

  /**
   * Alias of `muGrid::StaticFieldMap` over a second-rank tensor field you wish
   * to iterate over quadrature point by quadrature point.
   *
   * @tparam T scalar type stored in the field, must be one of `muGrid::Real`,
   * `muGrid::Int`, `muGrid::Uint`, `muGrid::Complex`
   * @tparam Mutability whether or not the map allows to modify the content of
   * the field
   * @tparam Dim spatial dimension of the tensor
   * @tparam IterationType describes the pixel-subdivision
   */
  template <typename T, Mapping Mutability, Dim_t Dim,
            IterUnit IterationType>
  using T2FieldMap =
      StaticFieldMap<T, Mutability, internal::MatrixMap<T, Dim, Dim>,
                     IterationType>;

  /**
   * Alias of `muGrid::StaticFieldMap` over a fourth-rank tensor field you wish
   * to iterate over quadrature point by quadrature point.
   *
   * @tparam T scalar type stored in the field, must be one of `muGrid::Real`,
   * `muGrid::Int`, `muGrid::Uint`, `muGrid::Complex`
   * @tparam Mutability whether or not the map allows to modify the content of
   * the field
   * @tparam Dim spatial dimension of the tensor
   * @tparam IterationType describes the pixel-subdivision
   */
  template <typename T, Mapping Mutability, Dim_t Dim,
            IterUnit IterationType>
  using T4FieldMap =
      StaticFieldMap<T, Mutability,
                     internal::MatrixMap<T, Dim * Dim, Dim * Dim>,
                     IterationType>;

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_FIELD_MAP_STATIC_HH_
