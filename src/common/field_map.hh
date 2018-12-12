/**
 * @file   field_map.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   26 Sep 2017
 *
 * @brief  just and indirection to include all iterables defined for fields
 *
 * Copyright © 2017 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 */

#include "common/field_map_tensor.hh"
#include "common/field_map_matrixlike.hh"
#include "common/field_map_scalar.hh"

#include <sstream>
#include <stdexcept>
#include <type_traits>

#ifndef SRC_COMMON_FIELD_MAP_HH_
#define SRC_COMMON_FIELD_MAP_HH_

namespace muSpectre {

  /**
   * allows to iterate over raw data as if it were a FieldMap. This is
   * particularly useful when interacting with external solvers, such
   * as scipy and Eigen
   * @param EigenMap needs to be statically sized a Eigen::Map<XXX>
   *
   * @warning This type is not safe for re-use. I.e., after there has
   * been an assignment to the underlying eigen array, the
   * `RawFieldMap` might be invalidated!
   */
  template <class EigenMap> class RawFieldMap {
   public:
    /**
     * determining the constness of the mapped array required in order
     * to formulate the constructors const-correctly
     */
    constexpr static bool IsConst{std::is_const<
        std::remove_pointer_t<typename EigenMap::PointerArgType>>::value};
    //! short-hand for the basic scalar type
    using T = typename EigenMap::Scalar;
    //! raw pointer type to store
    using T_ptr = std::conditional_t<IsConst, const T *, T *>;
    //! input array (~field) type to be mapped
    using FieldVec_t =
        std::conditional_t<IsConst, const Eigen::VectorXd, Eigen::VectorXd>;

    //! Plain mapped Eigen type
    using EigenPlain = typename EigenMap::PlainObject;

    //! Default constructor
    RawFieldMap() = delete;

    //! constructor from a *contiguous* array
    RawFieldMap(Eigen::Map<FieldVec_t> vec,
                Dim_t nb_rows = EigenMap::RowsAtCompileTime,
                Dim_t nb_cols = EigenMap::ColsAtCompileTime)
        : data{vec.data()}, nb_rows{nb_rows}, nb_cols{nb_cols},
          nb_components{nb_rows * nb_cols},
          nb_pixels(vec.size() / nb_components) {
      if ((nb_rows == Eigen::Dynamic) or(nb_cols == Eigen::Dynamic)) {
        throw FieldError(
            "You have to specify the number of rows and columns if you map a "
            "dynamically sized Eigen Map type.");
      }
      if ((nb_rows < 1) or(nb_cols < 1)) {
        throw FieldError("Only positive numbers of rows and columns make "
                         "sense");
      }
      if (vec.size() % this->nb_components != 0) {
        std::stringstream err{};
        err << "The vector size of " << vec.size()
            << " is not an integer multiple of the size of value_type, which "
            << "is " << this->nb_components << ".";
        throw std::runtime_error(err.str());
      }
    }

    //! constructor from a *contiguous* array
    RawFieldMap(Eigen::Ref<FieldVec_t> vec,
                Dim_t nb_rows = EigenMap::RowsAtCompileTime,
                Dim_t nb_cols = EigenMap::ColsAtCompileTime)
        : data{vec.data()}, nb_rows{nb_rows}, nb_cols{nb_cols},
          nb_components{nb_rows * nb_cols},
          nb_pixels(vec.size() / nb_components) {
      if (vec.size() % this->nb_components != 0) {
        std::stringstream err{};
        err << "The vector size of " << vec.size()
            << " is not an integer multiple of the size of value_type, which "
            << "is " << this->nb_components << ".";
        throw std::runtime_error(err.str());
      }
    }

    //! Copy constructor
    RawFieldMap(const RawFieldMap &other) = delete;

    //! Move constructor
    RawFieldMap(RawFieldMap &&other) = default;

    //! Destructor
    virtual ~RawFieldMap() = default;

    //! Copy assignment operator
    RawFieldMap &operator=(const RawFieldMap &other) = delete;

    //! Move assignment operator
    RawFieldMap &operator=(RawFieldMap &&other) = delete;

    //! returns number of EigenMaps stored within the array
    size_t size() const { return this->nb_pixels; }

    //! forward declaration of iterator type
    template <bool IsConst> class iterator_t;
    using iterator = iterator_t<false>;
    using const_iterator = iterator_t<true>;

    //! returns an iterator to the first element
    iterator begin() { return iterator{*this, 0}; }
    const_iterator begin() const { return const_iterator{*this, 0}; }
    //! returns an iterator past the last element
    iterator end() { return iterator{*this, this->size()}; }
    const_iterator end() const { return const_iterator{*this, this->size()}; }

    //! evaluates the average of the field
    EigenPlain mean() const {
      using T_t = EigenPlain;
      T_t mean(T_t::Zero(this->nb_rows, this->nb_cols));
      for (auto &&val : *this) {
        mean += val;
      }
      mean /= this->size();
      return mean;
    }

   protected:
    inline T_ptr get_data() { return data; }
    inline const T_ptr get_data() const { return data; }
    //! raw data pointer (ugly, I know)
    T_ptr data;
    const Dim_t nb_rows;
    const Dim_t nb_cols;
    const Dim_t nb_components;
    //! number of EigenMaps stored within the array
    size_t nb_pixels;

   private:
  };

  /**
   * Small iterator class to be used with the RawFieldMap
   */
  template <class EigenMap>
  template <bool IsConst>
  class RawFieldMap<EigenMap>::iterator_t {
   public:
    //! short hand for the raw field map's type
    using Parent = RawFieldMap<EigenMap>;

    //! the  map needs to be friend in order to access the protected constructor
    friend Parent;
    //! stl compliance
    using value_type = std::conditional_t<
        IsConst, Eigen::Map<const typename EigenMap::PlainObject>, EigenMap>;
    using T_ptr =
        std::conditional_t<IsConst, const Parent::T_ptr, Parent::T_ptr>;
    //! stl compliance
    using iterator_category = std::forward_iterator_tag;

    //! Default constructor
    iterator_t() = delete;

    //! Copy constructor
    iterator_t(const iterator_t &other) = default;

    //! Move constructor
    iterator_t(iterator_t &&other) = default;

    //! Destructor
    virtual ~iterator_t() = default;

    //! Copy assignment operator
    iterator_t &operator=(const iterator_t &other) = default;

    //! Move assignment operator
    iterator_t &operator=(iterator_t &&other) = default;

    //! pre-increment
    inline iterator_t &operator++() {
      ++this->index;
      return *this;
    }

    //! dereference
    inline value_type operator*() {
      return value_type(raw_ptr + this->map.nb_components * index,
                        this->map.nb_rows, this->map.nb_cols);
    }

    //! inequality
    inline bool operator!=(const iterator_t &other) const {
      return this->index != other.index;
    }

    //! equality
    inline bool operator==(const iterator_t &other) const {
      return this->index == other.index;
    }

   protected:
    //! protected constructor
    iterator_t(const Parent &map, size_t start)
        : raw_ptr{map.get_data()}, map{map}, index{start} {}

    template <bool dummy_non_const = not IsConst>
    iterator_t(std::enable_if_t<dummy_non_const, Parent &> map, size_t start)
        : raw_ptr{map.data}, map{map}, index{start} {
      static_assert(dummy_non_const == not IsConst, "SFINAE");
    }
    //! raw data
    T_ptr raw_ptr;
    //! ref to underlying map
    const Parent &map;
    //! currently pointed-to element
    size_t index;

   private:
  };
}  // namespace muSpectre

#endif  // SRC_COMMON_FIELD_MAP_HH_
