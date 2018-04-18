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
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Emacs; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */


#include "common/field_map_tensor.hh"
#include "common/field_map_matrixlike.hh"
#include "common/field_map_scalar.hh"

#include <sstream>
#include <stdexcept>
#include <type_traits>

#ifndef FIELD_MAP_H
#define FIELD_MAP_H

namespace muSpectre {

  /**
   * allows to iterate over raw data as if it were a FieldMap. This is
   * particularly useful when interacting with external solvers, such
   * as scipy and Eigen
   * @param EigenMap needs to be statically sized a Eigen::Map<XXX>
   */
  template <class EigenMap>
  class RawFieldMap
  {
  public:
    /**
     * determining the constness of the mapped array required in order
     * to formulate the constructors const-correctly
     */
    constexpr static bool IsConst{
      std::is_const<
        std::remove_pointer_t<typename EigenMap::PointerArgType>>::value};
    // short-hand for the basic scalar type
    using T = typename EigenMap::Scalar;
    // raw pointer type to store
    using T_ptr = std::conditional_t<IsConst,
                                     const T*,
                                     T*>;
    // input array (~field) type to be mapped
    using FieldVec_t = std::conditional_t<IsConst,
                                          const Eigen::VectorXd,
                                          Eigen::VectorXd>;
    //! Default constructor
    RawFieldMap() = delete;

    //! constructor from a *contiguous* array
    RawFieldMap(Eigen::Map<FieldVec_t> vec):
      data{vec.data()}, nb_pixels{vec.size()/NbComponents}
    {
      if (vec.size() % NbComponents != 0) {
        std::stringstream err{};
        err << "The vector size of " << vec.size()
            << " is not an integer multiple of the size of value_type, which "
            << "is " << NbComponents << ".";
        throw std::runtime_error(err.str());
      }
    }

    //! constructor from a *contiguous* array
    RawFieldMap(Eigen::Ref<FieldVec_t> vec):
      data{vec.data()}, nb_pixels{vec.size()/NbComponents}
    {
      if (vec.size() % NbComponents != 0) {
        std::stringstream err{};
        err << "The vector size of " << vec.size()
            << " is not an integer multiple of the size of value_type, which "
            << "is " << NbComponents << ".";
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
    RawFieldMap& operator=(const RawFieldMap &other) = delete;

    //! Move assignment operator
    RawFieldMap& operator=(RawFieldMap &&other) = delete;

    //! returns number of EigenMaps stored within the array
    size_t size() const {return this->nb_pixels;}

    //! forward declaration of iterator type
    class iterator;

    //! returns an iterator to the first element
    iterator begin() { return iterator{this->data, 0};}
    //! returns an iterator past the last element
    iterator end() {return iterator{this->data, this->size()};}

  protected:
    //! statically known size of the mapped type
    constexpr static size_t NbComponents{EigenMap::SizeAtCompileTime};
    //! raw data pointer (ugly, I know)
    T_ptr data;
    //! number of EigenMaps stored within the array
    size_t nb_pixels;
  private:
  };

  /**
   * Small iterator class to be used with the RawFieldMap
   */
  template <class EigenMap>
  class RawFieldMap<EigenMap>::iterator
  {
  public:
    //! short hand for the raw field map's type
    using Parent = RawFieldMap<EigenMap>;

    //! the  map needs to be friend in order to access the protected constructor
    friend Parent;
    //! stl compliance
    using value_type = EigenMap;
    //! stl compliance
    using iterator_category = std::forward_iterator_tag;

    //! Default constructor
    iterator() = delete;

    //! Copy constructor
    iterator(const iterator &other) = default;

    //! Move constructor
    iterator(iterator &&other) = default;

    //! Destructor
    virtual ~iterator() = default;

    //! Copy assignment operator
    iterator& operator=(const iterator &other) = default;

    //! Move assignment operator
    iterator& operator=(iterator &&other) = default;

    //! pre-increment
    inline iterator & operator++() {
      ++this->index;
      return *this;
    }

    //! dereference
    inline value_type operator *() {
      return EigenMap(raw_map + Parent::NbComponents*index);
    }

    //! inequality
    inline bool operator != (const iterator & other) const {
      return this->index != other.index;
    }

    //! equality
    inline bool operator == (const iterator & other) const {
      return this->index == other.index;
    }

  protected:

    //! protected constructor
    iterator (Parent::T_ptr raw_map, size_t start):
      raw_map{raw_map}, index{start} {}
    //! raw data
    Parent::T_ptr raw_map;
    //! currently pointed-to element
    size_t index;
  private:
  };
}  // muSpectre


#endif /* FIELD_MAP_H */
