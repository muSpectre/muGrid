/**
 * @file   ref_vector.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   21 Aug 2019
 *
 * @brief  convenience class providing a vector of references
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

#include <vector>
#ifndef SRC_LIBMUGRID_REF_VECTOR_HH_
#define SRC_LIBMUGRID_REF_VECTOR_HH_

namespace muGrid {

  /**
   * work-around to allow using vectors of references (which are
   * forbidden by the C++ stl
   */
  template <typename T>
  class RefVector : protected std::vector<T *> {
    class iterator;
    using Parent = std::vector<T *>;

   public:
    using Parent::reserve;
    //! Default constructor
    RefVector() = default;

    //! Copy constructor
    RefVector(const RefVector & other) = default;

    //! Move constructor
    RefVector(RefVector && other) = default;

    //! Destructor
    virtual ~RefVector() = default;

    //! Copy assignment operator
    RefVector & operator=(const RefVector & other) = default;

    //! Move assignment operator
    RefVector & operator=(RefVector && other) = default;

    //! stl
    void push_back(T & value) { Parent::push_back(&value); }

    //! stl
    T & at(size_t index) { return *Parent::at(index); }

    //! stl
    const T & at(size_t index) const { return *Parent::at(index); }

    //! random access operator
    T & operator[](size_t index) { return *Parent::operator[](index); }

    //! random const access operator
    const T & operator[](size_t index) const {
      return *Parent::operator[](index);
    }

    //! stl
    iterator begin() { return iterator{Parent::begin()}; }

    //! stl
    iterator end() { return iterator{Parent::end()}; }
  };

  /**
   * iterator over `muGrid::RefVector`
   */
  template <typename T>
  class RefVector<T>::iterator : public std::vector<T *>::iterator {
    using Parent = typename std::vector<T *>::iterator;

   public:
    using std::vector<T *>::iterator::iterator;

    //! copy constructor
    iterator(Parent & iterator) : Parent{iterator} {}

    //! move constructor
    iterator(Parent && iterator) : Parent{std::move(iterator)} {}

    //! dereference
    T & operator*() {
      T & retval{*Parent::operator*()};
      return retval;
    }
  };

}  // namespace muGrid
#endif  // SRC_LIBMUGRID_REF_VECTOR_HH_
