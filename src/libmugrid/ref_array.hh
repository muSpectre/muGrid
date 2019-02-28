/**
 * file   ref_array.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   04 Dec 2018
 *
 * @brief  convenience class to simulate an array of references
 *
 * Copyright © 2018 Till Junge
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
 * along with GNU Emacs; see the file COPYING. If not, write to the
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

#ifndef SRC_LIBMUGRID_REF_ARRAY_HH_
#define SRC_LIBMUGRID_REF_ARRAY_HH_

#include <array>
#include <initializer_list>
#include "iterators.hh"

namespace muGrid {
  namespace internal {

    template <typename T, typename FirstVal, typename... RestVals>
    struct TypeChecker {
      constexpr static bool value{
          std::is_same<T, std::remove_reference_t<FirstVal>>::value and
          TypeChecker<T, RestVals...>::value};
    };

    template <typename T, typename OnlyVal>
    struct TypeChecker<T, OnlyVal> {
      constexpr static bool value{
          std::is_same<T, std::remove_reference_t<OnlyVal>>::value};
    };

  }  // namespace internal

  template <typename T, size_t N>
  class RefArray {
   public:
    //! Default constructor
    RefArray() = delete;

    template <typename... Vals>
    explicit RefArray(Vals &... vals) : values{&vals...} {
      static_assert(internal::TypeChecker<T, Vals...>::value,
                    "Only refs to type T allowed");
    }

    //! Copy constructor
    RefArray(const RefArray & other) = default;

    //! Move constructor
    RefArray(RefArray && other) = default;

    //! Destructor
    virtual ~RefArray() = default;

    //! Copy assignment operator
    RefArray & operator=(const RefArray & other) = default;

    //! Move assignment operator
    RefArray & operator=(RefArray && other) = delete;

    T & operator[](size_t index) { return *this->values[index]; }
    constexpr T & operator[](size_t index) const {
      return *this->values[index];
    }

   protected:
    std::array<T *, N> values{};

   private:
  };

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_REF_ARRAY_HH_
