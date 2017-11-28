/**
 * file   test_goodies.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   27 Sep 2017
 *
 * @brief  helpers for testing
 *
 * @section LICENCE
 *
 * Copyright (C) 2017 Till Junge
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

#include <random>
#include <type_traits>

#ifndef TEST_GOODIES_H
#define TEST_GOODIES_H

namespace muSpectre {

  namespace testGoodies {

    template <Dim_t Dim>
    struct dimFixture{
      constexpr static Dim_t dim{Dim};
    };

    using dimlist = boost::mpl::list<dimFixture<oneD>,
                                   dimFixture<twoD>,
                                   dimFixture<threeD>>;

    template<typename T>
    class RandRange {
    public:
      RandRange(): rd(), gen(rd()) {}

      template <typename dummyT = T>
      std::enable_if_t<std::is_floating_point<dummyT>::value, dummyT>
      randval(T&& lower, T&& upper) {
        static_assert(std::is_same<T, dummyT>::value);
        auto distro = std::uniform_real_distribution<T>(lower, upper);
        return distro(this->gen);
      }

      template <typename dummyT = T>
      std::enable_if_t<std::is_integral<dummyT>::value, dummyT>
      randval(T&& lower, T&& upper) {
        static_assert(std::is_same<T, dummyT>::value);
        auto distro = std::uniform_int_distribution<T>(lower, upper);
        return distro(this->gen);
      }

    private:
      std::random_device rd;
      std::default_random_engine gen;
    };

  }  // testGoodies

}  // muSpectre

#endif /* TEST_GOODIES_H */
