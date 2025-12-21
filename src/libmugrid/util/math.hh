/**
 * @file   util/math.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   24 Jan 2019
 *
 * @brief  Mathematical utilities for muGrid
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

#ifndef SRC_LIBMUGRID_UTIL_MATH_HH_
#define SRC_LIBMUGRID_UTIL_MATH_HH_

#include "../core/types.hh"

#include <numeric>
#include <type_traits>

namespace muGrid {

    //! convenience definitions
    constexpr Real pi{3.1415926535897932384626433};

    //! compile-time potentiation required for field-size computations
    template <typename R, typename I>
    constexpr R ipow(R base, I exponent) {
        static_assert(std::is_integral<I>::value, "Type must be integer");
        R retval{1};
        for (I i = 0; i < exponent; ++i) {
            retval *= base;
        }
        return retval;
    }

    //! compile-time square root (helper)
    static constexpr Dim_t ct_sqrt(Dim_t res, Dim_t l, Dim_t r) {
        if (l == r) {
            return r;
        } else {
            const auto mid = (r + l) / 2;

            if (mid * mid >= res) {
                return ct_sqrt(res, l, mid);
            } else {
                return ct_sqrt(res, mid + 1, r);
            }
        }
    }

    //! compile-time square root
    static constexpr Dim_t ct_sqrt(Dim_t res) { return ct_sqrt(res, 1, res); }

    //! helper to get the number of elements from a shape
    template <typename T>
    Index_t get_nb_from_shape(const T & shape) {
        return shape.size() > 0 ? std::accumulate(shape.begin(), shape.end(), 1,
                                                  std::multiplies<Index_t>())
                                : 1;
    }

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_UTIL_MATH_HH_
