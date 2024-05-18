/**
 * @file   test_goodies.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   24 Apr 2020
 *
 * @brief  helpers for testing
 *
 * Copyright © 2020 Till Junge
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

#include "test_goodies.hh"

namespace muGrid {

  namespace testGoodies {

    /* ---------------------------------------------------------------------- */
    std::vector<Int> generate_primes(const Uint & lower, const Uint & upper) {
      // list of all consecutive integers from 2 to upper limit
      std::vector<Uint> all_integers{};
      std::vector<Int> primes{};
      all_integers.reserve(upper - 2);
      // vector of tags (true means that an integer has been identified as
      // *non*-prime
      std::vector<bool> excluded(upper - 2, false);

      for (Uint i{2}; i < upper; ++i) {
        all_integers.push_back(i);
      }

      for (Uint i{0}; i < all_integers.size(); ++i) {
        const bool tag{excluded[i]};
        if (tag == false) {  // val has not been tagged, so it's prime
          const auto & val{all_integers[i]};
          if (val >= lower) {
            primes.push_back(val);
          }
          for (Uint p{2 * val}; p < upper; p += val) {  // tag all multiples
            excluded[p-2] = true;
          }
        }
      }
      return primes;
    }

  }  // namespace testGoodies

}  // namespace muGrid
