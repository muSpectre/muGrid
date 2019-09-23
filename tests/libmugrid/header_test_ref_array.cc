/**
 * @file   header_test_ref_array.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   04 Dec 2018
 *
 * @brief  tests for the RefArray convenience struct
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

#include "tests.hh"

#include <libmugrid/ref_array.hh>

namespace muGrid {

  BOOST_AUTO_TEST_SUITE(RefArray_tests);

  BOOST_AUTO_TEST_CASE(two_d_test) {
    std::array<int, 2> values{2, 3};
    RefArray<int, 2> refs{values[0], values[1]};

    BOOST_CHECK_EQUAL(values[0], refs[0]);
    BOOST_CHECK_EQUAL(values[1], refs[1]);
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muGrid
