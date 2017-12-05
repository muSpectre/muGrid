/**
 * file   test_ccoord_operations.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   03 Dec 2017
 *
 * @brief  tests for cell coordinate operations
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

#include <iostream>

#include "common/common.hh"
#include "common/ccoord_operations.hh"
#include "common/test_goodies.hh"
#include "tests.hh"


namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(ccoords_operations);

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_cube, Fix, testGoodies::dimlist, Fix) {
    constexpr auto dim{Fix::dim};
    using Ccoord = Ccoord_t<dim>;
    constexpr Dim_t size{5};

    constexpr Ccoord cube = CcoordOps::get_cube<dim>(size);
    Ccoord ref_cube;
    for (Dim_t i = 0; i < dim; ++i) {
      ref_cube[i] = size;
    }

    BOOST_CHECK_EQUAL_COLLECTIONS(ref_cube.begin(), ref_cube.end(),
                                  cube.begin(), cube.end());
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_hermitian, Fix, testGoodies::dimlist,
                                   Fix) {
    constexpr auto dim{Fix::dim};
    using Ccoord = Ccoord_t<dim>;
    constexpr Dim_t size{5};

    constexpr Ccoord cube = CcoordOps::get_cube<dim>(size);
    constexpr Ccoord herm = CcoordOps::get_hermitian_sizes(cube);
    Ccoord ref_cube = cube;
    ref_cube.back() = (cube.back() + 1) / 2;

    BOOST_CHECK_EQUAL_COLLECTIONS(ref_cube.begin(), ref_cube.end(),
                                  herm.begin(), herm.end());
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_size, Fix, testGoodies::dimlist,
                                   Fix) {
    constexpr auto dim{Fix::dim};
    using Ccoord = Ccoord_t<dim>;
    constexpr Dim_t size{5};

    constexpr Ccoord cube = CcoordOps::get_cube<dim>(size);

    BOOST_CHECK_EQUAL(CcoordOps::get_size(cube),
                      ipow(size, dim));

  }

  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre
