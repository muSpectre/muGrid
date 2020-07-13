/**
 * @file   test_ccoord_operations.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   01 Oct 2019
 *
 * @brief  ccoord ops tests which cannot rely only on headers
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

#include <libmugrid/ccoord_operations.hh>
#include <libmugrid/iterators.hh>

#include "test_goodies.hh"
#include "tests.hh"

namespace muGrid {
  BOOST_AUTO_TEST_SUITE(ccoords_operations);

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_dynamic_pixels, Fix,
                                   testGoodies::dimlist, Fix) {
    constexpr auto Dim{Fix::dim};
    using Ccoord = Ccoord_t<Dim>;
    constexpr Index_t size{5};
    Ccoord nb_grid_pts{CcoordOps::get_cube<Dim>(size)};
    for (int i{0}; i < Dim; ++i) {
      nb_grid_pts[i] += i;
    }

    CcoordOps::Pixels<Dim> static_pix(nb_grid_pts);
    CcoordOps::DynamicPixels dynamic_pix(nb_grid_pts);

    for (auto && tup : akantu::zip(static_pix, dynamic_pix)) {
      auto && stat{std::get<0>(tup)};
      auto && dyna{std::get<1>(tup)};
      for (Dim_t i{0}; i < Dim; ++i) {
        BOOST_CHECK_EQUAL(stat[i], dyna[i]);
      }
    }
    BOOST_CHECK_EQUAL(static_pix.size(), dynamic_pix.size());
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_enumeration, Fix, testGoodies::dimlist,
                                   Fix) {
    constexpr auto Dim{Fix::dim};
    using Ccoord = Ccoord_t<Dim>;
    constexpr Index_t size{5};
    Ccoord nb_grid_pts{CcoordOps::get_cube<Dim>(size)};
    for (int i{0}; i < Dim; ++i) {
      nb_grid_pts[i] += i;
    }

    CcoordOps::DynamicPixels dynamic_pix(nb_grid_pts);
    for (auto && tup :
         akantu::zip(akantu::enumerate(dynamic_pix), dynamic_pix.enumerate())) {
      auto && aka_tup{std::get<0>(tup)};
      auto && msp_tup{std::get<1>(tup)};

      auto && aka_id{std::get<0>(aka_tup)};
      auto && msp_id{std::get<0>(msp_tup)};
      BOOST_CHECK_EQUAL(aka_id, msp_id);

      auto && aka_pix{std::get<1>(aka_tup)};
      auto && msp_pix{std::get<1>(msp_tup)};
      BOOST_CHECK_EQUAL(aka_pix, msp_pix);
    }
  }

  BOOST_AUTO_TEST_CASE(test_is_buffer_contiguous) {
    DynCcoord_t nb_grid_pts2({2, 3});
    DynCcoord_t column_major2({1, 2});
    DynCcoord_t row_major2({3, 1});
    BOOST_CHECK(CcoordOps::is_buffer_contiguous(
        nb_grid_pts2, CcoordOps::get_col_major_strides(nb_grid_pts2)));
    BOOST_CHECK(CcoordOps::is_buffer_contiguous(nb_grid_pts2, column_major2));
    BOOST_CHECK(CcoordOps::is_buffer_contiguous(nb_grid_pts2, row_major2));
    DynCcoord_t non_contiguous2({6, 1});
    BOOST_CHECK(!CcoordOps::is_buffer_contiguous(nb_grid_pts2,
                                                 non_contiguous2));

    DynCcoord_t nb_grid_pts3({5, 3, 4});
    DynCcoord_t column_major3({1, 5, 15});
    DynCcoord_t row_major3({12, 4, 1});
    BOOST_CHECK(CcoordOps::is_buffer_contiguous(
        nb_grid_pts3, CcoordOps::get_col_major_strides(nb_grid_pts3)));
    BOOST_CHECK(CcoordOps::is_buffer_contiguous(nb_grid_pts3, column_major3));
    BOOST_CHECK(CcoordOps::is_buffer_contiguous(nb_grid_pts3, row_major3));
    DynCcoord_t partial_transpose3({1, 20, 5});
    BOOST_CHECK(CcoordOps::is_buffer_contiguous(nb_grid_pts3,
                                               partial_transpose3));
    DynCcoord_t non_contiguous3({6, 5, 15});
    BOOST_CHECK(!CcoordOps::is_buffer_contiguous(nb_grid_pts3,
                                                 non_contiguous3));
  }

  BOOST_AUTO_TEST_CASE(test_get_size_large) {
    muGrid::DynCcoord_t nb_grid_pts{65536, 65536}, nb_grid_pts2{131072, 131072};
    BOOST_CHECK(CcoordOps::get_size(nb_grid_pts) == 65536L * 65536L);
    BOOST_CHECK(CcoordOps::get_size(nb_grid_pts2) == 131072L * 131072L);
  }

  BOOST_AUTO_TEST_CASE(arithmetic_operators) {
    DynCcoord_t a{3, 4};
    DynCcoord_t b{1, 2};
    DynCcoord_t sum{4, 6};
    DynCcoord_t diff{2, 2};

    DynCcoord_t c{a + b};

    BOOST_CHECK_EQUAL(sum, c);

    c = a - b;
    BOOST_CHECK_EQUAL(diff, c);

    DynRcoord_t c1{1, 2, 3};
    DynRcoord_t c2{1.3, 2.8, 5.7};

    DynRcoord_t c1pc2{c1[0] + c2[0], c1[1] + c2[1], c1[2] + c2[2]};
    DynRcoord_t c1mc2{c1[0] - c2[0], c1[1] - c2[1], c1[2] - c2[2]};

    BOOST_CHECK_EQUAL(c1 + c2, c1pc2);
    BOOST_CHECK_EQUAL(c1 - c2, c1mc2);
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muGrid
