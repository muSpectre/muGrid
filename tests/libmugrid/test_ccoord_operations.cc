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
    constexpr Dim_t size{5};
    Ccoord nb_grid_pts = CcoordOps::get_cube<Dim>(size);
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
    constexpr Dim_t size{5};
    Ccoord nb_grid_pts = CcoordOps::get_cube<Dim>(size);
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

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muGrid
