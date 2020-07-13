/**
 * @file   header_test_ccoord_operations.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   03 Dec 2017
 *
 * @brief  tests for cell coordinate operations
 *
 * Copyright © 2017 Till Junge
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
 * * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 *
 */

#include <iostream>

#include <libmugrid/ccoord_operations.hh>
#include "test_goodies.hh"
#include "tests.hh"

namespace muGrid {
  BOOST_AUTO_TEST_SUITE(ccoords_operations);

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_cube, Fix, testGoodies::dimlist, Fix) {
    constexpr auto dim{Fix::dim};
    using Ccoord = Ccoord_t<dim>;
    constexpr Index_t size{5};

    constexpr Ccoord cube{CcoordOps::get_cube<dim>(size)};
    Ccoord ref_cube;
    for (Dim_t i = 0; i < dim; ++i) {
      ref_cube[i] = size;
    }

    BOOST_CHECK_EQUAL_COLLECTIONS(ref_cube.begin(), ref_cube.end(),
                                  cube.begin(), cube.end());
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_size, Fix, testGoodies::dimlist, Fix) {
    constexpr auto dim{Fix::dim};
    using Ccoord = Ccoord_t<dim>;
    constexpr Index_t size{5};

    constexpr Ccoord cube{CcoordOps::get_cube<dim>(size)};

    BOOST_CHECK_EQUAL(CcoordOps::get_size(cube), ipow(size, dim));
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_stride_size, Fix, testGoodies::dimlist,
                                   Fix) {
    constexpr auto dim{Fix::dim};
    using Ccoord = Ccoord_t<dim>;
    constexpr Index_t size{5};

    constexpr Ccoord cube{CcoordOps::get_cube<dim>(size)};
    constexpr Ccoord stride{CcoordOps::get_col_major_strides(cube)};

    BOOST_CHECK_EQUAL(CcoordOps::get_buffer_size(cube, stride),
                      ipow(size, dim));
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_index, Fix, testGoodies::dimlist, Fix) {
    constexpr auto dim{Fix::dim};
    using Ccoord = Ccoord_t<dim>;

    testGoodies::RandRange<Dim_t> rng;

    Ccoord sizes{};
    for (Dim_t i{0}; i < dim; ++i) {
      sizes[i] = rng.randval(2, 5);
    }
    Ccoord locations{};

    const size_t nb_pix{CcoordOps::get_size(sizes)};

    for (size_t i{0}; i < nb_pix; ++i) {
      BOOST_CHECK_EQUAL(
          i, CcoordOps::get_index(sizes, locations,
                                  CcoordOps::get_ccoord(sizes, locations, i)));
    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_index_col_major, Fix,
                                   testGoodies::dimlist, Fix) {
    constexpr auto dim{Fix::dim};
    using Ccoord = Ccoord_t<dim>;

    testGoodies::RandRange<Dim_t> rng;

    Ccoord sizes{};
    for (Dim_t i{0}; i < dim; ++i) {
      sizes[i] = rng.randval(2, 5);
    }
    Ccoord strides{CcoordOps::get_col_major_strides(sizes)};
    Ccoord locations{};

    const size_t nb_pix{CcoordOps::get_size(sizes)};

    for (size_t i{0}; i < nb_pix; ++i) {
      BOOST_CHECK_EQUAL(
          i, CcoordOps::get_index_from_strides(
                 strides, Ccoord{},
                 CcoordOps::get_ccoord(sizes, locations, i)));
    }

    for (size_t i{0}; i < nb_pix; ++i) {
      BOOST_CHECK_EQUAL(
          i, CcoordOps::get_index(
          sizes, Ccoord{},
          CcoordOps::get_ccoord_from_strides(sizes, locations, strides, i)));
    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_index_row_major, Fix,
                                   testGoodies::dimlist, Fix) {
    constexpr auto dim{Fix::dim};
    using Ccoord = Ccoord_t<dim>;

    testGoodies::RandRange<Dim_t> rng;

    Ccoord sizes{};
    for (Dim_t i{0}; i < dim; ++i) {
      sizes[i] = rng.randval(2, 5);
    }
    Ccoord strides{CcoordOps::get_row_major_strides(sizes)};
    Ccoord locations{};

    const size_t nb_pix{CcoordOps::get_size(sizes)};

    for (size_t i{0}; i < nb_pix; ++i) {
      BOOST_CHECK_EQUAL(
          i, CcoordOps::get_index_from_strides(
          strides, Ccoord{},
          CcoordOps::get_ccoord_from_strides(sizes, locations, strides, i)));
    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_index_non_contiguous_strides, Fix,
                                 testGoodies::dimlist, Fix) {
    constexpr auto dim{Fix::dim};
    using Ccoord = Ccoord_t<dim>;

    testGoodies::RandRange<Dim_t> rng;

    Ccoord sizes{};
    for (Dim_t i{0}; i < dim; ++i) {
      sizes[i] = rng.randval(2, 5);
    }
    Ccoord strides{CcoordOps::get_col_major_strides(sizes)};
    strides[dim-1] += 13;
    Ccoord locations{};

    const size_t nb_pix{CcoordOps::get_size(sizes)};

    for (size_t i{0}; i < nb_pix; ++i) {
      auto coord{CcoordOps::get_ccoord(sizes, locations, i)};
      auto j{CcoordOps::get_index_from_strides(strides, Ccoord{}, coord)};
      auto coord2{
          CcoordOps::get_ccoord_from_strides(sizes, Ccoord{}, strides, j)};
      for (Dim_t k{0}; k < dim; ++k) {
        BOOST_CHECK_EQUAL(coord[k], coord2[k]);
      }
    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_index_transposed_strides, Fix,
                                   testGoodies::dimlist, Fix) {
    constexpr auto dim{Fix::dim};
    using Ccoord = Ccoord_t<dim>;

    testGoodies::RandRange<Dim_t> rng;

    Ccoord sizes{};
    for (Dim_t i{0}; i < dim; ++i) {
      sizes[i] = rng.randval(2, 5);
    }
    Ccoord strides{CcoordOps::get_col_major_strides(sizes)};
    if (dim > 2) {
      strides[dim - 1] = strides[dim - 3] * sizes[dim - 3];
    } else {
      strides[dim - 1] = 1;
    }
    strides[dim - 2] = strides[dim - 1] * sizes[dim - 1];
    Ccoord locations{};

    const size_t nb_pix{CcoordOps::get_size(sizes)};

    for (size_t i{0}; i < nb_pix; ++i) {
      auto coord{CcoordOps::get_ccoord(sizes, locations, i)};
      auto j{CcoordOps::get_index_from_strides(strides, Ccoord{}, coord)};
      auto coord2{
          CcoordOps::get_ccoord_from_strides(sizes, Ccoord{}, strides, j)};
      for (Dim_t k{0}; k < dim; ++k) {
        BOOST_CHECK_EQUAL(coord[k], coord2[k]);
      }
    }
  }

  BOOST_AUTO_TEST_CASE(test_ccoord_unit_grid_pts) {
    DynCcoord_t nb_grid_pts1{1, 2};
    DynCcoord_t strides1{1, 1};
    DynCcoord_t ccoord1{0, 1};
    BOOST_CHECK_EQUAL(CcoordOps::get_ccoord_from_strides(
                          nb_grid_pts1, DynCcoord_t{}, strides1, 1), ccoord1);

    DynCcoord_t nb_grid_pts2{1, 1, 2};
    DynCcoord_t strides2{1, 1, 1};
    DynCcoord_t ccoord2{0, 0, 1};
    BOOST_CHECK_EQUAL(CcoordOps::get_ccoord_from_strides(
                          nb_grid_pts2, DynCcoord_t{}, strides2, 1), ccoord2);

    DynCcoord_t nb_grid_pts3{1, 2, 1};
    DynCcoord_t strides3{1, 1, 1};
    DynCcoord_t ccoord3{0, 1, 0};
    BOOST_CHECK_EQUAL(CcoordOps::get_ccoord_from_strides(
                          nb_grid_pts3, DynCcoord_t{}, strides3, 1), ccoord3);
  }

  BOOST_AUTO_TEST_CASE(vector_test) {
    constexpr Ccoord_t<threeD> c3{1, 2, 3};
    constexpr Ccoord_t<twoD> c2{c3[0], c3[1]};
    constexpr Rcoord_t<threeD> s3{1.3, 2.8, 5.7};
    constexpr Rcoord_t<twoD> s2{s3[0], s3[1]};

    Eigen::Matrix<Real, twoD, 1> v2;
    v2 << s3[0], s3[1];
    Eigen::Matrix<Real, threeD, 1> v3;
    v3 << s3[0], s3[1], s3[2];

    auto vec2{CcoordOps::get_vector(c2, v2(1))};
    auto vec3{CcoordOps::get_vector(c3, v3(1))};

    for (Dim_t i = 0; i < twoD; ++i) {
      BOOST_CHECK_EQUAL(c2[i] * v2(1), vec2[i]);
    }
    for (Dim_t i = 0; i < threeD; ++i) {
      BOOST_CHECK_EQUAL(c3[i] * v3(1), vec3[i]);
    }

    vec2 = CcoordOps::get_vector(c2, v2);
    vec3 = CcoordOps::get_vector(c3, v3);

    for (Dim_t i = 0; i < twoD; ++i) {
      BOOST_CHECK_EQUAL(c2[i] * v2(i), vec2[i]);
      BOOST_CHECK_EQUAL(vec2[i], CcoordOps::get_vector(c2, s2)[i]);
    }
    for (Dim_t i = 0; i < threeD; ++i) {
      BOOST_CHECK_EQUAL(c3[i] * v3(i), vec3[i]);
      BOOST_CHECK_EQUAL(vec3[i], CcoordOps::get_vector(c3, s3)[i]);
    }
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muGrid
