/**
 * @file   header_test_t4_map.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   20 Nov 2017
 *
 * @brief  Test the fourth-order map on second-order tensor implementation
 *
 * Copyright © 2017 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 */

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <boost/mpl/list.hpp>

#include "common/common.hh"
#include "tests.hh"
#include "common/T4_map_proxy.hh"

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(T4map_tests);

  /**
   * Test fixture for construction of T4Map for the time being, symmetry is not
   * exploited
   */
  template <typename T, Dim_t Dim> struct T4_fixture {
    T4_fixture() : matrix{}, tensor(matrix.data()) {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using M4 = T4Mat<T, Dim>;
    using T4 = T4MatMap<T, Dim>;
    constexpr static Dim_t dim{Dim};
    M4 matrix;
    T4 tensor;
  };

  using fix_collection =
      boost::mpl::list<T4_fixture<Real, twoD>, T4_fixture<Real, threeD>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(Simple_construction_test, F, fix_collection,
                                   F) {
    BOOST_CHECK_EQUAL(F::tensor.cols(), F::dim * F::dim);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(write_access_test, F, fix_collection, F) {
    auto &t4 = F::tensor;
    constexpr Dim_t dim{F::dim};
    Eigen::TensorFixedSize<Real, Eigen::Sizes<dim, dim, dim, dim>> t4c;
    Eigen::Map<typename F::M4> t4c_map(t4c.data());
    for (Dim_t i = 0; i < F::dim; ++i) {
      for (Dim_t j = 0; j < F::dim; ++j) {
        for (Dim_t k = 0; k < F::dim; ++k) {
          for (Dim_t l = 0; l < F::dim; ++l) {
            get(t4, i, j, k, l) =
                1000 * (i + 1) + 100 * (j + 1) + 10 * (k + 1) + l + 1;
            t4c(i, j, k, l) =
                1000 * (i + 1) + 100 * (j + 1) + 10 * (k + 1) + l + 1;
          }
        }
      }
    }
    for (Dim_t i = 0; i < ipow(dim, 4); ++i) {
      BOOST_CHECK_EQUAL(F::matrix.data()[i], t4c.data()[i]);
    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(assign_matrix_test, F, fix_collection, F) {
    decltype(F::matrix) matrix;
    matrix.setRandom();
    F::tensor = matrix;
    for (Dim_t i = 0; i < ipow(F::dim, 4); ++i) {
      BOOST_CHECK_EQUAL(F::matrix.data()[i], matrix.data()[i]);
    }
  }

  BOOST_AUTO_TEST_CASE(Return_ref_from_const_test) {
    constexpr Dim_t dim{2};
    using T = int;
    using M4 = Eigen::Matrix<T, dim * dim, dim * dim>;
    using M4c = const Eigen::Matrix<T, dim * dim, dim * dim>;
    using T4 = T4MatMap<T, dim>;
    using T4c = T4MatMap<T, dim, true>;

    M4 mat;
    mat.setRandom();
    M4c cmat{mat};
    T4 tensor{mat.data()};
    T4c ctensor{mat.data()};

    T a = get(tensor, 0, 0, 0, 1);
    T b = get(ctensor, 0, 0, 0, 1);
    BOOST_CHECK_EQUAL(a, b);
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muSpectre
