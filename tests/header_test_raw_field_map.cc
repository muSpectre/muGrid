/**
 * file   header_test_raw_field_map.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   17 Apr 2018
 *
 * @brief  tests for the raw field map type
 *
 * @section LICENSE
 *
 * Copyright © 2018 Till Junge
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


#include "test_field_collections.hh"
#include "common/field_map.hh"

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(raw_field_map_tests);

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(iter_field_test, F, iter_collections, F) {
    using FC_t = typename F::Parent::FC_t;
    using MSqMap = MatrixFieldMap<FC_t, Real, F::Parent::mdim(), F::Parent::mdim()>;
    MSqMap Mmap{F::fc["Tensorfield Real o2"]};
    auto m_it = Mmap.begin();
    auto m_it_end = Mmap.end();

    RawFieldMap<Eigen::Map<Eigen::Matrix<Real, F::Parent::mdim(), F::Parent::mdim()>>>
      raw_map{Mmap.get_field().eigenvec()};

    for (auto && mat: Mmap) {
      mat.setRandom();
    }

    for (auto tup: akantu::zip(Mmap, raw_map)) {
      auto & mat_A = std::get<0>(tup);
      auto & mat_B = std::get<1>(tup);

      BOOST_CHECK_EQUAL((mat_A-mat_B).norm(), 0.);
    }

    Mmap.get_field().eigenvec().setZero();

    for (auto && mat: raw_map) {
      mat.setIdentity();
    }

    for (auto && mat: Mmap) {
      BOOST_CHECK_EQUAL((mat-mat.Identity()).norm(), 0.);
    }

  }

  BOOST_AUTO_TEST_CASE(Const_correctness_test) {
    Eigen::VectorXd vec1(12);
    vec1.setRandom();

    RawFieldMap<Eigen::Map<Eigen::Vector3d>> map1{vec1};
    static_assert(not map1.IsConst, "should not have been const");
    RawFieldMap<Eigen::Map<const Eigen::Vector3d>> cmap1{vec1};
    static_assert(cmap1.IsConst, "should have been const");

    const Eigen::VectorXd vec2{vec1};

    RawFieldMap<Eigen::Map<const Eigen::Vector3d>> cmap2{vec2};
  }

  BOOST_AUTO_TEST_CASE(incompatible_size_check) {
    Eigen::VectorXd vec1(11);
    using RawFieldMap_t = RawFieldMap<Eigen::Map<Eigen::Vector3d>>;
    BOOST_CHECK_THROW(RawFieldMap_t {vec1}, std::runtime_error);
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre
