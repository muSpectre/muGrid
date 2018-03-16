/**
 * @file   test_field_collections_2.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   23 Nov 2017
 *
 * @brief Continuation of tests from test_field_collection.cc, split for faster
 *        compilation
 *
 * Copyright © 2017 Till Junge
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

#include "test_field_collections_header.hh"
namespace muSpectre {
  BOOST_AUTO_TEST_SUITE(field_collection_tests);

    BOOST_FIXTURE_TEST_CASE_TEMPLATE(iter_field_test, F, iter_collections, F) {
    using FC_t = typename F::Parent::FC_t;
    using Tensor4Map = TensorFieldMap<FC_t, Real, order, F::Parent::mdim()>;
    Tensor4Map T4map{F::fc["Tensorfield Real o4"]};
    F::fc["Tensorfield Real o4"].set_zero();
    for (auto && tens:T4map) {
      BOOST_CHECK_EQUAL(Real(Eigen::Tensor<Real, 0>(tens.abs().sum().eval())()), 0);
    }

    using Tensor2Map = TensorFieldMap<FC_t, Real, matrix_order, F::Parent::mdim()>;
    using MSqMap = MatrixFieldMap<FC_t, Real, F::Parent::mdim(), F::Parent::mdim()>;
    using ASqMap =  ArrayFieldMap<FC_t, Real, F::Parent::mdim(), F::Parent::mdim()>;
    Tensor2Map T2map{F::fc["Tensorfield Real o2"]};
    MSqMap Mmap{F::fc["Tensorfield Real o2"]};
    ASqMap Amap{F::fc["Tensorfield Real o2"]};
    auto t2_it = T2map.begin();
    auto t2_it_end = T2map.end();
    auto m_it = Mmap.begin();
    auto a_it = Amap.begin();
    for (; t2_it != t2_it_end; ++t2_it, ++m_it, ++a_it) {
      t2_it->setRandom();
      auto && m = *m_it;
      bool comp = (m == a_it->matrix());
      BOOST_CHECK(comp);
    }

    using ScalarMap = ScalarFieldMap<FC_t, Int>;
    ScalarMap s_map{F::fc["integer Scalar"]};
    for (Uint i = 0; i < s_map.size(); ++i) {
      s_map[i] = i;
    }
    Uint counter{0};

    for (const auto& val: s_map) {
      BOOST_CHECK_EQUAL(counter++, val);
    }

  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(ccoord_indexing_test, F, glob_iter_colls, F) {
    using FC_t = typename F::Parent::FC_t;
    using ScalarMap = ScalarFieldMap<FC_t, Int>;
    ScalarMap s_map{F::fc["integer Scalar"]};
    for (Uint i = 0; i < s_map.size(); ++i) {
      s_map[i] = i;
    }


    for (size_t i = 0; i < CcoordOps::get_size(F::fc.get_sizes()); ++i) {
      BOOST_CHECK_EQUAL(CcoordOps::get_index(F::fc.get_sizes(),
                                             F::fc.get_locations(),
                                             CcoordOps::get_ccoord(F::fc.get_sizes(),
                                                                   F::fc.get_locations(),
                                                                   i)), i);
    }

  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(iterator_methods_test, F, iter_collections, F) {
    using FC_t = typename F::Parent::FC_t;
    using Tensor4Map = TensorFieldMap<FC_t, Real, order, F::Parent::mdim()>;
    Tensor4Map T4map{F::fc["Tensorfield Real o4"]};
    using it_t = typename Tensor4Map::iterator;
    std::ptrdiff_t diff{3}; // arbitrary, as long as it is smaller than the container size

    // check constructors
    auto itstart = T4map.begin(); // standard way of obtaining iterator
    auto itend = T4map.end(); // ditto

    it_t it1{T4map};
    it_t it2{T4map, false};
    it_t it3{T4map, size_t(diff)};
    BOOST_CHECK(itstart == itstart);
    BOOST_CHECK(itstart != itend);
    BOOST_CHECK_EQUAL(itstart, it1);
    BOOST_CHECK_EQUAL(itend, it2);

    // check ostream operator
    std::stringstream response;
    response << it3;
    BOOST_CHECK_EQUAL
      (response.str(),
       std::string
       ("iterator on field 'Tensorfield Real o4', entry ") +
       std::to_string(diff));

    // check copy, move, and assigment constructor (and operator+)
    it_t itcopy = it1;
    it_t itmove = std::move(T4map.begin());
    it_t it4 = it1+diff;
    it_t it5(it2);
    it_t tmp(it2);
    it_t it6(std::move(tmp));
    it_t it7 = it4 -diff;
    BOOST_CHECK_EQUAL(itcopy, it1);
    BOOST_CHECK_EQUAL(itmove, it1);
    BOOST_CHECK_EQUAL(it4, it3);
    BOOST_CHECK_EQUAL(it5, it2);
    BOOST_CHECK_EQUAL(it6, it5);
    BOOST_CHECK_EQUAL(it7, it1);

    // check increments/decrements
    BOOST_CHECK_EQUAL(it1++, itcopy);    // post-increment
    BOOST_CHECK_EQUAL(it1, itcopy+1);
    BOOST_CHECK_EQUAL(--it1, itcopy);    // pre -decrement
    BOOST_CHECK_EQUAL(++it1, itcopy+1);  // pre -increment
    BOOST_CHECK_EQUAL(it1--, itcopy+1);  // post-decrement
    BOOST_CHECK_EQUAL(it1, itcopy);


    // dereference and member-of-pointer check
    Eigen::Tensor<Real, 4> Tens = *it1;
    Eigen::Tensor<Real, 4> Tens2 = *itstart;
    Eigen::Tensor<bool, 0> check = (Tens==Tens2).all();
    BOOST_CHECK_EQUAL(bool(check()), true);

    BOOST_CHECK_NO_THROW(itstart->setZero());

    //check access subscripting
    auto T3a = *it3;
    auto T3b = itstart[diff];
    BOOST_CHECK(bool(Eigen::Tensor<bool, 0>((T3a==T3b).all())()));

    // div. comparisons
    BOOST_CHECK_LT(itstart, itend);
    BOOST_CHECK(!(itend < itstart));
    BOOST_CHECK(!(itstart < itstart));

    BOOST_CHECK_LE(itstart, itend);
    BOOST_CHECK_LE(itstart, itstart);
    BOOST_CHECK(!(itend <= itstart));

    BOOST_CHECK_GT(itend, itstart);
    BOOST_CHECK(!(itend>itend));
    BOOST_CHECK(!(itstart>itend));

    BOOST_CHECK_GE(itend, itstart);
    BOOST_CHECK_GE(itend, itend);
    BOOST_CHECK(!(itstart >= itend));

    // check assignment increment/decrement
    BOOST_CHECK_EQUAL(it1+=diff, it3);
    BOOST_CHECK_EQUAL(it1-=diff, itstart);

    // check cell coordinates
    using Ccoord = Ccoord_t<F::sdim()>;
    Ccoord a{itstart.get_ccoord()};
    Ccoord b{0};

    // Weirdly, boost::has_left_shift<std::ostream, T>::value is false for Ccoords, even though the operator is implemented :(
    //BOOST_CHECK_EQUAL(a, b);
    bool check2 = (a==b);
    BOOST_CHECK(check2);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(const_tensor_iter_test, F, iter_collections, F) {
    using FC_t = typename F::Parent::FC_t;
    using Tensor4Map = TensorFieldMap<FC_t, Real, order, F::Parent::mdim()>;
    Tensor4Map T4map{F::fc["Tensorfield Real o4"]};

    using T_t = typename Tensor4Map::T_t;
    Eigen::TensorMap<const T_t> Tens2(T4map[0].data(), F::Parent::sdim(), F::Parent::sdim(), F::Parent::sdim(), F::Parent::sdim());

    for (auto it = T4map.cbegin(); it != T4map.cend(); ++it) {
      // maps to const tensors can't be initialised with a const pointer this sucks
      auto&&  tens = *it;
      auto&&  ptr = tens.data();

      static_assert(std::is_pointer<std::remove_reference_t<decltype(ptr)>>::value, "should be getting a pointer");
      //static_assert(std::is_const<std::remove_pointer_t<decltype(ptr)>>::value, "should be const");

      // If Tensor were written well, above static_assert should pass, and the
      // following check shouldn't. If it get triggered, it means that a newer
      // version of Eigen now does have const-correct
      // TensorMap<const Tensor<...>. This means that const-correct field maps
      // are then also possible for tensors
      BOOST_CHECK(!std::is_const<std::remove_pointer_t<decltype(ptr)>>::value);

    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(const_matrix_iter_test, F, iter_collections, F) {
    using FC_t = typename F::Parent::FC_t;
    using MatrixMap = MatrixFieldMap<FC_t, Complex, F::sdim(), F::mdim()>;
    MatrixMap Mmap{F::fc["Matrixfield Complex sdim x mdim"]};

    for (auto it = Mmap.cbegin(); it != Mmap.cend(); ++it) {
      // maps to const tensors can't be initialised with a const pointer this sucks
      auto&&  mat = *it;
      auto&&  ptr = mat.data();

      static_assert(std::is_pointer<std::remove_reference_t<decltype(ptr)>>::value, "should be getting a pointer");
      //static_assert(std::is_const<std::remove_pointer_t<decltype(ptr)>>::value, "should be const");

      // If Matrix were written well, above static_assert should pass, and the
      // following check shouldn't. If it get triggered, it means that a newer
      // version of Eigen now does have const-correct
      // MatrixMap<const Matrix<...>. This means that const-correct field maps
      // are then also possible for matrices
      BOOST_CHECK(!std::is_const<std::remove_pointer_t<decltype(ptr)>>::value);

    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(const_scalar_iter_test, F, iter_collections, F) {
    using FC_t = typename F::Parent::FC_t;
    using ScalarMap = ScalarFieldMap<FC_t, Int>;
    ScalarMap Smap{F::fc["integer Scalar"]};

    for (auto it = Smap.cbegin(); it != Smap.cend(); ++it) {
      auto&& scal = *it;
      static_assert(std::is_const<std::remove_reference_t<decltype(scal)>>::value,
                    "referred type should be const");
      static_assert(std::is_lvalue_reference<decltype(scal)>::value,
                    "Should have returned an lvalue ref");

    }
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre
