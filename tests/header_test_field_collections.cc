/**
 * @file   header_test_field_collections_1.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   20 Sep 2017
 *
 * @brief  Test the FieldCollection classes which provide fast optimized
 * iterators over run-time typed fields
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

#include "test_field_collections.hh"
#include "common/field_map_dynamic.hh"

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(field_collection_tests);

  BOOST_AUTO_TEST_CASE(simple) {
    constexpr Dim_t sdim = 2;
    using FC_t = GlobalFieldCollection<sdim>;
    FC_t fc;

    BOOST_CHECK_EQUAL(FC_t::spatial_dim(), sdim);
    BOOST_CHECK_EQUAL(fc.get_spatial_dim(), sdim);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(Simple_construction_test, F,
                                   test_collections, F) {
    BOOST_CHECK_EQUAL(F::FC_t::spatial_dim(), F::sdim());
    BOOST_CHECK_EQUAL(F::fc.get_spatial_dim(), F::sdim());
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(get_field2_test, F, test_collections, F) {
    const auto order{2};

    using FC_t = typename F::FC_t;
    using TF_t = TensorField<FC_t, Real, order, F::mdim()>;
    auto &&myfield = make_field<TF_t>("TensorField real 2", F::fc);

    using TensorMap = TensorFieldMap<FC_t, Real, order, F::mdim()>;
    using MatrixMap = MatrixFieldMap<FC_t, Real, F::mdim(), F::mdim()>;
    using ArrayMap = ArrayFieldMap<FC_t, Real, F::mdim(), F::mdim()>;

    TensorMap TFM(myfield);
    MatrixMap MFM(myfield);
    ArrayMap AFM(myfield);

    BOOST_CHECK_EQUAL(TFM.info_string(), "Tensor(d, " + std::to_string(order) +
                                             "_o, " +
                                             std::to_string(F::mdim()) + "_d)");
    BOOST_CHECK_EQUAL(MFM.info_string(), "Matrix(d, " +
                                             std::to_string(F::mdim()) + "x" +
                                             std::to_string(F::mdim()) + ")");
    BOOST_CHECK_EQUAL(AFM.info_string(), "Array(d, " +
                                             std::to_string(F::mdim()) + "x" +
                                             std::to_string(F::mdim()) + ")");
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(multi_field_test, F, mult_collections, F) {
    using FC_t = typename F::FC_t;
    // possible maptypes for Real tensor fields
    using T_type = Real;
    using T_TFM1_t = TensorFieldMap<FC_t, T_type, order, F::mdim()>;
    using T_TFM2_t =
        TensorFieldMap<FC_t, T_type, 2, F::mdim() * F::mdim()>;  //! dangerous
    using T4_Map_t = T4MatrixFieldMap<FC_t, Real, F::mdim()>;

    // impossible maptypes for Real tensor fields
    using T_SFM_t = ScalarFieldMap<FC_t, T_type>;
    using T_MFM_t = MatrixFieldMap<FC_t, T_type, 1, 1>;
    using T_AFM_t = ArrayFieldMap<FC_t, T_type, 1, 1>;
    using T_MFMw1_t = MatrixFieldMap<FC_t, Int, 1, 2>;
    using T_MFMw2_t = MatrixFieldMap<FC_t, Real, 1, 2>;
    using T_MFMw3_t = MatrixFieldMap<FC_t, Complex, 1, 2>;
    const std::string T_name{"Tensorfield Real o4"};
    const std::string T_name_w{"TensorField Real o4 wrongname"};

    BOOST_CHECK_THROW(T_SFM_t(F::fc.at(T_name)), FieldInterpretationError);
    BOOST_CHECK_NO_THROW(T_TFM1_t(F::fc.at(T_name)));
    BOOST_CHECK_NO_THROW(T_TFM2_t(F::fc.at(T_name)));
    BOOST_CHECK_NO_THROW(T4_Map_t(F::fc.at(T_name)));
    BOOST_CHECK_THROW(T4_Map_t(F::fc.at(T_name_w)), std::out_of_range);
    BOOST_CHECK_THROW(T_MFM_t(F::fc.at(T_name)), FieldInterpretationError);
    BOOST_CHECK_THROW(T_AFM_t(F::fc.at(T_name)), FieldInterpretationError);
    BOOST_CHECK_THROW(T_MFMw1_t(F::fc.at(T_name)), FieldInterpretationError);
    BOOST_CHECK_THROW(T_MFMw2_t(F::fc.at(T_name)), FieldInterpretationError);
    BOOST_CHECK_THROW(T_MFMw2_t(F::fc.at(T_name)), FieldInterpretationError);
    BOOST_CHECK_THROW(T_MFMw3_t(F::fc.at(T_name)), FieldInterpretationError);
    BOOST_CHECK_THROW(T_SFM_t(F::fc.at(T_name_w)), std::out_of_range);

    // possible maptypes for integer scalar fields
    using S_type = Int;
    using S_SFM_t = ScalarFieldMap<FC_t, S_type>;
    using S_TFM1_t = TensorFieldMap<FC_t, S_type, 1, 1>;
    using S_TFM2_t = TensorFieldMap<FC_t, S_type, 2, 1>;
    using S_MFM_t = MatrixFieldMap<FC_t, S_type, 1, 1>;
    using S_AFM_t = ArrayFieldMap<FC_t, S_type, 1, 1>;
    using S4_Map_t = T4MatrixFieldMap<FC_t, S_type, 1>;
    // impossible maptypes for integer scalar fields
    using S_MFMw1_t = MatrixFieldMap<FC_t, Int, 1, 2>;
    using S_MFMw2_t = MatrixFieldMap<FC_t, Real, 1, 2>;
    using S_MFMw3_t = MatrixFieldMap<FC_t, Complex, 1, 2>;
    const std::string S_name{"integer Scalar"};
    const std::string S_name_w{"integer Scalar wrongname"};

    BOOST_CHECK_NO_THROW(S_SFM_t(F::fc.at(S_name)));
    BOOST_CHECK_NO_THROW(S_TFM1_t(F::fc.at(S_name)));
    BOOST_CHECK_NO_THROW(S_TFM2_t(F::fc.at(S_name)));
    BOOST_CHECK_NO_THROW(S_MFM_t(F::fc.at(S_name)));
    BOOST_CHECK_NO_THROW(S_AFM_t(F::fc.at(S_name)));
    BOOST_CHECK_NO_THROW(S4_Map_t(F::fc.at(S_name)));
    BOOST_CHECK_THROW(S_MFMw1_t(F::fc.at(S_name)), FieldInterpretationError);
    BOOST_CHECK_THROW(T4_Map_t(F::fc.at(S_name)), FieldInterpretationError);
    BOOST_CHECK_THROW(S_MFMw2_t(F::fc.at(S_name)), FieldInterpretationError);
    BOOST_CHECK_THROW(S_MFMw2_t(F::fc.at(S_name)), FieldInterpretationError);
    BOOST_CHECK_THROW(S_MFMw3_t(F::fc.at(S_name)), FieldInterpretationError);
    BOOST_CHECK_THROW(S_SFM_t(F::fc.at(S_name_w)), std::out_of_range);

    // possible maptypes for complex matrix fields
    using M_type = Complex;
    using M_MFM_t = MatrixFieldMap<FC_t, M_type, F::sdim(), F::mdim()>;
    using M_AFM_t = ArrayFieldMap<FC_t, M_type, F::sdim(), F::mdim()>;
    // impossible maptypes for complex matrix fields
    using M_SFM_t = ScalarFieldMap<FC_t, M_type>;
    using M_MFMw1_t = MatrixFieldMap<FC_t, Int, 1, 2>;
    using M_MFMw2_t = MatrixFieldMap<FC_t, Real, 1, 2>;
    using M_MFMw3_t = MatrixFieldMap<FC_t, Complex, 1, 2>;
    const std::string M_name{"Matrixfield Complex sdim x mdim"};
    const std::string M_name_w{"Matrixfield Complex sdim x mdim wrongname"};

    BOOST_CHECK_THROW(M_SFM_t(F::fc.at(M_name)), FieldInterpretationError);
    BOOST_CHECK_NO_THROW(M_MFM_t(F::fc.at(M_name)));
    BOOST_CHECK_NO_THROW(M_AFM_t(F::fc.at(M_name)));
    BOOST_CHECK_THROW(M_MFMw1_t(F::fc.at(M_name)), FieldInterpretationError);
    BOOST_CHECK_THROW(M_MFMw2_t(F::fc.at(M_name)), FieldInterpretationError);
    BOOST_CHECK_THROW(M_MFMw2_t(F::fc.at(M_name)), FieldInterpretationError);
    BOOST_CHECK_THROW(M_MFMw3_t(F::fc.at(M_name)), FieldInterpretationError);
    BOOST_CHECK_THROW(M_SFM_t(F::fc.at(M_name_w)), std::out_of_range);
  }

  /* ---------------------------------------------------------------------- */
  //! Check whether fields can be initialized
  using mult_collections_t = boost::mpl::list<FC_multi_fixture<2, 2, true>,
                                              FC_multi_fixture<2, 3, true>,
                                              FC_multi_fixture<3, 3, true>>;
  using mult_collections_f = boost::mpl::list<FC_multi_fixture<2, 2, false>,
                                              FC_multi_fixture<2, 3, false>,
                                              FC_multi_fixture<3, 3, false>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(init_test_glob, F, mult_collections_t, F) {
    Ccoord_t<F::sdim()> size;
    Ccoord_t<F::sdim()> loc{};
    for (auto &&s : size) {
      s = 3;
    }
    BOOST_CHECK_NO_THROW(F::fc.initialise(size, loc));
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(init_test_loca, F, mult_collections_f, F) {
    testGoodies::RandRange<Int> rng;
    for (int i = 0; i < 7; ++i) {
      Ccoord_t<F::sdim()> pixel;
      for (auto &&s : pixel) {
        s = rng.randval(0, 7);
      }
      F::fc.add_pixel(pixel);
    }

    BOOST_CHECK_NO_THROW(F::fc.initialise());
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(init_test_loca_with_push_back, F,
                                   mult_collections_f, F) {
    constexpr auto mdim{F::mdim()};
    constexpr int nb_pix{7};
    testGoodies::RandRange<Int> rng{};
    using ftype = internal::TypedSizedFieldBase<decltype(F::fc), Real,
                                                mdim * mdim * mdim * mdim>;
    using stype = Eigen::Array<Real, mdim * mdim * mdim * mdim, 1>;
    auto &field = static_cast<ftype &>(F::fc["Tensorfield Real o4"]);
    field.push_back(stype());
    for (int i = 0; i < nb_pix; ++i) {
      Ccoord_t<F::sdim()> pixel;
      for (auto &&s : pixel) {
        s = rng.randval(0, 7);
      }
      F::fc.add_pixel(pixel);
    }

    BOOST_CHECK_THROW(F::fc.initialise(), FieldCollectionError);
    for (int i = 0; i < nb_pix - 1; ++i) {
      field.push_back(stype());
    }
    BOOST_CHECK_NO_THROW(F::fc.initialise());
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(iter_field_test, F, iter_collections, F) {
    using FC_t = typename F::Parent::FC_t;
    using Tensor4Map = TensorFieldMap<FC_t, Real, order, F::Parent::mdim()>;
    Tensor4Map T4map{F::fc["Tensorfield Real o4"]};
    TypedFieldMap<FC_t, Real> dyn_map{F::fc["Tensorfield Real o4"]};
    F::fc["Tensorfield Real o4"].set_zero();

    for (auto &&tens : T4map) {
      BOOST_CHECK_EQUAL(Real(Eigen::Tensor<Real, 0>(tens.abs().sum().eval())()),
                        0);
    }
    for (auto &&tens : T4map) {
      tens.setRandom();
    }

    for (auto &&tup : akantu::zip(T4map, dyn_map)) {
      auto &tens = std::get<0>(tup);
      auto &dyn = std::get<1>(tup);
      constexpr Dim_t nb_comp{ipow(F::mdim(), order)};
      Eigen::Map<Eigen::Array<Real, nb_comp, 1>> tens_arr(tens.data());
      Real error{(dyn - tens_arr).matrix().norm()};
      BOOST_CHECK_EQUAL(error, 0);
    }

    using Tensor2Map =
        TensorFieldMap<FC_t, Real, matrix_order, F::Parent::mdim()>;
    using MSqMap =
        MatrixFieldMap<FC_t, Real, F::Parent::mdim(), F::Parent::mdim()>;
    using ASqMap =
        ArrayFieldMap<FC_t, Real, F::Parent::mdim(), F::Parent::mdim()>;
    using A2Map = ArrayFieldMap<FC_t, Real, 3, 4>;
    using WrongMap = ArrayFieldMap<FC_t, Real, 7, 4>;
    Tensor2Map T2map{F::fc["Tensorfield Real o2"]};
    MSqMap Mmap{F::fc["Tensorfield Real o2"]};
    ASqMap Amap{F::fc["Tensorfield Real o2"]};
    A2Map DynMap{F::fc["Dynamically sized Field"]};
    auto &fc_ref{F::fc};
    BOOST_CHECK_THROW(WrongMap{fc_ref["Dynamically sized Field"]},
                      FieldInterpretationError);
    auto t2_it = T2map.begin();
    auto t2_it_end = T2map.end();
    auto m_it = Mmap.begin();
    auto a_it = Amap.begin();
    for (; t2_it != t2_it_end; ++t2_it, ++m_it, ++a_it) {
      t2_it->setRandom();
      auto &&m = *m_it;
      bool comp = (m == a_it->matrix());
      BOOST_CHECK(comp);
    }

    size_t counter{0};
    for (auto val : DynMap) {
      ++counter;
      val += val.Ones() * counter;
    }

    counter = 0;
    for (auto val : DynMap) {
      ++counter;
      val -= val.Ones() * counter;
      auto error{val.matrix().norm()};
      BOOST_CHECK_LT(error, tol);
    }

    using ScalarMap = ScalarFieldMap<FC_t, Int>;
    ScalarMap s_map{F::fc["integer Scalar"]};
    for (Uint i = 0; i < s_map.size(); ++i) {
      s_map[i] = i;
    }

    counter = 0;
    for (const auto &val : s_map) {
      BOOST_CHECK_EQUAL(counter++, val);
    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(ccoord_indexing_test, F, glob_iter_colls,
                                   F) {
    using FC_t = typename F::Parent::FC_t;
    using ScalarMap = ScalarFieldMap<FC_t, Int>;
    ScalarMap s_map{F::fc["integer Scalar"]};
    for (Uint i = 0; i < s_map.size(); ++i) {
      s_map[i] = i;
    }

    for (size_t i = 0; i < CcoordOps::get_size(F::fc.get_sizes()); ++i) {
      BOOST_CHECK_EQUAL(
          CcoordOps::get_index(F::fc.get_sizes(), F::fc.get_locations(),
                               CcoordOps::get_ccoord(F::fc.get_sizes(),
                                                     F::fc.get_locations(), i)),
          i);
    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(iterator_methods_test, F, iter_collections,
                                   F) {
    using FC_t = typename F::Parent::FC_t;
    using Tensor4Map = TensorFieldMap<FC_t, Real, order, F::Parent::mdim()>;
    Tensor4Map T4map{F::fc["Tensorfield Real o4"]};
    using it_t = typename Tensor4Map::iterator;
    std::ptrdiff_t diff{
        3};  // arbitrary, as long as it is smaller than the container size

    // check constructors
    auto itstart = T4map.begin();  // standard way of obtaining iterator
    auto itend = T4map.end();      // ditto

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
    BOOST_CHECK_EQUAL(
        response.str(),
        std::string("iterator on field 'Tensorfield Real o4', entry ") +
            std::to_string(diff));

    // check move and assigment constructor (and operator+)
    it_t it_repl{T4map};
    it_t itmove = std::move(T4map.begin());
    it_t it4 = it1 + diff;
    it_t it7 = it4 - diff;
    // BOOST_CHECK_EQUAL(itcopy, it1);
    BOOST_CHECK_EQUAL(itmove, it1);
    BOOST_CHECK_EQUAL(it4, it3);
    BOOST_CHECK_EQUAL(it7, it1);

    // check increments/decrements
    BOOST_CHECK_EQUAL(it1++, it_repl);  // post-increment
    BOOST_CHECK_EQUAL(it1, it_repl + 1);
    BOOST_CHECK_EQUAL(--it1, it_repl);      // pre -decrement
    BOOST_CHECK_EQUAL(++it1, it_repl + 1);  // pre -increment
    BOOST_CHECK_EQUAL(it1--, it_repl + 1);  // post-decrement
    BOOST_CHECK_EQUAL(it1, it_repl);

    // dereference and member-of-pointer check
    Eigen::Tensor<Real, 4> Tens = *it1;
    Eigen::Tensor<Real, 4> Tens2 = *itstart;
    Eigen::Tensor<bool, 0> check = (Tens == Tens2).all();
    BOOST_CHECK_EQUAL(static_cast<bool>(check()), true);

    BOOST_CHECK_NO_THROW(itstart->setZero());

    // check access subscripting
    auto T3a = *it3;
    auto T3b = itstart[diff];
    BOOST_CHECK(static_cast<bool>
                (Eigen::Tensor<bool, 0>((T3a == T3b).all())()));

    // div. comparisons
    BOOST_CHECK_LT(itstart, itend);
    BOOST_CHECK(!(itend < itstart));
    BOOST_CHECK(!(itstart < itstart));

    BOOST_CHECK_LE(itstart, itend);
    BOOST_CHECK_LE(itstart, itstart);
    BOOST_CHECK(!(itend <= itstart));

    BOOST_CHECK_GT(itend, itstart);
    BOOST_CHECK(!(itend > itend));
    BOOST_CHECK(!(itstart > itend));

    BOOST_CHECK_GE(itend, itstart);
    BOOST_CHECK_GE(itend, itend);
    BOOST_CHECK(!(itstart >= itend));

    // check assignment increment/decrement
    BOOST_CHECK_EQUAL(it1 += diff, it3);
    BOOST_CHECK_EQUAL(it1 -= diff, itstart);

    // check cell coordinates
    using Ccoord = Ccoord_t<F::sdim()>;
    Ccoord a{itstart.get_ccoord()};
    Ccoord b{0};

    // Weirdly, boost::has_left_shift<std::ostream, T>::value is false for
    // Ccoords, even though the operator is implemented :(
    // BOOST_CHECK_EQUAL(a, b);
    bool check2 = (a == b);
    BOOST_CHECK(check2);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(const_tensor_iter_test, F, iter_collections,
                                   F) {
    using FC_t = typename F::Parent::FC_t;
    using Tensor4Map = TensorFieldMap<FC_t, Real, order, F::Parent::mdim()>;
    Tensor4Map T4map{F::fc["Tensorfield Real o4"]};

    using T_t = typename Tensor4Map::T_t;
    Eigen::TensorMap<const T_t> Tens2(T4map[0].data(), F::Parent::sdim(),
                                      F::Parent::sdim(), F::Parent::sdim(),
                                      F::Parent::sdim());

    for (auto it = T4map.cbegin(); it != T4map.cend(); ++it) {
      // maps to const tensors can't be initialised with a const pointer this
      // sucks
      auto &&tens = *it;
      auto &&ptr = tens.data();

      static_assert(
          std::is_pointer<std::remove_reference_t<decltype(ptr)>>::value,
          "should be getting a pointer");
      // static_assert(std::is_const<std::remove_pointer_t<decltype(ptr)>>::value,
      // "should be const");

      // If Tensor were written well, above static_assert should pass, and the
      // following check shouldn't. If it get triggered, it means that a newer
      // version of Eigen now does have const-correct
      // TensorMap<const Tensor<...>. This means that const-correct field maps
      // are then also possible for tensors
      BOOST_CHECK(!std::is_const<std::remove_pointer_t<decltype(ptr)>>::value);
    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(const_matrix_iter_test, F, iter_collections,
                                   F) {
    using FC_t = typename F::Parent::FC_t;
    using MatrixMap = MatrixFieldMap<FC_t, Complex, F::sdim(), F::mdim()>;
    MatrixMap Mmap{F::fc["Matrixfield Complex sdim x mdim"]};

    for (auto it = Mmap.cbegin(); it != Mmap.cend(); ++it) {
      // maps to const tensors can't be initialised with a const pointer this
      // sucks
      auto &&mat = *it;
      auto &&ptr = mat.data();

      static_assert(
          std::is_pointer<std::remove_reference_t<decltype(ptr)>>::value,
          "should be getting a pointer");
      // static_assert(std::is_const<std::remove_pointer_t<decltype(ptr)>>::value,
      // "should be const");

      // If Matrix were written well, above static_assert should pass, and the
      // following check shouldn't. If it get triggered, it means that a newer
      // version of Eigen now does have const-correct
      // MatrixMap<const Matrix<...>. This means that const-correct field maps
      // are then also possible for matrices
      BOOST_CHECK(!std::is_const<std::remove_pointer_t<decltype(ptr)>>::value);
    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(const_scalar_iter_test, F, iter_collections,
                                   F) {
    using FC_t = typename F::Parent::FC_t;
    using ScalarMap = ScalarFieldMap<FC_t, Int>;
    ScalarMap Smap{F::fc["integer Scalar"]};

    for (auto it = Smap.cbegin(); it != Smap.cend(); ++it) {
      auto &&scal = *it;
      static_assert(
          std::is_const<std::remove_reference_t<decltype(scal)>>::value,
          "referred type should be const");
      static_assert(std::is_lvalue_reference<decltype(scal)>::value,
                    "Should have returned an lvalue ref");
    }
  }
  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(assignment_test, Fix, iter_collections,
                                   Fix) {
    auto t4map{Fix::t4_field.get_map()};
    auto t2map{Fix::t2_field.get_map()};
    auto scmap{Fix::sc_field.get_map()};
    auto m2map{Fix::m2_field.get_map()};
    auto dymap{Fix::dyn_field.get_map()};

    auto t4map_c{Fix::t4_field.get_const_map()};
    auto t2map_c{Fix::t2_field.get_const_map()};
    auto scmap_c{Fix::sc_field.get_const_map()};
    auto m2map_c{Fix::m2_field.get_const_map()};
    auto dymap_c{Fix::dyn_field.get_const_map()};

    const auto t4map_val{Matrices::Isymm<Fix::mdim()>()};
    t4map = t4map_val;
    const auto t2map_val{Matrices::I2<Fix::mdim()>()};
    t2map = t2map_val;
    const Int scmap_val{1};
    scmap = scmap_val;
    Eigen::Matrix<Complex, Fix::sdim(), Fix::mdim()> m2map_val;
    m2map_val.setRandom();
    m2map = m2map_val;
    const size_t nb_pts{Fix::fc.size()};

    testGoodies::RandRange<size_t> rnd{};
    BOOST_CHECK_EQUAL((t4map[rnd.randval(0, nb_pts - 1)] - t4map_val).norm(),
                      0.);
    BOOST_CHECK_EQUAL((t2map[rnd.randval(0, nb_pts - 1)] - t2map_val).norm(),
                      0.);
    BOOST_CHECK_EQUAL((scmap[rnd.randval(0, nb_pts - 1)] - scmap_val), 0.);
    BOOST_CHECK_EQUAL((m2map[rnd.randval(0, nb_pts - 1)] - m2map_val).norm(),
                      0.);
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(Eigentest, Fix, iter_collections, Fix) {
    auto t4eigen = Fix::t4_field.eigen();
    auto t2eigen = Fix::t2_field.eigen();

    BOOST_CHECK_EQUAL(t4eigen.rows(), ipow(Fix::mdim(), 4));
    BOOST_CHECK_EQUAL(t4eigen.cols(), Fix::t4_field.size());

    using T2_t = typename Eigen::Matrix<Real, Fix::mdim(), Fix::mdim()>;
    T2_t test_mat;
    test_mat.setRandom();
    Eigen::Map<Eigen::Array<Real, ipow(Fix::mdim(), 2), 1>> test_map(
        test_mat.data());
    t2eigen.col(0) = test_map;

    BOOST_CHECK_EQUAL((Fix::t2_field.get_map()[0] - test_mat).norm(), 0.);
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(field_proxy_test, Fix, iter_collections,
                                   Fix) {
    Eigen::VectorXd t4values{Fix::t4_field.eigenvec()};

    using FieldProxy_t = TypedField<typename Fix::FC_t, Real>;

    //! create a field proxy
    FieldProxy_t proxy("proxy to 'Tensorfield Real o4'", Fix::fc, t4values,
                       Fix::t4_field.get_nb_components());

    Eigen::VectorXd wrong_size_not_multiple{
        Eigen::VectorXd::Zero(t4values.size() + 1)};
    BOOST_CHECK_THROW(FieldProxy_t("size not a multiple of nb_components",
                                   Fix::fc, wrong_size_not_multiple,
                                   Fix::t4_field.get_nb_components()),
                      FieldError);

    Eigen::VectorXd wrong_size_but_multiple{Eigen::VectorXd::Zero(
        t4values.size() + Fix::t4_field.get_nb_components())};
    BOOST_CHECK_THROW(FieldProxy_t("size wrong multiple of nb_components",
                                   Fix::fc, wrong_size_but_multiple,
                                   Fix::t4_field.get_nb_components()),
                      FieldError);

    using Tensor4Map =
        T4MatrixFieldMap<typename Fix::FC_t, Real, Fix::Parent::mdim()>;
    Tensor4Map ref_map{Fix::t4_field};
    Tensor4Map proxy_map{proxy};

    for (auto tup : akantu::zip(ref_map, proxy_map)) {
      auto &ref = std::get<0>(tup);
      auto &prox = std::get<1>(tup);
      BOOST_CHECK_EQUAL((ref - prox).norm(), 0);
    }
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(field_proxy_of_existing_field, Fix,
                                   iter_collections, Fix) {
    Eigen::Ref<Eigen::VectorXd> t4values{Fix::t4_field.eigenvec()};
    using FieldProxy_t = TypedField<typename Fix::FC_t, Real>;

    //! create a field proxy
    FieldProxy_t proxy("proxy to 'Tensorfield Real o4'", Fix::fc, t4values,
                       Fix::t4_field.get_nb_components());

    using Tensor4Map =
        T4MatrixFieldMap<typename Fix::FC_t, Real, Fix::Parent::mdim()>;
    Tensor4Map ref_map{Fix::t4_field};
    Tensor4Map proxy_map{proxy};
    for (auto tup : akantu::zip(ref_map, proxy_map)) {
      auto &ref = std::get<0>(tup);
      auto &prox = std::get<1>(tup);
      prox += prox.Identity();
      BOOST_CHECK_EQUAL((ref - prox).norm(), 0);
    }
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(typed_field_getter, Fix, mult_collections,
                                   Fix) {
    constexpr auto mdim{Fix::mdim()};
    auto &fc{Fix::fc};
    auto &field = fc.template get_typed_field<Real>("Tensorfield Real o4");
    BOOST_CHECK_EQUAL(field.get_nb_components(), ipow(mdim, fourthOrder));
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(enumeration, Fix, iter_collections, Fix) {
    auto t4map{Fix::t4_field.get_map()};
    auto t2map{Fix::t2_field.get_map()};
    auto scmap{Fix::sc_field.get_map()};
    auto m2map{Fix::m2_field.get_map()};
    auto dymap{Fix::dyn_field.get_map()};

    for (auto &&tup :
         akantu::zip(scmap.get_collection(), scmap, scmap.enumerate())) {
      const auto &ccoord_ref = std::get<0>(tup);
      const auto &val_ref = std::get<1>(tup);
      const auto &key_val = std::get<2>(tup);
      const auto &ccoord = std::get<0>(key_val);
      const auto &val = std::get<1>(key_val);

      for (auto &&ccoords : akantu::zip(ccoord_ref, ccoord)) {
        const auto &ref{std::get<0>(ccoords)};
        const auto &val{std::get<1>(ccoords)};
        BOOST_CHECK_EQUAL(ref, val);
      }

      const auto error{std::abs(val - val_ref)};
      BOOST_CHECK_EQUAL(error, 0);
    }

    for (auto &&tup :
         akantu::zip(t4map.get_collection(), t4map, t4map.enumerate())) {
      const auto &ccoord_ref = std::get<0>(tup);
      const auto &val_ref = std::get<1>(tup);
      const auto &key_val = std::get<2>(tup);
      const auto &ccoord = std::get<0>(key_val);
      const auto &val = std::get<1>(key_val);

      for (auto &&ccoords : akantu::zip(ccoord_ref, ccoord)) {
        const auto &ref{std::get<0>(ccoords)};
        const auto &val{std::get<1>(ccoords)};
        BOOST_CHECK_EQUAL(ref, val);
      }

      const auto error{(val - val_ref).norm()};
      BOOST_CHECK_EQUAL(error, 0);
    }

    for (auto &&tup :
         akantu::zip(t2map.get_collection(), t2map, t2map.enumerate())) {
      const auto &ccoord_ref = std::get<0>(tup);
      const auto &val_ref = std::get<1>(tup);
      const auto &key_val = std::get<2>(tup);
      const auto &ccoord = std::get<0>(key_val);
      const auto &val = std::get<1>(key_val);

      for (auto &&ccoords : akantu::zip(ccoord_ref, ccoord)) {
        const auto &ref{std::get<0>(ccoords)};
        const auto &val{std::get<1>(ccoords)};
        BOOST_CHECK_EQUAL(ref, val);
      }

      const auto error{(val - val_ref).norm()};
      BOOST_CHECK_EQUAL(error, 0);
    }

    for (auto &&tup :
         akantu::zip(m2map.get_collection(), m2map, m2map.enumerate())) {
      const auto &ccoord_ref = std::get<0>(tup);
      const auto &val_ref = std::get<1>(tup);
      const auto &key_val = std::get<2>(tup);
      const auto &ccoord = std::get<0>(key_val);
      const auto &val = std::get<1>(key_val);

      for (auto &&ccoords : akantu::zip(ccoord_ref, ccoord)) {
        const auto &ref{std::get<0>(ccoords)};
        const auto &val{std::get<1>(ccoords)};
        BOOST_CHECK_EQUAL(ref, val);
      }

      const auto error{(val - val_ref).norm()};
      BOOST_CHECK_EQUAL(error, 0);
    }

    for (auto &&tup :
         akantu::zip(dymap.get_collection(), dymap, dymap.enumerate())) {
      const auto &ccoord_ref = std::get<0>(tup);
      const auto &val_ref = std::get<1>(tup);
      const auto &key_val = std::get<2>(tup);
      const auto &ccoord = std::get<0>(key_val);
      const auto &val = std::get<1>(key_val);

      for (auto &&ccoords : akantu::zip(ccoord_ref, ccoord)) {
        const auto &ref{std::get<0>(ccoords)};
        const auto &val{std::get<1>(ccoords)};
        BOOST_CHECK_EQUAL(ref, val);
      }

      const auto error{(val - val_ref).matrix().norm()};
      BOOST_CHECK_EQUAL(error, 0);
    }
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muSpectre
