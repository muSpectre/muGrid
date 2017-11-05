/**
 * file   test_field_collections.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   20 Sep 2017
 *
 * @brief  Test the FieldCollection classes which provide fast optimized iterators
 *         over run-time typed fields
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

#include <stdexcept>
#include <boost/mpl/list.hpp>
#include <random>
#include <type_traits>
#include <sstream>
#include <string>

#include "common/common.hh"
#include "common/ccoord_operations.hh"
#include "common/test_goodies.hh"
#include "tests.hh"
#include "common/field_collection.hh"
#include "common/field.hh"
#include "common/field_map.hh"

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(field_collection_tests);

  //! Test fixture for simple tests on single field in collection
  template <Dim_t DimS, Dim_t DimM, bool Global>
  struct FC_fixture:
    public FieldCollection<DimS, DimM, Global> {
    FC_fixture()
      :fc() {}
    inline static constexpr Dim_t sdim(){return DimS;}
    inline static constexpr Dim_t mdim(){return DimM;}
    inline static constexpr bool global(){return Global;}
    using FC_t = FieldCollection<DimS, DimM, Global>;
    FC_t fc;
  };

  using test_collections = boost::mpl::list<FC_fixture<2, 2, true>,
                                            FC_fixture<2, 3, true>,
                                            FC_fixture<3, 3, true>,
                                            FC_fixture<2, 2, false>,
                                            FC_fixture<2, 3, false>,
                                            FC_fixture<3, 3, false>>;

  BOOST_AUTO_TEST_CASE(simple) {
    const Dim_t sdim = 2;
    const Dim_t mdim = 2;
    using FC_t = FieldCollection<sdim, mdim>;
    FC_t fc;

    BOOST_CHECK_EQUAL(FC_t::spatial_dim(), sdim);
    BOOST_CHECK_EQUAL(fc.get_spatial_dim(), sdim);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(Simple_construction_test, F, test_collections, F) {
    BOOST_CHECK_EQUAL(F::FC_t::spatial_dim(), F::sdim());
    BOOST_CHECK_EQUAL(F::fc.get_spatial_dim(), F::sdim());

  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(get_field2_test, F, test_collections, F) {
    const auto order{2};

    using FC_t = typename F::FC_t;
    using TF_t = TensorField<FC_t, Real, order, F::mdim()>;
    auto && myfield = make_field<TF_t>("TensorField real 2", F::fc);

    using TensorMap = TensorFieldMap<FC_t, Real, order,F::mdim()>;
    using MatrixMap = MatrixFieldMap<FC_t, Real, F::mdim(), F::mdim()>;
    using ArrayMap = ArrayFieldMap<FC_t, Real, F::mdim(), F::mdim()>;

    TensorMap TFM(myfield);
    MatrixMap MFM(myfield);
    ArrayMap  AFM(myfield);

    BOOST_CHECK_EQUAL(TFM.info_string(),
                      "Tensor(d, "+ std::to_string(order) +
                      "_o, " + std::to_string(F::mdim()) + "_d)");
    BOOST_CHECK_EQUAL(MFM.info_string(),
                      "Matrix(d, "+ std::to_string(F::mdim()) +
                      "x" + std::to_string(F::mdim()) + ")");
    BOOST_CHECK_EQUAL(AFM.info_string(),
                      "Array(d, "+ std::to_string(F::mdim()) +
                      "x" + std::to_string(F::mdim()) + ")");

  }

  constexpr Dim_t order{4}, matrix_order{2};
  //! Test fixture for multiple fields in the collection
  template <Dim_t DimS, Dim_t DimM, bool Global>
  struct FC_multi_fixture{
    FC_multi_fixture()
      :fc() {
      //add Real tensor field
      make_field<TensorField<FC_t, Real, order, DimM>>
        ("Tensorfield Real o4", fc);
      //add Real tensor field
      make_field<TensorField<FC_t, Real, matrix_order, DimM>>
        ("Tensorfield Real o2", fc);
      //add integer scalar field
      make_field<ScalarField<FC_t, Int>>
        ("integer Scalar", fc);
      //add complex matrix field
      make_field<MatrixField<FC_t, Complex, DimS, DimM>>
        ("Matrixfield Complex sdim x mdim", fc);

    }
    inline static constexpr Dim_t sdim(){return DimS;}
    inline static constexpr Dim_t mdim(){return DimM;}
    inline static constexpr bool global(){return Global;}
    using FC_t = FieldCollection<DimS, DimM, Global>;
    FC_t fc;
  };

  using mult_collections = boost::mpl::list<FC_multi_fixture<2, 2, true>,
                                            FC_multi_fixture<2, 3, true>,
                                            FC_multi_fixture<3, 3, true>,
                                            FC_multi_fixture<2, 2, false>,
                                            FC_multi_fixture<2, 3, false>,
                                            FC_multi_fixture<3, 3, false>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(multi_field_test, F, mult_collections, F) {
    using FC_t = typename F::FC_t;
    // possible maptypes for Real tensor fields
    using T_type = Real;
    using T_TFM1_t = TensorFieldMap<FC_t, T_type, order, F::mdim()>;
    using T_TFM2_t = TensorFieldMap<FC_t, T_type, 2, F::mdim()*F::mdim()>; //! dangerous
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
    BOOST_CHECK_THROW(S_MFMw1_t(F::fc.at(S_name)), FieldInterpretationError);
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
    for (auto && s: size) {
      s = 3;
    }
    BOOST_CHECK_NO_THROW(F::fc.initialise(size));
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(init_test_loca, F, mult_collections_f, F) {
    testGoodies::RandRange<Int> rng;
    for (int i = 0; i < 7; ++i) {
      Ccoord_t<F::sdim()> pixel;
      for (auto && s: pixel) {
        s = rng.randval(0, 7);
      }
      F::fc.add_pixel(pixel);
    }

    BOOST_CHECK_NO_THROW(F::fc.initialise());
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(init_test_loca_with_push_back, F, mult_collections_f, F) {
    constexpr auto mdim = F::mdim();
    constexpr int nb_pix = 7;
    testGoodies::RandRange<Int> rng;
    using ftype = internal::TypedFieldBase<decltype(F::fc), Real,  mdim*mdim*mdim*mdim>;
    using stype = Eigen::Array<Real, mdim*mdim*mdim*mdim, 1>;
    auto & field = reinterpret_cast<ftype&>(F::fc["Tensorfield Real o4"]);
    field.push_back(stype());
    for (int i = 0; i < nb_pix; ++i) {
      Ccoord_t<F::sdim()> pixel;
      for (auto && s: pixel) {
        s = rng.randval(0, 7);
      }
      F::fc.add_pixel(pixel);
    }

    BOOST_CHECK_THROW(F::fc.initialise(), FieldCollectionError);
    for (int i = 0; i < nb_pix-1; ++i) {
      field.push_back(stype());
    }
    BOOST_CHECK_NO_THROW(F::fc.initialise());

  }

  //! Test fixture for iterators over multiple fields
  template <Dim_t DimS, Dim_t DimM, bool Global>
  struct FC_iterator_fixture
    : public FC_multi_fixture<DimS, DimM, Global> {
    using Parent = FC_multi_fixture<DimS, DimM, Global>;
    FC_iterator_fixture()
      :Parent() {
      this-> fill();
    }

    template <bool isGlobal = Global>
    std::enable_if_t<isGlobal> fill() {
      static_assert(Global==isGlobal, "You're breaking my SFINAE plan");
      Ccoord_t<Parent::sdim()> size;
      for (auto && s: size) {
        s = cube_size();
      }
      this->fc.initialise(size);
    }

    template <bool notGlobal = !Global>
    std::enable_if_t<notGlobal> fill (int dummy = 0) {
      static_assert(notGlobal != Global, "You're breaking my SFINAE plan");
      testGoodies::RandRange<Int> rng;
      for (int i = 0*dummy; i < sele_size(); ++i) {
        Ccoord_t<Parent::sdim()> pixel;
        for (auto && s: pixel) {
          s = rng.randval(0, 7);
        }
        this->fc.add_pixel(pixel);
      }

      this->fc.initialise();
    }

    constexpr static Dim_t cube_size() {return 3;}
    constexpr static Dim_t sele_size() {return 7;}
  };

  using iter_collections = boost::mpl::list<FC_iterator_fixture<2, 2, true>,
                                            FC_iterator_fixture<2, 3, true>,
                                            FC_iterator_fixture<3, 3, true>,
                                            FC_iterator_fixture<2, 2, false>,
                                            FC_iterator_fixture<2, 3, false>,
                                            FC_iterator_fixture<3, 3, false>>;

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

  using glob_iter_colls = boost::mpl::list<FC_iterator_fixture<2, 2, true>,
                                           FC_iterator_fixture<2, 3, true>,
                                           FC_iterator_fixture<3, 3, true>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(ccoord_indexing_test, F, glob_iter_colls, F) {
    using FC_t = typename F::Parent::FC_t;
    using ScalarMap = ScalarFieldMap<FC_t, Int>;
    ScalarMap s_map{F::fc["integer Scalar"]};
    for (Uint i = 0; i < s_map.size(); ++i) {
      s_map[i] = i;
    }


    for (size_t i = 0; i < CcoordOps::get_size(F::fc.get_sizes()); ++i) {
      Ccoord_t<F::sdim()> sizes(F::fc.get_sizes());
      Ccoord_t<F::sdim()> ccoords(CcoordOps::get_ccoord(sizes, i));
      BOOST_CHECK_EQUAL(CcoordOps::get_index(F::fc.get_sizes(),
                                             CcoordOps::get_ccoord(F::fc.get_sizes(), i)), i);
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
    Ccoord b{};

    // Weirdly, boost::has_left_shift<std::ostream, T>::value is falso for Ccoords, even though the operator is implemented :(
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
