/**
 * @file   test_field_collections.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   20 Sep 2017
 *
 * @brief  Test the FieldCollection classes which provide fast optimized iterators
 *         over run-time typed fields
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

  BOOST_AUTO_TEST_CASE(simple) {
    const Dim_t sdim = 2;
    const Dim_t mdim = 2;
    using FC_t = GlobalFieldCollection<sdim, mdim>;
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


  BOOST_FIXTURE_TEST_CASE_TEMPLATE(multi_field_test, F, mult_collections, F) {
    using FC_t = typename F::FC_t;
    // possible maptypes for Real tensor fields
    using T_type = Real;
    using T_TFM1_t = TensorFieldMap<FC_t, T_type, order, F::mdim()>;
    using T_TFM2_t = TensorFieldMap<FC_t, T_type, 2, F::mdim()*F::mdim()>; //! dangerous
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

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(init_test_loca_with_push_back, F,
                                   mult_collections_f, F) {
    constexpr auto mdim = F::mdim();
    constexpr int nb_pix = 7;
    testGoodies::RandRange<Int> rng;
    using ftype = internal::TypedSizedFieldBase<
      decltype(F::fc), Real,  mdim*mdim*mdim*mdim>;
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



  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre
