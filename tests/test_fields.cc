/**
 * @file   test_fields.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   20 Sep 2017
 *
 * @brief  Test Fields that are used in FieldCollections
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

#include "tests.hh"
#include "common/field_collection.hh"
#include "common/field.hh"
#include "common/ccoord_operations.hh"

#include <boost/mpl/list.hpp>

#include <type_traits>

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(field_test);

  template <bool Global>
  struct FieldFixture
  {
    constexpr static bool IsGlobal{Global};
    constexpr static Dim_t Order{secondOrder};
    constexpr static Dim_t SDim{twoD};
    constexpr static Dim_t MDim{threeD};
    constexpr static Dim_t NbComponents{ipow(MDim, Order)};
    using FieldColl_t = std::conditional_t<Global,
                                           GlobalFieldCollection<SDim>,
                                           LocalFieldCollection<SDim>>;
    using TField_t = TensorField<FieldColl_t, Real, Order, MDim>;
    using DField_t = TypedField<FieldColl_t, Real>;

    FieldFixture()
      : tensor_field{make_field<TField_t>("TensorField", this->fc)},
        dynamic_field1{
          make_field<DField_t>("Dynamically sized field with correct number of"
                               " components", this->fc, ipow(MDim, Order))},
        dynamic_field2{
          make_field<DField_t>("Dynamically sized field with incorrect number"
                               "of components", this->fc, NbComponents+1)}
    {}
    ~FieldFixture() = default;

    FieldColl_t fc{};
    TField_t & tensor_field;
    DField_t & dynamic_field1;
    DField_t & dynamic_field2;
  };

  using field_fixtures = boost::mpl::list<FieldFixture<true>, FieldFixture<false>>;

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE(size_check_global, FieldFixture<true>) {
    // check that fields are initialised with empty vector
    BOOST_CHECK_EQUAL(tensor_field.size(), 0);
    BOOST_CHECK_EQUAL(dynamic_field1.size(), 0);
    BOOST_CHECK_EQUAL(dynamic_field2.size(), 0);

    // check that returned size is correct
    Dim_t len{2};
    auto sizes{CcoordOps::get_cube<SDim>(len)};
    fc.initialise(sizes, {});

    auto nb_pixels{CcoordOps::get_size(sizes)};
    BOOST_CHECK_EQUAL(tensor_field.size(),   nb_pixels);
    BOOST_CHECK_EQUAL(dynamic_field1.size(), nb_pixels);
    BOOST_CHECK_EQUAL(dynamic_field2.size(), nb_pixels);

    constexpr Dim_t pad_size{3};
    tensor_field.set_pad_size(pad_size);
    dynamic_field1.set_pad_size(pad_size);
    dynamic_field2.set_pad_size(pad_size);

    // check that setting pad size won't change logical size
    BOOST_CHECK_EQUAL(tensor_field.size(),   nb_pixels);
    BOOST_CHECK_EQUAL(dynamic_field1.size(), nb_pixels);
    BOOST_CHECK_EQUAL(dynamic_field2.size(), nb_pixels);
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE(size_check_local, FieldFixture<false>) {
    // check that fields are initialised with empty vector
    BOOST_CHECK_EQUAL(tensor_field.size(), 0);
    BOOST_CHECK_EQUAL(dynamic_field1.size(), 0);
    BOOST_CHECK_EQUAL(dynamic_field2.size(), 0);

    // check that returned size is correct
    Dim_t nb_pixels{3};

    Eigen::Array<Real, NbComponents, 1> new_elem;
    Eigen::Array<Real, 1, NbComponents> wrong_elem;
    for (Dim_t i{0}; i<NbComponents; ++i) {
      new_elem(i) = i;
      wrong_elem(i) = .1*i;
    }

    for (Dim_t i{0}; i < nb_pixels; ++i) {
      tensor_field.push_back(new_elem);
      dynamic_field1.push_back(new_elem);
      BOOST_CHECK_THROW(dynamic_field1.push_back(wrong_elem), FieldError);
      BOOST_CHECK_THROW(dynamic_field2.push_back(new_elem), FieldError);
    }
    BOOST_CHECK_EQUAL(tensor_field.size(),   nb_pixels);
    BOOST_CHECK_EQUAL(dynamic_field1.size(), nb_pixels);
    BOOST_CHECK_EQUAL(dynamic_field2.size(), 0);

    for (Dim_t i{0}; i < nb_pixels; ++i) {
      fc.add_pixel({i});
    }

    fc.initialise();

    BOOST_CHECK_EQUAL(tensor_field.size(),   nb_pixels);
    BOOST_CHECK_EQUAL(dynamic_field1.size(), nb_pixels);
    BOOST_CHECK_EQUAL(dynamic_field2.size(), nb_pixels);

    BOOST_CHECK_EQUAL(tensor_field.get_pad_size(),   0);
    BOOST_CHECK_EQUAL(dynamic_field1.get_pad_size(), 0);
    BOOST_CHECK_EQUAL(dynamic_field2.get_pad_size(), 0);

    constexpr Dim_t pad_size{3};
    tensor_field.set_pad_size(pad_size);
    dynamic_field1.set_pad_size(pad_size);
    dynamic_field2.set_pad_size(pad_size);

    BOOST_CHECK_EQUAL(tensor_field.get_pad_size(),   pad_size);
    BOOST_CHECK_EQUAL(dynamic_field1.get_pad_size(), pad_size);
    BOOST_CHECK_EQUAL(dynamic_field2.get_pad_size(), pad_size);

    // check that setting pad size won't change logical size
    BOOST_CHECK_EQUAL(tensor_field.size(),   nb_pixels);
    BOOST_CHECK_EQUAL(dynamic_field1.size(), nb_pixels);
    BOOST_CHECK_EQUAL(dynamic_field2.size(), nb_pixels);
  }

  BOOST_AUTO_TEST_CASE(simple_creation) {
    constexpr Dim_t sdim{twoD};
    constexpr Dim_t mdim{twoD};
    constexpr Dim_t order{fourthOrder};
    using FC_t = GlobalFieldCollection<sdim>;
    FC_t fc;

    using TF_t = TensorField<FC_t, Real, order, mdim>;
    auto & field{make_field<TF_t>("TensorField 1", fc)};

    // check that fields are initialised with empty vector
    BOOST_CHECK_EQUAL(field.size(), 0);
    Dim_t len{2};
    fc.initialise(CcoordOps::get_cube<sdim>(len), {});
    // check that returned size is correct
    BOOST_CHECK_EQUAL(field.size(), ipow(len, sdim));
    // check that setting pad size won't change logical size
    field.set_pad_size(24);
    BOOST_CHECK_EQUAL(field.size(), ipow(len, sdim));
  }

  BOOST_AUTO_TEST_CASE(dynamic_field_creation) {
    constexpr Dim_t sdim{threeD};
    Dim_t nb_components{2};

    using FC_t = GlobalFieldCollection<sdim>;
    FC_t fc{};
    make_field<TypedField<FC_t, Real>>("Dynamic Field", fc, nb_components);
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre
