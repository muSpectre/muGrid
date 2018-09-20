/**
 * @file   header_test_fields.cc
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
    using MField_t = MatrixField<FieldColl_t, Real, SDim, MDim>;
    using DField_t = TypedField<FieldColl_t, Real>;

    FieldFixture()
      : tensor_field{make_field<TField_t>("TensorField", this->fc)},
        matrix_field{make_field<MField_t>("MatrixField", this->fc)},
        dynamic_field1{
          make_field<DField_t>("Dynamically sized field with correct number of"
                               " components", this->fc, ipow(MDim, Order))},
        dynamic_field2{
          make_field<DField_t>("Dynamically sized field with incorrect number"
                               " of components", this->fc, NbComponents+1)}
    {}
    ~FieldFixture() = default;

    FieldColl_t fc{};
    TField_t & tensor_field;
    MField_t & matrix_field;
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

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(get_zeros_like, Fix, field_fixtures, Fix) {
    auto & t_clone{Fix::tensor_field.get_zeros_like("tensor clone")};
    static_assert(std::is_same<
                  std::remove_reference_t<decltype(t_clone)>,
                  typename Fix::TField_t>::value, "wrong overloaded function");

    auto & m_clone{Fix::matrix_field.get_zeros_like("matrix clone")};
    static_assert(std::is_same<
                  std::remove_reference_t<decltype(m_clone)>,
                  typename Fix::MField_t>::value, "wrong overloaded function");
    using FieldColl_t = typename Fix::FieldColl_t;
    using T = typename Fix::TField_t::Scalar;
    TypedField<FieldColl_t, T> & t_ref{t_clone};

    auto & typed_clone{t_ref.get_zeros_like("dynamically sized clone")};
    static_assert(std::is_same<
                  std::remove_reference_t<decltype(typed_clone)>,
                  TypedField<FieldColl_t, T>>::value,
                  "Field type incorrectly deduced");
    BOOST_CHECK_EQUAL(typed_clone.get_nb_components(), t_clone.get_nb_components());

    auto & dyn_clone{Fix::dynamic_field1.get_zeros_like("dynamic clone")};
    static_assert(std::is_same<decltype(dyn_clone), decltype(typed_clone)>::value,
                  "mismatch");
    BOOST_CHECK_EQUAL(typed_clone.get_nb_components(), dyn_clone.get_nb_components());
  }

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(fill_global_local) {
    FieldFixture<true> global;
    FieldFixture<false> local;
    constexpr Dim_t len{2};
    constexpr auto sizes{CcoordOps::get_cube<FieldFixture<true>::SDim>(len)};

    global.fc.initialise(sizes,{});

    local.fc.add_pixel({1, 1});
    local.fc.add_pixel({0, 1});
    local.fc.initialise();

    // fill the local matrix field and then transfer it to the global field
    for (auto mat: local.matrix_field.get_map()) {
      mat.setRandom();
    }
    global.matrix_field.fill_from_local(local.matrix_field);

    for (const auto & ccoord: local.fc) {
      const auto & a{local.matrix_field.get_map()[ccoord]};
      const auto & b{global.matrix_field.get_map()[ccoord]};
      const Real error{(a -b).norm()};
      BOOST_CHECK_EQUAL(error, 0.);
    }

    // fill the global tensor field and then transfer it to the global field
    for (auto mat: global.tensor_field.get_map()) {
      mat.setRandom();
    }
    local.tensor_field.fill_from_global(global.tensor_field);
    for (const auto & ccoord: local.fc) {
      const auto & a{local.matrix_field.get_map()[ccoord]};
      const auto & b{global.matrix_field.get_map()[ccoord]};
      const Real error{(a -b).norm()};
      BOOST_CHECK_EQUAL(error, 0.);
    }

    BOOST_CHECK_THROW(local.tensor_field.fill_from_global(global.matrix_field),
                      std::runtime_error);
    BOOST_CHECK_THROW(global.tensor_field.fill_from_local(local.matrix_field),
                      std::runtime_error);
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre
