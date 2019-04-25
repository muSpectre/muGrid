/**
 * @file   header_test_mapped_fields.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   27 Mar 2019
 *
 * @brief  Tests for mapped field combos
 *
 * Copyright © 2019 Till Junge
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
 * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 */


#include "libmugrid/mapped_field.hh"
#include "libmugrid/field_map_matrixlike.hh"
#include "libmugrid/field.hh"
#include "libmugrid/field_collection_local.hh"
#include "test_goodies.hh"
#include "tests.hh"
#include "libmugrid/iterators.hh"

#include <boost/mpl/list.hpp>
#include <boost/mpl/insert_range.hpp>


namespace muGrid {

  using testGoodies::rel_error;

  template<class MappedField_t>
  struct MappedFieldFixture {
    using MField_t = MappedField_t;
    using Collection_t = typename MappedField_t::Collection_t;
    using Ccoord = Ccoord_t<Collection_t::spatial_dim()>;

    MappedFieldFixture() : mfield{coll, "name"} {
      this->coll.add_pixel(Ccoord{0});
      this->coll.add_pixel(Ccoord{1});
      this->coll.initialise();
    }
    Collection_t coll{};
    MappedField_t mfield;
  };

  using simple_fixtures = boost::mpl::list<
      MappedFieldFixture<MappedField<
          LocalFieldCollection<twoD>,
          MatrixFieldMap<LocalFieldCollection<twoD>, Real, twoD, twoD>,
          TensorField<LocalFieldCollection<twoD>, Real, secondOrder, twoD>>>,
      MappedFieldFixture<MappedMatrixField<Int, threeD, 2, 1>>,
      MappedFieldFixture<MappedArrayField<Int, threeD, 2, 1>>,
      MappedFieldFixture<MappedT2Field<Complex, twoD>>,
      MappedFieldFixture<MappedT4Field<Complex, threeD>>>;

  using state_fixtures = boost::mpl::list<
      MappedFieldFixture<MappedField<
          LocalFieldCollection<threeD>,
          MatrixFieldMap<LocalFieldCollection<threeD>, Real, 1, 1>,
          StateField<MatrixField<LocalFieldCollection<threeD>, Real, 1, 1>>>>,
      MappedFieldFixture<MappedT2StateField<Int, twoD>>,
      MappedFieldFixture<MappedT2StateField<Int, twoD, 3>>,
      MappedFieldFixture<MappedMatrixStateField<Int, threeD, 2, 1>>,
      MappedFieldFixture<MappedArrayStateField<Int, threeD, 2, 1>>,
      MappedFieldFixture<MappedT4StateField<Complex, threeD>>>;

  using non_standard_fixtures = boost::mpl::list<
      MappedFieldFixture<MappedScalarField<Real, twoD>>,
      MappedFieldFixture<MappedScalarStateField<Real, twoD>>>;

  using all_arithmetic_fixtures =
      boost::mpl::insert_range<simple_fixtures,
                               boost::mpl::end<simple_fixtures>::type,
                               state_fixtures>::type;
  using all_fixtures =
      boost::mpl::insert_range<all_arithmetic_fixtures,
                               boost::mpl::end<all_arithmetic_fixtures>::type,
                               non_standard_fixtures>::type;

  BOOST_AUTO_TEST_SUITE(mapped_field_tests);

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(constructor_test, Fix, all_fixtures, Fix) {}

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(trivial_iteration_test, Fix, all_fixtures,
                                   Fix) {
    auto fun = [](auto && /*bla*/) {/*don't do anything*/};
    for (auto && tup : akantu::zip(Fix::mfield, Fix::mfield.get_map())) {
      fun(tup);
    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(trivial_access_test, Fix, all_fixtures,
                                   Fix) {
    auto fun = [](auto && /*bla*/) { /*don't do anything*/ };
    auto && value{Fix::mfield[0]};
    fun(value);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(get_map_test, Fix, all_fixtures,
                                   Fix) {
    auto fun = [](auto && /*bla*/) { /*don't do anything*/ };

    typename Fix::MField_t::FieldMap_t & field(Fix::mfield.get_map());
    fun(field);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(get_field_test, Fix, all_fixtures,
                                   Fix) {
    auto fun = [](auto && /*bla*/) { /*don't do anything*/ };

    typename Fix::MField_t::Field_t & field(Fix::mfield.get_field());
    fun(field);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(iteration_test, Fix, simple_fixtures, Fix) {
    for (auto && tup : akantu::zip(Fix::mfield, Fix::mfield.get_map())) {
      auto && direct = std::get<0>(tup);
      auto && indirect = std::get<1>(tup);
      indirect.setRandom();
      auto error{rel_error(indirect, direct)};
      BOOST_CHECK_EQUAL(error, 0);
    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(state_iteration_test, Fix, state_fixtures,
                                   Fix) {
    for (auto && tup : akantu::zip(Fix::mfield, Fix::mfield.get_map())) {
      auto && direct = std::get<0>(tup);
      auto && indirect = std::get<1>(tup);
      indirect.current().setRandom();
      Real error{rel_error(indirect.current(), direct.current())};
      BOOST_CHECK_EQUAL(error, 0);
    }
  }
  BOOST_AUTO_TEST_SUITE_END()
}  // namespace muGrid
