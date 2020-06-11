/**
 * @file   test_state_field_maps.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   23 Aug 2019
 *
 * @brief  tests for state field maps
 *
 * Copyright © 2019 Till Junge
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
 * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 *
 */

#include "tests.hh"
#include "libmugrid/state_field_map.hh"
#include "libmugrid/state_field.hh"
#include "libmugrid/field_typed.hh"
#include "libmugrid/field_collection_local.hh"
#include "libmugrid/state_field_map_static.hh"

#include <boost/mpl/list.hpp>

namespace muGrid {

  BOOST_AUTO_TEST_SUITE(state_field_maps_tests);

  struct LocalFieldBasicFixture {
    LocalFieldCollection fc{Unknown};
    const std::string SubDivision() { return "sub_pt"; }
  };

  template <Dim_t NbMemoryChoice>
  struct LocalFieldBasicFixtureFilled : public LocalFieldBasicFixture {
    static constexpr Index_t NbMemory() { return NbMemoryChoice; }
    static constexpr Index_t NbComponents() { return twoD * twoD; }
    static constexpr Index_t NbSubPt() { return 2; }

    LocalFieldBasicFixtureFilled()
        : array_field{fc.register_state_field<Int>(
              "test", NbMemory(), NbComponents(), SubDivision())},
          scalar_field{fc.register_state_field<Int>("test scalar", NbMemory(),
                                                    1, SubDivision())},
          t4_field{fc.register_state_field<Int>("test_t4", NbMemory(),
                                                NbComponents() * NbComponents(),
                                                SubDivision())} {
      this->fc.add_pixel(4);
      this->fc.add_pixel(8);
      this->fc.set_nb_sub_pts(SubDivision(), NbSubPt());
      this->fc.initialise();
    }
    TypedStateField<Int> & array_field;
    TypedStateField<Int> & scalar_field;
    TypedStateField<Int> & t4_field;
  };

  template <Dim_t NbMemoryChoice>
  struct StateFieldMapFixture
      : public LocalFieldBasicFixtureFilled<NbMemoryChoice> {
    StateFieldMapFixture()
        : array_field_map{this->array_field}, const_array_field_map{
                                                  this->array_field} {}
    StateFieldMap<Int, Mapping::Mut> array_field_map;
    StateFieldMap<Int, Mapping::Const> const_array_field_map;
  };

  template <Dim_t NbMemoryChoice>
  struct StaticStateFieldMapFixture
      : public LocalFieldBasicFixtureFilled<NbMemoryChoice> {
    StaticStateFieldMapFixture()
        : static_array_field_map{this->array_field},
          static_const_array_field_map{this->array_field},
          pixel_map{this->array_field}, t2_map{this->array_field},
          t4_map{this->t4_field}, scalar_map{this->scalar_field} {}
    using Parent = LocalFieldBasicFixtureFilled<NbMemoryChoice>;
    ArrayStateFieldMap<Int, Mapping::Mut, Parent::NbComponents(), 1,
                       Parent::NbMemory(), IterUnit::SubPt>
        static_array_field_map;
    ArrayStateFieldMap<Int, Mapping::Const, Parent::NbComponents(), 1,
                       Parent::NbMemory(), IterUnit::SubPt>
        static_const_array_field_map;
    ArrayStateFieldMap<Int, Mapping::Mut, Parent::NbComponents(),
                       Parent::NbSubPt(), Parent::NbMemory(),
                       IterUnit::Pixel>
        pixel_map;
    T2StateFieldMap<Int, Mapping::Mut, twoD, Parent::NbMemory(),
                    IterUnit::SubPt>
        t2_map;
    T4StateFieldMap<Int, Mapping::Mut, twoD, Parent::NbMemory(),
                    IterUnit::SubPt>
        t4_map;
    ScalarStateFieldMap<Int, Mapping::Mut, Parent::NbMemory(),
                        IterUnit::SubPt>
        scalar_map;
  };

  using StateFieldMapFixtures =
      boost::mpl::list<StateFieldMapFixture<1>, StateFieldMapFixture<3>>;
  using StaticStateFieldMapFixtures =
      boost::mpl::list<StaticStateFieldMapFixture<1>,
                       StaticStateFieldMapFixture<3>>;
  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(iteration_test, Fix, StateFieldMapFixtures,
                                   Fix) {
    Fix::array_field.current().eigen_vec().setZero();
    for (Dim_t i{1}; i < this->NbMemory() + 1; ++i) {
      Fix::array_field.cycle();
      Fix::array_field.current().eigen_vec().setOnes();
      Fix::array_field.current().eigen_vec() *= i;
    }
    for (auto && iterate : Fix::array_field_map) {
      auto && current{iterate.current()};
      for (Dim_t i{1}; i < this->NbMemory() + 1; ++i) {
        auto && old{iterate.old(i)};
        BOOST_CHECK_EQUAL((current - old).array().mean(), i);
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(direct_access_test, Fix,
                                   StateFieldMapFixtures, Fix) {
    Fix::array_field.current().eigen_vec().setZero();
    for (Dim_t i{1}; i < this->NbMemory() + 1; ++i) {
      Fix::array_field.cycle();
      Fix::array_field.current().eigen_vec().setOnes();
      Fix::array_field.current().eigen_vec() *= i;
    }
    auto && iterate{Fix::array_field_map[0]};
    auto && current{iterate.current()};
    for (Dim_t i{1}; i < this->NbMemory() + 1; ++i) {
      auto && old{iterate.old(i)};
      BOOST_CHECK_EQUAL((current - old).array().mean(), i);
    }
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(static_size_test, Fix, StateFieldMapFixtures,
                                   Fix) {
    using Type1 = ArrayStateFieldMap<Int, Mapping::Mut, Fix::NbComponents(), 1,
                                     Fix::NbMemory(), IterUnit::SubPt>;
    BOOST_CHECK_NO_THROW(Type1{this->array_field};);
    using Type2 = ArrayStateFieldMap<Int, Mapping::Const, Fix::NbComponents(),
                                     1, Fix::NbMemory(), IterUnit::SubPt>;
    BOOST_CHECK_NO_THROW(Type2{this->array_field});
    using WrongSize_t =
        ArrayStateFieldMap<Int, Mapping::Mut, Fix::NbComponents(),
                           Fix::NbComponents(), Fix::NbMemory(),
                           IterUnit::SubPt>;
    BOOST_CHECK_THROW(WrongSize_t{this->array_field}, FieldMapError);

    using WrongNbMemory_t =
        ArrayStateFieldMap<Int, Mapping::Mut, 1, Fix::NbComponents(),
                           Fix::NbMemory() + 1, IterUnit::SubPt>;
    BOOST_CHECK_THROW(WrongNbMemory_t{this->array_field}, FieldMapError);

    using Pixel_t =
        ArrayStateFieldMap<Int, Mapping::Mut, Fix::NbComponents(),
                           Fix::NbSubPt(), Fix::NbMemory(), IterUnit::Pixel>;
    BOOST_CHECK_NO_THROW(Pixel_t{this->array_field});

    using Matrix_t =
        MatrixStateFieldMap<Int, Mapping::Mut, Fix::NbComponents(), 1,
                            Fix::NbMemory(), IterUnit::SubPt>;
    BOOST_CHECK_NO_THROW(Matrix_t{this->array_field});

    using T2_t = T2StateFieldMap<Int, Mapping::Mut, twoD, Fix::NbMemory(),
                                 IterUnit::SubPt>;
    BOOST_CHECK_NO_THROW(T2_t{this->array_field});
    using T4_t = T4StateFieldMap<Int, Mapping::Mut, twoD, Fix::NbMemory(),
                                 IterUnit::SubPt>;
    BOOST_CHECK_NO_THROW(T4_t{this->t4_field});
    using Scalar_t = ScalarStateFieldMap<Int, Mapping::Mut, Fix::NbMemory(),
                                         IterUnit::SubPt>;
    BOOST_CHECK_NO_THROW(Scalar_t{this->scalar_field});
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(static_iteration_test, Fix,
                                   StaticStateFieldMapFixtures, Fix) {
    size_t counter{0};
    ScalarFieldMap<Int, Mapping::Mut, IterUnit::SubPt> & current{
        this->scalar_map.get_current_static()};
    BOOST_CHECK_NO_THROW(current[0]++);
    static_assert(
        std::is_same<std::remove_reference_t<decltype(current[0])>, int>::value,
        "make sure this resolved to the static state field map's "
        "method, and not to the dynamic base class");
    for (auto && entry : this->scalar_map) {
      auto & entry_val(entry.current());
      entry_val = counter++;
    }
    this->scalar_field.cycle();
    for (int i{0}; i < this->scalar_field.current().eigen_vec().size(); ++i) {
      BOOST_CHECK_EQUAL(i, this->scalar_field.old(1).eigen_vec()(i));
    }
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(static_direct_access_test, Fix,
                                   StaticStateFieldMapFixtures, Fix) {
    Fix::array_field.current().eigen_vec().setZero();
    for (Dim_t i{1}; i < this->NbMemory() + 1; ++i) {
      Fix::array_field.cycle();
      Fix::array_field.current().eigen_vec().setOnes();
      Fix::array_field.current().eigen_vec() *= i;
    }
    auto && iterate{Fix::static_array_field_map[0]};
    auto && current{iterate.current()};
    for (Dim_t i{1}; i < this->NbMemory() + 1; ++i) {
      auto && old{iterate.old(i)};
      BOOST_CHECK_EQUAL((current - old).array().mean(), i);
    }
  }
  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(static_iteration_testb, Fix,
                                   StaticStateFieldMapFixtures, Fix) {
    Fix::array_field.current().eigen_vec().setZero();
    for (Dim_t i{1}; i < this->NbMemory() + 1; ++i) {
      Fix::array_field.cycle();
      Fix::array_field.current().eigen_vec().setOnes();
      Fix::array_field.current().eigen_vec() *= i;
    }
    for (auto && iterate : Fix::static_array_field_map) {
      auto && current{iterate.current()};
      for (Dim_t i{1}; i < this->NbMemory() + 1; ++i) {
        auto && old{iterate.old(i)};
        BOOST_CHECK_EQUAL((current - old).array().mean(), i);
      }
    }
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muGrid
