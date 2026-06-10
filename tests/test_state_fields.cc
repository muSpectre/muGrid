/**
 * @file   test_state_fields.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   21 Aug 2019
 *
 * @brief  Tests for state fields
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
#include "field/state_field.hh"
#include "field/field_typed.hh"
#include "collection/field_collection_local.hh"

namespace muGrid {

  BOOST_AUTO_TEST_SUITE(state_field_tests);

  struct LocalFieldBasicFixture {
    LocalFieldCollection fc{Unknown};
    static std::string SubDivision() { return "quad"; }
  };

  struct LocalFieldBasicFixtureFilled : public LocalFieldBasicFixture {
    static constexpr Dim_t NbMemory() { return 3; }
    static constexpr Dim_t NbComponents() { return 21; }

    LocalFieldBasicFixtureFilled()
        : state_field{fc.register_state_field<Real>(
              "test", NbMemory(), NbComponents(), SubDivision())} {
      this->fc.add_pixel(4);
      this->fc.add_pixel(8);
      this->fc.set_nb_sub_pts(SubDivision(), 2);
      this->fc.initialise();
    }
    TypedStateField<Real> & state_field;
  };

  BOOST_FIXTURE_TEST_CASE(construction_test, LocalFieldBasicFixture) {
    constexpr Dim_t NbMemory{1}, NbComponents{21};
    auto & state_field{fc.register_state_field<Real>(
        "test", NbMemory, NbComponents, SubDivision())};

    state_field.current();
    BOOST_CHECK(fc.state_field_exists("test"));

    BOOST_CHECK_THROW(fc.register_state_field<Real>(
                          "test", NbMemory, NbComponents, SubDivision()),
                      FieldCollectionError);
  }

  BOOST_FIXTURE_TEST_CASE(cycle_test, LocalFieldBasicFixtureFilled) {
    auto & current_field{state_field.current()};
    current_field.eigen_sub_pt().setZero();
    for (Dim_t i{0}; i < NbMemory(); ++i) {
      state_field.cycle();
      auto map{state_field.current().eigen_sub_pt()};
      map.setOnes();
      map *= (i + 1);
    }

    BOOST_CHECK_EQUAL(state_field.current().eigen_vec()(0), NbMemory());

    for (Dim_t i{0}; i < NbMemory(); ++i) {
      BOOST_CHECK_EQUAL(state_field.old(i + 1).eigen_vec()(0),
                        NbMemory() - 1 - i);
    }
  }

  // Regression test: the initial index map used a bitwise & instead of a
  // modulo, which aliased current() and old() to the same field for nb_memory
  // values not of the form 2^k - 1 (e.g. 2). With nb_memory = 2 the indices
  // must be a permutation of {0, 1, 2}; writing a distinct value into each
  // slot (via the mutable current(), cycling between writes) and reading them
  // back via old() must round-trip. If two indices alias, a later write
  // overwrites an earlier slot and the read-back fails.
  BOOST_FIXTURE_TEST_CASE(distinct_indices_nb_memory_2,
                          LocalFieldBasicFixture) {
    constexpr Dim_t NbMemory{2}, NbComponents{1};
    auto & state_field{fc.register_state_field<Real>(
        "test", NbMemory, NbComponents, SubDivision())};
    this->fc.add_pixel(4);
    this->fc.set_nb_sub_pts(SubDivision(), 1);
    this->fc.initialise();

    // Write 0 into the current slot, then 1 and 2 into the slots that become
    // current after each cycle.
    state_field.current().eigen_vec().setConstant(0.);
    for (Dim_t i{1}; i <= NbMemory; ++i) {
      state_field.cycle();
      state_field.current().eigen_vec().setConstant(static_cast<Real>(i));
    }

    // The three physical buffers now hold the distinct values 0, 1, 2; after
    // the cycles current() == 2, old(1) == 1, old(2) == 0.
    BOOST_CHECK_EQUAL(state_field.current().eigen_vec()(0), 2.);
    BOOST_CHECK_EQUAL(state_field.old(1).eigen_vec()(0), 1.);
    BOOST_CHECK_EQUAL(state_field.old(2).eigen_vec()(0), 0.);
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muGrid
