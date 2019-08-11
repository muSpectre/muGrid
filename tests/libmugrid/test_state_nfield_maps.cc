/**
 * @file   test_state_nfield_maps.cc
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
 * General Public License for more details.
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
 */

#include "tests.hh"
#include "libmugrid/state_nfield_map.hh"
#include "libmugrid/state_nfield.hh"
#include "libmugrid/nfield_typed.hh"
#include "libmugrid/nfield_collection_local.hh"
#include "libmugrid/state_nfield_map_static.hh"

namespace muGrid {

  BOOST_AUTO_TEST_SUITE(state_nfield_maps_tests);

  struct LocalNFieldBasicFixture {
    LocalNFieldCollection fc{Unknown, Unknown};
  };

  struct LocalNFieldBasicFixtureFilled : public LocalNFieldBasicFixture {
    static constexpr Dim_t NbMemory() { return 3; }
    static constexpr Dim_t NbComponents() { return twoD * twoD; }
    static constexpr Dim_t NbQuad() { return 2; }

    LocalNFieldBasicFixtureFilled()
        : array_field{fc.register_state_field<TypedStateNField<Int>>(
              "test", NbMemory(), NbComponents())},
          scalar_field{fc.register_state_field<TypedStateNField<Int>>(
              "test scalar", NbMemory(), 1)},
          t4_field{fc.register_state_field<TypedStateNField<Int>>(
              "test_t4", NbMemory(), NbComponents() * NbComponents())} {
      this->fc.add_pixel(4);
      this->fc.add_pixel(8);
      this->fc.set_nb_quad(NbQuad());
      this->fc.initialise();
    }
    TypedStateNField<Int> & array_field;
    TypedStateNField<Int> & scalar_field;
    TypedStateNField<Int> & t4_field;
  };

  struct StateNFieldMapFixture : public LocalNFieldBasicFixtureFilled {
    StateNFieldMapFixture()
        : array_field_map{this->array_field}, const_array_field_map{
                                                  this->array_field} {}
    StateNFieldMap<Int, false> array_field_map;
    StateNFieldMap<Int, true> const_array_field_map;
  };

  struct StaticStateNFieldMapFixture : public LocalNFieldBasicFixtureFilled {
    StaticStateNFieldMapFixture()
        : array_field_map{this->array_field},
          const_array_field_map{this->array_field},
          pixel_map{this->array_field}, t2_map{this->array_field},
          t4_map{this->t4_field}, scalar_map{this->scalar_field} {}
    ArrayStateNFieldMap<Int, false, NbComponents(), 1, NbMemory()>
        array_field_map;
    ArrayStateNFieldMap<Int, true, NbComponents(), 1, NbMemory()>
        const_array_field_map;
    ArrayStateNFieldMap<Int, false, NbComponents(), NbQuad(), NbMemory(),
                        Iteration::Pixel>
        pixel_map;
    T2StateNFieldMap<Int, false, twoD, NbMemory()> t2_map;
    T4StateNFieldMap<Int, false, twoD, NbMemory()> t4_map;
    ScalarStateNFieldMap<Int, false, NbMemory()> scalar_map;
  };

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE(iteration_test, StateNFieldMapFixture) {
    array_field_map.initialise();
    const_array_field_map.initialise();
    array_field.current().eigen_vec().setZero();
    for (Dim_t i{1}; i < this->NbMemory() + 1; ++i) {
      array_field.cycle();
      array_field.current().eigen_vec().setOnes();
      array_field.current().eigen_vec() *= i;
    }
    for (Dim_t i{1}; i < this->NbMemory() + 1; ++i) {
      for (auto && iterate : array_field_map) {
        auto && current{iterate.current()};
        auto && old{iterate.old(1)};
        BOOST_CHECK_EQUAL((current - old).array().mean() % this->NbMemory(), 1);
      }
      for (auto && iterate : const_array_field_map) {
        BOOST_CHECK_EQUAL((iterate.current() - iterate.old(1)).array().mean() %
                              this->NbMemory(),
                          1);  //! dummy test meant to fail for now
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE(static_size_test, StateNFieldMapFixture) {
    using Type1 = ArrayStateNFieldMap<Int, false, NbComponents(), 1, NbMemory(),
                                      Iteration::QuadPt>;
    BOOST_CHECK_NO_THROW(Type1{this->array_field};);
    using Type2 = ArrayStateNFieldMap<Int, true, NbComponents(), 1, NbMemory(),
                                      Iteration::QuadPt>;
    BOOST_CHECK_NO_THROW(Type2{this->array_field});
    using WrongSize_t =
        ArrayStateNFieldMap<Int, false, NbComponents(), NbComponents(),
                            NbMemory(), Iteration::QuadPt>;
    BOOST_CHECK_THROW(WrongSize_t{this->array_field}, NFieldMapError);

    using WrongNbMemory_t =
        ArrayStateNFieldMap<Int, false, 1, NbComponents(), NbMemory() + 1,
                            Iteration::QuadPt>;
    BOOST_CHECK_THROW(WrongNbMemory_t{this->array_field}, NFieldMapError);

    using Pixel_t = ArrayStateNFieldMap<Int, false, NbComponents(), NbQuad(),
                                        NbMemory(), Iteration::Pixel>;
    BOOST_CHECK_NO_THROW(Pixel_t{this->array_field});

    using Matrix_t = MatrixStateNFieldMap<Int, false, NbComponents(), 1,
                                          NbMemory(), Iteration::QuadPt>;
    BOOST_CHECK_NO_THROW(Matrix_t{this->array_field});

    using T2_t = T2StateNFieldMap<Int, false, twoD, NbMemory()>;
    BOOST_CHECK_NO_THROW(T2_t{this->array_field});
    using T4_t = T4StateNFieldMap<Int, false, twoD, NbMemory()>;
    BOOST_CHECK_NO_THROW(T4_t{this->t4_field});
    using Scalar_t = ScalarStateNFieldMap<Int, false, NbMemory()>;
    BOOST_CHECK_NO_THROW(Scalar_t{this->scalar_field});
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE(static_iteration_test, StaticStateNFieldMapFixture) {
    size_t counter{0};

    this->scalar_map.initialise();
    ScalarNFieldMap<Int, false> & current{
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

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muGrid
