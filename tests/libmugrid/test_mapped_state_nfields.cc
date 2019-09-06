/**
 * @file   test_mapped_state_nfields.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   09 Sep 2019
 *
 * @brief  Tests for the mapped state field classes
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
#include "libmugrid/nfield_collection_global.hh"
#include "libmugrid/mapped_state_nfield.hh"

namespace muGrid {

  BOOST_AUTO_TEST_SUITE(mapped_state_nfields);
  struct InitialiserBase {
    constexpr static Dim_t DimS{twoD};
    constexpr static Dim_t NbRow{2}, NbCol{3};
    constexpr static Dim_t NbQuad() { return 2; }
    constexpr static size_t NbMemory() { return 1; }
    GlobalNFieldCollection<DimS> fc{NbQuad()};
    InitialiserBase() { this->fc.initialise({2, 3}); }
  };

  struct MappedStateNFieldFixture : public InitialiserBase {
    MappedMatrixStateNField<Real, Mapping::Mut, NbRow, NbCol, NbMemory()>
        mapped_matrix;
    MappedArrayStateNField<Real, Mapping::Mut, NbRow, NbCol, NbMemory()>
        mapped_array;
    MappedScalarStateNField<Real, Mapping::Mut, NbMemory()> mapped_scalar;
    MappedT2StateNField<Real, Mapping::Mut, DimS, NbMemory()> mapped_t2;
    MappedT4StateNField<Real, Mapping::Mut, DimS, NbMemory()> mapped_t4;

    MappedStateNFieldFixture()
        : InitialiserBase{}, mapped_matrix{"matrix", this->fc},
          mapped_array{"array", this->fc}, mapped_scalar{"scalar", this->fc},
          mapped_t2{"t2", this->fc}, mapped_t4{"t4", this->fc} {};
  };

  BOOST_FIXTURE_TEST_CASE(access_and_iteration_test, MappedStateNFieldFixture) {
    for (auto && iterate : this->mapped_t2) {
      iterate.current().setRandom();
    }
    for (auto && iterate : this->mapped_matrix) {
      iterate.current().setRandom();
    }
    this->mapped_array.get_state_field().current().eigen_quad_pt() =
        this->mapped_matrix.get_state_field().current().eigen_quad_pt();
    for (auto && tup : akantu::zip(this->mapped_matrix, this->mapped_array)) {
      const auto & matrix{std::get<0>(tup).current()};
      const auto & array{std::get<1>(tup).current()};
      BOOST_CHECK_EQUAL((matrix - array.matrix()).norm(), 0);
    }

    BOOST_CHECK_EQUAL((this->mapped_array[4].current().matrix() -
                       this->mapped_matrix[4].current())
                          .norm(),
                      0);
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muGrid
