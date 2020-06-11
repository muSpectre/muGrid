/**
 * @file   test_mapped_state_fields.cc
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
#include "libmugrid/field_collection_global.hh"
#include "libmugrid/mapped_state_field.hh"

namespace muGrid {

  BOOST_AUTO_TEST_SUITE(mapped_state_fields);
  struct InitialiserBase {
    constexpr static Index_t DimS{twoD};
    constexpr static Index_t NbRow{2}, NbCol{3};
    constexpr static Index_t NbSubPt{2};
    constexpr static size_t NbMemory{1};
    static const std::string sub_division_tag() { return "sub_pt"; }
    GlobalFieldCollection fc{DimS};
    InitialiserBase() {
      this->fc.initialise(Ccoord_t<twoD>{2, 3});
      this->fc.set_nb_sub_pts(sub_division_tag(), NbSubPt);
      }
  };
  constexpr Index_t InitialiserBase::DimS;
  constexpr Index_t InitialiserBase::NbRow;
  constexpr Index_t InitialiserBase::NbCol;
  constexpr Index_t InitialiserBase::NbSubPt;
  constexpr size_t InitialiserBase::NbMemory;

  struct MappedStateFieldFixture : public InitialiserBase {
    MappedMatrixStateField<Real, Mapping::Mut, NbRow, NbCol,
                           IterUnit::SubPt, NbMemory>
        mapped_matrix;
    MappedArrayStateField<Real, Mapping::Mut, NbRow, NbCol, IterUnit::SubPt,
                          NbMemory>
        mapped_array;
    MappedScalarStateField<Real, Mapping::Mut, IterUnit::SubPt, NbMemory>
        mapped_scalar;
    MappedT2StateField<Real, Mapping::Mut, DimS, IterUnit::SubPt, NbMemory>
        mapped_t2;
    MappedT4StateField<Real, Mapping::Mut, DimS, IterUnit::SubPt, NbMemory>
        mapped_t4;

    MappedStateFieldFixture()
        : InitialiserBase{}, mapped_matrix{"matrix", this->fc,
                                           sub_division_tag()},
          mapped_array{"array", this->fc, sub_division_tag()},
          mapped_scalar{"scalar", this->fc, sub_division_tag()},
          mapped_t2{"t2", this->fc, sub_division_tag()},
          mapped_t4{"t4", this->fc, sub_division_tag()} {};
  };

  BOOST_FIXTURE_TEST_CASE(access_and_iteration_test, MappedStateFieldFixture) {
    for (auto && iterate : this->mapped_t2) {
      iterate.current().setRandom();
    }
    for (auto && iterate : this->mapped_matrix) {
      iterate.current().setRandom();
    }
    this->mapped_array.get_state_field().current().eigen_sub_pt() =
        this->mapped_matrix.get_state_field().current().eigen_sub_pt();
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
