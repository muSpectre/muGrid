/**
 * @file   test_cell_data.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   24 Jul 2020
 *
 * @brief  Fixtures for CellData
 *
 * Copyright © 2020 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
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
 *
 */

#include "tests.hh"
#include "libmugrid/test_goodies.hh"

#include <cell/cell_data.hh>

#include <boost/mpl/list.hpp>

#ifndef TESTS_TEST_CELL_DATA_HH_
#define TESTS_TEST_CELL_DATA_HH_

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Index_t Dim>
  struct CellDataFixture {
    constexpr static Index_t SpatialDim{Dim};
    static DynCcoord_t get_size() {
      switch (SpatialDim) {
      case twoD: {
        return {3, 5};
        break;
      }
      case threeD: {
        return {3, 5, 7};
        break;
      }
      default:
        std::stringstream err_msg{};
        err_msg << "can't give you a size for Dim = " << SpatialDim << ". "
                << "I can only handle two- and three-dimensional problems.";
        throw muGrid::RuntimeError{err_msg.str()};
        break;
      }
    }

    static DynRcoord_t get_length() {
      switch (SpatialDim) {
      case twoD: {
        return {1, 2};
        break;
      }
      case threeD: {
        return {1, 2, 3};
        break;
      }
      default:
        std::stringstream err_msg{};
        err_msg << "can't give you a size for Dim = " << SpatialDim << ". "
                << "I can only handle two- and three-dimensional problems.";
        throw muGrid::RuntimeError{err_msg.str()};
        break;
      }
    }
    CellDataFixture() : cell_data(CellData::make(get_size(), get_length())) {}

    CellData_ptr cell_data;
  };

  template <Index_t Dim>
  constexpr Index_t CellDataFixture<Dim>::SpatialDim;

  /* ---------------------------------------------------------------------- */
  using CellDataFixtures =
      boost::mpl::list<CellDataFixture<twoD>, CellDataFixture<threeD>>;

}  // namespace muSpectre

#endif  // TESTS_TEST_CELL_DATA_HH_
