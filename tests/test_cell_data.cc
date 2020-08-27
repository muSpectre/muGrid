/**
 * @file   test_cell_data.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   26 Jun 2020
 *
 * @brief  tests the cell data class
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

#include "test_cell_data.hh"

namespace muSpectre {


  BOOST_AUTO_TEST_SUITE(cell_data);

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(constructor_test, Fix, CellDataFixtures,
                                   Fix) {}

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muSpectre
