/**
 * @file   tests.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   10 May 2017
 *
 * @brief  common defs for tests
 *
 * Copyright © 2017 Till Junge
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
 * * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 *
 */
#include <libmugrid/grid_common.hh>

#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>

#include "iostream"

#ifndef TESTS_LIBMUGRID_TESTS_HH_
#define TESTS_LIBMUGRID_TESTS_HH_

namespace muGrid {

  constexpr Real tol = 1e-14 * 100;       // it's in percent
  constexpr Real finite_diff_tol = 1e-7;  // it's in percent

}  // namespace muGrid

#endif  // TESTS_LIBMUGRID_TESTS_HH_
