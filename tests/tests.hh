/**
 * file   tests.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   10 May 2017
 *
 * @brief  common defs for tests
 *
 * @section LICENCE
 *
 * Copyright (C) 2017 Till Junge
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

#include "common/common.hh"
#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>

#ifndef TESTS_H
#define TESTS_H

namespace muSpectre {


  const Real tol = 1e-14*100; //it's in percent

}  // muSpectre


#endif /* TESTS_H */
