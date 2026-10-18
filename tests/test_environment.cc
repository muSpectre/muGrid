/**
 * @file   test_environment.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   18 Oct 2026
 *
 * @brief  Tests for the boolean environment switch parser
 *
 * Copyright © 2026 Lars Pastewka
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
#include "core/environment.hh"
#include "core/exception.hh"

#include <cstdlib>

namespace muGrid {

  BOOST_AUTO_TEST_SUITE(environment_tests);

  //! The switches used to look at the first character only ("on" read as
  //! false) or accept only '1'; one parser now serves all of them.
  BOOST_AUTO_TEST_CASE(boolean_spellings) {
    const char * name{"MUGRID_TEST_ENVIRONMENT_FLAG"};
    unsetenv(name);
    BOOST_CHECK_EQUAL(environment_flag(name), -1);
    setenv(name, "", 1);
    BOOST_CHECK_EQUAL(environment_flag(name), -1);
    for (const char * on : {"1", "y", "Yes", "t", "TRUE", "on", "On"}) {
      setenv(name, on, 1);
      BOOST_CHECK_EQUAL(environment_flag(name), 1);
    }
    for (const char * off : {"0", "n", "NO", "f", "false", "off", "OFF"}) {
      setenv(name, off, 1);
      BOOST_CHECK_EQUAL(environment_flag(name), 0);
    }
    // A value that is neither is refused rather than read as one of them
    for (const char * bad : {"2", "maybe", "onn", "yes please"}) {
      setenv(name, bad, 1);
      BOOST_CHECK_THROW(environment_flag(name), RuntimeError);
    }
    unsetenv(name);
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muGrid
