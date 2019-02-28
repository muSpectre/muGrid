# =============================================================================
# file   CMakeLists.txt
#
# @author Till Junge <till.junge@epfl.ch>
#
# @date   08 Jan 2018
#
# @brief  Main configuration file
#
# @section LICENSE
#
# Copyright © 2018 Till Junge
#
# µSpectre is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3, or (at
# your option) any later version.
#
# µSpectre is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with µSpectre; see the file COPYING. If not, write to the
# Free Software Foundation, Inc., 59 Temple Place - Suite 330,
# Boston, MA 02111-1307, USA.
#
# Additional permission under GNU GPL version 3 section 7
#
# If you modify this Program, or any covered work, by linking or combining it
# with proprietary FFT implementations or numerical libraries, containing parts
# covered by the terms of those libraries' licenses, the licensors of this
# Program grant you additional permission to convey the resulting work.
# =============================================================================
set(mugrid_tests)
macro(muGrid_add_test)
  muTools_add_test(${ARGN} LINK_LIBRARIES ${MUGRID_NAMESPACE}muGrid TEST_LIST mugrid_tests)
endmacro()

file(GLOB HEADER_TEST_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/header*.cc")
foreach(header_test ${HEADER_TEST_SRCS})
  get_test_name(test_name ${header_test} ".*_test")

  muGrid_add_test(${test_name}
    SOURCES main_test_suite.cc ${header_test}
    TYPE BOOST --report_level=detailed)
endforeach(header_test ${HEADER_TEST_SRCS})

# build library tests
file(GLOB TEST_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/test_*.cc")
muGrid_add_test(main_mugrid_test_suite
  SOURCES main_test_suite.cc ${TEST_SRCS}
  TYPE BOOST --report_level=detailed)

add_custom_target(test_mugrid DEPENDS ${mugrid_tests})
