#==============================================================================
# file   CMakeLists.txt
#
# @author Till Junge <till.junge@altermail.ch>
#
# @date   22 Mar 2019
#
# @brief  adds linters to tests
#
# Copyright © 2019 Till Junge
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
# along with GNU Emacs; see the file COPYING. If not, write to the
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


if(MUCHOICE MATCHES "muGrid")
  cpplint_add_subdirectory("${CMAKE_SOURCE_DIR}/src/libmugrid" "")
elseif(MUCHOICE MATCHES "muFFT")
  cpplint_add_subdirectory("${CMAKE_SOURCE_DIR}/src/libmugrid" "")
  cpplint_add_subdirectory("${CMAKE_SOURCE_DIR}/src/libmufft" "")
else()
  cpplint_add_subdirectory("${CMAKE_SOURCE_DIR}/src" "")
  cpplint_add_subdirectory("${CMAKE_SOURCE_DIR}/tests" "")
  cpplint_add_subdirectory("${CMAKE_SOURCE_DIR}/language_bindings" "")
  cpplint_add_subdirectory("${CMAKE_SOURCE_DIR}/examples" "--filter=-build/namespaces")
endif()
