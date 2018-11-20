#==============================================================================
# file   CMakeLists.txt
#
# @author Till Junge <till.junge@epfl.ch>
#
# @date   01 Feb 2018
#
# @brief  builds the µSpectre documentation
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

find_package(Sphinx REQUIRED)

# configured documentation tools and intermediate build results
set(BINARY_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/_build")

# Sphinx cache with pickled ReST documents
set(SPHINX_CACHE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_doctrees")

# HTML output directory
set(SPHINX_HTML_DIR "${CMAKE_CURRENT_BINARY_DIR}/html")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/dev-docs/source/conf.py"
  "${CMAKE_CURRENT_BINARY_DIR}/conf.py"
  @ONLY)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/dev-docs/source/Doxyfile"
  "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
  @ONLY)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/dev-docs/source/input_def.in"
  "${CMAKE_CURRENT_BINARY_DIR}/input_def"
  @ONLY)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/dev-docs/source/xml_output_def.in"
  "${CMAKE_CURRENT_BINARY_DIR}/xml_output_def"
  @ONLY)

add_custom_target(dev_doc
  ${SPHINX_EXECUTABLE}
  -j 4
  -q -b html
  -c "${CMAKE_CURRENT_BINARY_DIR}"
  -d "${SPHINX_CACHE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/dev-docs/source"
  "${SPHINX_HTML_DIR}"
  COMMENT "Building HTML documentation with Sphinx")
