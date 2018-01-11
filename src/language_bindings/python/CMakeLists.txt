#==============================================================================
# file   CMakeLists.txt
#
# @author Till Junge <till.junge@epfl.ch>
#
# @date   08 Jan 2018
#
# @brief  configuration for python binding using pybind11
#
# @section LICENCE
#
# Copyright © 2018 Till Junge
#
# µSpectre is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3, or (at
# your option) any later version.
#
# µSpectre is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GNU Emacs; see the file COPYING. If not, write to the
# Free Software Foundation, Inc., 59 Temple Place - Suite 330,
# Boston, MA 02111-1307, USA.
#

pybind11_add_module(common bind_py_common.cc)
target_link_libraries(common PRIVATE muSpectre)

pybind11_add_module(system bind_py_system.cc)
target_link_libraries(system PRIVATE muSpectre)

pybind11_add_module(material bind_py_material.cc)
target_link_libraries(material PRIVATE muSpectre)

pybind11_add_module(solvers bind_py_solvers.cc)
target_link_libraries(solvers PRIVATE muSpectre)
