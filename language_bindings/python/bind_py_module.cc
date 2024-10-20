/**
 * @file   bind_py_module.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   10 Oct 2019
 *
 * @brief  Python bindings for µGrid
 *
 * Copyright © 2018 Till Junge
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

#include "bind_py_declarations.hh"

#include <pybind11/pybind11.h>

PYBIND11_MODULE(_muGrid, mod) {
  mod.doc() = "Python bindings to the µGrid library";

  add_common_mugrid(mod);
  add_communicator(mod);
  add_field_classes(mod);
  add_state_field_classes(mod);
  add_field_collection_classes(mod);
  add_options_dictionary(mod);
  add_testing(mod);
#ifdef WITH_NETCDF_IO
  add_file_io_classes(mod);
#endif
}
