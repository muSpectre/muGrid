/**
 * @file   bind_py_module.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   12 Jan 2018
 *
 * @brief  Python bindings for µSpectre
 *
 * Copyright © 2018 Till Junge
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
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#include "bind_py_declarations.hh"

#include <pybind11/pybind11.h>

using namespace pybind11::literals;
namespace py=pybind11;

PYBIND11_MODULE(_muSpectre, mod) {
  mod.doc() = "Python bindings to the µSpectre library";

  add_common(mod);
  add_cell(mod);
  add_material(mod);
  add_solvers(mod);
  add_fft_engines(mod);
  add_field_collections(mod);
}
