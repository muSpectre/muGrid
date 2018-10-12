/**
 * @file   bind_py_common.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   12 Jan 2018
 *
 * @brief  header for python bindings for the common part of µSpectre
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

#ifndef BIND_PY_DECLARATIONS_H
#define BIND_PY_DECLARATIONS_H

#include <pybind11/pybind11.h>
namespace py = pybind11;

void add_common(py::module & mod);
void add_cell(py::module & mod);
void add_material(py::module & mod);
void add_solvers(py::module & mod);
void add_fft_engines(py::module & mod);
void add_projections(py::module & submodule);
void add_field_collections(py::module & submodule);

#endif /* BIND_PY_DECLARATIONS_H */
