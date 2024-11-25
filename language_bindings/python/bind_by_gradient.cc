/**
* @file   bind_py_field.cc
*
* @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
*
* @date   16 Oct 2019
*
* @brief  Python bindings for fields
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

#include "libmugrid/grid_common.hh"
#include "libmugrid/exception.hh"
#include "libmugrid/gradient_operator_default.hh"
#include "libmugrid/numpy_tools.hh"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <sstream>

using muGrid::Field;
using muGrid::FieldCollection;
using muGrid::GlobalFieldCollection;
using muGrid::Index_t;
using muGrid::RuntimeError;
using muGrid::Shape_t;
using muGrid::TypedField;
using muGrid::TypedFieldBase;
using muGrid::WrappedField;
using muGrid::operator<<;
using muGrid::raw_mem_ops::strided_copy;
using pybind11::literals::operator""_a;

namespace py = pybind11;

void add_gradient_operator_default(py::module & mod) {
 py::class_<GradientOperatorDefault>(mod, "GradientOperatorDefault")
     .def("set_zero", &Field::set_zero)
     .def("stride", &Field::get_stride)
     .def_property_readonly("buffer_size", &Field::get_buffer_size)
     .def_property_readonly("shape",
                            [](Field & field) {
                              return field.get_shape(muGrid::IterUnit::SubPt);
                            })
     .def_property_readonly("pad_size", &Field::get_pad_size)
     .def_property_readonly("name", &Field::get_name)
     .def_property_readonly("collection", &Field::get_collection)
     .def_property_readonly("nb_components", &Field::get_nb_components)
     .def_property_readonly("components_shape", &Field::get_components_shape)
     .def_property_readonly("nb_entries", &Field::get_nb_entries)
     .def_property_readonly("nb_buffer_entries", &Field::get_nb_buffer_entries)
     .def_property_readonly("is_global", &Field::is_global)
     .def_property_readonly("sub_division", &Field::get_sub_division_tag);
}

void add_gradient_classes(py::module & mod) {
  add_gradient_operator_default(mod);
}
