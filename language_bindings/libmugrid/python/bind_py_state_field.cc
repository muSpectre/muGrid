/**
 * @file   bind_py_state_field.cc
 *
 * @author Richard Leute <richard.leute@imtek.uni-freiburg.de>
 *
 * @date   23 Sep 2020
 *
 * @brief  python bindings for state fields
 *
 * Copyright © 2020 Till Junge
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

#include <sstream>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "libmugrid/state_field.hh"
#include "libmugrid/field.hh"
#include "libmugrid/field_typed.hh"
#include "libmugrid/field_collection.hh"

using muGrid::StateField;
using muGrid::Field;
using muGrid::TypedStateField;
using muGrid::TypedField;
using muGrid::Unit;
using muGrid::Index_t;
using muGrid::FieldCollection;
using muGrid::Real;
using muGrid::Complex;
using muGrid::Int;
using muGrid::Uint;
using pybind11::literals::operator""_a;

namespace py = pybind11;

void add_state_field(py::module & mod) {
  py::class_<StateField> state_field(mod, "StateField");
  state_field
      .def("cycle", &StateField::cycle)
      .def("current", (Field & (StateField::*)()) & StateField::current,
           py::return_value_policy::reference_internal)
      .def("current",
           (const Field & (StateField::*)() const) & StateField::current,
           py::return_value_policy::reference_internal)
      .def("old", &StateField::old, "nb_steps_ago"_a = 1,
           py::return_value_policy::reference_internal)
      .def("get_nb_memory", &StateField::get_nb_memory)
      .def("get_indices", &StateField::get_indices);
}

template <class T>
void add_typed_state_field(py::module & mod, std::string name) {
  py::class_<TypedStateField<T>, StateField> typed_state_field(
      mod, (name + "StateField").c_str(), py::buffer_protocol());
  typed_state_field
      .def("current",
           (TypedField<T> & (TypedStateField<T>::*)()) &
               TypedStateField<T>::current,
           py::return_value_policy::reference_internal)
      .def("current",
           (const TypedField<T> & (TypedStateField<T>::*)() const) &
               TypedStateField<T>::current,
           py::return_value_policy::reference_internal)
      .def("old", &TypedStateField<T>::old, "nb_steps_ago"_a = 1,
           py::return_value_policy::reference_internal);
}

void add_state_field_classes(py::module & mod) {
  add_state_field(mod);

  add_typed_state_field<Real>(mod, "Real");
  add_typed_state_field<Complex>(mod, "Complex");
  add_typed_state_field<Int>(mod, "Int");
  add_typed_state_field<Uint>(mod, "Uint");
}
