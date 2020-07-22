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

#include "libmugrid/exception.hh"
#include "libmugrid/field.hh"
#include "libmugrid/field_typed.hh"
#include "libmugrid/field_collection.hh"
#include "libmugrid/field_collection_global.hh"
#include "libmugrid/numpy_tools.hh"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <sstream>
#include <stdexcept>

using muGrid::Field;
using muGrid::FieldCollection;
using muGrid::GlobalFieldCollection;
using muGrid::Index_t;
using muGrid::RuntimeError;
using muGrid::TypedField;
using muGrid::TypedFieldBase;
using muGrid::WrappedField;
using pybind11::literals::operator""_a;

namespace py = pybind11;

void add_field(py::module & mod) {
  py::class_<Field>(mod, "Field")
      .def("set_zero", &Field::set_zero)
      .def_property_readonly("buffer_size", &Field::get_buffer_size)
      .def_property_readonly("shape",
                             [](Field & field) {
                               return field.get_shape(
                                   muGrid::IterUnit::SubPt);
                             })
      .def_property_readonly("stride", &Field::get_stride)
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

template <class T>
void add_typed_field(py::module & mod, std::string name) {
  py::class_<TypedFieldBase<T>, Field>(mod, (name + "Base").c_str(),
                                       py::buffer_protocol())
      .def_buffer([](TypedFieldBase<T> & self) {
        auto & coll = self.get_collection();
        if (not coll.is_initialised()) {
          throw RuntimeError("Field collection isn't initialised yet");
        }
        auto subdivision{muGrid::IterUnit::SubPt};
        return py::buffer_info(
            self.data(), self.get_shape(subdivision),
            self.get_strides(subdivision, sizeof(T)));
      })
      .def(
          "array",
          [](TypedFieldBase<T> & self, const muGrid::IterUnit & it) {
            return muGrid::numpy_wrap(self, it);
          },
          "iteration_type"_a = muGrid::IterUnit::SubPt, py::keep_alive<0, 1>());

  py::class_<TypedField<T>, TypedFieldBase<T>>(mod, name.c_str())
      .def("clone", &TypedField<T>::clone, "new_name"_a,
           "allow_overwrite"_a, py::return_value_policy::reference_internal);
}

void add_field_classes(py::module & mod) {
  add_field(mod);

  add_typed_field<muGrid::Real>(mod, "RealField");
  add_typed_field<muGrid::Complex>(mod, "ComplexField");
  add_typed_field<muGrid::Int>(mod, "IntField");
  add_typed_field<muGrid::Uint>(mod, "UintField");
}
