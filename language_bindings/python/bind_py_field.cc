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
#include "libmugrid/field.hh"
#include "libmugrid/field_map.hh"
#include "libmugrid/field_typed.hh"
#include "libmugrid/field_collection.hh"
#include "libmugrid/field_collection_global.hh"
#include "libmugrid/mapped_field.hh"
#include "libmugrid/numpy_tools.hh"
#include "libmugrid/raw_memory_operations.hh"

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

void add_field(py::module & mod) {
  py::class_<Field>(mod, "Field")
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

template <class T, muGrid::IterUnit iter_unit>
decltype(auto) array_getter(TypedFieldBase<T> & self) {
  return muGrid::numpy_wrap(self, iter_unit);
}

template <class T, muGrid::IterUnit iter_unit>
void array_setter(TypedFieldBase<T> & self, py::array_t<T> array) {
  const Shape_t array_shape(array.shape(), array.shape() + array.ndim());
  if (array_shape != self.get_shape(iter_unit)) {
    std::stringstream error{};
    error << "Dimension mismatch: The shape " << array_shape
          << " is not equal to the field shape " << self.get_shape(iter_unit)
          << ".";
    throw RuntimeError{error.str()};
  }
  Shape_t array_strides(array.strides(), array.strides() + array.ndim());
  // numpy arrays have stride in bytes
  for (auto && s : array_strides)  s /= sizeof(T);
  strided_copy(self.get_shape(iter_unit), array_strides,
               self.get_strides(iter_unit), array.data(), self.data());
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
        return py::buffer_info(self.data(), self.get_shape(subdivision),
                               self.get_strides(subdivision, sizeof(T)));
      })
      .def(
          "array",
          [](TypedFieldBase<T> & self, const muGrid::IterUnit & it) {
            return muGrid::numpy_wrap(self, it);
          },
          "iteration_type"_a = muGrid::IterUnit::SubPt, py::keep_alive<0, 1>())
      .def_property("s", array_getter<T, muGrid::IterUnit::SubPt>,
                    array_setter<T, muGrid::IterUnit::SubPt>,
                    py::keep_alive<0, 1>())
      .def_property("p", array_getter<T, muGrid::IterUnit::Pixel>,
                    array_setter<T, muGrid::IterUnit::Pixel>,
                    py::keep_alive<0, 1>())
      .def(
          "get_pixel_map",
          [](TypedFieldBase<T> & field, const Index_t & nb_rows) {
            return field.get_pixel_map(nb_rows);
          },
          "nb_rows"_a = muGrid::Unknown,
          py::return_value_policy::reference_internal)
      .def(
          "get_sub_pt_map",
          [](TypedFieldBase<T> & field, const Index_t & nb_rows) {
            return field.get_sub_pt_map(nb_rows);
          },
          "nb_rows"_a = muGrid::Unknown,
          py::return_value_policy::reference_internal);

  py::class_<TypedField<T>, TypedFieldBase<T>>(mod, name.c_str())
      .def("clone", &TypedField<T>::clone, "new_name"_a, "allow_overwrite"_a,
           py::return_value_policy::reference_internal);
}

template <typename T, muGrid::Mapping Mutability>
decltype(auto) add_field_map_const(py::module & mod, const std::string & name) {
  std::string full_name{name +
                        (Mutability == muGrid::Mapping::Mut ? "Mut" : "Const")};
  using Map_t = muGrid::FieldMap<T, Mutability>;

  py::class_<Map_t> pyclass(mod, full_name.c_str());
  pyclass.def("mean", [](const Map_t & map) { return map.mean(); });
  return pyclass;
}
template <typename T>
void add_field_map(py::module & mod, const std::string & name) {
  add_field_map_const<T, muGrid::Mapping::Const>(mod, name);
  add_field_map_const<T, muGrid::Mapping::Mut>(mod, name).def(
      "set_uniform",
      [](muGrid::FieldMap<T, muGrid::Mapping::Mut> & map,
         py::EigenDRef<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> val) {
        map = val;
      },
      "value"_a);
}

template <typename T>
void add_mutable_mapped_field(py::module & mod, const std::string & name) {
  using MappedField_t =
      muGrid::MappedField<muGrid::FieldMap<T, muGrid::Mapping::Mut>>;

  py::class_<MappedField_t>(mod, name.c_str())
      .def_property_readonly("field",
                             [](MappedField_t & mf) -> muGrid::TypedField<T> & {
                               return mf.get_field();
                             })
      .def_property_readonly(
          "map",
          [](MappedField_t & mf)
              -> muGrid::FieldMap<T, muGrid::Mapping::Mut> & {
            return mf.get_map();
          });
}

void add_field_classes(py::module & mod) {
  add_field(mod);

  add_typed_field<muGrid::Real>(mod, "RealField");
  add_typed_field<muGrid::Complex>(mod, "ComplexField");
  add_typed_field<muGrid::Int>(mod, "IntField");
  add_typed_field<muGrid::Uint>(mod, "UintField");

  add_field_map<muGrid::Real>(mod, "RealFieldMap");
  add_field_map<muGrid::Complex>(mod, "ComplexFieldMap");
  add_field_map<muGrid::Int>(mod, "IntFieldMap");
  add_field_map<muGrid::Uint>(mod, "UintFieldMap");

  add_mutable_mapped_field<muGrid::Real>(mod, "RealMappedField");
  add_mutable_mapped_field<muGrid::Complex>(mod, "ComplexMappedField");
  add_mutable_mapped_field<muGrid::Int>(mod, "IntMappedField");
  add_mutable_mapped_field<muGrid::Uint>(mod, "UintMappedField");
}
