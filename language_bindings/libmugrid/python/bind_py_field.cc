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

#include "libmugrid/field.hh"
#include "libmugrid/field_typed.hh"
#include "libmugrid/field_collection.hh"
#include "libmugrid/field_collection_global.hh"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <sstream>
#include <stdexcept>

using muGrid::Dim_t;
using muGrid::GlobalFieldCollection;
using muGrid::Field;
using muGrid::FieldCollection;
using muGrid::TypedField;
using muGrid::TypedFieldBase;
using muGrid::WrappedField;
using pybind11::literals::operator""_a;

namespace py = pybind11;

void add_field(py::module & mod) {
  py::class_<Field>(mod, "Field")
      .def("buffer_size", &Field::buffer_size)
      .def("set_zero", &Field::set_zero)
      .def("stride", &Field::get_stride)
      .def_property_readonly("size", &Field::size)
      .def_property_readonly("pad_size", &Field::get_pad_size)
      .def_property_readonly("name", &Field::get_name)
      .def_property_readonly("collection", &Field::get_collection)
      .def_property_readonly("nb_components", &Field::get_nb_components)
      .def_property_readonly("is_global", &Field::is_global);
}

template <class T>
void add_typed_field(py::module & mod, std::string name) {
  auto && array_computer = [](TypedFieldBase<T> & self,
                              const std::vector<Dim_t> & shape,
                              const muGrid::Iteration & it) {
    // py_class will be passed as the `base` class to the array
    // constructors below. This ties the lifetime of the array that does
    // not own its own data to the field object. (Without this
    // parameter, the constructor makes a copy of the array.)
    std::vector<size_t> return_shape;
    const size_t dim{shape.size()};

    // If shape is given, then we return a field of tensors of this
    // shape
    Dim_t ntotal{1};
    if (dim != 0) {
      for (auto & n : shape) {
        return_shape.push_back(n);
        ntotal *= n;
      }
    }

    const auto nb_quad{self.get_collection().get_nb_quad()};
    if (it == muGrid::Iteration::QuadPt) {
      // If shape is not given, we just return column vectors with the
      // components
      if (dim == 0) {
        return_shape.push_back(self.get_nb_components());
      } else if (ntotal != self.get_nb_components()) {
        std::stringstream error{};
        error << "Field has " << self.get_nb_components() << " components "
              << "per quadrature point, but shape requested would "
                 "require "
              << ntotal << " components.";
        throw std::runtime_error(error.str());
      }
      return_shape.push_back(nb_quad);
    } else {
      // If shape is not given, we just return column vectors with the
      // components
      if (dim == 0) {
        return_shape.push_back(self.get_nb_components() * nb_quad);
      } else if (ntotal != self.get_nb_components() * nb_quad) {
        std::stringstream error{};
        error << "Field has " << self.get_nb_components() * nb_quad
              << " components per pixel, but shape requested would "
                 "require "
              << ntotal << " components.";
        throw std::runtime_error(error.str());
      }
    }

    const auto & coll{self.get_collection()};
    if (coll.get_domain() == FieldCollection::ValidityDomain::Global) {
      // We have a global field collection and can return array that
      // have the correct shape corresponding to the grid (on the local
      // MPI process).
      const GlobalFieldCollection & global_coll =
          dynamic_cast<const GlobalFieldCollection &>(coll);
      const auto & nb_grid_pts{global_coll.get_pixels().get_nb_grid_pts()};
      for (auto & n : nb_grid_pts) {
        return_shape.push_back(n);
      }
    } else {
      return_shape.push_back(coll.get_nb_pixels());
    }
    return py::array_t<T, py::array::f_style>(return_shape, self.data(),
                                              py::capsule([]() {}));
  };

  py::class_<TypedFieldBase<T>, Field>(mod, (name + "Base").c_str(),
                                         py::buffer_protocol())
      .def_buffer([](TypedFieldBase<T> & self) {
        auto & coll = self.get_collection();
        return py::buffer_info(
            self.data(),
            {static_cast<size_t>(coll.get_nb_pixels()),
             static_cast<size_t>(coll.get_nb_quad()),
             static_cast<size_t>(self.get_nb_components())},
            {sizeof(T) * static_cast<size_t>(self.get_nb_components() *
                                             coll.get_nb_quad()),
             sizeof(T) * static_cast<size_t>(self.get_nb_components()),
             sizeof(T)});
      })
      .def("array", array_computer, "shape"_a = std::vector<Dim_t>{},
           "iteration_type"_a = muGrid::Iteration::QuadPt,
           py::return_value_policy::reference_internal)
      .def("array",
           [&array_computer](TypedFieldBase<T> & self,
                             const muGrid::Iteration & it) {
             return array_computer(self, std::vector<Dim_t>{}, it);
           },
           "iteration_type"_a = muGrid::Iteration::QuadPt,
           py::return_value_policy::reference_internal);

  py::class_<TypedField<T>, TypedFieldBase<T>>(mod, name.c_str());
}

void add_field_classes(py::module & mod) {
  add_field(mod);

  add_typed_field<muGrid::Real>(mod, "RealField");
  add_typed_field<muGrid::Complex>(mod, "ComplexField");
  add_typed_field<muGrid::Int>(mod, "IntField");
  add_typed_field<muGrid::Uint>(mod, "UintField");
}
