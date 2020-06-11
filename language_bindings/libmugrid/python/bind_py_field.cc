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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <sstream>
#include <stdexcept>

using muGrid::Index_t;
using muGrid::Field;
using muGrid::FieldCollection;
using muGrid::GlobalFieldCollection;
using muGrid::RuntimeError;
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
      .def_property_readonly("nb_dof_per_sub_pt", &Field::get_nb_dof_per_sub_pt)
      .def_property_readonly("nb_entries", &Field::get_nb_entries)
      .def_property_readonly("is_global", &Field::is_global);
}

template <class T>
void add_typed_field(py::module & mod, std::string name) {
  auto && array_computer = [](TypedFieldBase<T> & self,
                              const std::vector<Index_t> & shape,
                              const muGrid::IterUnit & it) {
    // py_class will be passed as the `base` class to the array
    // constructors below. This ties the lifetime of the array that does
    // not own its own data to the field object. (Without this
    // parameter, the constructor makes a copy of the array.)
    std::vector<size_t> return_shape, return_strides{};
    const size_t dim{shape.size()};

    // If shape is given, then we return a field of tensors of this
    // shape
    Index_t ntotal{1}, stride{sizeof(T)};
    if (dim != 0) {
      for (auto & n : shape) {
        return_shape.push_back(n);
        return_strides.push_back(stride);
        ntotal *= n;
        stride *= n;
      }
    }

    auto && nb_sub_pts{self.get_nb_sub_pts()};
    auto && nb_dof_per_sub_pt{self.get_nb_dof_per_sub_pt()};

    switch (it) {
    case muGrid::IterUnit::SubPt: {
      // If shape is not given, we just return column vectors with the
      // components
      if (dim == 0) {
        return_shape.push_back(nb_dof_per_sub_pt);
        return_strides.push_back(stride);
        stride *= nb_dof_per_sub_pt;
      } else if (ntotal != self.get_nb_dof_per_sub_pt()) {
        std::stringstream error{};
        error << "Field has " << nb_dof_per_sub_pt << " components "
              << "per quadrature point, but shape requested would "
                 "require "
              << ntotal << " components.";
        throw RuntimeError(error.str());
      }
      return_shape.push_back(nb_sub_pts);
      return_strides.push_back(stride);
      stride *= nb_sub_pts;
      break;
    }
    case muGrid::IterUnit::Pixel: {
      // If shape is not given, we just return column vectors with the
      // components
      if (dim == 0) {
        return_shape.push_back(nb_dof_per_sub_pt * nb_sub_pts);
        return_strides.push_back(stride);
        stride *= nb_dof_per_sub_pt * nb_sub_pts;
      } else if (ntotal != nb_dof_per_sub_pt * nb_sub_pts) {
        std::stringstream error{};
        error << "Field has " << nb_dof_per_sub_pt * nb_sub_pts
              << " components per pixel, but shape requested would "
                 "require "
              << ntotal << " components.";
        throw RuntimeError(error.str());
      }

      break;
    }
    default:
      throw RuntimeError{"unknown pixel sub-division"};
      break;
    }

    const auto & coll{self.get_collection()};
    if (coll.get_domain() == FieldCollection::ValidityDomain::Global) {
      // We have a global field collection and can return array that
      // have the correct shape corresponding to the grid (on the local
      // MPI process).
      const GlobalFieldCollection & global_coll =
          dynamic_cast<const GlobalFieldCollection &>(coll);
      const auto & nb_grid_pts{
          global_coll.get_pixels().get_nb_subdomain_grid_pts()};
      const auto & strides{global_coll.get_pixels().get_strides()};
      for (const auto & tup : akantu::zip(nb_grid_pts, strides)) {
        const auto n = std::get<0>(tup);
        const auto s = std::get<1>(tup);
        return_shape.push_back(n);
        return_strides.push_back(stride * s);
      }
    } else {
      if (not coll.is_initialised()) {
        throw RuntimeError("Field collection isn't initialised yet");
      }
      return_shape.push_back(coll.get_nb_pixels());
      return_strides.push_back(stride);
    }
    return py::array_t<T, py::array::f_style>(
        return_shape, return_strides, self.data(), py::capsule([]() {}));
  };

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
      .def_property_readonly("shape",
                             [](TypedFieldBase<T> & field) {
                               return field.get_shape(
                                   muGrid::IterUnit::SubPt);
                             })
      .def("array", array_computer, "shape"_a = std::vector<Index_t>{},
           "iteration_type"_a = muGrid::IterUnit::SubPt,
           py::keep_alive<0, 1>())
      .def(
          "array",
          [&array_computer](TypedFieldBase<T> & self,
                            const muGrid::IterUnit & it) {
            return array_computer(self, std::vector<Index_t>{}, it);
          },
          "iteration_type"_a = muGrid::IterUnit::SubPt,
          py::keep_alive<0, 1>());

  py::class_<TypedField<T>, TypedFieldBase<T>>(mod, name.c_str());
}

void add_field_classes(py::module & mod) {
  add_field(mod);

  add_typed_field<muGrid::Real>(mod, "RealField");
  add_typed_field<muGrid::Complex>(mod, "ComplexField");
  add_typed_field<muGrid::Int>(mod, "IntField");
  add_typed_field<muGrid::Uint>(mod, "UintField");
}
