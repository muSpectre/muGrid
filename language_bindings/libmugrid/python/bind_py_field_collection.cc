/**
 * @file   bind_py_field_collection.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   16 Oct 2019
 *
 * @brief  Python bindings for field collections
 *
 * Copyright © 2018 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 */

#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "libmugrid/field.hh"
#include "libmugrid/field_typed.hh"
#include "libmugrid/field_collection.hh"
#include "libmugrid/field_collection_global.hh"
#include "libmugrid/field_collection_local.hh"
#include "libmugrid/state_field.hh"

using muGrid::Complex;
using muGrid::Dim_t;
using muGrid::DynCcoord_t;
using muGrid::GlobalFieldCollection;
using muGrid::Int;
using muGrid::LocalFieldCollection;
using muGrid::FieldCollection;
using muGrid::Real;
using muGrid::Uint;
using pybind11::literals::operator""_a;

namespace py = pybind11;

void add_field_collection(py::module & mod) {
  py::class_<FieldCollection> field_collection(mod, "FieldCollection");
  field_collection
      .def("register_real_field", &FieldCollection::register_real_field,
           py::return_value_policy::reference_internal)
      .def("register_complex_field", &FieldCollection::register_complex_field,
           py::return_value_policy::reference_internal)
      .def("register_int_field", &FieldCollection::register_int_field,
           py::return_value_policy::reference_internal)
      .def("register_uint_field", &FieldCollection::register_uint_field,
           py::return_value_policy::reference_internal)
      .def("field_exists", &FieldCollection::field_exists)
      .def("state_field_exists", &FieldCollection::state_field_exists)
      .def_property_readonly("nb_entries", &FieldCollection::get_nb_entries)
      .def_property_readonly("nb_pixels", &FieldCollection::get_nb_pixels)
      .def_property("nb_quad", &FieldCollection::get_nb_quad,
                    &FieldCollection::set_nb_quad)
      .def_property_readonly("domain", &FieldCollection::get_domain)
      .def_property_readonly("is_initialised",
                             &FieldCollection::is_initialised)
      .def("get_field", &FieldCollection::get_field,
           py::return_value_policy::reference_internal)
      .def(
          "get_real_field",
          [](FieldCollection & collection, const std::string & unique_name)
              -> muGrid::TypedFieldBase<Real> & {
            return dynamic_cast<muGrid::TypedFieldBase<Real> &>(
                collection.get_field(unique_name));
          },
          "unique_name"_a, py::return_value_policy::reference_internal)
      .def(
          "get_complex_field",
          [](FieldCollection & collection, const std::string & unique_name)
              -> muGrid::TypedFieldBase<Complex> & {
            return dynamic_cast<muGrid::TypedFieldBase<Complex> &>(
                collection.get_field(unique_name));
          },
          "unique_name"_a, py::return_value_policy::reference_internal)
      .def(
          "get_int_field",
          [](FieldCollection & collection, const std::string & unique_name)
              -> muGrid::TypedFieldBase<Int> & {
            return dynamic_cast<muGrid::TypedFieldBase<Int> &>(
                collection.get_field(unique_name));
          },
          "unique_name"_a, py::return_value_policy::reference_internal)
      .def(
          "get_uint_field",
          [](FieldCollection & collection, const std::string & unique_name)
              -> muGrid::TypedFieldBase<Uint> & {
            return dynamic_cast<muGrid::TypedFieldBase<Uint> &>(
                collection.get_field(unique_name));
          },
          "unique_name"_a, py::return_value_policy::reference_internal)
      .def("get_state_field", &FieldCollection::get_state_field,
           py::return_value_policy::reference_internal)
      .def("keys", &FieldCollection::list_fields)
      .def_property_readonly("field_names", &FieldCollection::list_fields);

  py::class_<muGrid::FieldCollection::IndexIterable>(mod, "IndexIterable")
      .def("__len__", &muGrid::FieldCollection::IndexIterable::size)
      .def("__iter__", [](muGrid::FieldCollection::IndexIterable & iterable) {
        return py::make_iterator(iterable.begin(), iterable.end());
      });

  py::class_<muGrid::FieldCollection::PixelIndexIterable>(mod,
                                                           "PixelIndexIterable")
      .def("__len__", &muGrid::FieldCollection::PixelIndexIterable::size)
      .def("__iter__",
           [](muGrid::FieldCollection::PixelIndexIterable & iterable) {
             return py::make_iterator(iterable.begin(), iterable.end());
           });

  py::enum_<FieldCollection::ValidityDomain>(field_collection,
                                              "ValidityDomain")
      .value("Global", FieldCollection::ValidityDomain::Global)
      .value("Local", FieldCollection::ValidityDomain::Local)
      .export_values();
}

void add_global_field_collection(py::module & mod) {
  py::class_<GlobalFieldCollection, FieldCollection>(mod,
                                                       "GlobalFieldCollection")
      .def(py::init<Dim_t, Dim_t>())
      .def("initialise",
           [](GlobalFieldCollection & self, const DynCcoord_t & nb_grid_pts) {
             self.initialise(nb_grid_pts);
           })
      .def("initialise", (void (GlobalFieldCollection::*)(  // NOLINT
                             const DynCcoord_t &, const DynCcoord_t &)) &
                             GlobalFieldCollection::initialise)
      .def("initialise",
           (void (GlobalFieldCollection::*)(  // NOLINT
               const DynCcoord_t &, const DynCcoord_t &, const DynCcoord_t &)) &
               GlobalFieldCollection::initialise)
      .def_property_readonly("pixels", &GlobalFieldCollection::get_pixels);
}

void add_local_field_collection(py::module & mod) {
  py::class_<LocalFieldCollection, FieldCollection>(mod,
                                                      "LocalFieldCollection");
}

void add_field_collection_classes(py::module & mod) {
  add_field_collection(mod);
  add_global_field_collection(mod);
  add_local_field_collection(mod);
}
