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

#include "libmugrid/nfield.hh"
#include "libmugrid/nfield_typed.hh"
#include "libmugrid/nfield_collection.hh"
#include "libmugrid/nfield_collection_global.hh"
#include "libmugrid/nfield_collection_local.hh"
#include "libmugrid/state_nfield.hh"

using muGrid::Complex;
using muGrid::Dim_t;
using muGrid::DynCcoord_t;
using muGrid::GlobalNFieldCollection;
using muGrid::Int;
using muGrid::LocalNFieldCollection;
using muGrid::NFieldCollection;
using muGrid::Real;
using muGrid::Uint;
using pybind11::literals::operator""_a;

namespace py = pybind11;

void add_field_collection(py::module & mod) {
  py::class_<NFieldCollection> field_collection(mod, "FieldCollection");
  field_collection
      .def("register_real_field", &NFieldCollection::register_real_field,
           py::return_value_policy::reference_internal)
      .def("register_complex_field", &NFieldCollection::register_complex_field,
           py::return_value_policy::reference_internal)
      .def("register_int_field", &NFieldCollection::register_int_field,
           py::return_value_policy::reference_internal)
      .def("register_uint_field", &NFieldCollection::register_uint_field,
           py::return_value_policy::reference_internal)
      .def("field_exists", &NFieldCollection::field_exists)
      .def("state_field_exists", &NFieldCollection::state_field_exists)
      .def_property_readonly("nb_entries", &NFieldCollection::get_nb_entries)
      .def_property_readonly("nb_pixels", &NFieldCollection::get_nb_pixels)
      .def_property("nb_quad", &NFieldCollection::get_nb_quad,
                    &NFieldCollection::set_nb_quad)
      .def_property_readonly("domain", &NFieldCollection::get_domain)
      .def_property_readonly("is_initialised",
                             &NFieldCollection::is_initialised)
      .def("get_field", &NFieldCollection::get_field,
           py::return_value_policy::reference_internal)
      .def(
          "get_real_field",
          [](NFieldCollection & collection, const std::string & unique_name)
              -> muGrid::TypedNFieldBase<Real> & {
            return dynamic_cast<muGrid::TypedNFieldBase<Real> &>(
                collection.get_field(unique_name));
          },
          "unique_name"_a, py::return_value_policy::reference_internal)
      .def(
          "get_complex_field",
          [](NFieldCollection & collection, const std::string & unique_name)
              -> muGrid::TypedNFieldBase<Complex> & {
            return dynamic_cast<muGrid::TypedNFieldBase<Complex> &>(
                collection.get_field(unique_name));
          },
          "unique_name"_a, py::return_value_policy::reference_internal)
      .def(
          "get_int_field",
          [](NFieldCollection & collection, const std::string & unique_name)
              -> muGrid::TypedNFieldBase<Int> & {
            return dynamic_cast<muGrid::TypedNFieldBase<Int> &>(
                collection.get_field(unique_name));
          },
          "unique_name"_a, py::return_value_policy::reference_internal)
      .def(
          "get_uint_field",
          [](NFieldCollection & collection, const std::string & unique_name)
              -> muGrid::TypedNFieldBase<Uint> & {
            return dynamic_cast<muGrid::TypedNFieldBase<Uint> &>(
                collection.get_field(unique_name));
          },
          "unique_name"_a, py::return_value_policy::reference_internal)
      .def("get_state_field", &NFieldCollection::get_state_field,
           py::return_value_policy::reference_internal)
      .def("keys", &NFieldCollection::list_fields)
      .def_property_readonly("field_names", &NFieldCollection::list_fields);

  py::class_<muGrid::NFieldCollection::IndexIterable>(mod, "IndexIterable")
      .def("__len__", &muGrid::NFieldCollection::IndexIterable::size)
      .def("__iter__", [](muGrid::NFieldCollection::IndexIterable & iterable) {
        return py::make_iterator(iterable.begin(), iterable.end());
      });

  py::class_<muGrid::NFieldCollection::PixelIndexIterable>(mod,
                                                           "PixelIndexIterable")
      .def("__len__", &muGrid::NFieldCollection::PixelIndexIterable::size)
      .def("__iter__",
           [](muGrid::NFieldCollection::PixelIndexIterable & iterable) {
             return py::make_iterator(iterable.begin(), iterable.end());
           });

  py::enum_<NFieldCollection::ValidityDomain>(field_collection,
                                              "ValidityDomain")
      .value("Global", NFieldCollection::ValidityDomain::Global)
      .value("Local", NFieldCollection::ValidityDomain::Local)
      .export_values();
}

void add_global_field_collection(py::module & mod) {
  py::class_<GlobalNFieldCollection, NFieldCollection>(mod,
                                                       "GlobalFieldCollection")
      .def(py::init<Dim_t, Dim_t>())
      .def("initialise",
           [](GlobalNFieldCollection & self, const DynCcoord_t & nb_grid_pts) {
             self.initialise(nb_grid_pts);
           })
      .def("initialise", (void (GlobalNFieldCollection::*)(  // NOLINT
                             const DynCcoord_t &, const DynCcoord_t &)) &
                             GlobalNFieldCollection::initialise)
      .def("initialise",
           (void (GlobalNFieldCollection::*)(  // NOLINT
               const DynCcoord_t &, const DynCcoord_t &, const DynCcoord_t &)) &
               GlobalNFieldCollection::initialise)
      .def_property_readonly("pixels", &GlobalNFieldCollection::get_pixels);
}

void add_local_field_collection(py::module & mod) {
  py::class_<LocalNFieldCollection, NFieldCollection>(mod,
                                                      "LocalFieldCollection");
}

void add_field_collection_classes(py::module & mod) {
  add_field_collection(mod);
  add_global_field_collection(mod);
  add_local_field_collection(mod);
}
