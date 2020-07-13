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
using muGrid::DynCcoord_t;
using muGrid::FieldCollection;
using muGrid::GlobalFieldCollection;
using muGrid::Index_t;
using muGrid::Int;
using muGrid::LocalFieldCollection;
using muGrid::Real;
using muGrid::StorageOrder;
using muGrid::Uint;
using pybind11::literals::operator""_a;

namespace py = pybind11;

void add_field_collection(py::module & mod) {
  py::class_<FieldCollection> field_collection(mod, "FieldCollection");
  field_collection
      .def(
          "register_real_field",
          [](FieldCollection & collection, const std::string & unique_name,
             const Index_t & nb_components,
             const std::string & sub_division,
             const muGrid::Unit & unit) -> muGrid::TypedField<Real> & {
            return collection.register_real_field(
                unique_name, nb_components, sub_division, unit);
          },
          "unique_name"_a, "nb_components"_a,
          "sub_division"_a = muGrid::PixelTag,
          "unit"_a = muGrid::Unit::unitless(),
          py::return_value_policy::reference_internal)
      .def(
          "register_real_field",
          [](FieldCollection & collection, const std::string & unique_name,
             const muGrid::Shape_t & components_shape,
             const std::string & sub_division,
             const muGrid::Unit & unit) -> muGrid::TypedField<Real> & {
            return collection.register_real_field(
                unique_name, components_shape, sub_division, unit);
          },
          "unique_name"_a, "components_shape"_a,
          "sub_division"_a = muGrid::PixelTag,
          "unit"_a = muGrid::Unit::unitless(),
          py::return_value_policy::reference_internal)
      .def(
          "register_complex_field",
          [](FieldCollection & collection, const std::string & unique_name,
             const Index_t & nb_components,
             const std::string & sub_division,
             const muGrid::Unit & unit) -> muGrid::TypedField<Complex> & {
            return collection.register_complex_field(
                unique_name, nb_components, sub_division, unit);
          },
          "unique_name"_a, "nb_components"_a,
          "sub_division"_a = muGrid::PixelTag,
          "unit"_a = muGrid::Unit::unitless(),
          py::return_value_policy::reference_internal)
      .def(
          "register_complex_field",
          [](FieldCollection & collection, const std::string & unique_name,
             const muGrid::Shape_t & components_shape,
             const std::string & sub_division,
             const muGrid::Unit & unit)
              -> muGrid::TypedField<Complex> & {
            return collection.register_complex_field(
                unique_name, components_shape, sub_division, unit);
          },
          "unique_name"_a, "components_shape"_a,
          "sub_division"_a = muGrid::PixelTag,
          "unit"_a = muGrid::Unit::unitless(),
          py::return_value_policy::reference_internal)
      .def(
          "register_uint_field",
          [](FieldCollection & collection, const std::string & unique_name,
             const Index_t & nb_components,
             const std::string & sub_division,
             const muGrid::Unit & unit) -> muGrid::TypedField<Uint> & {
            return collection.register_uint_field(
                unique_name, nb_components, sub_division, unit);
          },
          "unique_name"_a, "nb_components"_a,
          "sub_division"_a = muGrid::PixelTag,
          "unit"_a = muGrid::Unit::unitless(),
          py::return_value_policy::reference_internal)
      .def(
          "register_uint_field",
          [](FieldCollection & collection, const std::string & unique_name,
             const muGrid::Shape_t & components_shape,
             const std::string & sub_division,
             const muGrid::Unit & unit) -> muGrid::TypedField<Uint> & {
            return collection.register_uint_field(
                unique_name, components_shape, sub_division, unit);
          },
          "unique_name"_a, "components_shape"_a,
          "sub_division"_a = muGrid::PixelTag,
          "unit"_a = muGrid::Unit::unitless(),
          py::return_value_policy::reference_internal)
      .def(
          "register_int_field",
          [](FieldCollection & collection, const std::string & unique_name,
             const Index_t & nb_components,
             const std::string & sub_division,
             const muGrid::Unit & unit) -> muGrid::TypedField<Int> & {
            return collection.register_int_field(unique_name, nb_components,
                                                 sub_division, unit);
          },
          "unique_name"_a, "nb_components"_a,
          "sub_division"_a = muGrid::PixelTag,
          "unit"_a = muGrid::Unit::unitless(),
          py::return_value_policy::reference_internal)
      .def(
          "register_int_field",
          [](FieldCollection & collection, const std::string & unique_name,
             const muGrid::Shape_t & components_shape,
             const std::string & sub_division,
             const muGrid::Unit & unit) -> muGrid::TypedField<Int> & {
            return collection.register_int_field(
                unique_name, components_shape, sub_division, unit);
          },
          "unique_name"_a, "components_shape"_a,
          "sub_division"_a = muGrid::PixelTag,
          "unit"_a = muGrid::Unit::unitless(),
          py::return_value_policy::reference_internal)
      .def("field_exists", &FieldCollection::field_exists)
      .def("state_field_exists", &FieldCollection::state_field_exists)
      .def_property_readonly("nb_pixels", &FieldCollection::get_nb_pixels)
      .def(
          "get_nb_sub_pts",
          [](const FieldCollection & coll, const std::string & tag) {
            return coll.get_nb_sub_pts(tag);
          },
          "tag"_a, py::return_value_policy::copy)
      .def("set_nb_sub_pts", &FieldCollection::set_nb_sub_pts, "tag"_a,
           "nb_sub_pts"_a)
      .def_property_readonly("domain", &FieldCollection::get_domain)
      .def_property_readonly("is_initialised", &FieldCollection::is_initialised)
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
          [](FieldCollection & collection,
             const std::string & unique_name) -> muGrid::TypedFieldBase<Int> & {
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

  py::enum_<FieldCollection::ValidityDomain>(field_collection, "ValidityDomain")
      .value("Global", FieldCollection::ValidityDomain::Global)
      .value("Local", FieldCollection::ValidityDomain::Local)
      .export_values();
}

void add_global_field_collection(py::module & mod) {
  py::class_<GlobalFieldCollection, FieldCollection>(mod,
                                                     "GlobalFieldCollection")
      .def(py::init<const Index_t &>(), "spatial_dimension"_a)
      .def(py::init<const Index_t &, const FieldCollection::SubPtMap_t &,
                    StorageOrder>(),
           "spatial_dimension"_a, "sub_pts"_a,
           "storage_order"_a = StorageOrder::ColMajor)
      .def(py::init<const Index_t &, const DynCcoord_t &, const DynCcoord_t &,
                    const FieldCollection::SubPtMap_t &, StorageOrder>(),
           "spatial_dimension"_a, "nb_subdomain_grid_pts"_a,
           "subdomain_locations"_a, "sub_pts"_a,
           "storage_order"_a = StorageOrder::ColMajor)
      .def(py::init<const Index_t &, const DynCcoord_t &, const DynCcoord_t &,
                    const DynCcoord_t &, const FieldCollection::SubPtMap_t &,
                    StorageOrder>(),
           "spatial_dimension"_a, "nb_subdomain_grid_pts"_a,
           "subdomain_locations"_a, "pixels_strides"_a, "sub_pts"_a,
           "storage_order"_a = StorageOrder::ColMajor)
      .def(py::init<const Index_t &, const DynCcoord_t &, const DynCcoord_t &,
               StorageOrder, const FieldCollection::SubPtMap_t &,
               StorageOrder>(),
           "spatial_dimension"_a, "nb_subdomain_grid_pts"_a,
           "subdomain_locations"_a, "pixels_storage_order"_a, "sub_pts"_a,
           "storage_order"_a = StorageOrder::ColMajor)
      .def("initialise",
           (void (GlobalFieldCollection::*)(  // NOLINT
               const DynCcoord_t &, const DynCcoord_t &, const DynCcoord_t &)) &
               GlobalFieldCollection::initialise,
           "nb_subdomain_grid_pts"_a, "subdomain_locations"_a,
           "pixels_strides"_a)
      .def("initialise",
           (void (GlobalFieldCollection::*)(  // NOLINT
               const DynCcoord_t &, const DynCcoord_t &, StorageOrder)) &
               GlobalFieldCollection::initialise,
           "nb_subdomain_grid_pts"_a, "subdomain_locations"_a = DynCcoord_t{},
           "pixels_storage_order"_a = StorageOrder::Automatic)
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
