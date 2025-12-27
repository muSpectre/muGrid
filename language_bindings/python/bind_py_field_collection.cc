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

#include "field/field.hh"
#include "field/field_typed.hh"
#include "collection/field_collection.hh"
#include "collection/field_collection_global.hh"
#include "collection/field_collection_local.hh"
#include "field/state_field.hh"

#include <map>

using muGrid::Complex;
using muGrid::DynGridIndex;
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
    py::class_<FieldCollection> field_collection(mod, "FieldCollection",
        R"pbdoc(
        Base class for managing collections of fields on structured grids.

        A FieldCollection groups fields that share the same spatial discretization.
        Each field can have different numbers of components and sub-points, but all
        fields in a collection share the same pixel grid.

        Fields are created using accessor methods like ``real_field()`` or
        ``register_real_field()``. The difference is that accessor methods return
        an existing field if one with the same name exists, while register methods
        raise an exception if the field already exists.

        There are two concrete implementations:

        - ``GlobalFieldCollection``: Fields defined at all grid points
        - ``LocalFieldCollection``: Fields defined at a subset of grid points

        See Also
        --------
        GlobalFieldCollection : Collection for fields at all grid points
        LocalFieldCollection : Collection for fields at selected grid points
        )pbdoc");
    field_collection
        .def(
            "register_real_field",
            [](FieldCollection & collection, const std::string & unique_name,
               const Index_t & nb_components, const std::string & sub_division,
               const muGrid::Unit & unit) -> muGrid::Field & {
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
               const muGrid::Unit & unit) -> muGrid::Field & {
                return collection.register_real_field(
                    unique_name, components_shape, sub_division, unit);
            },
            "unique_name"_a, "components_shape"_a = muGrid::Shape_t{},
            "sub_division"_a = muGrid::PixelTag,
            "unit"_a = muGrid::Unit::unitless(),
            py::return_value_policy::reference_internal)
        .def(
            "register_complex_field",
            [](FieldCollection & collection, const std::string & unique_name,
               const Index_t & nb_components, const std::string & sub_division,
               const muGrid::Unit & unit) -> muGrid::Field & {
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
               const muGrid::Unit & unit) -> muGrid::Field & {
                return collection.register_complex_field(
                    unique_name, components_shape, sub_division, unit);
            },
            "unique_name"_a, "components_shape"_a = muGrid::Shape_t{},
            "sub_division"_a = muGrid::PixelTag,
            "unit"_a = muGrid::Unit::unitless(),
            py::return_value_policy::reference_internal)
        .def(
            "register_uint_field",
            [](FieldCollection & collection, const std::string & unique_name,
               const Index_t & nb_components, const std::string & sub_division,
               const muGrid::Unit & unit) -> muGrid::Field & {
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
               const muGrid::Unit & unit) -> muGrid::Field & {
                return collection.register_uint_field(
                    unique_name, components_shape, sub_division, unit);
            },
            "unique_name"_a, "components_shape"_a = muGrid::Shape_t{},
            "sub_division"_a = muGrid::PixelTag,
            "unit"_a = muGrid::Unit::unitless(),
            py::return_value_policy::reference_internal)
        .def(
            "register_int_field",
            [](FieldCollection & collection, const std::string & unique_name,
               const Index_t & nb_components, const std::string & sub_division,
               const muGrid::Unit & unit) -> muGrid::Field & {
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
               const muGrid::Unit & unit) -> muGrid::Field & {
                return collection.register_int_field(
                    unique_name, components_shape, sub_division, unit);
            },
            "unique_name"_a, "components_shape"_a = muGrid::Shape_t{},
            "sub_division"_a = muGrid::PixelTag,
            "unit"_a = muGrid::Unit::unitless(),
            py::return_value_policy::reference_internal)
        .def(
            "register_real_state_field",
            [](FieldCollection & collection, const std::string & unique_prefix,
               const Index_t & nb_memory, const Index_t & nb_components,
               const std::string & sub_division_tag,
               const muGrid::Unit & unit) -> muGrid::TypedStateField<Real> & {
                return collection.register_real_state_field(
                    unique_prefix, nb_memory, nb_components, sub_division_tag,
                    unit);
            },
            "unique_prefix"_a, "nb_memory"_a, "nb_components"_a,
            "sub_division_tag"_a = muGrid::PixelTag,
            "unit"_a = muGrid::Unit::unitless(),
            py::return_value_policy::reference_internal)
        .def(
            "register_complex_state_field",
            [](FieldCollection & collection, const std::string & unique_prefix,
               const Index_t & nb_memory, const Index_t & nb_components,
               const std::string & sub_division_tag, const muGrid::Unit & unit)
                -> muGrid::TypedStateField<Complex> & {
                return collection.register_complex_state_field(
                    unique_prefix, nb_memory, nb_components, sub_division_tag,
                    unit);
            },
            "unique_prefix"_a, "nb_memory"_a, "nb_components"_a,
            "sub_division_tag"_a = muGrid::PixelTag,
            "unit"_a = muGrid::Unit::unitless(),
            py::return_value_policy::reference_internal)
        .def(
            "register_int_state_field",
            [](FieldCollection & collection, const std::string & unique_prefix,
               const Index_t & nb_memory, const Index_t & nb_components,
               const std::string & sub_division_tag,
               const muGrid::Unit & unit) -> muGrid::TypedStateField<Int> & {
                return collection.register_int_state_field(
                    unique_prefix, nb_memory, nb_components, sub_division_tag,
                    unit);
            },
            "unique_prefix"_a, "nb_memory"_a, "nb_components"_a,
            "sub_division_tag"_a = muGrid::PixelTag,
            "unit"_a = muGrid::Unit::unitless(),
            py::return_value_policy::reference_internal)
        .def(
            "register_unint_state_field",
            [](FieldCollection & collection, const std::string & unique_prefix,
               const Index_t & nb_memory, const Index_t & nb_components,
               const std::string & sub_division_tag,
               const muGrid::Unit & unit) -> muGrid::TypedStateField<Uint> & {
                return collection.register_uint_state_field(
                    unique_prefix, nb_memory, nb_components, sub_division_tag,
                    unit);
            },
            "unique_prefix"_a, "nb_memory"_a, "nb_components"_a = 1,
            "sub_division_tag"_a = muGrid::PixelTag,
            "unit"_a = muGrid::Unit::unitless(),
            py::return_value_policy::reference_internal)
        .def(
            "real_field",
            [](FieldCollection & collection, const std::string & unique_name,
               const Index_t & nb_components, const std::string & sub_division,
               const muGrid::Unit & unit) -> muGrid::Field & {
                return collection.real_field(unique_name, nb_components,
                                             sub_division, unit);
            },
            "unique_name"_a, "nb_components"_a,
            "sub_division"_a = muGrid::PixelTag,
            "unit"_a = muGrid::Unit::unitless(),
            py::return_value_policy::reference_internal)
        .def(
            "real_field",
            [](FieldCollection & collection, const std::string & unique_name,
               const muGrid::Shape_t & components_shape,
               const std::string & sub_division,
               const muGrid::Unit & unit) -> muGrid::Field & {
                return collection.real_field(unique_name, components_shape,
                                             sub_division, unit);
            },
            "unique_name"_a, "components_shape"_a = muGrid::Shape_t{},
            "sub_division"_a = muGrid::PixelTag,
            "unit"_a = muGrid::Unit::unitless(),
            py::return_value_policy::reference_internal)
        .def(
            "complex_field",
            [](FieldCollection & collection, const std::string & unique_name,
               const Index_t & nb_components, const std::string & sub_division,
               const muGrid::Unit & unit) -> muGrid::Field & {
                return collection.complex_field(unique_name, nb_components,
                                                sub_division, unit);
            },
            "unique_name"_a, "nb_components"_a,
            "sub_division"_a = muGrid::PixelTag,
            "unit"_a = muGrid::Unit::unitless(),
            py::return_value_policy::reference_internal)
        .def(
            "complex_field",
            [](FieldCollection & collection, const std::string & unique_name,
               const muGrid::Shape_t & components_shape,
               const std::string & sub_division,
               const muGrid::Unit & unit) -> muGrid::Field & {
                return collection.complex_field(unique_name, components_shape,
                                                sub_division, unit);
            },
            "unique_name"_a, "components_shape"_a = muGrid::Shape_t{},
            "sub_division"_a = muGrid::PixelTag,
            "unit"_a = muGrid::Unit::unitless(),
            py::return_value_policy::reference_internal)
        .def(
            "uint_field",
            [](FieldCollection & collection, const std::string & unique_name,
               const Index_t & nb_components, const std::string & sub_division,
               const muGrid::Unit & unit) -> muGrid::Field & {
                return collection.uint_field(unique_name, nb_components,
                                             sub_division, unit);
            },
            "unique_name"_a, "nb_components"_a,
            "sub_division"_a = muGrid::PixelTag,
            "unit"_a = muGrid::Unit::unitless(),
            py::return_value_policy::reference_internal)
        .def(
            "uint_field",
            [](FieldCollection & collection, const std::string & unique_name,
               const muGrid::Shape_t & components_shape,
               const std::string & sub_division,
               const muGrid::Unit & unit) -> muGrid::Field & {
                return collection.uint_field(unique_name, components_shape,
                                             sub_division, unit);
            },
            "unique_name"_a, "components_shape"_a = muGrid::Shape_t{},
            "sub_division"_a = muGrid::PixelTag,
            "unit"_a = muGrid::Unit::unitless(),
            py::return_value_policy::reference_internal)
        .def(
            "int_field",
            [](FieldCollection & collection, const std::string & unique_name,
               const Index_t & nb_components, const std::string & sub_division,
               const muGrid::Unit & unit) -> muGrid::Field & {
                return collection.int_field(unique_name, nb_components,
                                            sub_division, unit);
            },
            "unique_name"_a, "nb_components"_a,
            "sub_division"_a = muGrid::PixelTag,
            "unit"_a = muGrid::Unit::unitless(),
            py::return_value_policy::reference_internal)
        .def(
            "int_field",
            [](FieldCollection & collection, const std::string & unique_name,
               const muGrid::Shape_t & components_shape,
               const std::string & sub_division,
               const muGrid::Unit & unit) -> muGrid::Field & {
                return collection.int_field(unique_name, components_shape,
                                            sub_division, unit);
            },
            "unique_name"_a, "components_shape"_a = muGrid::Shape_t{},
            "sub_division"_a = muGrid::PixelTag,
            "unit"_a = muGrid::Unit::unitless(),
            py::return_value_policy::reference_internal)
        .def(
            "real_state_field",
            [](FieldCollection & collection, const std::string & unique_prefix,
               const Index_t & nb_memory, const Index_t & nb_components,
               const std::string & sub_division_tag,
               const muGrid::Unit & unit) -> muGrid::TypedStateField<Real> & {
                return collection.real_state_field(unique_prefix, nb_memory,
                                                   nb_components,
                                                   sub_division_tag, unit);
            },
            "unique_prefix"_a, "nb_memory"_a, "nb_components"_a,
            "sub_division_tag"_a = muGrid::PixelTag,
            "unit"_a = muGrid::Unit::unitless(),
            py::return_value_policy::reference_internal)
        .def(
            "complex_state_field",
            [](FieldCollection & collection, const std::string & unique_prefix,
               const Index_t & nb_memory, const Index_t & nb_components,
               const std::string & sub_division_tag, const muGrid::Unit & unit)
                -> muGrid::TypedStateField<Complex> & {
                return collection.complex_state_field(unique_prefix, nb_memory,
                                                      nb_components,
                                                      sub_division_tag, unit);
            },
            "unique_prefix"_a, "nb_memory"_a, "nb_components"_a,
            "sub_division_tag"_a = muGrid::PixelTag,
            "unit"_a = muGrid::Unit::unitless(),
            py::return_value_policy::reference_internal)
        .def(
            "int_state_field",
            [](FieldCollection & collection, const std::string & unique_prefix,
               const Index_t & nb_memory, const Index_t & nb_components,
               const std::string & sub_division_tag,
               const muGrid::Unit & unit) -> muGrid::TypedStateField<Int> & {
                return collection.int_state_field(unique_prefix, nb_memory,
                                                  nb_components,
                                                  sub_division_tag, unit);
            },
            "unique_prefix"_a, "nb_memory"_a, "nb_components"_a,
            "sub_division_tag"_a = muGrid::PixelTag,
            "unit"_a = muGrid::Unit::unitless(),
            py::return_value_policy::reference_internal)
        .def(
            "unint_state_field",
            [](FieldCollection & collection, const std::string & unique_prefix,
               const Index_t & nb_memory, const Index_t & nb_components,
               const std::string & sub_division_tag,
               const muGrid::Unit & unit) -> muGrid::TypedStateField<Uint> & {
                return collection.uint_state_field(unique_prefix, nb_memory,
                                                   nb_components,
                                                   sub_division_tag, unit);
            },
            "unique_prefix"_a, "nb_memory"_a, "nb_components"_a,
            "sub_division_tag"_a = muGrid::PixelTag,
            "unit"_a = muGrid::Unit::unitless(),
            py::return_value_policy::reference_internal)
        .def("field_exists", &FieldCollection::field_exists)
        .def("state_field_exists", &FieldCollection::state_field_exists)
        .def_property_readonly("nb_pixels", &FieldCollection::get_nb_pixels)
        .def_property_readonly("nb_pixels_without_ghosts",
                               &FieldCollection::get_nb_pixels_without_ghosts)
        .def(
            "get_nb_sub_pts",
            [](const FieldCollection & coll, const std::string & tag) {
                return coll.get_nb_sub_pts(tag);
            },
            "tag"_a, py::return_value_policy::copy)
        .def("set_nb_sub_pts", &FieldCollection::set_nb_sub_pts, "tag"_a,
             "nb_sub_pts"_a)
        .def_property_readonly("domain", &FieldCollection::get_domain)
        .def_property_readonly("is_initialised",
                               &FieldCollection::is_initialised)
        .def("get_field",
             py::overload_cast<const std::string &>(&FieldCollection::get_field,
                                                    py::const_),
             py::return_value_policy::reference_internal)
        .def(
            "get_real_field",
            [](FieldCollection & collection, const std::string & unique_name)
                -> muGrid::TypedFieldBase<Real> & {
                auto & field{collection.get_field(unique_name)};
                field.assert_typeid(typeid(Real));
                return static_cast<muGrid::TypedFieldBase<Real> &>(field);
            },
            "unique_name"_a, py::return_value_policy::reference_internal)
        .def(
            "get_complex_field",
            [](FieldCollection & collection, const std::string & unique_name)
                -> muGrid::TypedFieldBase<Complex> & {
                auto & field{collection.get_field(unique_name)};
                field.assert_typeid(typeid(Complex));
                return static_cast<muGrid::TypedFieldBase<Complex> &>(field);
            },
            "unique_name"_a, py::return_value_policy::reference_internal)
        .def(
            "get_int_field",
            [](FieldCollection & collection, const std::string & unique_name)
                -> muGrid::TypedFieldBase<Int> & {
                auto & field{collection.get_field(unique_name)};
                field.assert_typeid(typeid(Int));
                return static_cast<muGrid::TypedFieldBase<Int> &>(field);
            },
            "unique_name"_a, py::return_value_policy::reference_internal)
        .def(
            "get_uint_field",
            [](FieldCollection & collection, const std::string & unique_name)
                -> muGrid::TypedFieldBase<Uint> & {
                auto & field{collection.get_field(unique_name)};
                field.assert_typeid(typeid(Uint));
                return static_cast<muGrid::TypedFieldBase<Uint> &>(field);
            },
            "unique_name"_a, py::return_value_policy::reference_internal)
        .def("get_state_field", &FieldCollection::get_state_field,
             py::return_value_policy::reference_internal, "unique_prefix"_a)
        .def("keys", &FieldCollection::list_fields)
        .def_property_readonly("field_names", &FieldCollection::list_fields)
        .def_property_readonly("is_on_device", &FieldCollection::is_on_device)
        .def_property_readonly("memory_location",
                               &FieldCollection::get_memory_location);

    py::class_<muGrid::FieldCollection::IndexIterable>(mod, "IndexIterable")
        .def("__len__", &muGrid::FieldCollection::IndexIterable::size)
        .def("__iter__", [](muGrid::FieldCollection::IndexIterable & iterable) {
            return py::make_iterator(iterable.begin(), iterable.end());
        });

    py::class_<muGrid::FieldCollection::PixelIndexIterable>(
        mod, "PixelIndexIterable")
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

    py::enum_<FieldCollection::MemoryLocation>(field_collection,
                                               "MemoryLocation")
        .value("Host", FieldCollection::MemoryLocation::Host)
        .value("Device", FieldCollection::MemoryLocation::Device)
        .export_values();
}

void add_global_field_collection(py::module & mod) {
    using MemoryLocation = FieldCollection::MemoryLocation;
    py::class_<GlobalFieldCollection, FieldCollection>(mod,
                                                       "GlobalFieldCollection",
        R"pbdoc(
        Field collection for fields defined at all grid points.

        A GlobalFieldCollection manages fields on a structured Cartesian grid.
        All fields in the collection share the same grid dimensions and can
        optionally include ghost layers for domain decomposition in parallel
        computations.

        Parameters
        ----------
        nb_domain_grid_pts : list of int
            Global grid dimensions [Nx, Ny] or [Nx, Ny, Nz]
        nb_subdomain_grid_pts : list of int, optional
            Local subdomain dimensions (for MPI decomposition)
        subdomain_locations : list of int, optional
            Starting indices of the local subdomain in the global grid
        sub_pts : dict, optional
            Mapping of sub-point names to counts (e.g., {"quad": 4})
        storage_order : StorageOrder, optional
            Memory layout for field data (default: ColMajor)
        nb_ghosts_left : list of int, optional
            Ghost layers on low-index side of each dimension
        nb_ghosts_right : list of int, optional
            Ghost layers on high-index side of each dimension
        memory_location : MemoryLocation, optional
            Where to allocate field memory (Host or Device)

        Examples
        --------
        >>> fc = GlobalFieldCollection([64, 64, 64])
        >>> displacement = fc.real_field("displacement", (3,))
        >>> stress = fc.real_field("stress", (3, 3), "quad")
        )pbdoc")
        // Primary constructor: creates and initializes the collection
        // Following Python's "initialization is instantiation" idiom
        .def(py::init<const DynGridIndex &, const DynGridIndex &,
                      const DynGridIndex &, const FieldCollection::SubPtMap_t &,
                      StorageOrder, const DynGridIndex &, const DynGridIndex &,
                      MemoryLocation>(),
             "nb_domain_grid_pts"_a, "nb_subdomain_grid_pts"_a = DynGridIndex{},
             "subdomain_locations"_a = DynGridIndex{},
             "sub_pts"_a = FieldCollection::SubPtMap_t{},
             "storage_order"_a = StorageOrder::ColMajor,
             "nb_ghosts_left"_a = DynGridIndex{},
             "nb_ghosts_right"_a = DynGridIndex{},
             "memory_location"_a = MemoryLocation::Host)
        // Constructor with explicit pixel strides
        .def(
            py::init<const DynGridIndex &, const DynGridIndex &, const DynGridIndex &,
                     const DynGridIndex &, const FieldCollection::SubPtMap_t &,
                     StorageOrder, const DynGridIndex &, const DynGridIndex &,
                     MemoryLocation>(),
            "nb_domain_grid_pts"_a, "nb_subdomain_grid_pts"_a,
            "subdomain_locations"_a, "pixels_strides"_a, "sub_pts"_a,
            "storage_order"_a = StorageOrder::ColMajor,
            "nb_ghosts_left"_a = DynGridIndex{},
            "nb_ghosts_right"_a = DynGridIndex{},
            "memory_location"_a = MemoryLocation::Host)
        // Constructor with explicit pixel storage order
        .def(
            py::init<const DynGridIndex &, const DynGridIndex &, const DynGridIndex &,
                     StorageOrder, const FieldCollection::SubPtMap_t &,
                     StorageOrder, const DynGridIndex &, const DynGridIndex &,
                     MemoryLocation>(),
            "nb_domain_grid_pts"_a, "nb_subdomain_grid_pts"_a,
            "subdomain_locations"_a, "pixels_storage_order"_a, "sub_pts"_a,
            "storage_order"_a = StorageOrder::ColMajor,
            "nb_ghosts_left"_a = DynGridIndex{},
            "nb_ghosts_right"_a = DynGridIndex{},
            "memory_location"_a = MemoryLocation::Host)
        .def_property_readonly("pixels", &GlobalFieldCollection::get_pixels_with_ghosts);
}

void add_local_field_collection(py::module & mod) {
    using MemoryLocation = FieldCollection::MemoryLocation;
    py::class_<LocalFieldCollection, FieldCollection> fc_local(
        mod, "LocalFieldCollection",
        R"pbdoc(
        Field collection for fields defined at a subset of grid points.

        A LocalFieldCollection manages fields that only exist at selected pixels
        rather than the entire grid. This is useful for material properties that
        only apply to certain regions (e.g., inclusion properties in a composite).

        Pixels must be added explicitly using ``add_pixel()`` before calling
        ``initialise()``. After initialization, no new pixels can be added.

        Parameters
        ----------
        spatial_dimension : int
            Number of spatial dimensions (2 or 3)
        name : str, optional
            Name for this collection (for identification)
        nb_sub_pts : dict, optional
            Mapping of sub-point names to counts
        memory_location : MemoryLocation, optional
            Where to allocate field memory (Host or Device)

        Examples
        --------
        >>> lfc = LocalFieldCollection(3, "inclusions")
        >>> lfc.add_pixel(42)  # Add pixel at global index 42
        >>> lfc.add_pixel(100)
        >>> lfc.initialise()
        >>> props = lfc.real_field("elastic_modulus", 1)
        )pbdoc");
    fc_local
        .def(py::init<const Index_t &,
                      const muGrid::FieldCollection::SubPtMap_t &,
                      MemoryLocation>(),
             "spatial_dimension"_a,
             "nb_sub_pts"_a = std::map<std::string, Index_t>{},
             "memory_location"_a = MemoryLocation::Host)
        .def(py::init<const Index_t &, const std::string &,
                      const muGrid::FieldCollection::SubPtMap_t &,
                      MemoryLocation>(),
             "spatial_dimension"_a, "name"_a,
             "nb_sub_pts"_a = std::map<std::string, Index_t>{},
             "memory_location"_a = MemoryLocation::Host)
        .def(
            "add_pixel",
            [](LocalFieldCollection & fc_local, const size_t & global_index) {
                return fc_local.add_pixel(global_index);
            },
            "global_index"_a)
        .def("initialise", &LocalFieldCollection::initialise)
        .def("get_name", &LocalFieldCollection::get_name);
}

void add_field_collection_classes(py::module & mod) {
    add_field_collection(mod);
    add_global_field_collection(mod);
    add_local_field_collection(mod);
}
