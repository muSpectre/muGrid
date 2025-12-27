#include "bind_py_declarations.hh"

#include "mpi/decomposition.hh"
#include "mpi/cartesian_decomposition.hh"
#include "field/field.hh"
#include "collection/field_collection.hh"
#include "util/python_helpers.hh"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

using muGrid::Decomposition;
using muGrid::CartesianDecomposition;
using muGrid::Communicator;
using muGrid::Int;
using muGrid::Index_t;
using muGrid::DynGridIndex;
using muGrid::Real;
using muGrid::Field;
using muGrid::py_coords;
using Pixels_t = muGrid::CcoordOps::Pixels;
using pybind11::literals::operator""_a;

namespace py = pybind11;

// A helper class that bounces calls to virtual methods back to Python
class PyDecomposition : public Decomposition {
   public:
    // Inherit the constructors
    using Decomposition::Decomposition;

    // Trampoline (one for each virtual function)

    void communicate_ghosts(const Field & field) const override {
        PYBIND11_OVERRIDE_PURE(void, Decomposition, communicate_ghosts, field);
    }

    void communicate_ghosts(const std::string & field_name) const override {
        PYBIND11_OVERRIDE_PURE(void, Decomposition, communicate_ghosts,
                               field_name);
    }

    void reduce_ghosts(const Field & field) const override {
        PYBIND11_OVERRIDE_PURE(void, Decomposition, reduce_ghosts, field);
    }

    void reduce_ghosts(const std::string & field_name) const override {
        PYBIND11_OVERRIDE_PURE(void, Decomposition, reduce_ghosts, field_name);
    }
};

// Bind abstract class Decomposition
void add_decomposition(py::module & mod) {
    py::class_<Decomposition, PyDecomposition>(mod, "Decomposition",
        R"pbdoc(
        Abstract base class for domain decomposition strategies.

        A Decomposition handles the distribution of grid points across MPI ranks
        and the communication of ghost (halo) values between neighboring ranks.

        This is an abstract class - use ``CartesianDecomposition`` for the
        concrete implementation.

        See Also
        --------
        CartesianDecomposition : Cartesian domain decomposition with ghost handling
        )pbdoc")
        .def(py::init<>())
        .def(
            "communicate_ghosts",
            [](const Decomposition & self, const Field & field) {
                self.communicate_ghosts(field);
            },
            "field"_a,
            R"pbdoc(
            Fill ghost buffers with values from neighboring subdomains.

            This copies boundary values from neighboring ranks into the local
            ghost regions, enabling stencil operations that access neighbor values.

            Parameters
            ----------
            field : Field
                The field whose ghost values should be updated
            )pbdoc")
        .def(
            "communicate_ghosts",
            [](const Decomposition & self, std::string field_name) {
                self.communicate_ghosts(field_name);
            },
            "field_name"_a,
            "Fill ghost buffers by field name.")
        .def(
            "reduce_ghosts",
            [](const Decomposition & self, const Field & field) {
                self.reduce_ghosts(field);
            },
            "field"_a,
            R"pbdoc(
            Accumulate ghost buffer contributions back to interior domain.

            This is the adjoint (transpose) of ``communicate_ghosts()``. It sums
            contributions from ghost regions back into the interior domain values.
            Used for transpose operations like divergence (adjoint of gradient).

            Parameters
            ----------
            field : Field
                The field whose ghost contributions should be accumulated
            )pbdoc")
        .def(
            "reduce_ghosts",
            [](const Decomposition & self, std::string field_name) {
                self.reduce_ghosts(field_name);
            },
            "field_name"_a,
            "Accumulate ghost contributions by field name.");
}

// Bind class Cartesian Decomposition
void add_cartesian_decomposition(py::module & mod) {
    using MemoryLocation = muGrid::FieldCollection::MemoryLocation;
    py::class_<CartesianDecomposition, Decomposition>(mod,
                                                      "CartesianDecomposition",
        R"pbdoc(
        Cartesian domain decomposition with ghost layer handling.

        CartesianDecomposition partitions a global grid into subdomains for
        parallel processing with MPI. Each rank owns a rectangular subdomain
        and can have ghost (halo) layers for accessing neighboring values.

        This class owns a GlobalFieldCollection and provides methods to register
        fields and communicate ghost values between MPI ranks.

        Parameters
        ----------
        comm : Communicator
            MPI communicator
        nb_domain_grid_pts : list of int
            Global grid dimensions [Nx, Ny] or [Nx, Ny, Nz]
        nb_subdivisions : list of int
            Number of subdivisions in each direction for domain decomposition
        nb_ghosts_left : list of int
            Ghost layers on low-index side of each dimension
        nb_ghosts_right : list of int
            Ghost layers on high-index side of each dimension
        sub_pts : dict, optional
            Mapping of sub-point names to counts (e.g., {"quad": 4})
        memory_location : MemoryLocation, optional
            Where to allocate field memory (Host or Device)

        Attributes
        ----------
        collection : GlobalFieldCollection
            The underlying field collection
        coords : tuple of ndarray
            Physical coordinates at each pixel (excluding ghosts)
        coordsg : tuple of ndarray
            Physical coordinates including ghost pixels
        icoords : tuple of ndarray
            Integer indices at each pixel (excluding ghosts)
        icoordsg : tuple of ndarray
            Integer indices including ghost pixels
        nb_domain_grid_pts : list of int
            Global grid dimensions
        nb_subdomain_grid_pts : list of int
            Local subdomain dimensions (excluding ghosts)
        subdomain_locations : list of int
            Starting indices of local subdomain in global grid

        Examples
        --------
        >>> from muGrid import Communicator, CartesianDecomposition
        >>> comm = Communicator()
        >>> decomp = CartesianDecomposition(
        ...     comm, [64, 64, 64], [1, 1, 1], [1, 1, 1], [1, 1, 1]
        ... )
        >>> field = decomp.collection.real_field("temperature")
        >>> decomp.communicate_ghosts(field)
        )pbdoc")
        .def(py::init<const Communicator &, const DynGridIndex &,
                      const DynGridIndex &, const DynGridIndex &,
                      const DynGridIndex &,
                      const muGrid::FieldCollection::SubPtMap_t &,
                      MemoryLocation>(),
             "comm"_a, "nb_domain_grid_pts"_a, "nb_subdivisions"_a,
             "nb_ghosts_left"_a, "nb_ghosts_right"_a,
             "sub_pts"_a = muGrid::FieldCollection::SubPtMap_t{},
             "memory_location"_a = MemoryLocation::Host)
        .def_property_readonly(
            "collection",
            py::overload_cast<>(&CartesianDecomposition::get_collection,
                                py::const_),
            "Get the underlying GlobalFieldCollection.")
        .def_property_readonly("nb_subdivisions",
                               &CartesianDecomposition::get_nb_subdivisions,
                               "Number of subdivisions in each direction.")
        .def_property_readonly("nb_domain_grid_pts",
                               &CartesianDecomposition::get_nb_domain_grid_pts,
                               "Global grid dimensions.")
        .def_property_readonly(
            "nb_subdomain_grid_pts",
            &CartesianDecomposition::get_nb_subdomain_grid_pts_without_ghosts,
            "Local subdomain dimensions (excluding ghosts).")
        .def_property_readonly(
            "subdomain_locations",
            &CartesianDecomposition::get_subdomain_locations_without_ghosts,
            "Starting indices of local subdomain in global grid.")
        .def_property_readonly(
            "nb_subdomain_grid_pts_with_ghosts",
            &CartesianDecomposition::get_nb_subdomain_grid_pts_with_ghosts,
            "Local subdomain dimensions (including ghosts).")
        .def_property_readonly(
            "subdomain_locations_with_ghosts",
            &CartesianDecomposition::get_subdomain_locations_with_ghosts,
            "Starting indices including ghost offset.")
        .def_property_readonly("coords",
                               [](const CartesianDecomposition & self) {
                                   return py_coords<Real, false>(self);
                               },
                               "Physical coordinates (excluding ghosts).")
        .def_property_readonly("icoords",
                               [](const CartesianDecomposition & self) {
                                   return py_coords<Int, false>(self);
                               },
                               "Integer indices (excluding ghosts).")
        .def_property_readonly("coordsg",
                               [](const CartesianDecomposition & self) {
                                   return py_coords<Real, true>(self);
                               },
                               "Physical coordinates (including ghosts).")
        .def_property_readonly("icoordsg",
                               [](const CartesianDecomposition & self) {
                                   return py_coords<Int, true>(self);
                               },
                               "Integer indices (including ghosts).")
        .def_property_readonly("is_on_device",
                               &CartesianDecomposition::is_on_device,
                               "True if field data is on GPU device.")
        .def_property_readonly("memory_location",
                               &CartesianDecomposition::get_memory_location,
                               "Memory location (Host or Device).");
}

// Combined binding function
void add_decomposition_classes(py::module & mod) {
    add_decomposition(mod);
    add_cartesian_decomposition(mod);
}
