#include "bind_py_declarations.hh"

#include "libmugrid/decomposition.hh"
#include "libmugrid/cartesian_decomposition.hh"
#include "libmugrid/field.hh"
#include "libmugrid/python_helpers.hh"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

using muGrid::Decomposition;
using muGrid::CartesianDecomposition;
using muGrid::Communicator;
using muGrid::Int;
using muGrid::Index_t;
using muGrid::IntCoord_t;
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
};

// Bind abstract class Decomposition
void add_decomposition(py::module & mod) {
    py::class_<Decomposition, PyDecomposition>(mod, "Decomposition")
        .def(py::init<>())
        .def(
            "communicate_ghosts",
            [](const Decomposition & self, const Field & field) {
                self.communicate_ghosts(field);
            },
            "field"_a)
        .def(
            "communicate_ghosts",
            [](const Decomposition & self, std::string field_name) {
                self.communicate_ghosts(field_name);
            },
            "field_name"_a);
}

// Bind class Cartesian Decomposition
void add_cartesian_decomposition(py::module & mod) {
    py::class_<CartesianDecomposition, Decomposition>(mod,
                                                      "CartesianDecomposition")
        .def(py::init<const Communicator &, const IntCoord_t &,
                      const IntCoord_t &, const IntCoord_t &,
                      const IntCoord_t &>(),
             "comm"_a, "nb_domain_grid_pts"_a, "nb_subdivisions"_a,
             "nb_ghosts_left"_a, "nb_ghosts_right"_a)
        .def_property_readonly(
            "collection",
            py::overload_cast<>(&CartesianDecomposition::get_collection,
                                py::const_))
        .def_property_readonly("nb_subdivisions",
                               &CartesianDecomposition::get_nb_subdivisions)
        .def_property_readonly("nb_domain_grid_pts",
                               &CartesianDecomposition::get_nb_domain_grid_pts)
        .def_property_readonly(
            "nb_subdomain_grid_pts",
            &CartesianDecomposition::get_nb_subdomain_grid_pts_without_ghosts)
        .def_property_readonly(
            "subdomain_locations",
            &CartesianDecomposition::get_subdomain_locations_without_ghosts)
        .def_property_readonly(
            "nb_subdomain_grid_pts_with_ghosts",
            &CartesianDecomposition::get_nb_subdomain_grid_pts_with_ghosts)
        .def_property_readonly(
            "subdomain_locations_with_ghosts",
            &CartesianDecomposition::get_subdomain_locations_with_ghosts)
        .def_property_readonly("coords",
                               [](const CartesianDecomposition & self) {
                                   return py_coords<Real, false>(self);
                               })
        .def_property_readonly("icoords",
                               [](const CartesianDecomposition & self) {
                                   return py_coords<Int, false>(self);
                               })
        .def_property_readonly("coordsg",
                               [](const CartesianDecomposition & self) {
                                   return py_coords<Real, true>(self);
                               })
        .def_property_readonly("icoordsg",
                               [](const CartesianDecomposition & self) {
                                   return py_coords<Int, true>(self);
                               });
}

// Combined binding function
void add_decomposition_classes(py::module & mod) {
    add_decomposition(mod);
    add_cartesian_decomposition(mod);
}
