#include "bind_py_declarations.hh"

#include "libmugrid/decomposition.hh"
#include "libmugrid/cartesian_decomposition.hh"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

using muGrid::Decomposition;
using muGrid::CartesianDecomposition;
using muGrid::Communicator;
using muGrid::Int;
using muGrid::Index_t;
using muGrid::DynCcoord_t;
using Pixels_t = muGrid::CcoordOps::DynamicPixels;
using pybind11::literals::operator""_a;

namespace py = pybind11;

// A helper class that bounces calls to virtual methods back to Python
class PyDecomposition : public Decomposition {
 public:
  // Inherit the constructors
  using Decomposition::Decomposition;

  // Trampoline (one for each virtual function)

  void communicate_ghosts(std::string field_name) const override {
    PYBIND11_OVERRIDE_PURE(void, Decomposition, communicate_ghosts, field_name);
  }
};

// Bind abstract class Decomposition
void add_decomposition(py::module & mod) {
  py::class_<Decomposition, PyDecomposition>(mod, "Decomposition")
      .def(py::init<>())
      .def("communicate_ghosts", &Decomposition::communicate_ghosts,
           "field_name"_a);
}

// Bind class Cartesian Decomposition
void add_cartesian_decomposition(py::module & mod) {
  py::class_<CartesianDecomposition, Decomposition>(mod,
                                                    "CartesianDecomposition")
      .def(py::init<const Communicator &, const DynCcoord_t &,
                    const DynCcoord_t &, const DynCcoord_t &,
                    const DynCcoord_t &>(),
           "comm"_a, "nb_domain_grid_pts"_a, "nb_subdivisions"_a,
           "nb_ghost_left"_a, "nb_ghost_right"_a)
      .def("communicate_ghosts", &CartesianDecomposition::communicate_ghosts,
           "field_name"_a)
      .def_property_readonly("collection",
                             &CartesianDecomposition::get_collection)
      .def_property_readonly("nb_subdivisions",
                             &CartesianDecomposition::get_nb_subdivisions)
      .def_property_readonly("nb_domain_grid_pts",
                             &CartesianDecomposition::get_nb_domain_grid_pts)
      .def_property_readonly("nb_subdomain_grid_pts",
                             &CartesianDecomposition::get_nb_subdomain_grid_pts)
      .def_property_readonly("subdomain_locations",
                             &CartesianDecomposition::get_subdomain_locations)
      .def_property_readonly("global_coords", [](CartesianDecomposition & cart_decomp) {
        // Create a NumPy array with shape = (dim, nx, ny, ...)
        std::vector<Index_t> shape{};
        const auto & nb_subdomain_grid_pts{
            cart_decomp.get_nb_subdomain_grid_pts()};
        auto dim{nb_subdomain_grid_pts.size()};
        shape.push_back(dim);
        for (auto nb : nb_subdomain_grid_pts) {
          shape.push_back(nb);
        }
        py::array_t<Int, py::array::f_style> coords(shape);

        // Get necessary information
        const auto & nb_domain_grid_pts{cart_decomp.get_nb_domain_grid_pts()};
        const auto & subdomain_locations{cart_decomp.get_subdomain_locations()};
        auto * ptr{static_cast<Int *>(coords.request().ptr)};

        // Fill the array with global coordinates
        Pixels_t pixels{nb_subdomain_grid_pts};
        for (auto pixel_id_coords : pixels.enumerate()) {
          auto local_coords{std::get<1>(pixel_id_coords)};
          auto global_coords{(subdomain_locations + local_coords) %
                             nb_domain_grid_pts};
          for (int i{0}; i < dim; ++i) {
            *ptr = global_coords[i];
            ptr++;
          }
        }
        return coords;
      });
}

// Combined binding function
void add_decomposition_classes(py::module & mod) {
  add_decomposition(mod);
  add_cartesian_decomposition(mod);
}
