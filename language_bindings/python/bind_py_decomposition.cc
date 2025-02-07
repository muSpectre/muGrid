#include "bind_py_declarations.hh"

#include "libmugrid/decomposition.hh"
#include "libmugrid/cartesian_decomposition.hh"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

using muGrid::Decomposition;
using muGrid::CartesianDecomposition;
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
                    const DynCcoord_t &, const SubPtMap_t &>(),
           " comm"_a, "nb_domain_grid_pts"_a, "nb_subdivisions"_a,
           "nb_ghost_left"_a, "nb_ghost_right"_a, "nb_sub_pts"_a = {})
      .def("communicate_ghosts", &CartesianDecomposition::communicate_ghosts,
           "field_name"_a);
}

// Combined binding function
void add_decomposition_classses(py::module & mod) {
  add_decomposition(mod);
  add_cartesian_decomposition(mod);
}
