/**
 * @file   bind_py_class_solvers.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   04 Sep 2020
 *
 * @brief  binding for the new solver classes
 *
 * Copyright © 2020 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 *
 */

#include "solver/matrix_adaptor.hh"
#include "solver/solver_newton_cg.hh"
#include "solver/solver_fem_newton_cg.hh"
#include "solver/solver_fem_newton_pcg.hh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

using muSpectre::Index_t;
using muSpectre::Real;
using muSpectre::Uint;
using muSpectre::Verbosity;

using pybind11::literals::operator""_a;

using muSpectre::CellData;
using muSpectre::SolverBase;
using muSpectre::SolverFEMNewtonCG;
using muSpectre::SolverFEMNewtonPCG;
using muSpectre::SolverNewtonCG;
using muSpectre::SolverSinglePhysics;

void add_solver_base(py::module & mod) {
  py::class_<SolverBase, muSpectre::MatrixAdaptable,
             std::shared_ptr<SolverBase>>(mod, "SolverBase")
      .def_property_readonly("counter",
                             [](const SolverBase & solver_base) {
                               return solver_base.get_counter();
                             })
      .def_property("formulation", &SolverBase::get_formulation,
                    &SolverBase::set_formulation)
      .def("initialise_cell", &SolverBase::initialise_cell)
      .def_property_readonly("communicator", &SolverBase::get_communicator);
}

void add_single_physics_solver(py::module & mod) {
  py::class_<SolverSinglePhysics, SolverBase,
             std::shared_ptr<SolverSinglePhysics>>(mod, "SolverSinglePhysics")
      .def(
          "solve_load_increment",
          [](SolverSinglePhysics & solver,
             py::EigenDRef<Eigen::MatrixXd> load_step) {
            return solver.solve_load_increment(load_step);
          },
          "load_step"_a)
      .def_property_readonly("is_mechanic", &SolverSinglePhysics::is_mechanics)
      .def("evaluate_stress",
           [](SolverSinglePhysics & solver) { solver.evaluate_stress(); })
      .def("evaluate_stress_tangent", [](SolverSinglePhysics & solver) {
        solver.evaluate_stress_tangent();
      });
}

void add_spectral_newton_cg_solver(py::module & mod) {
  py::class_<SolverNewtonCG, SolverSinglePhysics,
             std::shared_ptr<SolverNewtonCG>>(mod, "SolverNewtonCG")
      .def(py::init<std::shared_ptr<CellData>,
                    std::shared_ptr<muSpectre::KrylovSolverBase>,
                    const Verbosity &, const Real &, const Real &,
                    const Uint &>(),
           "cell_data"_a, "krylov_solver"_a, "verbosity"_a, "newton_tol"_a,
           "equil_tol"_a, "max_iter"_a)
      .def_property_readonly("nb_dof", &SolverNewtonCG::get_nb_dof);
}

void add_fem_newton_cg_solver(py::module & mod) {
  py::class_<SolverFEMNewtonCG, SolverSinglePhysics,
             std::shared_ptr<SolverFEMNewtonCG>>(mod, "SolverFEMNewtonCG")
      .def(py::init<std::shared_ptr<muSpectre::Discretisation>,
                    std::shared_ptr<muSpectre::KrylovSolverBase>,
                    const Verbosity &, const Real &, const Real &,
                    const Uint &>(),
           "discretisation"_a, "krylov_solver"_a, "verbosity"_a, "newton_tol"_a,
           "equil_tol"_a, "max_iter"_a)
      .def_property_readonly("displacement_rank",
                             &SolverFEMNewtonCG::get_displacement_rank)
      .def_property_readonly("eval_grad", &SolverFEMNewtonCG::get_eval_grad,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("flux", &SolverFEMNewtonCG::get_flux,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("tangent", &SolverFEMNewtonCG::get_tangent,
                             py::return_value_policy::reference_internal);

  py::class_<SolverFEMNewtonPCG, SolverFEMNewtonCG,
             std::shared_ptr<SolverFEMNewtonPCG>>(mod, "SolverFEMNewtonPCG")
      .def(py::init<std::shared_ptr<muSpectre::Discretisation>,
                    std::shared_ptr<muSpectre::KrylovSolverPreconditionedBase>,
                    const Verbosity &, const Real &, const Real &,
                    const Uint &>(),
           "discretisation"_a, "krylov_solver"_a, "verbosity"_a, "newton_tol"_a,
           "equil_tol"_a, "max_iter"_a)
      .def(
          "set_reference_material",
          [](SolverFEMNewtonPCG & solver,
             py::EigenDRef<Eigen::MatrixXd> material_properties) {
            solver.set_reference_material(material_properties);
          },
          "material_properties"_a);
}

void add_class_solvers(py::module & mod) {
  add_solver_base(mod);
  add_single_physics_solver(mod);
  add_spectral_newton_cg_solver(mod);
  add_fem_newton_cg_solver(mod);
}
