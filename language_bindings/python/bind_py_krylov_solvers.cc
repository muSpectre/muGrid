/**
 * @file   bind_py_krylov_solvers.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   04 Sep 2020
 *
 * @brief  python bindings for µSpectre's Krylov solver suite
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
#include "solver/krylov_solver_cg.hh"
#include "solver/krylov_solver_pcg.hh"
#include "solver/krylov_solver_eigen.hh"
#include "solver/krylov_solver_trust_region_cg.hh"
#include "solver/krylov_solver_trust_region_pcg.hh"

#include <pybind11/stl.h>
#include <pybind11/eigen.h>

using muSpectre::Real;
using muSpectre::Uint;
using muSpectre::Verbosity;
using muSpectre::ResetCG;
namespace py = pybind11;
using pybind11::literals::operator""_a;

/**
 * Solvers instanciated for cells with equal spatial and material dimension
 */

void add_enum_reset(py::module & mod) {
  py::enum_<ResetCG>(mod, "ResetCG")
      .value("no_reset", ResetCG::no_reset)
      .value("iter_count", ResetCG::iter_count)
      .value("gradient_orthogonality", ResetCG::gradient_orthogonality)
      .value("valid_direction", ResetCG::valid_direction);
}

template <class KrylovSolver>
void add_krylov_solver_helper(py::module & mod, std::string name) {
  py::class_<KrylovSolver, typename KrylovSolver::Parent,
             std::shared_ptr<KrylovSolver>>(mod, name.c_str())
      .def(py::init<std::shared_ptr<muSpectre::MatrixAdaptable>, const Real &,
                    const Uint &, const Verbosity &>(),
           "system_matrix"_a, "tol"_a, "maxiter"_a,
           "verbose"_a = Verbosity::Silent)
      .def(py::init<const Real &, const Uint &, const Verbosity &>(), "tol"_a,
           "maxiter"_a, "verbose"_a = Verbosity::Silent);
}

template <class KrylovSolver>
void add_preconditioned_krylov_solver_helper(py::module & mod,
                                             std::string name) {
  py::class_<KrylovSolver, typename KrylovSolver::Parent,
             std::shared_ptr<KrylovSolver>>(mod, name.c_str())
      .def(py::init<std::shared_ptr<muSpectre::MatrixAdaptable>,
                    std::shared_ptr<muSpectre::MatrixAdaptable>, const Real &,
                    const Uint &, const Verbosity &>(),
           "system_matrix"_a, "inv_preconditioner"_a, "tol"_a, "maxiter"_a,
           "verbose"_a = Verbosity::Silent)
      .def(py::init<const Real &, const Uint &, const Verbosity &>(), "tol"_a,
           "maxiter"_a, "verbose"_a = Verbosity::Silent);
}

template <class KrylovSolver>
void add_krylov_solver_trust_region_helper(py::module & mod, std::string name) {
  py::class_<KrylovSolver, typename KrylovSolver::Parent,
             std::shared_ptr<KrylovSolver>>(mod, name.c_str())
      .def(py::init<std::shared_ptr<muSpectre::MatrixAdaptable>, const Real &,
                    const Uint &, const Real &, const Verbosity &,
                    const ResetCG &, const Uint &>(),
           "cell"_a, "tol"_a = -1.0, "maxiter"_a = 1000, "trust_region"_a = 1.0,
           "verbose"_a = Verbosity::Silent, "reset"_a = ResetCG::no_reset,
           "reset_iter_count"_a = 0)
      .def(py::init<const Real &, const Uint &, const Real &, const Verbosity &,
                    const ResetCG &, const Uint &>(),
           "tol"_a, "maxiter"_a, "trust_region"_a = 1.0,
           "verbose"_a = Verbosity::Silent, "reset"_a = ResetCG::no_reset,
           "reset_iter_count"_a = 0)
      .def("initialise", &KrylovSolver::initialise)
      .def("solve", &KrylovSolver::solve, "rhs"_a)
      .def("set_trust_region",
           &muSpectre::KrylovSolverTrustRegionCG::set_trust_region,
           "new_trust_region"_a)
      .def_property_readonly("counter", &KrylovSolver::get_counter)
      .def_property_readonly("maxiter", &KrylovSolver::get_maxiter)
      .def_property_readonly("name", &KrylovSolver::get_name)
      .def_property_readonly("tol", &KrylovSolver::get_tol);
}

template <class KrylovSolver>
void add_preconditioned_krylov_solver_trust_region_helper(py::module & mod,
                                                          std::string name) {
  py::class_<KrylovSolver, typename KrylovSolver::Parent,
             std::shared_ptr<KrylovSolver>>(mod, name.c_str())
      .def(py::init<std::shared_ptr<muSpectre::MatrixAdaptable>,
                    std::shared_ptr<muSpectre::MatrixAdaptable>, const Real &,
                    const Uint &, const Real &, const Verbosity &,
                    const ResetCG &, const Uint &>(),
           "cell"_a, "inv_preconditioner"_a, "tol"_a = -1.0, "maxiter"_a = 1000,
           "trust_region"_a = 1.0, "verbose"_a = Verbosity::Silent,
           "reset"_a = ResetCG::no_reset, "reset_iter_count"_a = 0)
      .def(py::init<const Real &, const Uint &, const Real &, const Verbosity &,
                    const ResetCG &, const Uint &>(),
           "tol"_a, "maxiter"_a, "trust_region"_a = 1.0,
           "verbose"_a = Verbosity::Silent, "reset"_a = ResetCG::no_reset,
           "reset_iter_count"_a = 0)
      .def("initialise", &KrylovSolver::initialise)
      .def("solve", &KrylovSolver::solve, "rhs"_a)
      .def("set_trust_region",
           &muSpectre::KrylovSolverTrustRegionPCG::set_trust_region,
           "new_trust_region"_a)
      .def_property_readonly("counter", &KrylovSolver::get_counter)
      .def_property_readonly("maxiter", &KrylovSolver::get_maxiter)
      .def_property_readonly("name", &KrylovSolver::get_name)
      .def_property_readonly("tol", &KrylovSolver::get_tol);
}

void add_krylov_solver(py::module & mod) {
  py::class_<muSpectre::MatrixAdaptable,
             std::shared_ptr<muSpectre::MatrixAdaptable>>(mod,
                                                          "MatrixAdaptable");
  std::stringstream name{};
  name << "KrylovSolverBase";
  py::class_<muSpectre::KrylovSolverBase,
             std::shared_ptr<muSpectre::KrylovSolverBase>>(mod,
                                                           name.str().c_str())
      .def_property_readonly("name", &muSpectre::KrylovSolverBase::get_name)
      .def_property_readonly("counter",
                             &muSpectre::KrylovSolverBase::get_counter)
      .def("initialise", &muSpectre::KrylovSolverBase::initialise)
      .def(
          "set_matrix",
          [](muSpectre::KrylovSolverBase & solver,
             std::shared_ptr<muSpectre::MatrixAdaptable>
                 system_matrix_adaptable) -> void {
            solver.set_matrix(system_matrix_adaptable);
          },
          "system_matrix_adaptable"_a)
      .def("solve", &muSpectre::KrylovSolverBase::solve);

  py::class_<muSpectre::KrylovSolverTrustRegionFeatures,
             std::shared_ptr<muSpectre::KrylovSolverTrustRegionFeatures>>(
      mod, "KrylovSolverTrustRegionFeatures");

  class PyKrylovSolverPreconditionedFeatures
      : public muSpectre::KrylovSolverPreconditionedFeatures {
   public:
    using Parent = muSpectre::KrylovSolverPreconditionedFeatures;
    PyKrylovSolverPreconditionedFeatures(
        std::shared_ptr<muSpectre::MatrixAdaptable> preconditioner_adaptable)
        : Parent(preconditioner_adaptable) {}
    void set_preconditioner(std::shared_ptr<muSpectre::MatrixAdaptable>
                                preconditioner_adaptable) override {
      PYBIND11_OVERLOAD_PURE(void, Parent, set_precondtioner,
                             preconditioner_adaptable);
    }
  };

  py::class_<muSpectre::KrylovSolverPreconditionedFeatures,
             std::shared_ptr<muSpectre::KrylovSolverPreconditionedFeatures>,
             PyKrylovSolverPreconditionedFeatures>(
      mod, "KrylovSolverPreconditionedFeatures")
      .def("set_preconditioner",
           &muSpectre::KrylovSolverPreconditionedFeatures::set_preconditioner,
           "inv_preconditioner_matrix_adaptable"_a);

  // py::class_<muSpectre::KrylovSolverPreconditionedBase,
  //            muSpectre::KrylovSolverBase,
  //            std::shared_ptr<muSpectre::KrylovSolverPreconditionedBase>>(
  //     mod, "KrylovSolverPreconditionedBase")
  //     .def("set_preconditioner",
  //          &muSpectre::KrylovSolverPreconditionedBase::set_preconditioner,
  //          "inv_preconditioner_matrix_adaptable"_a);

  add_enum_reset(mod);
  add_krylov_solver_helper<muSpectre::KrylovSolverCG>(mod, "KrylovSolverCG");
  add_krylov_solver_helper<muSpectre::KrylovSolverCGEigen>(
      mod, "KrylovSolverCGEigen");
  add_krylov_solver_helper<muSpectre::KrylovSolverGMRESEigen>(
      mod, "KrylovSolverGMRESEigen");
  add_krylov_solver_helper<muSpectre::KrylovSolverBiCGSTABEigen>(
      mod, "KrylovSolverBiCGSTABEigen");
  add_krylov_solver_helper<muSpectre::KrylovSolverDGMRESEigen>(
      mod, "KrylovSolverDGMRESEigen");
  add_krylov_solver_helper<muSpectre::KrylovSolverMINRESEigen>(
      mod, "KrylovSolverMINRESEigen");
  add_preconditioned_krylov_solver_helper<muSpectre::KrylovSolverPCG>(
      mod, "KrylovSolverPCG");
  add_krylov_solver_trust_region_helper<muSpectre::KrylovSolverTrustRegionCG>(
      mod, "KrylovSolverTrustRegionCG");
  add_preconditioned_krylov_solver_trust_region_helper<
      muSpectre::KrylovSolverTrustRegionPCG>(mod, "KrylovSolverTrustRegionPCG");
}

void add_krylov_solvers(py::module & mod) { add_krylov_solver(mod); }
