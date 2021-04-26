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
#include "solver/solver_trust_region_newton_cg.cc"
#include "solver/solver_fem_trust_region_newton_cg.cc"
#include "solver/solver_fem_trust_region_newton_pcg.cc"

#include <libmugrid/numpy_tools.hh>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

using muSpectre::Index_t;
using muSpectre::Real;
using muSpectre::Uint;
using muSpectre::Verbosity;

using pybind11::literals::operator""_a;

using muFFT::Gradient_t;
using muSpectre::CellData;
using muSpectre::SolverBase;
using muSpectre::SolverFEMNewtonCG;
using muSpectre::SolverFEMNewtonPCG;
using muSpectre::SolverNewtonCG;
using SolverTRNewtonCG = muSpectre::SolverTrustRegionNewtonCG;
using SolverFEMTRNewtonCG = muSpectre::SolverFEMTrustRegionNewtonCG;
using SolverFEMTRNewtonPCG = muSpectre::SolverFEMTrustRegionNewtonPCG;
using muSpectre::SolverSinglePhysics;

class PySolverBase : public SolverBase {
 public:
  using Parent = SolverBase;

  PySolverBase(std::shared_ptr<CellData> cell_data,
               const muGrid::Verbosity & verbosity)
      : Parent(cell_data, verbosity) {}

  void initialise_cell() override {
    PYBIND11_OVERLOAD_PURE(void, SolverBase, initialise_cell);
  }

  muSpectre::OptimizeResult
  solve_load_increment(const LoadStep & load_step,
                       EigenStrainFunc_ref eigen_strain_func,
                       CellExtractFieldFunc_ref cell_extract_func) override {
    PYBIND11_OVERLOAD_PURE(muSpectre::OptimizeResult, SolverBase,
                           solve_load_increment, load_step, eigen_strain_func,
                           cell_extract_func);
  }

  // using muSpectre::MatrixAdaptable::MatrixAdaptable;
  Index_t get_nb_dof() const override {
    PYBIND11_OVERLOAD_PURE(Index_t, SolverBase, get_nb_dof);
  }
  void action_increment(EigenCVec_t delta_grad, const Real & alpha,
                        EigenVec_t del_flux) override {
    PYBIND11_OVERLOAD_PURE(void, SolverBase, solve_load_increment, delta_grad,
                           alpha, del_flux);
  }

  const muGrid::Communicator & get_communicator() const override {
    PYBIND11_OVERLOAD_PURE(const muGrid::Communicator &, SolverBase,
                           get_communicator);
  }
};

void add_solver_base(py::module & mod) {
  py::class_<SolverBase, muSpectre::MatrixAdaptable,
             std::shared_ptr<SolverBase>, PySolverBase>(mod, "SolverBase")
      // .def(py::init<>())
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
  using EigenStrainFunc_t = std::function<void(muGrid::TypedFieldBase<Real> &)>;
  using CellExtractFunc_t =
      std::function<void(const std::shared_ptr<muSpectre::CellData>)>;
  py::class_<SolverSinglePhysics, SolverBase,
             std::shared_ptr<SolverSinglePhysics>>(mod, "SolverSinglePhysics")
      .def(
          "solve_load_increment",
          [](SolverSinglePhysics & solver,
             py::EigenDRef<Eigen::MatrixXd> load_step,
             py::function & eigen_strain_pyfunc,
             py::function & cell_extract_pyfunc) {
            EigenStrainFunc_t eigen_strain_cpp_func{
                [&eigen_strain_pyfunc](
                    muGrid::TypedFieldBase<Real> & eigen_strain_field) {
                  eigen_strain_pyfunc(muGrid::numpy_wrap(
                      eigen_strain_field, muGrid::IterUnit::SubPt));
                }};
            CellExtractFunc_t cell_extract_cpp_func{
                [&cell_extract_pyfunc, &solver](
                    const std::shared_ptr<muSpectre::CellData> cell_data) {
                  cell_extract_pyfunc(solver, cell_data);
                }};
            return solver.solve_load_increment(load_step, eigen_strain_cpp_func,
                                               cell_extract_cpp_func);
          },
          "load_step"_a, "eigen_strain_func"_a = nullptr,
          "cell_extract_func"_a = nullptr)
      .def(
          "solve_load_increment",
          [](SolverSinglePhysics & solver,
             py::EigenDRef<Eigen::MatrixXd> load_step,
             py::function & eigen_strain_pyfunc) {
            EigenStrainFunc_t eigen_strain_cpp_func{
                [&eigen_strain_pyfunc](
                    muGrid::TypedFieldBase<Real> & eigen_strain_field) {
                  eigen_strain_pyfunc(muGrid::numpy_wrap(
                      eigen_strain_field, muGrid::IterUnit::SubPt));
                }};
            return solver.solve_load_increment(load_step,
                                               eigen_strain_cpp_func);
          },
          "load_step"_a, "eigen_strain_func"_a = nullptr)
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
      .def("evaluate_stress_tangent",
           [](SolverSinglePhysics & solver) {
             solver.evaluate_stress_tangent();
           })
      .def_property_readonly("grad", &SolverSinglePhysics::get_grad)
      .def_property_readonly("flux", &SolverSinglePhysics::get_flux)
      .def_property_readonly("tangent", &SolverSinglePhysics::get_tangent);
}

void add_spectral_newton_cg_solver(py::module & mod) {
  py::class_<SolverNewtonCG, SolverSinglePhysics,
             std::shared_ptr<SolverNewtonCG>>(mod, "SolverNewtonCG")
      .def(py::init<std::shared_ptr<CellData>,
                    std::shared_ptr<muSpectre::KrylovSolverBase>,
                    const Verbosity &, const Real &, const Real &, const Uint &,
                    const Gradient_t &>(),
           "cell_data"_a, "krylov_solver"_a, "verbosity"_a, "newton_tol"_a,
           "equil_tol"_a, "max_iter"_a, "gradient"_a)
      .def(py::init<std::shared_ptr<CellData>,
                    std::shared_ptr<muSpectre::KrylovSolverBase>,
                    const Verbosity &, const Real &, const Real &,
                    const Uint &>(),
           "cell_data"_a, "krylov_solver"_a, "verbosity"_a, "newton_tol"_a,
           "equil_tol"_a, "max_iter"_a)

      .def_property_readonly("projection", &SolverNewtonCG::get_projection)
      .def_property_readonly("flux", &SolverNewtonCG::get_flux)
      .def_property_readonly("grad", &SolverNewtonCG::get_grad)
      .def_property_readonly("eval_grad", &SolverNewtonCG::get_eval_grad)
      .def_property_readonly("tangent", &SolverNewtonCG::get_tangent)
      .def_property_readonly("nb_dof", &SolverNewtonCG::get_nb_dof)
      .def_property_readonly("projection", &SolverNewtonCG::get_projection);
}

void add_spectral_trust_region_newton_cg_solver(py::module & mod) {
  py::class_<SolverTRNewtonCG, SolverSinglePhysics,
             std::shared_ptr<SolverTRNewtonCG>>(mod, "SolverTRNewtonCG")
      .def(py::init<std::shared_ptr<CellData>,
                    std::shared_ptr<muSpectre::KrylovSolverTrustRegionCG>,
                    const Verbosity &, const Real &, const Real &, const Uint &,
                    const Real &, const Real &>(),
           "cell_data"_a, "krylov_solver"_a, "verbosity"_a, "newton_tol"_a,
           "equil_tol"_a, "max_iter"_a, "trust_region_max"_a, "eta"_a)
      .def(py::init<std::shared_ptr<CellData>,
                    std::shared_ptr<muSpectre::KrylovSolverTrustRegionCG>,
                    const Verbosity &, const Real &, const Real &, const Uint &,
                    const Real &, const Real &, const Gradient_t &>(),
           "cell_data"_a, "krylov_solver"_a, "verbosity"_a, "newton_tol"_a,
           "equil_tol"_a, "max_iter"_a, "trust_region_max"_a, "eta"_a,
           "gradient"_a)
      .def_property_readonly("nb_dof", &SolverTRNewtonCG::get_nb_dof)
      .def_property_readonly("projection", &SolverTRNewtonCG::get_projection);
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
                             py::return_value_policy::reference_internal)
      .def_property_readonly("disp", &SolverFEMNewtonCG::get_disp_fluctuation,
                             py::return_value_policy::reference_internal);

  py::class_<SolverFEMNewtonPCG, SolverFEMNewtonCG,
             std::shared_ptr<SolverFEMNewtonPCG>>(mod, "SolverFEMNewtonPCG")
      .def(py::init<std::shared_ptr<muSpectre::Discretisation>,
                    std::shared_ptr<muSpectre::KrylovSolverPCG>,
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

void add_fem_trust_region_newton_cg_solver(py::module & mod) {
  py::class_<SolverFEMTRNewtonCG, SolverSinglePhysics,
             std::shared_ptr<SolverFEMTRNewtonCG>>(mod, "SolverFEMTRNewtonCG")
      .def(py::init<std::shared_ptr<muSpectre::Discretisation>,
                    std::shared_ptr<muSpectre::KrylovSolverTrustRegionCG>,
                    const Verbosity &, const Real &, const Real &, const Uint &,
                    const Real &, const Real &>(),
           "discretisation"_a, "krylov_solver"_a, "verbosity"_a, "newton_tol"_a,
           "equil_tol"_a, "max_iter"_a, "trust_region_max"_a, "eta"_a)
      .def_property_readonly("displacement_rank",
                             &SolverFEMTRNewtonCG::get_displacement_rank)
      .def_property_readonly("eval_grad", &SolverFEMTRNewtonCG::get_eval_grad,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("flux", &SolverFEMTRNewtonCG::get_flux,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("tangent", &SolverFEMTRNewtonCG::get_tangent,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("disp", &SolverFEMTRNewtonCG::get_disp_fluctuation,
                             py::return_value_policy::reference_internal);

  py::class_<SolverFEMTRNewtonPCG, SolverFEMTRNewtonCG,
             std::shared_ptr<SolverFEMTRNewtonPCG>>(mod, "SolverFEMTRNewtonPCG")
      .def(py::init<std::shared_ptr<muSpectre::Discretisation>,
                    std::shared_ptr<muSpectre::KrylovSolverTrustRegionPCG>,
                    const Verbosity &, const Real &, const Real &, const Uint &,
                    const Real &, const Real &>(),
           "discretisation"_a, "krylov_solver"_a, "verbosity"_a, "newton_tol"_a,
           "equil_tol"_a, "max_iter"_a, "trust_region_max"_a, "eta"_a)
      .def(
          "set_reference_material",
          [](SolverFEMTRNewtonPCG & solver,
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
  add_spectral_trust_region_newton_cg_solver(mod);
  add_fem_trust_region_newton_cg_solver(mod);
}
