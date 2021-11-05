/**
 * @file   bind_py_solvers.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   09 Jan 2018
 *
 * @brief  python bindings for the muSpectre solvers
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
 * Lesser General Public License for more details.
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
 *
 */

#include "bind_py_declarations.hh"

#include "common/muSpectre_common.hh"
#include "solver/matrix_adaptor.hh"
#include "solver/solvers.hh"
#include "solver/krylov_solver_cg.hh"
#include "solver/krylov_solver_eigen.hh"

#include <libmugrid/numpy_tools.hh>

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <functional>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

using muSpectre::Index_t;
using muSpectre::OptimizeResult;
using muSpectre::Real;
using muSpectre::Uint;
using pybind11::literals::operator""_a;
using muSpectre::IsStrainInitialised;
using muSpectre::StoreNativeStress;
using muSpectre::Verbosity;
namespace py = pybind11;
using muGrid::NumpyProxy;

/**
 * Solvers instantiated for cells with equal spatial and material dimension
 */
void add_newton_cg_helper(py::module & mod) {
  const char name[]{"newton_cg"};
  using solver = muSpectre::KrylovSolverBase;
  using grad = py::EigenDRef<Eigen::MatrixXd>;
  using grad_vec = muSpectre::LoadSteps_t;
  // using PyField_t = py::array_t<Real, py::array::f_style>;

  // using Func_py_t = std::function<void(const size_t &, PyField_t &)>;
  using Func_t =
      std::function<void(const size_t &, muGrid::TypedFieldBase<Real> &)>;

  py::enum_<IsStrainInitialised>(mod, "IsStrainInitialised")
      .value("Yes", IsStrainInitialised::True)
      .value("No", IsStrainInitialised::False);

  mod.def(
      name,
      [](muSpectre::Cell & s, const grad & g, solver & so, Real nt, Real eqt,
         Verbosity verb, IsStrainInitialised strain_init,
         StoreNativeStress store_native_stress,
         py::function & eigen_strain_pyfunc) -> OptimizeResult {
        const grad_vec & g_vec{g};
        Func_t eigen_strain_cpp_func{
            [&eigen_strain_pyfunc](
                const size_t & step_nb,
                muGrid::TypedFieldBase<Real> & eigen_strain_field) {
              eigen_strain_pyfunc(step_nb, numpy_wrap(eigen_strain_field,
                                                      muGrid::IterUnit::SubPt));
            }};
        auto cell_ptr{
            std::dynamic_pointer_cast<muSpectre::Cell>(s.shared_from_this())};
        return newton_cg(cell_ptr, g_vec, so, nt, eqt, verb, strain_init,
                         store_native_stress, eigen_strain_cpp_func)
            .front();
      },
      "cell"_a, "ΔF₀"_a, "solver"_a, "newton_tol"_a, "equil_tol"_a,
      "verbose"_a = Verbosity::Silent,
      "IsStrainInitialised"_a = IsStrainInitialised::False,
      "store_native_stress"_a = StoreNativeStress::no,
      "eigen strain func"_a = nullptr);
  mod.def(
      name,
      [](std::shared_ptr<muSpectre::Cell> s, const grad & g, solver & so,
         Real nt, Real eqt, Verbosity verb, IsStrainInitialised strain_init,
         StoreNativeStress store_native_stress) -> OptimizeResult {
        const grad_vec & g_vec{g};
        return newton_cg(s, g_vec, so, nt, eqt, verb, strain_init,
                         store_native_stress)
            .front();
      },
      "cell"_a, "ΔF₀"_a, "solver"_a, "newton_tol"_a, "equil_tol"_a,
      "verbose"_a = Verbosity::Silent,
      "IsStrainInitialised"_a = IsStrainInitialised::False,
      "store_native_stress"_a = StoreNativeStress::no);
  mod.def(
      name,
      [](std::shared_ptr<muSpectre::Cell> s, const grad_vec & g, solver & so,
         Real nt, Real eqt, Verbosity verb, IsStrainInitialised strain_init,
         StoreNativeStress store_native_stress,
         const py::function & eigen_strain_pyfunc)
          -> std::vector<OptimizeResult> {
        Func_t eigen_strain_cpp_func{
            [&eigen_strain_pyfunc](
                const size_t & step_nb,
                muGrid::TypedFieldBase<Real> & eigen_strain_field) {
              eigen_strain_pyfunc(step_nb,
                                  muGrid::numpy_wrap(eigen_strain_field,
                                                     muGrid::IterUnit::SubPt));
            }};
        return newton_cg(s, g, so, nt, eqt, verb, strain_init,
                         store_native_stress, eigen_strain_cpp_func);
      },
      "cell"_a, "ΔF₀"_a, "solver"_a, "newton_tol"_a, "equil_tol"_a,
      "verbose"_a = Verbosity::Silent,
      "IsStrainInitialised"_a = IsStrainInitialised::False,
      "store_native_stress"_a = StoreNativeStress::no,
      "eigen strain func"_a = nullptr);
  mod.def(
      name,
      [](std::shared_ptr<muSpectre::Cell> s, const grad_vec & g, solver & so,
         Real nt, Real eqt, Verbosity verb, IsStrainInitialised strain_init,
         StoreNativeStress store_native_stress) -> std::vector<OptimizeResult> {
        return newton_cg(s, g, so, nt, eqt, verb, strain_init,
                         store_native_stress);
      },
      "cell"_a, "ΔF₀"_a, "solver"_a, "newton_tol"_a, "equil_tol"_a,
      "verbose"_a = Verbosity::Silent,
      "IsStrainInitialised"_a = IsStrainInitialised::False,
      "store_native_stress"_a = StoreNativeStress::no);
}

void add_de_geus_helper(py::module & mod) {
  const char name[]{"de_geus"};
  using solver = muSpectre::KrylovSolverBase;
  using grad = py::EigenDRef<Eigen::MatrixXd>;
  using grad_vec = muSpectre::LoadSteps_t;

  mod.def(
      name,
      [](std::shared_ptr<muSpectre::Cell> s, const grad & g, solver & so,
         Real nt, Real eqt, Verbosity verb) -> OptimizeResult {
        Eigen::MatrixXd tmp{g};
        return de_geus(s, tmp, so, nt, eqt, verb);
      },
      "cell"_a, "ΔF₀"_a, "solver"_a, "newton_tol"_a, "equilibrium_tol"_a,
      "verbose"_a = Verbosity::Silent);
  mod.def(
      name,
      [](std::shared_ptr<muSpectre::Cell> s, const grad_vec & g, solver & so,
         Real nt, Real eqt, Verbosity verb) -> std::vector<OptimizeResult> {
        return de_geus(s, g, so, nt, eqt, verb);
      },
      "cell"_a, "ΔF₀"_a, "solver"_a, "newton_tol"_a, "equilibrium_tol"_a,
      "verbose"_a = Verbosity::Silent);
}

void add_trust_region_newton_cg_helper(py::module & mod) {
  const char name[]{"trust_region_newton_cg"};
  using solver = muSpectre::KrylovSolverTrustRegionCG;
  using grad = py::EigenDRef<Eigen::MatrixXd>;
  using grad_vec = muSpectre::LoadSteps_t;

  using Func_t =
      std::function<void(const size_t &, muGrid::TypedFieldBase<Real> &)>;

  mod.def(
      name,
      [](std::shared_ptr<muSpectre::Cell> s, const grad & g, solver & so,
         Real tr, Real nt, Real eqt, Real it, Real dt, Verbosity verb,
         IsStrainInitialised strain_init) -> OptimizeResult {
        const grad_vec & g_vec{g};
        return trust_region_newton_cg(s, g_vec, so, tr, nt, eqt, it, dt, verb,
                                      strain_init)
            .front();
      },
      "cell"_a, "ΔF₀"_a, "solver"_a, "trust_region"_a, "newton_tol"_a,
      "equil_tol"_a, "inc_tr_tol"_a, "dec_tr_tol"_a,
      "verbose"_a = Verbosity::Silent,
      "IsStrainInitialised"_a = IsStrainInitialised::False);
  mod.def(
      name,
      [](std::shared_ptr<muSpectre::Cell> s, const grad & g, solver & so,
         Real tr, Real nt, Real eqt, Real it, Real dt, Verbosity verb,
         IsStrainInitialised strain_init,
         py::function & eigen_strain_pyfunc) -> OptimizeResult {
        const grad_vec & g_vec{g};
        Func_t eigen_strain_cpp_func{
            [&eigen_strain_pyfunc](
                const size_t & step_nb,
                muGrid::TypedFieldBase<Real> & eigen_strain_field) {
              eigen_strain_pyfunc(step_nb,
                                  muGrid::numpy_wrap(eigen_strain_field,
                                                     muGrid::IterUnit::SubPt));
            }};
        return trust_region_newton_cg(s, g_vec, so, tr, nt, eqt, it, dt, verb,
                                      strain_init, eigen_strain_cpp_func)
            .front();
      },
      "cell"_a, "ΔF₀"_a, "solver"_a, "trust_region"_a, "newton_tol"_a,
      "equil_tol"_a, "inc_tr_tol"_a, "dec_tr_tol"_a,
      "verbose"_a = Verbosity::Silent,
      "IsStrainInitialised"_a = IsStrainInitialised::False,
      "eigen_strain_func"_a = nullptr);
  mod.def(
      name,
      [](std::shared_ptr<muSpectre::Cell> s, const grad_vec & g, solver & so,
         Real tr, Real nt, Real eqt, Real it, Real dt, Verbosity verb,
         IsStrainInitialised strain_init) -> std::vector<OptimizeResult> {
        return trust_region_newton_cg(s, g, so, tr, nt, eqt, it, dt, verb,
                                      strain_init);
      },
      "cell"_a, "ΔF₀"_a, "solver"_a, "trust_region"_a, "newton_tol"_a,
      "equil_tol"_a, "inc_tr_tol"_a = 0.0, "dec_tr_tol"_a,
      "verbose"_a = Verbosity::Silent,
      "IsStrainInitialised"_a = IsStrainInitialised::False);
  mod.def(
      name,
      [](std::shared_ptr<muSpectre::Cell> s, const grad_vec & g, solver & so,
         Real tr, Real nt, Real eqt, Real it, Real dt, Verbosity verb,
         IsStrainInitialised strain_init,
         py::function & eigen_strain_pyfunc) -> std::vector<OptimizeResult> {
        Func_t eigen_strain_cpp_func{
            [&eigen_strain_pyfunc](
                const size_t & step_nb,
                muGrid::TypedFieldBase<Real> & eigen_strain_field) {
              eigen_strain_pyfunc(step_nb,
                                  muGrid::numpy_wrap(eigen_strain_field,
                                                     muGrid::IterUnit::SubPt));
            }};
        return trust_region_newton_cg(s, g, so, tr, nt, eqt, it, dt, verb,
                                      strain_init, eigen_strain_cpp_func);
      },
      "cell"_a, "ΔF₀"_a, "solver"_a, "trust_region"_a, "newton_tol"_a,
      "equil_tol"_a, "inc_tr_tol"_a = 0.0, "dec_tr_tol"_a,
      "verbose"_a = Verbosity::Silent,
      "IsStrainInitialised"_a = IsStrainInitialised::False,
      "eigen_strain_func"_a = nullptr);
}

void add_solver_helper(py::module & mod) {
  add_newton_cg_helper(mod);
  add_de_geus_helper(mod);
  add_trust_region_newton_cg_helper(mod);
}

void add_solvers(py::module & solvers) {
  py::class_<OptimizeResult>(solvers, "OptimizeResult")
      .def_readwrite("grad", &OptimizeResult::grad)
      .def_readwrite("stress", &OptimizeResult::stress)
      .def_readwrite("success", &OptimizeResult::success)
      .def_readwrite("status", &OptimizeResult::status)
      .def_readwrite("message", &OptimizeResult::message)
      .def_readwrite("nb_it", &OptimizeResult::nb_it)
      .def_readwrite("nb_fev", &OptimizeResult::nb_fev)
      .def_readwrite("newton_norm", &OptimizeResult::newton_norm)
      .def_readwrite("equil_norm", &OptimizeResult::equil_norm)
      .def_readwrite("formulation", &OptimizeResult::formulation);

  add_krylov_solvers(solvers);

  add_solver_helper(solvers);
}
