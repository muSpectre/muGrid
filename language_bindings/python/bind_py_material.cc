/**
 * @file   bind_py_material.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   09 Jan 2018
 *
 * @brief  python bindings for µSpectre's materials
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

#include <libmugrid/exception.hh>

#include "common/muSpectre_common.hh"
#include "materials/material_base.hh"
#include "materials/material_evaluator.hh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <sstream>
#include <string>

using muGrid::RuntimeError;
using muSpectre::Index_t;
using muSpectre::Real;
using pybind11::literals::operator""_a;
namespace py = pybind11;

/* ---------------------------------------------------------------------- */
template <Index_t Dim>
void add_material_linear_elastic_generic1_helper(py::module & mod);

template <Index_t Dim>
void add_material_linear_elastic_generic2_helper(py::module & mod);

/**
 * python binding for the optionally objective form of Hooke's law
 */
template <Index_t Dim>
void add_material_linear_elastic1_helper(py::module & mod);
template <Index_t Dim>
void add_material_linear_elastic2_helper(py::module & mod);
template <Index_t Dim>
void add_material_linear_elastic3_helper(py::module & mod);
template <Index_t Dim>
void add_material_linear_elastic4_helper(py::module & mod);
template <Index_t Dim>
void add_material_linear_elastic_damage1_helper(py::module & mod);
template <Index_t Dim>
void add_material_linear_elastic_damage2_helper(py::module & mod);
template <Index_t Dim>
void add_material_hyper_elasto_plastic1_helper(py::module & mod);
template <Index_t Dim>
void add_material_hyper_elasto_plastic2_helper(py::module & mod);
template <Index_t Dim>
void add_material_stochastic_plasticity_helper(py::module & mod);
template <Index_t Dim>
void add_material_visco_elastic_ss_helper(py::module & mod);
template <Index_t Dim>
void add_material_visco_elastic_damage_ss1_helper(py::module & mod);
template <Index_t Dim>
void add_material_visco_elastic_damage_ss2_helper(py::module & mod);
template <Index_t Dim>
void add_material_neo_hookean_elastic_helper(py::module & mod);

#ifdef WITH_SPLIT
template <Index_t Dim, muSpectre::Formulation Form>
void add_material_laminate_helper(py::module & mod);
#endif

/* ---------------------------------------------------------------------- */
class PyMaterialBase : public muSpectre::MaterialBase {
 public:
  /* Inherit the constructors */
  using Parent = muSpectre::MaterialBase;
  using Parent::Parent;

  /* Trampoline (need one for each virtual function) */
  void save_history_variables() override {
    PYBIND11_OVERLOAD_PURE(void,                     // Return type
                           Parent,                   // Parent class
                           save_history_variables);  // Name of function in C++
                                                     // (must match Python name)
  }

  /* Trampoline (need one for each virtual function) */
  void initialise() override {
    PYBIND11_OVERLOAD(
        void,         // Return type
        Parent,       // Parent class
        initialise);  // Name of function in C++ (must match Python name)
  }

  void compute_stresses(
      const muGrid::RealField & F, muGrid::RealField & P,
      const muSpectre::Formulation & form,
      const muSpectre::SplitCell & is_cell_split,
      const muSpectre::StoreNativeStress & store_native_stress) override {
    PYBIND11_OVERLOAD_PURE(
        void,              // Return type
        Parent,            // Parent class
        compute_stresses,  // Name of function in C++ (must match Python name)
        F, P, form, is_cell_split, store_native_stress);
  }

  void compute_stresses_tangent(
      const muGrid::RealField & F, muGrid::RealField & P, muGrid::RealField & K,
      const muSpectre::Formulation & form,
      const muSpectre::SplitCell & is_cell_split,
      const muSpectre::StoreNativeStress & store_native_stress) override {
    PYBIND11_OVERLOAD_PURE(
        void,              // Return type
        Parent,            // Parent class
        compute_stresses,  // Name of function in C++ (must match Python name)
        F, P, K, form, is_cell_split, store_native_stress);
  }

  using DynMatrix_t = typename Parent::DynMatrix_t;
  using StressTangent_t = std::tuple<DynMatrix_t, DynMatrix_t>;
  std::tuple<DynMatrix_t, DynMatrix_t>
  constitutive_law_dynamic(const Eigen::Ref<const DynMatrix_t> & strain,
                           const size_t & quad_pt_index,
                           const muSpectre::Formulation & form) override {
    PYBIND11_OVERLOAD_PURE(
        // Return type
        StressTangent_t,
        // Return type
        Parent,
        // Name of function in C++ (must match Python name)
        constitutive_law_dynamic,
        // arguments
        strain, quad_pt_index, form);
  }
};

template <Index_t Dim>
void add_material_evaluator(py::module & mod) {
  std::stringstream name_stream{};
  name_stream << "MaterialEvaluator_" << Dim << "d";
  std::string name{name_stream.str()};

  using MatEval_t = muSpectre::MaterialEvaluator<Dim>;
  py::class_<MatEval_t>(mod, name.c_str())
      .def(py::init<std::shared_ptr<muSpectre::MaterialBase>>())
      .def("save_history_variables", &MatEval_t::save_history_variables,
           "for materials with state variables")
      .def(
          "evaluate_stress",
          [](MatEval_t & mateval, py::EigenDRef<Eigen::MatrixXd> & grad,
             muSpectre::Formulation form) {
            if ((grad.cols() != Dim) or (grad.rows() != Dim)) {
              std::stringstream err{};
              err << "need matrix of shape (" << Dim << "×" << Dim
                  << ") but got (" << grad.rows() << "×" << grad.cols() << ").";
              throw RuntimeError(err.str());
            }
            return mateval.evaluate_stress(grad, form);
          },
          "strain"_a, "formulation"_a,
          "Evaluates stress for a given strain and formulation "
          "(Piola-Kirchhoff 1 stress as a function of the placement gradient "
          "P = P(F) for formulation=Formulation.finite_strain and Cauchy "
          "stress as a function of the infinitesimal strain tensor σ = σ(ε) "
          "for formulation=Formulation.small_strain)",
          py::return_value_policy::reference_internal)
      .def(
          "evaluate_stress_tangent",
          [](MatEval_t & mateval, py::EigenDRef<Eigen::MatrixXd> & grad,
             muSpectre::Formulation form) {
            if ((grad.cols() != Dim) or (grad.rows() != Dim)) {
              std::stringstream err{};
              err << "need matrix of shape (" << Dim << "×" << Dim
                  << ") but got (" << grad.rows() << "×" << grad.cols() << ").";
              throw RuntimeError(err.str());
            }
            return mateval.evaluate_stress_tangent(grad, form);
          },
          "strain"_a, "formulation"_a,
          "Evaluates stress and tangent moduli for a given strain and "
          "formulation (Piola-Kirchhoff 1 stress as a function of the "
          "placement gradient P = P(F) for "
          "formulation=Formulation.finite_strain and Cauchy stress as a "
          "function of the infinitesimal strain tensor σ = σ(ε) for "
          "formulation=Formulation.small_strain). The tangent moduli are K = "
          "∂P/∂F for formulation=Formulation.finite_strain and C = ∂σ/∂ε for "
          "formulation=Formulation.small_strain.",
          py::return_value_policy::reference_internal)
      .def(
          "estimate_tangent",
          [](MatEval_t & evaluator, py::EigenDRef<Eigen::MatrixXd> & grad,
             muSpectre::Formulation form, const Real step,
             const muSpectre::FiniteDiff diff_type) {
            if ((grad.cols() != Dim) or (grad.rows() != Dim)) {
              std::stringstream err{};
              err << "need matrix of shape (" << Dim << "×" << Dim
                  << ") but got (" << grad.rows() << "×" << grad.cols() << ").";
              throw RuntimeError(err.str());
            }
            return evaluator.estimate_tangent(grad, form, step, diff_type);
          },
          "strain"_a, "formulation"_a, "Delta_x"_a,
          "difference_type"_a = muSpectre::FiniteDiff::centred,
          "Numerical estimate of the tangent modulus using finite "
          "differences. The finite difference scheme as well as the finite "
          "step size can be chosen. If there are no special circumstances, "
          "the default scheme of centred finite differences yields the most "
          "accurate results at an increased computational cost.");
}

void add_material_base_helper(py::module & mod) {
  std::string name{"MaterialBase"};
  using Material = muSpectre::MaterialBase;
  using MaterialTrampoline = PyMaterialBase;

  py::class_<Material, MaterialTrampoline /* <--- trampoline*/,
             std::shared_ptr<Material>>(mod, name.c_str())
      .def(py::init<const std::string &, const Index_t &, const Index_t &,
                    const Index_t &,
                    std::shared_ptr<muGrid::LocalFieldCollection>>())
      .def("save_history_variables", &Material::save_history_variables)
      .def("list_fields", &Material::list_fields)
      .def("size", &Material::size)
      .def(
          "add_pixel", [](Material & mat, size_t pix) { mat.add_pixel(pix); },
          "pixel"_a)
      .def_property_readonly(
          "collection",
          [](Material & material) -> muGrid::FieldCollection & {
            return material.get_collection();
          },
          "returns the field collection containing internal "
          "fields of this material");
}

template <Index_t Dim>
void add_material_helper(py::module & mod) {
  add_material_linear_elastic1_helper<Dim>(mod);
  add_material_linear_elastic2_helper<Dim>(mod);
  add_material_linear_elastic3_helper<Dim>(mod);
  add_material_linear_elastic4_helper<Dim>(mod);
  add_material_linear_elastic_damage1_helper<Dim>(mod);
  add_material_linear_elastic_damage2_helper<Dim>(mod);
  add_material_hyper_elasto_plastic1_helper<Dim>(mod);
  add_material_hyper_elasto_plastic2_helper<Dim>(mod);
  add_material_linear_elastic_generic1_helper<Dim>(mod);
  add_material_linear_elastic_generic2_helper<Dim>(mod);
  add_material_stochastic_plasticity_helper<Dim>(mod);
  add_material_visco_elastic_ss_helper<Dim>(mod);
  add_material_visco_elastic_damage_ss1_helper<Dim>(mod);
  add_material_visco_elastic_damage_ss2_helper<Dim>(mod);
  add_material_neo_hookean_elastic_helper<Dim>(mod);
  add_material_evaluator<Dim>(mod);

#ifdef WITH_SPLIT
  add_material_laminate_helper<Dim, muSpectre::Formulation::finite_strain>(mod);
  add_material_laminate_helper<Dim, muSpectre::Formulation::small_strain>(mod);
#endif
}

void add_material(py::module & mod) {
  auto material{mod.def_submodule("material")};
  material.doc() = "bindings for constitutive laws";
  add_material_base_helper(material);
  add_material_helper<muGrid::twoD>(material);
  add_material_helper<muGrid::threeD>(material);
}
