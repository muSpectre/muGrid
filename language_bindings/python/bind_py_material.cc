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

#include "common/muSpectre_common.hh"
#include "materials/material_base.hh"
#include "materials/material_evaluator.hh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <sstream>
#include <string>

using muSpectre::Dim_t;
using muSpectre::Real;
using pybind11::literals::operator""_a;
namespace py = pybind11;

/* ---------------------------------------------------------------------- */
template <Dim_t Dim>
void add_material_linear_elastic_generic1_helper(py::module & mod);
/* ---------------------------------------------------------------------- */
template <Dim_t Dim>
void add_material_linear_elastic_generic2_helper(py::module & mod);

/**
 * python binding for the optionally objective form of Hooke's law
 */
template <Dim_t Dim>
void add_material_linear_elastic1_helper(py::module & mod);
template <Dim_t Dim>
void add_material_linear_elastic2_helper(py::module & mod);
template <Dim_t Dim>
void add_material_linear_elastic3_helper(py::module & mod);
template <Dim_t Dim>
void add_material_linear_elastic4_helper(py::module & mod);
template <Dim_t Dim>
void add_material_hyper_elasto_plastic1_helper(py::module & mod);
template <Dim_t Dim>
void add_material_hyper_elasto_plastic2_helper(py::module & mod);
template <Dim_t Dim>
void add_material_stochastic_plasticity_helper(py::module & mod);

#ifdef WITH_SPLIT
template <Dim_t Dim>
void add_material_laminate_helper(py::module & mod);
#endif

/* ---------------------------------------------------------------------- */
template <Dim_t Dim>
class PyMaterialBase : public muSpectre::MaterialBase<Dim, Dim> {
 public:
  /* Inherit the constructors */
  using Parent = muSpectre::MaterialBase<Dim, Dim>;
  using Parent::Parent;
  using Strain_t = typename Parent::Strain_t;
  using Stress_t = typename Parent::Stress_t;
  using Stiffness_t = typename Parent::Stiffness_t;
  using StressStiffness_t = typename std::tuple<Stress_t, Stiffness_t>;

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

  void compute_stresses(const typename Parent::StrainField_t & F,
                        typename Parent::StressField_t & P,
                        muSpectre::Formulation form,
                        muSpectre::SplitCell is_cell_split) override {
    PYBIND11_OVERLOAD_PURE(
        void,              // Return type
        Parent,            // Parent class
        compute_stresses,  // Name of function in C++ (must match Python name)
        F, P, form, is_cell_split);
  }

  void compute_stresses_tangent(const typename Parent::StrainField_t & F,
                                typename Parent::StressField_t & P,
                                typename Parent::TangentField_t & K,
                                muSpectre::Formulation form,
                                muSpectre::SplitCell is_cell_split) override {
    PYBIND11_OVERLOAD_PURE(
        void,             /* Return type */
        Parent,           /* Parent class */
        compute_stresses, /* Name of function in C++ (must match Python name) */
        F, P, K, form, is_cell_split);
  }

  Stress_t
  constitutive_law_small_strain(const Eigen::Ref<const Strain_t> & strain,
                                const size_t & pixel_index) override {
    PYBIND11_OVERLOAD_PURE(
        Stress_t, /* Return type */
        Parent,   /* Parent class */
        constitutive_law_small_strain,
        /* Name of function in C++ (must match Python name) */
        strain, pixel_index);
  }

  Stress_t
  constitutive_law_finite_strain(const Eigen::Ref<const Strain_t> & strain,
                                 const size_t & pixel_index) override {
    PYBIND11_OVERLOAD_PURE(
        Stress_t, /* Return type */
        Parent,   /* Parent class */
        constitutive_law_finite_strain,
        /* Name of function in C++ (must match Python name) */
        strain, pixel_index);
  }

  std::tuple<Stress_t, Stiffness_t> constitutive_law_tangent_small_strain(
      const Eigen::Ref<const Strain_t> & strain,
      const size_t & pixel_index) override {
    PYBIND11_OVERLOAD_PURE(
        StressStiffness_t, /* Return type */
        Parent,            /* Parent class */
        constitutive_law_tangent_small_strain,
        /* Name of function in C++ (must match Python name) */
        strain, pixel_index);
  }

  std::tuple<Stress_t, Stiffness_t> constitutive_law_tangent_finite_strain(
      const Eigen::Ref<const Strain_t> & strain,
      const size_t & pixel_index) override {
    PYBIND11_OVERLOAD_PURE(
        StressStiffness_t, /* Return type */
        Parent,            /* Parent class */
        constitutive_law_tangent_finite_strain,
        /* Name of function in C++ (must match Python name) */
        strain, pixel_index);
  }
};

template <Dim_t Dim>
void add_material_evaluator(py::module & mod) {
  std::stringstream name_stream{};
  name_stream << "MaterialEvaluator_" << Dim << "d";
  std::string name{name_stream.str()};

  using MatEval_t = muSpectre::MaterialEvaluator<Dim>;
  py::class_<MatEval_t>(mod, name.c_str())
      .def(py::init<std::shared_ptr<muSpectre::MaterialBase<Dim, Dim>>>())
      .def("save_history_variables", &MatEval_t::save_history_variables,
           "for materials with state variables")
      .def("evaluate_stress",
           [](MatEval_t & mateval, py::EigenDRef<Eigen::MatrixXd> & grad,
              muSpectre::Formulation form) {
             if ((grad.cols() != Dim) or (grad.rows() != Dim)) {
               std::stringstream err{};
               err << "need matrix of shape (" << Dim << "×" << Dim
                   << ") but got (" << grad.rows() << "×" << grad.cols()
                   << ").";
               throw std::runtime_error(err.str());
             }
             return mateval.evaluate_stress(grad, form);
           },
           "strain"_a, "formulation"_a,
           "Evaluates stress for a given strain and formulation "
           "(Piola-Kirchhoff 1 stress as a function of the placement gradient "
           "P = P(F) for formulation=Formulation.finite_strain and Cauchy "
           "stress as a function of the infinitesimal strain tensor σ = σ(ε) "
           "for formulation=Formulation.small_strain)")
      .def("evaluate_stress_tangent",
           [](MatEval_t & mateval, py::EigenDRef<Eigen::MatrixXd> & grad,
              muSpectre::Formulation form) {
             if ((grad.cols() != Dim) or (grad.rows() != Dim)) {
               std::stringstream err{};
               err << "need matrix of shape (" << Dim << "×" << Dim
                   << ") but got (" << grad.rows() << "×" << grad.cols()
                   << ").";
               throw std::runtime_error(err.str());
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
           "formulation=Formulation.small_strain.")
      .def("estimate_tangent",
           [](MatEval_t & evaluator, py::EigenDRef<Eigen::MatrixXd> & grad,
              muSpectre::Formulation form, const Real step,
              const muSpectre::FiniteDiff diff_type) {
             if ((grad.cols() != Dim) or (grad.rows() != Dim)) {
               std::stringstream err{};
               err << "need matrix of shape (" << Dim << "×" << Dim
                   << ") but got (" << grad.rows() << "×" << grad.cols()
                   << ").";
               throw std::runtime_error(err.str());
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

template <Dim_t Dim>
void add_material_helper(py::module & mod) {
  std::stringstream name_stream{};
  name_stream << "MaterialBase_" << Dim << "d";
  std::string name{name_stream.str()};
  using Material = muSpectre::MaterialBase<Dim, Dim>;
  using MaterialTrampoline = PyMaterialBase<Dim>;

  py::class_<Material, MaterialTrampoline /* <--- trampoline*/,
             std::shared_ptr<Material>>(mod, name.c_str())
      .def(py::init<const std::string &, const Dim_t &, const Dim_t &>())
      .def("save_history_variables", &Material::save_history_variables)
      .def("list_fields", &Material::list_fields)
      .def("get_real_field", &Material::get_real_field, "field_name"_a,
           py::return_value_policy::reference_internal)
      .def("size", &Material::size)
      .def(
          "add_pixel",
          [](Material & mat, size_t pix) { mat.add_pixel(pix); },
          "pixel"_a)
      .def_property_readonly(
          "collection",
          [](Material & material) -> muGrid::NFieldCollection & {
            return material.get_collection();
          },
          "returns the field collection containing internal "
          "fields of this material");

  add_material_linear_elastic1_helper<Dim>(mod);
  add_material_linear_elastic2_helper<Dim>(mod);
  add_material_linear_elastic3_helper<Dim>(mod);
  add_material_linear_elastic4_helper<Dim>(mod);
  add_material_hyper_elasto_plastic1_helper<Dim>(mod);
  add_material_hyper_elasto_plastic2_helper<Dim>(mod);
  add_material_linear_elastic_generic1_helper<Dim>(mod);
  add_material_linear_elastic_generic2_helper<Dim>(mod);
  add_material_stochastic_plasticity_helper<Dim>(mod);
#ifdef WITH_SPLIT
  add_material_laminate_helper<Dim>(mod);
#endif

  add_material_evaluator<Dim>(mod);
}

void add_material(py::module & mod) {
  auto material{mod.def_submodule("material")};
  material.doc() = "bindings for constitutive laws";
  add_material_helper<muGrid::twoD>(material);
  add_material_helper<muGrid::threeD>(material);
}
