/**
 * @file   bind_py_material_stochastic_plasticity.cc
 *
 * @author Richard Leute <richard.leute@imtek.uni-freiburg.de>
 *
 * @date   25 Jan 2019
 *
 * @brief  python binding for MaterialStochasticPlasticity
 *
 * Copyright © 2019 Till Junge
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
#include "materials/material_stochastic_plasticity.hh"
#include "cell/cell.hh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <sstream>
#include <string>

using muSpectre::Index_t;
using muSpectre::Real;
using pybind11::literals::operator""_a;
namespace py = pybind11;

/**
 * python binding for the optionally objective form of Hooke's law
 * with per-pixel elastic properties
 */
template <Index_t Dim>
void add_material_stochastic_plasticity_helper(py::module & mod) {
  std::stringstream name_stream{};
  name_stream << "MaterialStochasticPlasticity_" << Dim << 'd';
  const auto name{name_stream.str()};

  using Mat_t = muSpectre::MaterialStochasticPlasticity<Dim>;
  using Cell_t = muSpectre::Cell;

  //! dynamic vector type for interactions with numpy/scipy/solvers etc.
  using Vector_t = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
  using StressField_t = Eigen::Ref<Vector_t>;

  py::class_<Mat_t, muSpectre::MaterialBase, std::shared_ptr<Mat_t>>(
      mod, name.c_str())
      .def_static(
          "make",
          [](Cell_t & cell, std::string n) -> Mat_t & {
            return Mat_t::make(cell, n);
          },
          "cell"_a, "name"_a, py::return_value_policy::reference_internal)
      .def(
          "add_pixel",
          [](Mat_t & mat, Index_t pix_id, Real Young, Real Poisson,
             Real plastic_increment, Real stress_threshold,
             Eigen::Ref<
                 const Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>
                 eigen_strain) {
            mat.add_pixel(pix_id, Young, Poisson, plastic_increment,
                          stress_threshold, eigen_strain);
          },
          "pixel_index"_a, "Young"_a, "Poisson"_a, "increment"_a, "threshold"_a,
          "strain"_a)
      .def_static("make_evaluator", []() { return Mat_t::make_evaluator(); })
      .def(
          "identify_overloaded_pixels",
          [](Mat_t & mat, Cell_t & cell, StressField_t & stress) {
            return mat.identify_overloaded_quad_pts(cell, stress);
          },
          "cell"_a, "stress"_a)
      .def(
          "set_plastic_increment",
          [](Mat_t & mat, const size_t & quad_pt_id, const Real increment) {
            return mat.set_plastic_increment(quad_pt_id, increment);
          },
          "quad_pt_id"_a, "increment"_a)
      .def(
          "set_stress_threshold",
          [](Mat_t & mat, const size_t & quad_pt_id, const Real threshold) {
            return mat.set_stress_threshold(quad_pt_id, threshold);
          },
          "quad_pt_id"_a, "threshold"_a)
      .def(
          "set_eigen_strain",
          [](Mat_t & mat, const size_t & quad_pt_id,
             Eigen::Ref<Eigen::Matrix<Real, Dim, Dim>> eigen_strain) {
            return mat.set_eigen_strain(quad_pt_id, eigen_strain);
          },
          "pixel"_a, "eigen_strain"_a)
      .def(
          "get_plastic_increment",
          [](Mat_t & mat, const size_t & quad_pt_id) {
            return mat.get_plastic_increment(quad_pt_id);
          },
          "pixel"_a)
      .def(
          "get_stress_threshold",
          [](Mat_t & mat, const size_t & quad_pt_id) {
            return mat.get_stress_threshold(quad_pt_id);
          },
          "pixel"_a)
      .def(
          "get_eigen_strain",
          [](Mat_t & mat, const size_t & quad_pt_id) {
            return mat.get_eigen_strain(quad_pt_id);
          },
          "pixel"_a)
      .def("reset_overloaded_pixels",
           [](Mat_t & mat) { return mat.reset_overloaded_quad_pts(); });
}

template void
add_material_stochastic_plasticity_helper<muSpectre::twoD>(py::module &);
template void
add_material_stochastic_plasticity_helper<muSpectre::threeD>(py::module &);
