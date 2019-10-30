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
#include "cell/cell_base.hh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <sstream>
#include <string>

using muSpectre::Dim_t;
using muSpectre::Real;
using pybind11::literals::operator""_a;
namespace py = pybind11;

/**
 * python binding for the optionally objective form of Hooke's law
 * with per-pixel elastic properties
 */
template <Dim_t dim>
void add_material_stochastic_plasticity_helper(py::module & mod) {
  std::stringstream name_stream{};
  name_stream << "MaterialStochasticPlasticity_" << dim << 'd';
  const auto name{name_stream.str()};

  using Mat_t = muSpectre::MaterialStochasticPlasticity<dim, dim>;
  using Sys_t = muSpectre::CellBase<dim, dim>;

  //! dynamic vector type for interactions with numpy/scipy/solvers etc.
  using Vector_t = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
  using StressField_t = Eigen::Ref<Vector_t>;

  py::class_<Mat_t, muSpectre::MaterialBase<dim, dim>, std::shared_ptr<Mat_t>>(
      mod, name.c_str())
      .def(py::init<std::string>(), "name"_a)
      .def_static("make",
                  [](Sys_t & sys, std::string n) -> Mat_t & {
                    return Mat_t::make(sys, n);
                  },
                  "cell"_a, "name"_a, py::return_value_policy::reference,
                  py::keep_alive<1, 0>())
      .def("add_pixel",
           [](Mat_t & mat, muSpectre::Ccoord_t<dim> pix, Real Young,
              Real Poisson, Real plastic_increment, Real stress_threshold,
              Eigen::Ref<
                  const Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>
                  eigen_strain) {
             mat.add_pixel(pix, Young, Poisson, plastic_increment,
                           stress_threshold, eigen_strain);
           },
           "pixel"_a, "Young"_a, "Poisson"_a, "increment"_a, "threshold"_a,
           "strain"_a)
      .def_static("make_evaluator", []() { return Mat_t::make_evaluator(); })
      .def("identify_overloaded_pixels",
           [](Mat_t & mat, Sys_t & sys, StressField_t & stress) {
             return mat.identify_overloaded_pixels(sys, stress);
           },
           "sys"_a, "stress"_a)
      .def("set_plastic_increment",
           [](Mat_t & mat, const muSpectre::Ccoord_t<dim> pixel,
              const Real increment) {
             return mat.set_plastic_increment(pixel, increment);
           },
           "pixel"_a, "increment"_a)
      .def("set_stress_threshold",
           [](Mat_t & mat, const muSpectre::Ccoord_t<dim> pixel,
              const Real threshold) {
             return mat.set_stress_threshold(pixel, threshold);
           },
           "pixel"_a, "threshold"_a)
      .def("set_eigen_strain",
           [](Mat_t & mat, const muSpectre::Ccoord_t<dim> pixel,
              Eigen::Ref<Eigen::Matrix<Real, dim, dim>> eigen_strain) {
             return mat.set_eigen_strain(pixel, eigen_strain);
           },
           "pixel"_a, "eigen_strain"_a)
      .def("get_plastic_increment",
           [](Mat_t & mat, const muSpectre::Ccoord_t<dim> pixel) {
             return mat.get_plastic_increment(pixel);
           },
           "pixel"_a)
      .def("get_stress_threshold",
           [](Mat_t & mat, const muSpectre::Ccoord_t<dim> pixel) {
             return mat.get_stress_threshold(pixel);
           },
           "pixel"_a)
      .def("get_eigen_strain",
           [](Mat_t & mat, const muSpectre::Ccoord_t<dim> pixel) {
             return mat.get_eigen_strain(pixel);
           },
           "pixel"_a)
      .def("reset_overloaded_pixels",
           [](Mat_t & mat) { return mat.reset_overloaded_pixels(); });
}

template void
add_material_stochastic_plasticity_helper<muSpectre::twoD>(py::module &);
template void
add_material_stochastic_plasticity_helper<muSpectre::threeD>(py::module &);
