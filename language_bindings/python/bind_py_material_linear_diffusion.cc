/**
 * @file   bind_py_material_linear_diffusion.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   05 Sep 2020
 *
 * @brief  Python bindings for MaterialLinearDiffusion
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

#include "common/muSpectre_common.hh"
#include "materials/material_linear_diffusion.hh"
#include "cell/cell_data.hh"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

using muSpectre::Dim_t;
using muSpectre::Index_t;
using muSpectre::Real;
using pybind11::literals::operator""_a;
namespace py = pybind11;

template <Dim_t Dim>
void add_material_linear_diffusion_helper(py::module & mod) {
  std::stringstream name_stream{};
  name_stream << "MaterialLinearDiffusion_" << Dim << 'd';
  const auto name{name_stream.str()};

  using Mat_t = muSpectre::MaterialLinearDiffusion<Dim>;
  using CellData = muSpectre::CellData;

  py::class_<Mat_t, muSpectre::MaterialBase, std::shared_ptr<Mat_t>>(
      mod, name.c_str())
      .def_static(
          "make",
          [](std::shared_ptr<CellData> cell, std::string name,
             Real diffusion_coeff) -> Mat_t & {
            return Mat_t::make(cell, name, diffusion_coeff);
          },
          "cell"_a, "name"_a, "Young"_a,
          py::return_value_policy::reference_internal)
      .def_static(
          "make",
          [](std::shared_ptr<CellData> cell, std::string name,
             py::EigenDRef<Eigen::MatrixXd> diffusion_coeff) -> Mat_t & {
            return Mat_t::make(cell, name, diffusion_coeff);
          },
          "cell"_a, "name"_a, "Young"_a,
          py::return_value_policy::reference_internal)
      .def_static(
          "make_evaluator", [](Real e) { return Mat_t::make_evaluator(e); },
          "Young"_a)
      .def_property_readonly("diffusion_coeff", &Mat_t::get_diffusion_coeff);
}
template void
add_material_linear_diffusion_helper<muSpectre::twoD>(py::module &);
template void
add_material_linear_diffusion_helper<muSpectre::threeD>(py::module &);
