/**
 * @file   bind_py_material_phase_field_fracture2.cc
 *
 * @author W. Beck Andrews <william.beck.andrews@imtek.uni-freiburg.de>
 *
 * @date   02 Feb 2021
 *
 * @brief  python binding for MaterialPhaseFieldFracture
 *
 * Copyright © 2021 W. Beck Andrews
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
#include "materials/material_phase_field_fracture2.hh"
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
template <Index_t dim>
void add_material_phase_field_fracture2_helper(py::module & mod) {
  std::stringstream name_stream{};
  name_stream << "MaterialPhaseFieldFracture2_" << dim << 'd';
  const auto name{name_stream.str()};

  using Mat_t = muSpectre::MaterialPhaseFieldFracture2<dim>;
  using Cell_t = muSpectre::Cell;

  py::class_<Mat_t, muSpectre::MaterialBase, std::shared_ptr<Mat_t>>(
      mod, name.c_str())
      .def_static(
          "make",
          [](std::shared_ptr<Cell_t> & cell, std::string name, Real ksmall)
              -> Mat_t & {
            return Mat_t::make(cell, name, ksmall);
          },
          "cell"_a, "name"_a, "Ksmall"_a,
              py::return_value_policy::reference_internal)
      .def(
          "add_pixel",
          [](Mat_t & mat, size_t pixel_index, Real Young, Real Poisson,
              Real phase_field) {
            mat.add_pixel(pixel_index, Young, Poisson, phase_field);
          },
          "pixel"_a, "Young"_a, "Poisson"_a, "phase_field"_a)
      .def(
          "set_youngs_modulus",
          [](Mat_t & mat, const size_t & quad_pt_id, const Real & Young) {
            return mat.set_youngs_modulus(quad_pt_id, Young);
          },
          "quad_pt_id"_a, "Young"_a)
      .def(
          "set_poisson_ratio",
          [](Mat_t & mat, const size_t & quad_pt_id, const Real & Poisson) {
            return mat.set_poisson_ratio(quad_pt_id, Poisson);
          },
          "quad_pt_id"_a, "Poisson"_a)
      .def(
          "set_phase_field",
          [](Mat_t & mat, const size_t & quad_pt_id, const Real & phase_field) {
            return mat.set_phase_field(quad_pt_id, phase_field);
          },
          "quad_pt_id"_a, "phase_field"_a)
      .def(
          "get_youngs_modulus",
          [](Mat_t & mat, const size_t & quad_pt_id) {
            return mat.get_youngs_modulus(quad_pt_id);
          },
          "quad_pt_id"_a)
      .def(
          "get_poisson_ratio",
          [](Mat_t & mat, const size_t & quad_pt_id) {
            return mat.get_poisson_ratio(quad_pt_id);
          },
          "quad_pt_id"_a)
      .def(
          "get_phase_field",
          [](Mat_t & mat, const size_t & quad_pt_id) {
            return mat.get_phase_field(quad_pt_id);
          },
          "quad_pt_id"_a)
      .def_static("make_evaluator", [](Real ksmall){
          return Mat_t::make_evaluator(ksmall);
          }, "Ksmall"_a);
}

template void
add_material_phase_field_fracture2_helper<muSpectre::twoD>(py::module &);
template void
add_material_phase_field_fracture2_helper<muSpectre::threeD>(py::module &);
