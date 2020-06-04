/**
 * @file   bind_py_material_linear_anisotropic.cc
 *
 * @author Al Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   12 Mar 2019
 *
 * @brief  python bindings for MaterialLinearAnisotropic
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
#include "materials/material_linear_anisotropic.hh"
#include "cell/cell_base.hh"

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
 * python binding for the optionally objective form of Anisotropic material
 */
template <Index_t dim>
void add_material_linear_anisotropic_helper(py::module & mod) {
  std::stringstream name_stream{};
  name_stream << "MaterialLinearAnisotropic_" << dim << 'd';
  const auto name{name_stream.str()};
  using Mat_t = muSpectre::MaterialLinearAnisotropic<dim, dim>;
  using Sys_t = muSpectre::CellBase<dim, dim>;
  using MatBase_t = MaterialBase<dim, dim>;
  py::class_<Mat_t, MatBase_t>(mod, name.c_str())
      .def_static(
          "make",
          [](Sys_t & sys, std::string n, std::vector<Real> stiffness_coeffs)
              -> Mat_t & { return Mat_t::make(sys, n, stiffness_coeffs); },
          "cell"_a, "name"_a, "stiffness_coeffs"_a,
          py::return_value_policy::reference_internal)
      .def(
          "add_pixel",
          [](Mat_t & mat, Ccoord_t<dim> pix) { mat.add_pixel(pix); }, "pixel"_a)
      .def(
          "add_pixel_split",
          [](Mat_t & mat, Ccoord_t<dim> pix, Real ratio) {
            mat.add_pixel_split(pix, ratio);
          },
          "pixel"_a, "ratio"_a)
      .def("size", &Mat_t::size);
}
