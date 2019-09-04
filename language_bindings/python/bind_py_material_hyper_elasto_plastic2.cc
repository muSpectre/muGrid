/**
 * @file   bind_py_material_hyper_elasto_plastic2.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *         Richard Leute <richard.leute@imtek.uni-freiburg.de>
 *
 * @date   08 Jul 2019
 *
 * @brief  python binding for MaterialHyperElastoPlastic2
 *
 * Copyright © 2018 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#include "common/muSpectre_common.hh"
#include "materials/stress_transformations_Kirchhoff.hh"
#include "materials/material_hyper_elasto_plastic2.hh"
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
template <Dim_t Dim>
void add_material_hyper_elasto_plastic2_helper(py::module & mod) {
  std::stringstream name_stream{};
  name_stream << "MaterialHyperElastoPlastic2_" << Dim << "d";
  const auto name{name_stream.str()};

  using Mat_t = muSpectre::MaterialHyperElastoPlastic2<Dim, Dim>;
  using Cell_t = muSpectre::CellBase<Dim, Dim>;

  py::class_<Mat_t, muSpectre::MaterialBase<Dim, Dim>, std::shared_ptr<Mat_t>>(
      mod, name.c_str())
      .def_static("make",
                  [](Cell_t & cell, std::string name) -> Mat_t & {
                    return Mat_t::make(cell, name);
                  },
                  "cell"_a, "name"_a, py::return_value_policy::reference,
                  py::keep_alive<1, 0>())
      .def("add_pixel",
           [](Mat_t & mat, muSpectre::Ccoord_t<Dim> pix, const Real Young,
              const Real Poisson, const Real tau_y0,
              const Real H) { mat.add_pixel(pix, Young, Poisson, tau_y0, H); },
           "pixel"_a, "Youngs_modulus"_a, "Poisson_ratio"_a, "tau_y0"_a, "H"_a)
      .def_static("make_evaluator", []() { return Mat_t::make_evaluator(); });
}

template void
add_material_hyper_elasto_plastic2_helper<muSpectre::twoD>(py::module &);
template void
add_material_hyper_elasto_plastic2_helper<muSpectre::threeD>(py::module &);
