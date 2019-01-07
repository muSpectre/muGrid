/**
 * @file   bind_py_material_linear_elastic1.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   29 Oct 2018
 *
 * @brief  python bindings for MaterialLinearElastic1
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

#include "common/common.hh"
#include "materials/material_linear_elastic1.hh"
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
 */
template <Dim_t dim>
void add_material_linear_elastic1_helper(py::module & mod) {
  std::stringstream name_stream{};
  name_stream << "MaterialLinearElastic1_" << dim << 'd';
  const auto name{name_stream.str()};

  using Mat_t = muSpectre::MaterialLinearElastic1<dim, dim>;
  using Sys_t = muSpectre::CellBase<dim, dim>;
  py::class_<Mat_t, muSpectre::MaterialBase<dim, dim>>(mod, name.c_str())
      .def_static("make",
                  [](Sys_t & sys, std::string n, Real e, Real p) -> Mat_t & {
                    return Mat_t::make(sys, n, e, p);
                  },
                  "cell"_a, "name"_a, "Young"_a, "Poisson"_a,
                  py::return_value_policy::reference, py::keep_alive<1, 0>());
}

template void
add_material_linear_elastic1_helper<muSpectre::twoD>(py::module &);
template void
add_material_linear_elastic1_helper<muSpectre::threeD>(py::module &);
