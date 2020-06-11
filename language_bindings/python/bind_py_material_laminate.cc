/**
 * @file   bind_py_material_laminate.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   19 Jun 2019
 *
 * @brief  python bindings for MaterialLaminate (both FS and SS)
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
#include "materials/material_laminate.hh"
#include "materials/stress_transformations_PK2.hh"
#include "materials/stress_transformations_PK1.hh"

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
 */
template <Index_t dim, muSpectre::Formulation Form>
void add_material_laminate_helper(py::module & mod) {
  std::stringstream name_stream{};
  switch (Form) {
  case muSpectre::Formulation::finite_strain: {
    name_stream << "MaterialLaminate_fs_" << dim << 'd';
    break;
  }
  case muSpectre::Formulation::small_strain: {
    name_stream << "MaterialLaminate_ss_" << dim << 'd';
    break;
  }
  default:
    throw std::runtime_error(
        "The laminate material can only be made via MaterialLaminate_ss_dim_d "
        "or  MaterialLaminate_fs_dim_d.");
    break;
  }

  const auto name{name_stream.str()};

  using Mat_t = muSpectre::MaterialLaminate<dim, Form>;
  using Sys_t = muSpectre::Cell;
  py::class_<Mat_t, muSpectre::MaterialBase, std::shared_ptr<Mat_t>>(
      mod, name.c_str())
      .def_static(
          "make",
          [](Sys_t & sys, std::string n) -> Mat_t & {
            return Mat_t::make(sys, n);
          },
          "cell"_a, "name"_a, py::return_value_policy::reference_internal)
      .def_static("make_evaluator", []() { return Mat_t::make_evaluator(); });
}

template void add_material_laminate_helper<
    muSpectre::twoD, muSpectre::Formulation::finite_strain>(py::module &);
template void add_material_laminate_helper<
    muSpectre::threeD, muSpectre::Formulation::finite_strain>(py::module &);

template void add_material_laminate_helper<
    muSpectre::twoD, muSpectre::Formulation::small_strain>(py::module &);
template void add_material_laminate_helper<
    muSpectre::threeD, muSpectre::Formulation::small_strain>(py::module &);
