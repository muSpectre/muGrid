/**
 * @file bind_py_material_dunant.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   30 Jan 2020
 *
 * @brief python binding of MaterialmaterialDunant
 *
 * Copyright © 2020 Ali Falsafi
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
#include "materials/material_dunant.hh"
#include "materials/material_dunant_max.hh"
#include "materials/material_dunant_t.hh"
#include "materials/material_dunant_tc.hh"
#include "cell/cell.hh"
#include "cell/cell_data.hh"

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
 * python binding for the material with dunant's damage model (MaterialDunant)
 */
template <Index_t Dim>
void add_material_dunant_helper(py::module & mod) {
  std::stringstream name_stream{};
  name_stream << "MaterialDunant_" << Dim << "d";
  const auto name{name_stream.str()};

  using Mat_t = muSpectre::MaterialDunant<Dim>;
  using Cell = muSpectre::Cell;
  using CellData = muSpectre::CellData;

  py::class_<Mat_t, muSpectre::MaterialBase, std::shared_ptr<Mat_t>>(
      mod, name.c_str())
      .def_static(
          "make",
          [](std::shared_ptr<Cell> cell, std::string name, Real young,
             Real poisson, Real kappa, Real alpha) -> Mat_t & {
            return Mat_t::make(cell, name, young, poisson, kappa, alpha);
          },
          "cell"_a, "name"_a, "YoungModulus"_a, "PoissonRatio"_a, "Kappa"_a,
          "Alpha"_a, py::return_value_policy::reference_internal)
      .def_static(
          "make",
          [](std::shared_ptr<CellData> cell, std::string name, Real young,
             Real poisson, Real kappa, Real alpha) -> Mat_t & {
            return Mat_t::make(cell, name, young, poisson, kappa, alpha);
          },
          "cell"_a, "name"_a, "YoungModulus"_a, "PoissonRatio"_a, "Kappa"_a,
          "Alpha"_a, py::return_value_policy::reference_internal)
      .def_static(
          "make_evaluator",
          [](Real young, Real poisson, Real kappa, Real alpha) {
            return Mat_t::make_evaluator(young, poisson, kappa, alpha);
          },
          "YoungModulus"_a, "PoissonRatio"_a, "Kappa"_a, "Alpha"_a)
      .def(
          "add_pixel",
          [](Mat_t & mat, size_t pixel_index) { mat.add_pixel(pixel_index); },
          "pixel_index"_a)
      .def(
          "add_pixel",
          [](Mat_t & mat, size_t pixel_index, Real kappa_var) {
            mat.add_pixel(pixel_index, kappa_var);
          },
          "pixel_index"_a, "kappa_variarion"_a);
}

/**
 * python binding for the material with dunant's damage model (MaterialDunant)
 */
template <Index_t Dim>
void add_material_dunant_max_helper(py::module & mod) {
  std::stringstream name_stream{};
  name_stream << "MaterialDunantMax_" << Dim << "d";
  const auto name{name_stream.str()};

  using Mat_t = muSpectre::MaterialDunantMax<Dim>;
  using Cell = muSpectre::Cell;
  using CellData = muSpectre::CellData;

  py::class_<Mat_t, muSpectre::MaterialBase, std::shared_ptr<Mat_t>>(
      mod, name.c_str())
      .def_static(
          "make",
          [](std::shared_ptr<Cell> cell, std::string name, Real young,
             Real poisson, Real kappa, Real alpha) -> Mat_t & {
            return Mat_t::make(cell, name, young, poisson, kappa, alpha);
          },
          "cell"_a, "name"_a, "YoungModulus"_a, "PoissonRatio"_a, "Kappa"_a,
          "Alpha"_a, py::return_value_policy::reference_internal)
      .def_static(
          "make",
          [](std::shared_ptr<CellData> cell, std::string name, Real young,
             Real poisson, Real kappa, Real alpha) -> Mat_t & {
            return Mat_t::make(cell, name, young, poisson, kappa, alpha);
          },
          "cell"_a, "name"_a, "YoungModulus"_a, "PoissonRatio"_a, "Kappa"_a,
          "Alpha"_a, py::return_value_policy::reference_internal)
      .def_static(
          "make_evaluator",
          [](Real young, Real poisson, Real kappa, Real alpha) {
            return Mat_t::make_evaluator(young, poisson, kappa, alpha);
          },
          "YoungModulus"_a, "PoissonRatio"_a, "Kappa"_a, "Alpha"_a)
      .def(
          "add_pixel",
          [](Mat_t & mat, size_t pixel_index) { mat.add_pixel(pixel_index); },
          "pixel_index"_a)
      .def(
          "add_pixel",
          [](Mat_t & mat, size_t pixel_index, Real kappa_var) {
            mat.add_pixel(pixel_index, kappa_var);
          },
          "pixel_index"_a, "kappa_variarion"_a);
}

/**
 * python binding for the material with dunant's damage model (MaterialDunant)
 */
template <Index_t Dim>
void add_material_dunant_t_helper(py::module & mod) {
  std::stringstream name_stream{};
  name_stream << "MaterialDunantT_" << Dim << "d";
  const auto name{name_stream.str()};

  using Mat_t = muSpectre::MaterialDunantT<Dim>;
  using Cell = muSpectre::Cell;
  using CellData = muSpectre::CellData;

  py::class_<Mat_t, muSpectre::MaterialBase, std::shared_ptr<Mat_t>>(
      mod, name.c_str())
      .def_static(
          "make",
          [](std::shared_ptr<Cell> cell, std::string name, Real young,
             Real poisson, Real kappa, Real alpha) -> Mat_t & {
            return Mat_t::make(cell, name, young, poisson, kappa, alpha);
          },
          "cell"_a, "name"_a, "YoungModulus"_a, "PoissonRatio"_a, "Kappa"_a,
          "Alpha"_a, py::return_value_policy::reference_internal)
      .def_static(
          "make",
          [](std::shared_ptr<CellData> cell, std::string name, Real young,
             Real poisson, Real kappa, Real alpha) -> Mat_t & {
            return Mat_t::make(cell, name, young, poisson, kappa, alpha);
          },
          "cell"_a, "name"_a, "YoungModulus"_a, "PoissonRatio"_a, "Kappa"_a,
          "Alpha"_a, py::return_value_policy::reference_internal)
      .def_static(
          "make_evaluator",
          [](Real young, Real poisson, Real kappa, Real alpha) {
            return Mat_t::make_evaluator(young, poisson, kappa, alpha);
          },
          "YoungModulus"_a, "PoissonRatio"_a, "Kappa"_a, "Alpha"_a)
      .def(
          "add_pixel",
          [](Mat_t & mat, size_t pixel_index) { mat.add_pixel(pixel_index); },
          "pixel_index"_a)
      .def(
          "add_pixel",
          [](Mat_t & mat, size_t pixel_index, Real kappa_var) {
            mat.add_pixel(pixel_index, kappa_var);
          },
          "pixel_index"_a, "kappa_variarion"_a);
}

/**
 * python binding for the material with dunant's damage model (MaterialDunant)
 */
template <Index_t Dim>
void add_material_dunant_tc_helper(py::module & mod) {
  std::stringstream name_stream{};
  name_stream << "MaterialDunantTC_" << Dim << "d";
  const auto name{name_stream.str()};

  using Mat_t = muSpectre::MaterialDunantTC<Dim>;
  using Cell = muSpectre::Cell;
  using CellData = muSpectre::CellData;

  py::class_<Mat_t, muSpectre::MaterialBase, std::shared_ptr<Mat_t>>(
      mod, name.c_str())
      .def_static(
          "make",
          [](std::shared_ptr<Cell> cell, std::string name, Real young,
             Real poisson, Real kappa, Real alpha, Real rho_c,
             Real rho_t) -> Mat_t & {
            return Mat_t::make(cell, name, young, poisson, kappa, alpha, rho_c,
                               rho_t);
          },
          "cell"_a, "name"_a, "YoungModulus"_a, "PoissonRatio"_a, "Kappa"_a,
          "Alpha"_a, "rho_c"_a, "rho_t"_a,
          py::return_value_policy::reference_internal)
      .def_static(
          "make",
          [](std::shared_ptr<CellData> cell, std::string name, Real young,
             Real poisson, Real kappa, Real alpha, Real rho_c,
             Real rho_t) -> Mat_t & {
            return Mat_t::make(cell, name, young, poisson, kappa, alpha, rho_c,
                               rho_t);
          },
          "cell"_a, "name"_a, "YoungModulus"_a, "PoissonRatio"_a, "Kappa"_a,
          "Alpha"_a, "rho_c"_a, "rho_t"_a,
          py::return_value_policy::reference_internal)
      .def_static(
          "make_evaluator",
          [](Real young, Real poisson, Real kappa, Real alpha, Real rho_c,
             Real rho_t) {
            return Mat_t::make_evaluator(young, poisson, kappa, alpha, rho_c,
                                         rho_t);
          },
          "YoungModulus"_a, "PoissonRatio"_a, "Kappa"_a, "Alpha"_a, "rho_c"_a,
          "rho_t"_a)
      .def(
          "add_pixel",
          [](Mat_t & mat, size_t pixel_index) { mat.add_pixel(pixel_index); },
          "pixel_index"_a)
      .def(
          "add_pixel",
          [](Mat_t & mat, size_t pixel_index, Real kappa_var) {
            mat.add_pixel(pixel_index, kappa_var);
          },
          "pixel_index"_a, "kappa_variarion"_a);
}

template void add_material_dunant_helper<muSpectre::twoD>(py::module &);
template void add_material_dunant_helper<muSpectre::threeD>(py::module &);

template void add_material_dunant_max_helper<muSpectre::twoD>(py::module &);
template void add_material_dunant_max_helper<muSpectre::threeD>(py::module &);

template void add_material_dunant_t_helper<muSpectre::twoD>(py::module &);
template void add_material_dunant_t_helper<muSpectre::threeD>(py::module &);

template void add_material_dunant_tc_helper<muSpectre::twoD>(py::module &);
template void add_material_dunant_tc_helper<muSpectre::threeD>(py::module &);
