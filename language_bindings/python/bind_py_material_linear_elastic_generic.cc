/**
 * @file   bind_py_material_linear_elastic_generic.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   20 Dec 2018
 *
 * @brief bindings for the generic linear elastic law defined by its stiffness
 * tensor
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

#include "materials/material_linear_elastic_generic1.hh"
#include "materials/material_linear_elastic_generic2.hh"
#include "cell/cell.hh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <sstream>
#include <string>

using muSpectre::Cell;
using muSpectre::Dim_t;
using muSpectre::Index_t;
using muSpectre::MaterialBase;
using muSpectre::MaterialLinearElasticGeneric1;
using muSpectre::MaterialLinearElasticGeneric2;
using muSpectre::Real;
using muSpectre::threeD;
using muSpectre::twoD;
namespace py = pybind11;
using namespace pybind11::literals;  // NOLINT: recommended usage

/**
 * python binding for the generic linear elastic material
 */
template <Index_t Dim>
void add_material_linear_elastic_generic1_helper(py::module & mod) {
  std::stringstream name_stream{};
  name_stream << "MaterialLinearElasticGeneric1_" << Dim << "d";
  const auto name{name_stream.str()};

  using Mat_t = MaterialLinearElasticGeneric1<Dim>;
  using Cell_t = Cell;

  py::class_<Mat_t, MaterialBase, std::shared_ptr<Mat_t>>(mod, name.c_str())
      .def_static(
          "make",
          [](Cell_t & cell, std::string name,
             const py::EigenDRef<
                 Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &
                 elastic_tensor) -> Mat_t & {
            return Mat_t::make(cell, name, elastic_tensor);
          },
          "cell"_a, "name"_a, "elastic_tensor"_a,
          py::return_value_policy::reference_internal,
          "Factory function returning a MaterialLinearElastic instance. "
          "The elastic tensor has to be specified in Voigt notation.")
      .def(
          "add_pixel",
          [](Mat_t & mat, Index_t pixel_index) { mat.add_pixel(pixel_index); },
          "pixel"_a,
          "Register a new pixel to this material. subsequent evaluations of "
          "the "
          "stress and tangent in the cell will use this constitutive law for "
          "this "
          "particular pixel")
      .def("size", &Mat_t::size)
      .def_static(
          "make_evaluator",
          [](const py::EigenDRef<Eigen::Matrix<
                 Real, Eigen::Dynamic, Eigen::Dynamic>> & elastic_tensor) {
            return Mat_t::make_evaluator(elastic_tensor);
          },
          "elastic_tensor"_a);
}

/**
 * python binding for the generic linear elastic material with eigenstcain
 */
template <Index_t Dim>
void add_material_linear_elastic_generic2_helper(py::module & mod) {
  std::stringstream name_stream{};
  name_stream << "MaterialLinearElasticGeneric2_" << Dim << "d";
  const auto name{name_stream.str()};

  using Mat_t = MaterialLinearElasticGeneric2<Dim>;
  using Cell_t = Cell;

  py::class_<Mat_t, MaterialBase, std::shared_ptr<Mat_t>>(mod, name.c_str())
      .def_static(
          "make",
          [](Cell_t & cell, std::string name,
             const py::EigenDRef<
                 Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &
                 elastic_tensor) -> Mat_t & {
            return Mat_t::make(cell, name, elastic_tensor);
          },
          "cell"_a, "name"_a, "elastic_tensor"_a,
          py::return_value_policy::reference_internal,
          "Factory function returning a MaterialLinearElastic instance. "
          "The elastic tensor has to be specified in Voigt notation.")
      .def(
          "add_pixel",
          [](Mat_t & mat, size_t pixel_index) { mat.add_pixel(pixel_index); },
          "pixel"_a,
          "Register a new pixel to this material. Subsequent evaluations of "
          "the stress and tangent in the cell will use this constitutive law "
          "for this particular pixel")
      .def(
          "add_pixel",
          [](Mat_t & mat, size_t pixel_index,
             py::EigenDRef<Eigen::ArrayXXd> & eig) {
            Eigen::Matrix<Real, Dim, Dim> eig_strain{eig};
            mat.add_pixel(pixel_index, eig_strain);
          },
          "pixel"_a, "eigenstrain"_a,
          "Register a new pixel to this material and assign the eigenstrain. "
          "Subsequent Evaluations of the stress and tangent in the cell will "
          "use this constitutive law for this particular pixel")
      .def("size", &Mat_t::size)
      .def_static(
          "make_evaluator",
          [](const py::EigenDRef<Eigen::Matrix<
                 Real, Eigen::Dynamic, Eigen::Dynamic>> & elastic_tensor) {
            return Mat_t::make_evaluator(elastic_tensor);
          },
          "elastic_tensor"_a);
}

template void add_material_linear_elastic_generic1_helper<twoD>(py::module &);
template void add_material_linear_elastic_generic1_helper<threeD>(py::module &);
template void add_material_linear_elastic_generic2_helper<twoD>(py::module &);
template void add_material_linear_elastic_generic2_helper<threeD>(py::module &);
