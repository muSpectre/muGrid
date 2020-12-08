/**
 * @file   bind_py_fem_discretisation.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   04 Sep 2020
 *
 * @brief  python bindings for the fem discretisation
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

#include "projection/fem_library.hh"
#include "projection/discretisation.hh"
#include "cell/cell_data.hh"

#include <pybind11/pybind11.h>

using muSpectre::CellData;
using muSpectre::FEMStencil;
using muSpectre::FEMStencilBase;
using muSpectre::Index_t;
using muSpectre::Real;
using pybind11::literals::operator""_a;
namespace py = pybind11;

void add_fem_discretisation(py::module & mod) {
  py::class_<FEMStencilBase, std::shared_ptr<FEMStencilBase>>(mod,
                                                              "FEMStencilBase");
  using FEMStencil_t = FEMStencil<muGrid::GradientOperatorDefault>;
  py::class_<FEMStencil_t, FEMStencilBase, std::shared_ptr<FEMStencil_t>>(
      mod, "FEMStencil")
      .def(
          py::init<
              const Index_t &, const Index_t &, const Index_t &,
              const Index_t &,
              const std::vector<std::vector<Eigen::MatrixXd>> &,
              const std::vector<std::tuple<Eigen::VectorXi, Eigen::MatrixXi>> &,
              const std::vector<Real> &, std::shared_ptr<CellData>>(),
          "nb_quad_pts_per_element"_a, "nb_elements"_a,
          "nb_element_nodal_pts"_a, "nb_pixel_nodal_pts"_a,
          "shape_fn_gradients"_a, "nodal_pts"_a, "quadrature_weights"_a,
          "cell"_a)
      .def_static("linear_interval", &muSpectre::FEMLibrary::linear_1d,
                  "cell_data"_a)
      .def_static("linear_triangle",
                  &muSpectre::FEMLibrary::linear_triangle_straight,
                  "cell_data"_a)
      .def_static("bilinear_quadrangle",
                  &muSpectre::FEMLibrary::bilinear_quadrangle, "cell_data"_a);

  py::class_<muSpectre::Discretisation,
             std::shared_ptr<muSpectre::Discretisation>>(mod, "Discretisation")
      .def(py::init<std::shared_ptr<FEMStencilBase>>(), "fem_stencil"_a);
}
