/**
 * @file   bind_py_material_linear_elastic2.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   29 Oct 2018
 *
 * @brief  python bindings for MaterialLinearElastic2
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
#include "materials/material_linear_elastic2.hh"
#include "cell/cell_base.hh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <sstream>
#include <string>

using muSpectre::Dim_t;
using muSpectre::Real;

namespace py = pybind11;
using pybind11::literals::operator""_a;

/**
 * python binding for the optionally objective form of Hooke's law
 * with a per pixel eigenstrain
 */
template <Dim_t dim>
void add_material_linear_elastic2_helper(py::module & mod) {
  std::stringstream name_stream{};
  name_stream << "MaterialLinearElastic2_" << dim << 'd';
  const auto name{name_stream.str()};

  using Mat_t = muSpectre::MaterialLinearElastic2<dim, dim>;
  using Sys_t = muSpectre::CellBase<dim, dim>;

  py::class_<Mat_t, muSpectre::MaterialBase<dim, dim>, std::shared_ptr<Mat_t>>(
      mod, name.c_str())
      .def_static("make",
                  [](Sys_t & sys, std::string n, Real e, Real p) -> Mat_t & {
                    return Mat_t::make(sys, n, e, p);
                  },
                  "cell"_a, "name"_a, "Young"_a, "Poisson"_a,
                  py::return_value_policy::reference, py::keep_alive<1, 0>())
      .def("add_pixel",
           [](Mat_t & mat, muSpectre::Ccoord_t<dim> pix,
              py::EigenDRef<Eigen::ArrayXXd> & eig) {
             Eigen::Matrix<Real, dim, dim> eig_strain{eig};
             mat.add_pixel(pix, eig_strain);
           },
           "pixel"_a, "eigenstrain"_a)
      .def_static("make_evaluator",
                  [](Real e, Real p) { return Mat_t::make_evaluator(e, p); },
                  "Young"_a, "Poisson"_a);
}

template void
add_material_linear_elastic2_helper<muSpectre::twoD>(py::module &);
template void
add_material_linear_elastic2_helper<muSpectre::threeD>(py::module &);