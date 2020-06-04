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
#include "materials/material_linear_elastic1.hh"
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
template <Index_t dim>
void add_material_linear_elastic1_helper(py::module & mod) {
  std::stringstream name_stream{};
  name_stream << "MaterialLinearElastic1_" << dim << 'd';
  const auto name{name_stream.str()};

  using Mat_t = muSpectre::MaterialLinearElastic1<dim>;
  using Cell = muSpectre::Cell;
  using Mat_ptr = std::shared_ptr<Mat_t>;
  py::class_<Mat_t, muSpectre::MaterialBase, std::shared_ptr<Mat_t>>(
      mod, name.c_str())
      .def_static(
          "make",
          [](Cell & cell, std::string name, Real Young, Real Poisson)
              -> Mat_t & { return Mat_t::make(cell, name, Young, Poisson); },
          "cell"_a, "name"_a, "Young"_a, "Poisson"_a,
          py::return_value_policy::reference_internal)
      .def_static(
          "make_evaluator",
          [](Real e, Real p) { return Mat_t::make_evaluator(e, p); }, "Young"_a,
          "Poisson"_a)
      .def_static(
          "make_free",
          [](Cell & cell, std::string name, Real Young,
             Real Poisson) -> Mat_ptr {
            auto && sdim{cell.get_spatial_dim()};
            auto && nb_quad_pts{cell.get_nb_quad_pts()};
            Mat_ptr ret_mat{std::make_shared<Mat_t>(name, sdim, nb_quad_pts,
                                                    Young, Poisson)};
            return ret_mat;
          },
          "name"_a, "nb_quad_pts"_a, "Young"_a, "Poisson"_a)
      .def_property_readonly("C", &Mat_t::get_C);
}

template void
add_material_linear_elastic1_helper<muSpectre::twoD>(py::module &);
template void
add_material_linear_elastic1_helper<muSpectre::threeD>(py::module &);
