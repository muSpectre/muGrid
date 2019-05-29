/**
 * @file   bind_py_common.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   08 Jan 2018
 *
 * @brief  Python bindings for the common part of µSpectre
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
 * General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 */

#include <libmugrid/ccoord_operations.hh>
#include <libmufft/mufft_common.hh>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>

using pybind11::literals::operator""_a;

namespace py = pybind11;

template <muGrid::Dim_t dim, typename T>
void add_get_cube_helper(py::module & mod) {
  std::stringstream name{};
  name << "get_" << dim << "d_cube";
  mod.def(name.str().c_str(), &muGrid::CcoordOps::get_cube<dim, T>, "size"_a,
          "return a Ccoord with the value 'size' repeated in each dimension");
}

template <muGrid::Dim_t dim>
void add_get_hermitian_helper(py::module & mod) {
  mod.def("get_hermitian_sizes", &muGrid::CcoordOps::get_hermitian_sizes<dim>,
          "full_sizes"_a,
          "return the hermitian sizes corresponding to the true sizes");
}

template <muGrid::Dim_t dim>
void add_get_ccoord_helper(py::module & mod) {
  using Ccoord = muGrid::Ccoord_t<dim>;
  mod.def(
      "get_domain_ccoord",
      [](Ccoord nb_grid_pts, muGrid::Dim_t index) {
        return muGrid::CcoordOps::get_ccoord<dim>(nb_grid_pts, Ccoord{}, index);
      },
      "nb_grid_pts"_a, "i"_a,
      "return the cell coordinate corresponding to the i'th cell in a grid of "
      "shape nb_grid_pts");
}

void add_get_cube(py::module & mod) {
  add_get_cube_helper<muGrid::oneD, muGrid::Dim_t>(mod);
  add_get_cube_helper<muGrid::oneD, muGrid::Real>(mod);
  add_get_cube_helper<muGrid::twoD, muGrid::Dim_t>(mod);
  add_get_cube_helper<muGrid::twoD, muGrid::Real>(mod);
  add_get_cube_helper<muGrid::threeD, muGrid::Dim_t>(mod);
  add_get_cube_helper<muGrid::threeD, muGrid::Real>(mod);

  add_get_hermitian_helper<muGrid::oneD>(mod);
  add_get_hermitian_helper<muGrid::twoD>(mod);
  add_get_hermitian_helper<muGrid::threeD>(mod);

  add_get_ccoord_helper<muGrid::oneD>(mod);
  add_get_ccoord_helper<muGrid::twoD>(mod);
  add_get_ccoord_helper<muGrid::threeD>(mod);
}

template <muGrid::Dim_t dim>
void add_get_index_helper(py::module & mod) {
  using Ccoord = muGrid::Ccoord_t<dim>;
  mod.def("get_domain_index",
          [](Ccoord sizes, Ccoord ccoord) {
            return muGrid::CcoordOps::get_index<dim>(sizes, Ccoord{}, ccoord);
          },
          "sizes"_a, "ccoord"_a,
          "return the linear index corresponding to grid point 'ccoord' in a "
          "grid of size 'sizes'");
}

void add_get_index(py::module & mod) {
  add_get_index_helper<muGrid::oneD>(mod);
  add_get_index_helper<muGrid::twoD>(mod);
  add_get_index_helper<muGrid::threeD>(mod);
}

template <muGrid::Dim_t dim>
void add_Pixels_helper(py::module & mod) {
  std::stringstream name{};
  name << "Pixels" << dim << "d";
  using Ccoord = muGrid::Ccoord_t<dim>;
  py::class_<muGrid::CcoordOps::Pixels<dim>> Pixels(mod, name.str().c_str());
  Pixels.def(py::init<Ccoord>());
}

void add_Pixels(py::module & mod) {
  add_Pixels_helper<muGrid::oneD>(mod);
  add_Pixels_helper<muGrid::twoD>(mod);
  add_Pixels_helper<muGrid::threeD>(mod);
}

void add_common(py::module & mod) {
  py::enum_<muFFT::FFT_PlanFlags>(mod, "FFT_PlanFlags")
      .value("estimate", muFFT::FFT_PlanFlags::estimate)
      .value("measure", muFFT::FFT_PlanFlags::measure)
      .value("patient", muFFT::FFT_PlanFlags::patient);

  add_get_cube(mod);

  add_Pixels(mod);

  add_get_index(mod);
}
