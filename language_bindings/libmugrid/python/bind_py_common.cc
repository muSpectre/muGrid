/**
 * @file   bind_py_common.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   08 Jan 2018
 *
 * @brief  Python bindings for the common part of µGrid
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

#include <sstream>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "libmugrid/ccoord_operations.hh"

using muGrid::Dim_t;
using muGrid::DynCcoord;
using muGrid::Real;
using muGrid::threeD;
using pybind11::literals::operator""_a;

namespace py = pybind11;

void add_enums(py::module & mod) {
  py::enum_<muGrid::Iteration>(mod, "Iteration")
      .value("Pixel", muGrid::Iteration::Pixel)
      .value("QuadPt", muGrid::Iteration::QuadPt)
      .export_values();
}

template <size_t MaxDim, typename T = Dim_t>
void add_dyn_ccoord_helper(py::module & mod, std::string name) {
  py::class_<DynCcoord<MaxDim, T>>(mod, name.c_str())
      .def(py::init<const std::vector<T>>())
      .def(py::init<Dim_t>())
      .def("__len__", &DynCcoord<MaxDim, T>::get_dim)
      .def("__getitem__", [](const DynCcoord<MaxDim, T> & self,
                             const Dim_t & index) {
        if (index < 0 || index >= self.get_dim()) {
          std::stringstream s;
          s << "index " << index << " out of range 0.." << self.get_dim() - 1;
          throw std::out_of_range(s.str());
        }
        return self[index];
      });
  py::implicitly_convertible<py::list, DynCcoord<MaxDim, T>>();
  py::implicitly_convertible<py::tuple, DynCcoord<MaxDim, T>>();
}

template <Dim_t dim, typename T>
void add_get_cube_helper(py::module & mod) {
  std::stringstream name{};
  name << "get_" << dim << "d_cube";
  mod.def(name.str().c_str(), &muGrid::CcoordOps::get_cube<dim, T>, "size"_a,
          "return a Ccoord with the value 'size' repeated in each dimension");
}

template <Dim_t dim>
void add_get_ccoord_helper(py::module & mod) {
  using Ccoord = muGrid::Ccoord_t<dim>;
  mod.def(
      "get_domain_ccoord",
      [](Ccoord nb_grid_pts, Dim_t index) {
        return muGrid::CcoordOps::get_ccoord<dim>(nb_grid_pts, Ccoord{}, index);
      },
      "nb_grid_pts"_a, "i"_a,
      "return the cell coordinate corresponding to the i'th cell in a grid of "
      "shape nb_grid_pts");
}

void add_get_cube(py::module & mod) {
  add_get_cube_helper<muGrid::oneD, Dim_t>(mod);
  add_get_cube_helper<muGrid::oneD, muGrid::Real>(mod);
  add_get_cube_helper<muGrid::twoD, Dim_t>(mod);
  add_get_cube_helper<muGrid::twoD, muGrid::Real>(mod);
  add_get_cube_helper<muGrid::threeD, Dim_t>(mod);
  add_get_cube_helper<muGrid::threeD, muGrid::Real>(mod);

  add_get_ccoord_helper<muGrid::oneD>(mod);
  add_get_ccoord_helper<muGrid::twoD>(mod);
  add_get_ccoord_helper<muGrid::threeD>(mod);
}

template <Dim_t dim>
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

template <Dim_t dim>
void add_Pixels_helper(py::module & mod) {
  std::stringstream name{};
  name << "Pixels" << dim << "d";
  using Ccoord = muGrid::Ccoord_t<dim>;
  py::class_<muGrid::CcoordOps::Pixels<dim>> Pixels(mod, name.str().c_str());
  Pixels.def(py::init<Ccoord>());
}

void add_Pixels(py::module & mod) {
  py::class_<muGrid::CcoordOps::DynamicPixels::Enumerator>(mod, "Enumerator")
      .def("__len__", &muGrid::CcoordOps::DynamicPixels::Enumerator::size)
      .def("__iter__",
           [](muGrid::CcoordOps::DynamicPixels::Enumerator & enumerator) {
             return py::make_iterator(enumerator.begin(), enumerator.end());
           });
  py::class_<muGrid::CcoordOps::DynamicPixels>(mod, "DynamicPixels")
      .def("__len__", &muGrid::CcoordOps::DynamicPixels::size)
      .def("__iter__",
           [](muGrid::CcoordOps::DynamicPixels & pixels) {
             return py::make_iterator(pixels.begin(), pixels.end());
           })
      .def("enumerate", &muGrid::CcoordOps::DynamicPixels::enumerate);
  add_Pixels_helper<muGrid::oneD>(mod);
  add_Pixels_helper<muGrid::twoD>(mod);
  add_Pixels_helper<muGrid::threeD>(mod);
}

void add_common(py::module & mod) {
  add_enums(mod);

  add_dyn_ccoord_helper<threeD, Dim_t>(mod, "DynCcoord");
  add_dyn_ccoord_helper<threeD, Real>(mod, "DynRcoord");

  add_get_cube(mod);

  add_Pixels(mod);

  add_get_index(mod);
}
