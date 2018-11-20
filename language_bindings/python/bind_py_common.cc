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

#include "common/common.hh"
#include "common/ccoord_operations.hh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>

using namespace muSpectre;
namespace py = pybind11;
using namespace pybind11::literals;


template <Dim_t dim, typename T>
void add_get_cube_helper(py::module & mod) {
  std::stringstream name {};
  name << "get_" << dim << "d_cube";
  mod.def
    (name.str().c_str(), &CcoordOps::get_cube<dim, T>, "size"_a,
     "return a Ccoord with the value 'size' repeated in each dimension");
}

template <Dim_t dim>
void add_get_hermitian_helper(py::module & mod) {
  mod.def
    ("get_hermitian_sizes", &CcoordOps::get_hermitian_sizes<dim>,
     "full_sizes"_a,
     "return the hermitian sizes corresponding to the true sizes");
}

template <Dim_t dim>
void add_get_ccoord_helper(py::module & mod) {
  using Ccoord = Ccoord_t<dim>;
  mod.def
    ("get_domain_ccoord", [](Ccoord resolutions, Dim_t index){
      return CcoordOps::get_ccoord<dim>(resolutions, Ccoord{}, index);
      },
     "resolutions"_a,
     "i"_a,
     "return the cell coordinate corresponding to the i'th cell in a grid of "
     "shape resolutions");
}


void add_get_cube(py::module & mod) {
  add_get_cube_helper<twoD, Dim_t>(mod);
  add_get_cube_helper<twoD, Real>(mod);
  add_get_cube_helper<threeD, Dim_t>(mod);
  add_get_cube_helper<threeD, Real>(mod);

  add_get_hermitian_helper<  twoD>(mod);
  add_get_hermitian_helper<threeD>(mod);

  add_get_ccoord_helper<  twoD>(mod);
  add_get_ccoord_helper<threeD>(mod);
}

template <Dim_t dim>
void add_get_index_helper(py::module & mod) {
  using Ccoord = Ccoord_t<dim>;
  mod.def("get_domain_index",
          [](Ccoord sizes, Ccoord ccoord){
            return CcoordOps::get_index<dim>(sizes, Ccoord{}, ccoord);},
          "sizes"_a, "ccoord"_a,
          "return the linear index corresponding to grid point 'ccoord' in a "
          "grid of size 'sizes'");
}

void add_get_index(py::module & mod) {
  add_get_index_helper<  twoD>(mod);
  add_get_index_helper<threeD>(mod);
}

template <Dim_t dim>
void add_Pixels_helper(py::module & mod) {
  std::stringstream name{};
  name << "Pixels" << dim << "d";
  using Ccoord = Ccoord_t<dim>;
  py::class_<CcoordOps::Pixels<dim>> Pixels(mod, name.str().c_str());
  Pixels.def(py::init<Ccoord>());
}

void add_Pixels(py::module & mod) {
  add_Pixels_helper<twoD>(mod);
  add_Pixels_helper<threeD>(mod);
}


void add_common (py::module & mod) {
  py::enum_<Formulation>(mod, "Formulation")
    .value("finite_strain", Formulation::finite_strain)
    //"µSpectre handles a problem in terms of tranformation gradient F and"
    //" first Piola-Kirchhoff stress P")
    .value("small_strain", Formulation::small_strain)
    //"µSpectre handles a problem in terms of the infinitesimal strain "
    //"tensor ε and Cauchy stress σ");
    ;

  py::enum_<StressMeasure>(mod, "StressMeasure")
    .value("Cauchy", StressMeasure::Cauchy)
    .value("PK1", StressMeasure::PK1)
    .value("PK2", StressMeasure::PK2)
    .value("Kirchhoff", StressMeasure::Kirchhoff)
    .value("Biot", StressMeasure::Biot)
    .value("Mandel", StressMeasure::Mandel)
    .value("no_stress_", StressMeasure::no_stress_);

  py::enum_<StrainMeasure>(mod, "StrainMeasure")
    .value("Gradient", StrainMeasure::Gradient)
    .value("Infinitesimal", StrainMeasure::Infinitesimal)
    .value("GreenLagrange", StrainMeasure::GreenLagrange)
    .value("Biot", StrainMeasure::Biot)
    .value("Log", StrainMeasure::Log)
    .value("Almansi", StrainMeasure::Almansi)
    .value("RCauchyGreen", StrainMeasure::RCauchyGreen)
    .value("LCauchyGreen", StrainMeasure::LCauchyGreen)
    .value("no_strain_", StrainMeasure::no_strain_);

  py::enum_<FFT_PlanFlags>(mod, "FFT_PlanFlags")
    .value("estimate", FFT_PlanFlags::estimate)
    .value("measure", FFT_PlanFlags::measure)
    .value("patient", FFT_PlanFlags::patient);

  mod.def("banner", &banner, "name"_a, "year"_a, "copyright_holder"_a);

  add_get_cube(mod);

  add_Pixels(mod);

  add_get_index(mod);
}
