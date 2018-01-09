/**
 * file   bind_py_common.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   08 Jan 2018
 *
 * @brief  Python bindings for the common part of µSpectre
 *
 * @section LICENCE
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
 * along with GNU Emacs; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
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

void add_get_cube(py::module & mod) {
  add_get_cube_helper<twoD, Dim_t>(mod);
  add_get_cube_helper<twoD, Real>(mod);
  add_get_cube_helper<threeD, Dim_t>(mod);
  add_get_cube_helper<threeD, Real>(mod);
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


PYBIND11_PLUGIN(common) {
  py::module mod("common", "Common utilities for pymuSpectre");

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

  add_get_cube(mod);

  add_Pixels(mod);
  return mod.ptr();
}
