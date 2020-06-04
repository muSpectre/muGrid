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
 * Lesser General Public License for more details.
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
 *
 */

#include "common/muSpectre_common.hh"
#include <libmugrid/ccoord_operations.hh>

#include <pybind11/pybind11.h>

using muSpectre::Index_t;
using muSpectre::Formulation;
using muSpectre::Real;
using muSpectre::StoreNativeStress;
using muSpectre::StrainMeasure;
using muSpectre::StressMeasure;
using pybind11::literals::operator""_a;

namespace py = pybind11;

void add_version(py::module & mod) {
  auto version{mod.def_submodule("version")};

  version.doc() = "version information";

  version.def("info", &muSpectre::version::info)
      .def("hash", &muSpectre::version::hash)
      .def("description", &muSpectre::version::description)
      .def("is_dirty", &muSpectre::version::is_dirty);
}

void add_common(py::module & mod) {
  add_version(mod);
  py::enum_<Formulation>(mod, "Formulation")
      .value("finite_strain", Formulation::finite_strain)
      // "µSpectre handles a problem in terms of tranformation gradient F and"
      // " first Piola-Kirchhoff stress P")
      .value("small_strain", Formulation::small_strain);
  // "µSpectre handles a problem in terms of the infinitesimal strain "
  // "tensor ε and Cauchy stress σ");

  py::enum_<muSpectre::SplitCell>(mod, "SplitCell")
      // informs the µSpctre about the kind of cell (split or not_split)
      .value("laminate", muSpectre::SplitCell::laminate)
      .value("split", muSpectre::SplitCell::simple)
      .value("non_split", muSpectre::SplitCell::no);

  py::enum_<StoreNativeStress>(mod, "StoreNativeStress")
      .value("yes", StoreNativeStress::yes)
      .value("no", StoreNativeStress::no);

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

  py::enum_<muSpectre::FiniteDiff>(
      mod, "FiniteDiff",
      "Distinguishes between different options of numerical differentiation;\n "
      "  1) 'forward' finite differences: ∂f/∂x ≈ (f(x+Δx) - f(x))/Δx\n   2) "
      "'backward' finite differences: ∂f/∂x ≈ (f(x) - f(x-Δx))/Δx\n   3) "
      "'centred' finite differences: ∂f/∂x ≈ (f(x+Δx) - f(x-Δx))/2Δx")
      .value("forward", muSpectre::FiniteDiff::forward)
      .value("backward", muSpectre::FiniteDiff::backward)
      .value("centred", muSpectre::FiniteDiff::centred);

  mod.attr("OneQuadPt") = muGrid::OneQuadPt;
  mod.def("banner", &muSpectre::banner, "name"_a, "year"_a,
          "copyright_holder"_a);
}
