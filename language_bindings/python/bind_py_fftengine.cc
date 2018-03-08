/**
 * @file   bind_py_fftengine.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   17 Jan 2018
 *
 * @brief  Python bindings for the FFT engines
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

#include "fft/fftw_engine.hh"
#include "bind_py_declarations.hh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

using namespace muSpectre;
namespace py=pybind11;
using namespace pybind11::literals;

template <Dim_t dim, class Engine>
void add_engine_helper(py::module & mod, std::string name) {
  using Ccoord = Ccoord_t<dim>;
  using Rcoord = Rcoord_t<dim>;
  using ArrayXXc = Eigen::Array<Complex, Eigen::Dynamic,
                                Eigen::Dynamic>;
  py::class_<Engine>(mod, name.c_str())
    .def(py::init<Ccoord, Rcoord>())
    .def("fft",
         [](Engine & eng, py::EigenDRef<Eigen::ArrayXXd> v) {
           using Coll_t = typename Engine::GFieldCollection_t;
           using Field_t = typename Engine::Field_t;
           Coll_t coll{};
           coll.initialise(eng.get_resolutions());
           Field_t & temp{make_field<Field_t>("temp_field", coll)};
           temp.eigen() = v;
           return ArrayXXc{eng.fft(temp).eigen()};
         },
         "array"_a)
    .def("ifft",
         [](Engine & eng,
            py::EigenDRef<ArrayXXc> v) {
           using Coll_t = typename Engine::GFieldCollection_t;
           using Field_t = typename Engine::Field_t;
           Coll_t coll{};
           coll.initialise(eng.get_resolutions());
           Field_t & temp{make_field<Field_t>("temp_field", coll)};
           eng.get_work_space().eigen()=v;
           eng.ifft(temp);
           return Eigen::ArrayXXd{temp.eigen()};
         },
         "array"_a)
    .def("initialise", &Engine::initialise,
         "flags"_a=FFT_PlanFlags::estimate)
    .def("normalisation", &Engine::normalisation);
}

void add_engine(py::module & mod) {
  add_engine_helper<  twoD, FFTWEngine<  twoD,   twoD>>(mod, "FFTW_2d");
  add_engine_helper<threeD, FFTWEngine<threeD, threeD>>(mod, "FFTW_3d");
}

void add_fft_engines(py::module & mod) {
  auto fft{mod.def_submodule("fft")};
  fft.doc() = "bindings for µSpectre's fft engines";
  add_engine(fft);
  add_projections(fft);
}
