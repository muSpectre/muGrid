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
#ifdef WITH_FFTWMPI
#include "fft/fftwmpi_engine.hh"
#endif
#ifdef WITH_PFFT
#include "fft/pfft_engine.hh"
#endif
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
#ifdef WITH_MPI
    .def(py::init([](Ccoord res, Rcoord lengths, size_t comm) {
           return new Engine(res, lengths,
                             std::move(Communicator(MPI_Comm(comm))));
         }),
         "resolutions"_a,
         "lengths"_a,
         "communicator"_a=size_t(MPI_COMM_SELF))
#else
    .def(py::init<Ccoord, Rcoord>())
#endif
    .def("fft",
         [](Engine & eng, py::EigenDRef<Eigen::ArrayXXd> v) {
           using Coll_t = typename Engine::GFieldCollection_t;
           using Field_t = typename Engine::Field_t;
           Coll_t coll{};
           coll.initialise(eng.get_resolutions(), eng.get_locations());
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
           coll.initialise(eng.get_resolutions(), eng.get_locations());
           Field_t & temp{make_field<Field_t>("temp_field", coll)};
           eng.get_work_space().eigen() = v;
           eng.ifft(temp);
           return Eigen::ArrayXXd{temp.eigen()};
         },
         "array"_a)
    .def("initialise", &Engine::initialise,
         "flags"_a=FFT_PlanFlags::estimate)
    .def("normalisation", &Engine::normalisation);
}

void add_fft_engines(py::module & mod) {
  auto fft{mod.def_submodule("fft")};
  fft.doc() = "bindings for µSpectre's fft engines";
  add_engine_helper<  twoD, FFTWEngine<  twoD,   twoD>>(fft, "FFTW_2d");
  add_engine_helper<threeD, FFTWEngine<threeD, threeD>>(fft, "FFTW_3d");
#ifdef WITH_FFTWMPI
  add_engine_helper<  twoD, FFTWMPIEngine<  twoD,   twoD>>(fft, "FFTWMPI_2d");
  add_engine_helper<threeD, FFTWMPIEngine<threeD, threeD>>(fft, "FFTWMPI_3d");
#endif
#ifdef WITH_PFFT
  add_engine_helper<  twoD, PFFTEngine<  twoD,   twoD>>(fft, "PFFT_2d");
  add_engine_helper<threeD, PFFTEngine<threeD, threeD>>(fft, "PFFT_3d");
#endif
  add_projections(fft);
}
