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

#include "bind_py_declarations.hh"

#include <libmufft/fftw_engine.hh>
#ifdef WITH_FFTWMPI
#include <libmufft/fftwmpi_engine.hh>
#endif
#ifdef WITH_PFFT
#include <libmufft/pfft_engine.hh>
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

using muGrid::Ccoord_t;
using muGrid::Complex;
using muGrid::Dim_t;
using pybind11::literals::operator""_a;
namespace py = pybind11;

template <class Engine>
void add_engine_helper(py::module & mod, std::string name) {
  using Ccoord = typename Engine::Ccoord;
  using ArrayXXc = Eigen::Array<Complex, Eigen::Dynamic, Eigen::Dynamic>;
  py::class_<Engine>(mod, name.c_str())
      .def(py::init([](Ccoord res, Dim_t nb_components,
                       muFFT::Communicator & comm) {
             // Initialize with muFFT Communicator object
             return new Engine(res, nb_components, comm);
           }),
           "nb_grid_pts"_a, "nb_components"_a,
           "communicator"_a = muFFT::Communicator())
#ifdef WITH_MPI
      .def(py::init([](Ccoord res, Dim_t nb_components, size_t comm) {
             // Initialize with bare MPI handle
             return new Engine(res, nb_components,
                               std::move(muFFT::Communicator(MPI_Comm(comm))));
           }),
           "nb_grid_pts"_a, "nb_components"_a,
           "communicator"_a = size_t(MPI_COMM_SELF))
#endif
      .def("fft",
           [](Engine & eng,
              Eigen::Ref<typename Engine::Field_t::EigenRep_t> v) {
             using Coll_t = typename Engine::GFieldCollection_t;
             using Field_t = typename Engine::Field_t;
             Coll_t coll{};
             coll.initialise(eng.get_nb_subdomain_grid_pts(),
                             eng.get_subdomain_locations());
             // Do not make a copy, just wrap the Eigen array into a field that
             // does not manage its own data.
             Field_t & proxy{muGrid::make_field<Field_t>(
                 "proxy_field", coll, v, eng.get_nb_components())};
             // We need to tie the lifetime of the return value to the lifetime
             // of the engine object, because we are returning the internal work
             // space buffer that is managed by the engine;
             // see return_value_policy below.
             return eng.fft(proxy).eigen();
           },
           "array"_a,
           py::return_value_policy::reference_internal)
      .def("ifft",
           [](Engine & eng, py::EigenDRef<ArrayXXc> v) {
             using Coll_t = typename Engine::GFieldCollection_t;
             using Field_t = typename Engine::Field_t;
             // Create an Eigen array that will hold the result of the inverse
             // FFT. We don't want the storage managed by a field because we
             // want to transfer possession of storage to Python without a copy
             // operation.
             typename Field_t::EigenRep_t res{eng.get_nb_components(),
                                              eng.size()};
             Coll_t coll{};
             coll.initialise(eng.get_nb_subdomain_grid_pts(),
                             eng.get_subdomain_locations());
             // Wrap the Eigen array into a proxy field that does not manage
             // its own data.
             Field_t & proxy{muGrid::make_field<Field_t>(
                 "proxy_field", coll, res, eng.get_nb_components())};
             eng.get_work_space().eigen() = v;
             eng.ifft(proxy);
             // We can safely transfer possession to Python since the Eigen
             // array is not tied to the engine object;
             // see return_value_policy below.
             return res;
           },
           "array"_a,
           py::return_value_policy::move)
      .def("initialise", &Engine::initialise,
           "flags"_a = muFFT::FFT_PlanFlags::estimate)
      .def("normalisation", &Engine::normalisation)
      .def("get_communicator", &Engine::get_communicator)
      .def("get_nb_subdomain_grid_pts", &Engine::get_nb_subdomain_grid_pts)
      .def("get_subdomain_locations", &Engine::get_subdomain_locations)
      .def("get_nb_fourier_grid_pts", &Engine::get_nb_fourier_grid_pts)
      .def("get_fourier_locations", &Engine::get_fourier_locations)
      .def("get_nb_domain_grid_pts", &Engine::get_nb_domain_grid_pts);
}

void add_fft_engines(py::module & fft) {
  add_engine_helper<muFFT::FFTWEngine<muGrid::twoD>>(fft, "FFTW_2d");
  add_engine_helper<muFFT::FFTWEngine<muGrid::threeD>>(fft, "FFTW_3d");
#ifdef WITH_FFTWMPI
  add_engine_helper<muFFT::FFTWMPIEngine<muGrid::twoD>>(fft, "FFTWMPI_2d");
  add_engine_helper<muFFT::FFTWMPIEngine<muGrid::threeD>>(fft, "FFTWMPI_3d");
#endif
#ifdef WITH_PFFT
  add_engine_helper<muFFT::PFFTEngine<muGrid::twoD>>(fft, "PFFT_2d");
  add_engine_helper<muFFT::PFFTEngine<muGrid::threeD>>(fft, "PFFT_3d");
#endif
}
