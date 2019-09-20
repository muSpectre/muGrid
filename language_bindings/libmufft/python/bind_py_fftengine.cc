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
 * µFFT is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µFFT is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µFFT; see the file COPYING. If not, write to the
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

using muGrid::Complex;
using muGrid::Dim_t;
using muGrid::DynCcoord_t;
using muGrid::OneQuadPt;
using muGrid::Real;
using pybind11::literals::operator""_a;
namespace py = pybind11;

template <class Engine>
void add_engine_helper(py::module & mod, std::string name) {
  using ArrayXXc = Eigen::Array<Complex, Eigen::Dynamic, Eigen::Dynamic>;
  py::class_<Engine>(mod, name.c_str())
      .def(py::init([](std::vector<Dim_t> nb_grid_pts, Dim_t nb_dof_per_pixel,
                       muFFT::Communicator & comm) {
             // Initialize with muFFT Communicator object
             return new Engine(DynCcoord_t(nb_grid_pts), nb_dof_per_pixel,
                               comm);
           }),
           "nb_grid_pts"_a, "nb_dof_per_pixel"_a,
           "communicator"_a = muFFT::Communicator())
#ifdef WITH_MPI
      .def(py::init([](std::vector<Dim_t> nb_grid_pts, Dim_t nb_dof_per_pixel,
                       size_t comm) {
             // Initialize with bare MPI handle
             return new Engine(DynCcoord_t(nb_grid_pts), nb_dof_per_pixel,
                               std::move(muFFT::Communicator(MPI_Comm(comm))));
           }),
           "nb_grid_pts"_a, "nb_dof_per_pixel"_a,
           "communicator"_a = size_t(MPI_COMM_SELF))
#endif
      // Interface for passing Fields directly
      .def("fft", &Engine::fft)
      .def("ifft", &Engine::ifft)
      // Interface for passing numpy arrays
      .def(
          "fft",
          [](Engine & eng, Eigen::Ref<typename Engine::Field_t::EigenRep_t> v) {
            using Coll_t = typename Engine::GFieldCollection_t;
            using Field_t = muGrid::WrappedNField<Real>;
            Coll_t coll{eng.get_dim(), OneQuadPt};
            coll.initialise(eng.get_nb_subdomain_grid_pts(),
                            eng.get_subdomain_locations());
            // Do not make a copy, just wrap the Eigen array into a field that
            // does not manage its own data.
            Field_t proxy{"proxy_field", coll, eng.get_nb_dof_per_pixel(), v};
            // We need to tie the lifetime of the return value to the lifetime
            // of the engine object, because we are returning the internal work
            // space buffer that is managed by the engine;
            // see return_value_policy below.
            return eng.fft(proxy).eigen_pixel();
          },
          "array"_a, py::return_value_policy::reference_internal)
      .def(
          "ifft",
          [](Engine & eng, py::EigenDRef<ArrayXXc> v) {
            using Coll_t = typename Engine::GFieldCollection_t;
            using Field_t = muGrid::WrappedNField<Real>;
            // Create an Eigen array that will hold the result of the inverse
            // FFT. We don't want the storage managed by a field because we
            // want to transfer possession of storage to Python without a copy
            // operation.
            typename Field_t::EigenRep_t res{eng.get_nb_dof_per_pixel(),
                                             eng.size()};
            Coll_t coll{eng.get_dim(), OneQuadPt};
            coll.initialise(eng.get_nb_subdomain_grid_pts(),
                            eng.get_subdomain_locations());
            // Wrap the Eigen array into a proxy field that does not manage
            // its own data.
            Field_t proxy{"proxy_field", coll, eng.get_nb_dof_per_pixel(), res};
            eng.get_work_space().eigen_pixel() = v;
            eng.ifft(proxy);
            // We can safely transfer possession to Python since the Eigen
            // array is not tied to the engine object;
            // see return_value_policy below.
            return res;
          },
          "array"_a, py::return_value_policy::move)
      .def("initialise", &Engine::initialise,
           "flags"_a = muFFT::FFT_PlanFlags::estimate)
      .def_property_readonly("normalisation", &Engine::normalisation)
      .def_property_readonly("communicator", &Engine::get_communicator)
      .def_property_readonly(
          "nb_subdomain_grid_pts",
          [](const Engine & eng) {
            auto nb = eng.get_nb_subdomain_grid_pts();
            return py::array(nb.get_dim(), nb.data());
          },
          py::return_value_policy::reference)
      .def_property_readonly(
          "subdomain_locations",
          [](const Engine & eng) {
            auto nb = eng.get_subdomain_locations();
            return py::array(nb.get_dim(), nb.data());
          },
          py::return_value_policy::reference)
      .def_property_readonly(
          "nb_fourier_grid_pts",
          [](const Engine & eng) {
            auto nb = eng.get_nb_fourier_grid_pts();
            return py::array(nb.get_dim(), nb.data());
          },
          py::return_value_policy::reference)
      .def_property_readonly(
          "fourier_locations",
          [](const Engine & eng) {
            auto nb = eng.get_fourier_locations();
            return py::array(nb.get_dim(), nb.data());
          },
          py::return_value_policy::reference)
      .def_property_readonly(
          "nb_domain_grid_pts",
          [](const Engine & eng) {
            auto nb = eng.get_nb_domain_grid_pts();
            return py::array(nb.get_dim(), nb.data());
          },
          py::return_value_policy::reference);
}

void add_fft_engines(py::module & fft) {
  add_engine_helper<muFFT::FFTWEngine>(fft, "FFTW");
#ifdef WITH_FFTWMPI
  add_engine_helper<muFFT::FFTWMPIEngine>(fft, "FFTWMPI");
#endif
#ifdef WITH_PFFT
  add_engine_helper<muFFT::PFFTEngine>(fft, "PFFT");
#endif
}
