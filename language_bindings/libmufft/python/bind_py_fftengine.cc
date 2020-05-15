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

#include <libmugrid/exception.hh>
#include <libmugrid/numpy_tools.hh>

#include <libmufft/fft_utils.hh>
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

using muGrid::numpy_wrap;
using muGrid::operator<<;
using muGrid::Complex;
using muGrid::Dim_t;
using muGrid::DynCcoord_t;
using muGrid::GlobalFieldCollection;
using muGrid::RuntimeError;
using muGrid::NumpyProxy;
using muGrid::OneQuadPt;
using muGrid::Real;
using muGrid::WrappedField;
using muFFT::fft_freq;
using muFFT::FFTEngineBase;
using muFFT::Communicator;
using pybind11::literals::operator""_a;
namespace py = pybind11;

class FFTEngineBaseUnclonable : public FFTEngineBase {
 public:
  FFTEngineBaseUnclonable(DynCcoord_t nb_grid_pts, Dim_t nb_dof_per_pixel,
                          Communicator comm)
      : FFTEngineBase(nb_grid_pts, nb_dof_per_pixel, comm) {}

  std::unique_ptr<FFTEngineBase> clone() const final {
    throw RuntimeError("Python version of FFTEngine cannot be cloned");
  }
};
/**
 * "Trampoline" class for handling the pure virtual methods, see
 * [http://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python]
 * for details
 */
class PyFFTEngineBase : public FFTEngineBaseUnclonable {
 public:
  //! base class
  using Parent = FFTEngineBase;
  //! field type on which projection is applied
  using RealField_t = typename Parent::RealField_t;
  //! workspace type
  using FourierField_t = typename Parent::FourierField_t;

  PyFFTEngineBase(DynCcoord_t nb_grid_pts, Dim_t nb_dof_per_pixel,
                  Communicator comm)
      : FFTEngineBaseUnclonable(nb_grid_pts, nb_dof_per_pixel, comm) {}

  FourierField_t & fft(RealField_t & field) override {
    PYBIND11_OVERLOAD_PURE(FourierField_t &, Parent, fft, field);
  }

  void ifft(RealField_t & field) const override {
    PYBIND11_OVERLOAD_PURE(void, Parent, ifft, field);
  }
};

void add_fft_engine_base(py::module & mod) {
  py::class_<FFTEngineBase,                   // class
             std::shared_ptr<FFTEngineBase>,  // holder
             PyFFTEngineBase                  // trampoline base
             >(mod, "FFTEngineBase")
      .def(py::init<DynCcoord_t, Dim_t, Communicator>());
}

template <class Engine>
void add_engine_helper(py::module & mod, std::string name) {
  py::class_<Engine,                   // class
             std::shared_ptr<Engine>,  // holder
             FFTEngineBase             // trampoline base
             >(mod, name.c_str())
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
      .def_property_readonly("fourier_field", &Engine::get_fourier_field,
                             py::return_value_policy::reference_internal)
      .def("fft", &Engine::fft, py::return_value_policy::reference_internal)
      .def("ifft", &Engine::ifft)
      // Interface for passing numpy arrays
      .def(
          "fft",
          [](Engine & eng, py::array_t<Real, py::array::f_style> & array) {
            // We need to tie the lifetime of the return value to the lifetime
            // of the engine object, because we are returning the internal work
            // space buffer that is managed by the engine;
            // see return_value_policy below.
            NumpyProxy<Real> proxy(
                eng.get_nb_subdomain_grid_pts(),
                eng.get_subdomain_locations(),
                eng.get_nb_dof_per_pixel(),
                array);
            return numpy_wrap(eng.fft(proxy.get_field()),
                              proxy.get_components_shape());
          },
          "array"_a,
          py::keep_alive<0, 1>(),
          "Perform forward FFT on the input array. The method returns an array "
          "containing the Fourier-transformed field, but this array is "
          "borrowed from a buffer internal to the FFT engine object. (A second "
          "call to the forward FFT will override this array.)")
      .def(
          "ifft",
          [](Engine & eng, py::array_t<Complex, py::array::f_style> & array) {
            // Copy the input array to the FFT work space.
            std::vector<Dim_t> components_shape{
                numpy_copy(eng.get_fourier_field(), array)};
            // Create an numpy array that will hold the result of the inverse
            // FFT. We don't want the storage managed by a field because we
            // want to transfer possession of storage to Python without a copy
            // operation.
            std::vector<Dim_t> shape;
            Dim_t nb_components = 1;
            for (auto && n : components_shape) {
              shape.push_back(n);
              nb_components *= n;
            }
            Dim_t size = nb_components;
            for (auto && n : eng.get_nb_subdomain_grid_pts()) {
              shape.push_back(n);
              size *= n;
            }
            // Create the numpy array that holds the output data.
            py::array_t<Real, py::array::f_style> result(shape);
            // Wrap the numpy array into a proxy field that does not manage
            // its own data.
            NumpyProxy<Real> output_proxy(
                eng.get_nb_subdomain_grid_pts(),
                eng.get_subdomain_locations(),
                eng.get_nb_dof_per_pixel(),
                result);
            eng.ifft(output_proxy.get_field());
            // We can safely transfer possession to Python since the py::array
            // is not tied to the engine object; see return_value_policy below.
            return result;
          },
          "array"_a, py::return_value_policy::move,
          "Perform inverse FFT on the input array. The method returns an array "
          "containing the transformed field. Unlike the forward FFT, this "
          "array is *not* borrowed but belongs to the caller.")
      .def("initialise", &Engine::initialise,
           "flags"_a = muFFT::FFT_PlanFlags::estimate)
      .def_property_readonly("normalisation", &Engine::normalisation)
      .def_property_readonly("communicator", &Engine::get_communicator)
      .def_property_readonly(
          "nb_subdomain_grid_pts",
          [](const Engine & eng) {
            return to_tuple(eng.get_nb_subdomain_grid_pts());
          },
          py::return_value_policy::reference)
      .def_property_readonly(
          "subdomain_locations",
          [](const Engine & eng) {
            return to_tuple(eng.get_subdomain_locations());
          },
          py::return_value_policy::reference)
      .def_property_readonly(
          "subdomain_strides",
          [](const Engine & eng) {
            return to_tuple(eng.get_subdomain_strides());
          },
          py::return_value_policy::reference)
      .def_property_readonly(
          "nb_fourier_grid_pts",
          [](const Engine & eng) {
            return to_tuple(eng.get_nb_fourier_grid_pts());
          },
          py::return_value_policy::reference)
      .def_property_readonly(
          "fourier_locations",
          [](const Engine & eng) {
            return to_tuple(eng.get_fourier_locations());
          },
          py::return_value_policy::reference)
      .def_property_readonly(
          "fourier_strides",
          [](const Engine & eng) {
            return to_tuple(eng.get_fourier_strides());
          },
          py::return_value_policy::reference)
      .def_property_readonly(
          "nb_domain_grid_pts",
          [](const Engine & eng) {
            return to_tuple(eng.get_nb_domain_grid_pts());
          },
          py::return_value_policy::reference)
      .def_property_readonly(
          "subdomain_slices",
          [](const Engine & eng) {
            auto & nb_pts = eng.get_nb_subdomain_grid_pts();
            auto & locs = eng.get_subdomain_locations();
            py::tuple t(eng.get_spatial_dim());
            for (Dim_t dim = 0; dim < eng.get_spatial_dim(); ++dim) {
              t[dim] = py::slice(locs[dim], locs[dim] + nb_pts[dim], 1);
            }
            return t;
          },
          py::return_value_policy::reference)
      .def_property_readonly(
          "fourier_slices",
          [](const Engine & eng) {
            auto & nb_pts = eng.get_nb_fourier_grid_pts();
            auto & locs = eng.get_fourier_locations();
            py::tuple t(eng.get_spatial_dim());
            for (Dim_t dim = 0; dim < eng.get_spatial_dim(); ++dim) {
              t[dim] = py::slice(locs[dim], locs[dim] + nb_pts[dim], 1);
            }
            return t;
          },
          py::return_value_policy::reference)
      .def_property_readonly(
          "fftfreq",
          [](const Engine & eng) {
            std::vector<Dim_t> shape{}, strides{};
            Dim_t dim{eng.get_spatial_dim()};
            shape.push_back(dim);
            strides.push_back(sizeof(Real));
            for (auto && n : eng.get_nb_fourier_grid_pts()) {
              shape.push_back(n);
            }
            for (auto && s : eng.get_pixels().get_strides()) {
              strides.push_back(s*dim*sizeof(Real));
            }
            py::array_t<Real> fftfreqs(shape, strides);
            Real *ptr{static_cast<Real*>(fftfreqs.request().ptr)};
            auto & nb_domain_grid_pts{eng.get_nb_domain_grid_pts()};
            for (auto && pix : eng.get_pixels()) {
              for (int i = 0; i < dim; ++i) {
                ptr[i] =
                    static_cast<Real>(fft_freq(pix[i], nb_domain_grid_pts[i]))
                    / nb_domain_grid_pts[i];
              }
              ptr += dim;
            }
            return fftfreqs;
          });
}

void add_fft_engines(py::module & mod) {
  add_fft_engine_base(mod);
  add_engine_helper<muFFT::FFTWEngine>(mod, "FFTW");
#ifdef WITH_FFTWMPI
  add_engine_helper<muFFT::FFTWMPIEngine>(mod, "FFTWMPI");
#endif
#ifdef WITH_PFFT
  add_engine_helper<muFFT::PFFTEngine>(mod, "PFFT");
#endif
}
