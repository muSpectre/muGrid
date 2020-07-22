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
#include <libmugrid/field_typed.hh>
#include <libmugrid/raw_memory_operations.hh>

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
using muGrid::DynCcoord_t;
using muGrid::GlobalFieldCollection;
using muGrid::Index_t;
using muGrid::IterUnit;
using muGrid::NumpyProxy;
using muGrid::OneQuadPt;
using muGrid::Real;
using muGrid::RuntimeError;
using muGrid::Shape_t;
using muGrid::WrappedField;
using muFFT::Communicator;
using muFFT::fft_freq;
using muFFT::FFTEngineBase;
using pybind11::literals::operator""_a;
namespace py = pybind11;

class FFTEngineBaseUnclonable : public FFTEngineBase {
 public:
  FFTEngineBaseUnclonable(DynCcoord_t nb_grid_pts, Communicator comm,
                          const muFFT::FFT_PlanFlags & plan_flags,
                          bool allow_temporary_buffer, bool allow_destroy_input)
      : FFTEngineBase(nb_grid_pts, comm, plan_flags, allow_temporary_buffer,
                      allow_destroy_input) {}

  std::unique_ptr<FFTEngineBase> clone() const final {
    throw muFFT::FFTEngineError("Python version of FFTEngine cannot be cloned");
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

  PyFFTEngineBase(DynCcoord_t nb_grid_pts, Communicator comm,
                  const muFFT::FFT_PlanFlags & plan_flags,
                  bool allow_temporary_buffer, bool allow_destroy_input)
      : FFTEngineBaseUnclonable(nb_grid_pts, comm, plan_flags,
                                allow_temporary_buffer, allow_destroy_input) {}

  void compute_fft(const RealField_t & input_field,
                   FourierField_t & output_field) const override {
    PYBIND11_OVERLOAD_PURE(void, Parent, fft, input_field, output_field);
  }

  void compute_ifft(const FourierField_t & input_field,
                    RealField_t & output_field) const override {
    PYBIND11_OVERLOAD_PURE(void, Parent, ifft, input_field, output_field);
  }

  void create_plan(const Index_t & nb_dof_per_pixel) override {
    PYBIND11_OVERLOAD_PURE(void, Parent, create_plan, nb_dof_per_pixel);
  }
};

void add_fft_engine_base(py::module & mod) {
  py::class_<FFTEngineBase,                   // class
             std::shared_ptr<FFTEngineBase>,  // holder
             PyFFTEngineBase                  // trampoline base
             >(mod, "FFTEngineBase")
      .def(py::init<DynCcoord_t, Communicator, const muFFT::FFT_PlanFlags &,
                    bool, bool>());
}

template <class Engine>
void add_engine_helper(py::module & mod, const std::string & name) {
  py::class_<Engine,                   // class
             std::shared_ptr<Engine>,  // holder
             FFTEngineBase             // trampoline base
             >
      fft_engine(mod, name.c_str());
  fft_engine
      .def(py::init([](std::vector<Index_t> nb_grid_pts,
                       muFFT::Communicator & comm,
                       const muFFT::FFT_PlanFlags & plan_flags,
                       bool allow_temporary_buffer,
                       bool allow_destroy_input) {
             // Initialise with muFFT Communicator object
             return new Engine(DynCcoord_t(nb_grid_pts), comm, plan_flags,
                               allow_temporary_buffer, allow_destroy_input);
           }),
           "nb_grid_pts"_a, "communicator"_a = muFFT::Communicator(),
           "flags"_a = muFFT::FFT_PlanFlags::estimate,
           "allow_temporary_buffer"_a = true, "allow_destroy_input"_a = false)
#ifdef WITH_MPI
      .def(py::init([](std::vector<Index_t> nb_grid_pts,
                       const muFFT::FFT_PlanFlags & plan_flags,
                       bool allow_temporary_buffer,
                       bool allow_destroy_input,
                       size_t comm) {
             // Initialise with bare MPI handle
             return new Engine(DynCcoord_t(nb_grid_pts),
                               std::move(muFFT::Communicator(MPI_Comm(comm))),
                               plan_flags,
                               allow_temporary_buffer,
                               allow_destroy_input);
           }),
           "nb_grid_pts"_a, "communicator"_a = size_t(MPI_COMM_SELF),
           "flags"_a = muFFT::FFT_PlanFlags::estimate,
           "allow_temporary_buffer"_a = true,
           "allow_destroy_input"_a = false)
#endif
      .def("fft", &Engine::fft)
      .def("ifft", &Engine::ifft)
      .def("create_plan", &Engine::create_plan, "nb_dof_per_pixel"_a)
      .def("register_real_space_field",
           (FFTEngineBase::RealField_t &
            (Engine::*)(const std::string &, const Index_t &)) &
               Engine::register_real_space_field,
           "unique_name"_a, "nb_dof_per_pixel"_a,
           py::return_value_policy::reference_internal)
      .def("register_real_space_field",
           (FFTEngineBase::RealField_t &
            (Engine::*)(const std::string &, const Shape_t &)) &
               Engine::register_real_space_field,
           "unique_name"_a, "shape"_a,
           py::return_value_policy::reference_internal)
      .def("fetch_or_register_real_space_field",
           (FFTEngineBase::RealField_t &
            (Engine::*)(const std::string &, const Index_t &)) &
               Engine::fetch_or_register_real_space_field,
           "unique_name"_a, "nb_dof_per_pixel"_a,
           py::return_value_policy::reference_internal)
      .def("fetch_or_register_real_space_field",
           (FFTEngineBase::RealField_t &
            (Engine::*)(const std::string &, const Shape_t &)) &
               Engine::fetch_or_register_real_space_field,
           "unique_name"_a, "shape"_a,
           py::return_value_policy::reference_internal)
      .def("register_fourier_space_field",
           (FFTEngineBase::FourierField_t &
            (Engine::*)(const std::string &, const Index_t &)) &
               Engine::register_fourier_space_field,
           "unique_name"_a, "nb_dof_per_pixel"_a,
           py::return_value_policy::reference_internal)
      .def("register_fourier_space_field",
           (FFTEngineBase::FourierField_t &
            (Engine::*)(const std::string &, const Shape_t &)) &
               Engine::register_fourier_space_field,
           "unique_name"_a, "shape"_a,
           py::return_value_policy::reference_internal)
      .def("fetch_or_register_fourier_space_field",
           (FFTEngineBase::FourierField_t &
            (Engine::*)(const std::string &, const Index_t &)) &
               Engine::fetch_or_register_fourier_space_field,
           "unique_name"_a, "nb_dof_per_pixel"_a,
           py::return_value_policy::reference_internal)
      .def("fetch_or_register_fourier_space_field",
           (FFTEngineBase::FourierField_t &
            (Engine::*)(const std::string &, const Shape_t &)) &
               Engine::fetch_or_register_fourier_space_field,
           "unique_name"_a, "shape"_a,
           py::return_value_policy::reference_internal)
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
            for (Index_t dim = 0; dim < eng.get_spatial_dim(); ++dim) {
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
            for (Index_t dim = 0; dim < eng.get_spatial_dim(); ++dim) {
              t[dim] = py::slice(locs[dim], locs[dim] + nb_pts[dim], 1);
            }
            return t;
          },
          py::return_value_policy::reference)
      .def_property_readonly("spatial_dim", &Engine::get_spatial_dim)
      .def("has_plan_for", &Engine::has_plan_for, "nb_dof_per_pixel"_a)
      .def_property_readonly("fftfreq", [](const Engine & eng) {
        std::vector<Index_t> shape{}, strides{};
        Index_t dim{eng.get_spatial_dim()};
        shape.push_back(dim);
        strides.push_back(sizeof(Real));
        for (auto && n : eng.get_nb_fourier_grid_pts()) {
          shape.push_back(n);
        }
        for (auto && s : eng.get_pixels().get_strides()) {
          strides.push_back(s * dim * sizeof(Real));
        }
        py::array_t<Real> fftfreqs(shape, strides);
        Real * ptr{static_cast<Real *>(fftfreqs.request().ptr)};
        auto & nb_domain_grid_pts{eng.get_nb_domain_grid_pts()};
        for (auto && pix : eng.get_pixels()) {
          for (int i = 0; i < dim; ++i) {
            ptr[i] =
                static_cast<Real>(fft_freq(pix[i], nb_domain_grid_pts[i])) /
                nb_domain_grid_pts[i];
          }
          ptr += dim;
        }
        return fftfreqs;
      })
      .def(
          "fft",
          [](Engine & eng,
             py::array_t<Real> & input_array,
             py::array_t<Complex> & output_array) {
            auto nb_dof_per_pixel{input_array.size() / eng.size()};
            NumpyProxy<Real> input_proxy(eng.get_nb_subdomain_grid_pts(),
                                         eng.get_subdomain_locations(),
                                         nb_dof_per_pixel, input_array);
            NumpyProxy<Complex> output_proxy(eng.get_nb_fourier_grid_pts(),
                                             eng.get_fourier_locations(),
                                             nb_dof_per_pixel, output_array);
            auto && input_proxy_field{input_proxy.get_field()};
            eng.fft(input_proxy_field, output_proxy.get_field());
          },
          "real_input_array"_a, "complex_output_array"_a,
          "Perform forward FFT of the input array into the output array")
      .def(
          "ifft",
          [](Engine & eng,
             py::array_t<Complex> & input_array,
             py::array_t<Real> & output_array) {
            auto nb_dof_per_pixel{output_array.size() / eng.size()};
            NumpyProxy<Complex> input_proxy(eng.get_nb_fourier_grid_pts(),
                                            eng.get_fourier_locations(),
                                            nb_dof_per_pixel, input_array);
            NumpyProxy<Real> output_proxy(eng.get_nb_subdomain_grid_pts(),
                                          eng.get_subdomain_locations(),
                                          nb_dof_per_pixel, output_array);
            eng.ifft(input_proxy.get_field(), output_proxy.get_field());
          },
          "fourier_input_array"_a, "real_output_array"_a,
          "Perform inverse FFT of the input array into the output array.");
}

void add_fft_engines(py::module & mod) {
  add_fft_engine_base(mod);
  add_engine_helper<muFFT::FFTWEngine>(mod, "FFTW");
#ifdef WITH_FFTWMPI
  add_engine_helper<muFFT::FFTWMPIEngine>(mod, "FFTWMPI");
#endif
#ifdef WITH_PFFT
  add_engine_helper<muFFT::PFFTEngine>(mod, "PFFT", not_col_major);
#endif
}
