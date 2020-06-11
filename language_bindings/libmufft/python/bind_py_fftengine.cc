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
using muFFT::Communicator;
using muFFT::fft_freq;
using muFFT::FFTEngineBase;
using muGrid::Complex;
using muGrid::DynCcoord_t;
using muGrid::GlobalFieldCollection;
using muGrid::Index_t;
using muGrid::NumpyProxy;
using muGrid::OneQuadPt;
using muGrid::Real;
using muGrid::RuntimeError;
using muGrid::WrappedField;
using pybind11::literals::operator""_a;
namespace py = pybind11;

class FFTEngineBaseUnclonable : public FFTEngineBase {
 public:
  FFTEngineBaseUnclonable(DynCcoord_t nb_grid_pts, Communicator comm)
      : FFTEngineBase(nb_grid_pts, comm) {}

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

  PyFFTEngineBase(DynCcoord_t nb_grid_pts, Communicator comm)
      : FFTEngineBaseUnclonable(nb_grid_pts, comm) {}

  void fft(const RealField_t & input_field,
           FourierField_t & output_field) const override {
    PYBIND11_OVERLOAD_PURE(void, Parent, fft, input_field, output_field);
  }

  void ifft(const FourierField_t & input_field,
            RealField_t & output_field) const override {
    PYBIND11_OVERLOAD_PURE(void, Parent, ifft, input_field, output_field);
  }

  void initialise(const Index_t & nb_dof_per_pixel,
                  const muFFT::FFT_PlanFlags & plan_flags) override {
    PYBIND11_OVERLOAD_PURE(void, Parent, initialise, nb_dof_per_pixel,
                           plan_flags);
  }
};

void add_fft_engine_base(py::module & mod) {
  py::class_<FFTEngineBase,                   // class
             std::shared_ptr<FFTEngineBase>,  // holder
             PyFFTEngineBase                  // trampoline base
             >(mod, "FFTEngineBase")
      .def(py::init<DynCcoord_t, Communicator>());
}

template <class Engine>
void add_engine_helper(py::module & mod, const std::string & name,
                       const bool & fourier_space_arrays_are_col_major) {
  py::class_<Engine,                   // class
             std::shared_ptr<Engine>,  // holder
             FFTEngineBase             // trampoline base
             >
      fft_engine(mod, name.c_str());
  fft_engine
      .def(py::init([](std::vector<Index_t> nb_grid_pts,
                       muFFT::Communicator & comm) {
             // Initialise with muFFT Communicator object
             return new Engine(DynCcoord_t(nb_grid_pts), comm);
           }),
           "nb_grid_pts"_a, "communicator"_a = muFFT::Communicator())
#ifdef WITH_MPI
      .def(py::init([](std::vector<Index_t> nb_grid_pts, size_t comm) {
             // Initialise with bare MPI handle
             return new Engine(DynCcoord_t(nb_grid_pts),
                               std::move(muFFT::Communicator(MPI_Comm(comm))));
           }),
           "nb_grid_pts"_a, "communicator"_a = size_t(MPI_COMM_SELF))
#endif
      .def("fft", &Engine::fft)
      .def("ifft", &Engine::ifft)
      .def(
          "initialise",
          [](Engine & engine, const Index_t & nb_dof_per_pixel,
             const muFFT::FFT_PlanFlags & plan_flags) {
            engine.initialise(nb_dof_per_pixel, plan_flags);
          },
          "nb_dof_per_pixel"_a, "flags"_a = muFFT::FFT_PlanFlags::estimate)
      .def("register_fourier_space_field",
           &muFFT::FFTEngineBase::register_fourier_space_field, "unique_name"_a,
           "nb_dof_per_pixel"_a, py::return_value_policy::reference_internal)
      .def(
          "create_or_fetch_fourier_space_field",
          [](Engine & engine, const std::string & unique_name,
             const Index_t & nb_dof_per_pixel) ->
          typename Engine::FourierField_t & {
            auto && collection{engine.get_fourier_field_collection()};
            if (collection.field_exists(unique_name)) {
              auto & field{dynamic_cast<typename Engine::FourierField_t &>(
                  collection.get_field(unique_name))};
              if (field.get_nb_dof_per_pixel() != nb_dof_per_pixel) {
                std::stringstream message{};
                message << "There is already a field named '" << unique_name
                        << "', but it holds " << field.get_nb_dof_per_pixel()
                        << " degrees of freedom per pixel, and not "
                        << nb_dof_per_pixel << ", as requested";
                throw muFFT::FFTEngineError{message.str()};
              }
              return field;
            } else {
              return engine.register_fourier_space_field(unique_name,
                                                         nb_dof_per_pixel);
            }
          },
          "unique_name"_a, "nb_dof_per_pixel"_a,
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
      });
  if (fourier_space_arrays_are_col_major) {
    fft_engine  // Interface for passing numpy arrays
        .def(
            "fft",
            [](Engine & eng,
               py::array_t<Real, py::array::f_style> & input_array,
               py::array_t<Complex> & output_array) {
              if (not(output_array.flags() & py::detail::npy_api::constants::
                                                 NPY_ARRAY_F_CONTIGUOUS_)) {
                throw muFFT::FFTEngineError(
                    "Can't use row_major output arrays, as the result would be "
                    "written into a temporary copy.");
              }
              auto && nb_dof_per_pixel{input_array.size() / eng.size()};
              if (nb_dof_per_pixel * eng.size() !=
                  static_cast<size_t>(input_array.size())) {
                std::stringstream message{};
                message
                    << "Cannot determine the number of degrees of freedom per "
                       "pixel: The supplied input array's size ("
                    << input_array.size()
                    << ") is not an integer multiple of the number of pixels ("
                    << eng.size() << " = " << input_array.size() << " / "
                    << Real(input_array.size()) / eng.size() << ".";
                throw muFFT::FFTEngineError{message.str()};
              }

              NumpyProxy<Real> input_proxy(eng.get_nb_subdomain_grid_pts(),
                                           eng.get_subdomain_locations(),
                                           nb_dof_per_pixel, input_array);
              auto info{output_array.request()};
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
               py::array_t<Complex, py::array::f_style> & input_array,
               py::array_t<Real> & output_array) {
              if (not(output_array.flags() & py::detail::npy_api::constants::
                                                 NPY_ARRAY_F_CONTIGUOUS_)) {
                throw muFFT::FFTEngineError(
                    "Can't use row_major output arrays, as the result would be "
                    "written into a temporary copy.");
              }
              auto && nb_dof_per_pixel{output_array.size() / eng.size()};
              if (nb_dof_per_pixel * eng.size() !=
                  static_cast<size_t>(output_array.size())) {
                std::stringstream message{};
                message
                    << "Cannot determine the number of degrees of freedom per "
                       "pixel: The supplied output array's size ("
                    << output_array.size()
                    << ") is not an integer multiple of the number of pixels ("
                    << eng.size() << " = " << output_array.size() << " / "
                    << Real(output_array.size()) / eng.size() << ".";
                throw muFFT::FFTEngineError{message.str()};
              }

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
  } else {
    fft_engine  // Interface for passing numpy arrays
        .def(
            "fft",
            [](Engine & eng,
               py::array_t<Real, py::array::f_style> & input_array,
               py::array_t<Complex> & output_array) {
              auto && nb_dof_per_pixel{input_array.size() / eng.size()};
              if (nb_dof_per_pixel * eng.size() !=
                  static_cast<size_t>(input_array.size())) {
                std::stringstream message{};
                message
                    << "Cannot determine the number of degrees of freedom per "
                       "pixel: The supplied input array's size ("
                    << input_array.size()
                    << ") is not an integer multiple of the number of pixels ("
                    << eng.size() << " = " << input_array.size() << " / "
                    << Real(input_array.size()) / eng.size() << ".";
                throw muFFT::FFTEngineError{message.str()};
              }

              NumpyProxy<Real> input_proxy(eng.get_nb_subdomain_grid_pts(),
                                           eng.get_subdomain_locations(),
                                           nb_dof_per_pixel, input_array);

              py::buffer_info info{output_array.request()};
              NumpyProxy<Complex> output_proxy(eng.get_nb_fourier_grid_pts(),
                                               eng.get_fourier_locations(),
                                               nb_dof_per_pixel, output_array);
              auto && input_proxy_field{input_proxy.get_field()};
              // create or fetch a temporary output buffer with proper memory
              // layout
              std::stringstream identifier{};
              identifier << "temporary_fourier_field_nb_dof_"
                         << nb_dof_per_pixel;
              auto & temp_output{eng.fetch_or_register_fourier_space_field(
                  identifier.str(), nb_dof_per_pixel)};
              eng.fft(input_proxy_field, temp_output);
              output_proxy.get_field() = temp_output;
            },
            "real_input_array"_a, "complex_output_array"_a,
            "Perform forward FFT of the input array into the output array")
        .def(
            "ifft",
            [](Engine & eng, py::array_t<Complex> & input_array,
               py::array_t<Real> & output_array) {
              auto && nb_dof_per_pixel{output_array.size() / eng.size()};
              if (nb_dof_per_pixel * eng.size() !=
                  static_cast<size_t>(output_array.size())) {
                std::stringstream message{};
                message
                    << "Cannot determine the number of degrees of freedom per "
                       "pixel: The supplied output array's size ("
                    << output_array.size()
                    << ") is not an integer multiple of the number of pixels ("
                    << eng.size() << " = " << output_array.size() << " / "
                    << Real(output_array.size()) / eng.size() << ".";
                throw muFFT::FFTEngineError{message.str()};
              }

              GlobalFieldCollection collection(eng.get_spatial_dim(),
                                               eng.get_nb_subdomain_grid_pts(),
                                               eng.get_subdomain_locations());
              // static variable in order to make it reusable
              thread_local std::vector<Real> temp_output_data{};
              temp_output_data.resize(eng.size() * nb_dof_per_pixel);
              muGrid::WrappedField<Real> temp_output_field{
                  "output",
                  collection,
                  static_cast<Index_t>(nb_dof_per_pixel),
                  nb_dof_per_pixel * collection.get_nb_pixels(),
                  temp_output_data.data(),
                  muFFT::PixelTag};

              // create or fetch a temporary output buffer with proper memory
              // layout
              std::stringstream identifier{};
              identifier << "temporary_fourier_field_nb_dof_"
                         << nb_dof_per_pixel;
              auto & temp_input{eng.fetch_or_register_fourier_space_field(
                  identifier.str(), nb_dof_per_pixel)};
              {
                auto && info{input_array.request()};
                auto && np_strides{info.strides};
                std::vector<Index_t> input_strides{};
                input_strides.reserve(np_strides.size());
                for (auto && val : np_strides) {
                  input_strides.push_back(val / info.itemsize);
                }
                std::vector<Index_t> shape{};
                shape.reserve(info.shape.size());
                for (auto && val : info.shape) {
                  shape.push_back(val);
                }
                // the first entries of the shape can be chosen by the user
                std::vector<Index_t> component_shape{};
                for (size_t i{0}; i < shape.size() - eng.get_spatial_dim();
                     ++i) {
                  component_shape.push_back(shape[i]);
                }

                auto && field_strides{temp_input.get_strides(component_shape)};
                muGrid::raw_mem_ops::strided_copy(
                    shape, input_strides, field_strides, info.ptr,
                    temp_input.data(), sizeof(Complex));
                eng.ifft(temp_input, temp_output_field);
              }

              // compute strides from numpy strides
              {
                auto && info{output_array.request()};
                auto && np_strides{info.strides};
                std::vector<Index_t> return_strides{};
                return_strides.reserve(np_strides.size());
                for (auto && val : np_strides) {
                  return_strides.push_back(val / info.itemsize);
                }
                std::vector<Index_t> shape{};
                shape.reserve(info.shape.size());
                for (auto && val : info.shape) {
                  shape.push_back(val);
                }
                // the first entries of the shape can be chosen by the user
                std::vector<Index_t> component_shape{};
                for (size_t i{0}; i < shape.size() - eng.get_spatial_dim();
                     ++i) {
                  component_shape.push_back(shape[i]);
                }
                auto && field_strides{
                    temp_output_field.get_strides(component_shape)};
                muGrid::raw_mem_ops::strided_copy(
                    shape, field_strides, return_strides,
                    temp_output_data.data(), info.ptr, sizeof(Real));
              }
            },
            "fourier_input_array"_a, "real_output_array"_a,
            "Perform inverse FFT of the input array into the output array.");
  }
}

void add_fft_engines(py::module & mod) {
  add_fft_engine_base(mod);
  constexpr bool col_major{true};
#if defined WITH_FFTWMPI || defined WITH_PFFT
  constexpr bool not_col_major{false};
#endif
  add_engine_helper<muFFT::FFTWEngine>(mod, "FFTW", col_major);
#ifdef WITH_FFTWMPI
  add_engine_helper<muFFT::FFTWMPIEngine>(mod, "FFTWMPI", not_col_major);
#endif
#ifdef WITH_PFFT
  add_engine_helper<muFFT::PFFTEngine>(mod, "PFFT", not_col_major);
#endif
}
