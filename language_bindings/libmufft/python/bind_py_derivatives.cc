/**
 * @file   bind_py_derivatives.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   17 Jun 2019
 *
 * @brief  Python bindings for the derivative operators
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
 * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 *
 */

#include <memory>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <libmugrid/ccoord_operations.hh>
#include <libmugrid/numpy_tools.hh>

#include <libmufft/derivative.hh>
#include <libmufft/fft_engine_base.hh>

using muFFT::DerivativeBase;
using muFFT::DiscreteDerivative;
using muFFT::FourierDerivative;
using muGrid::Complex;
using muGrid::DynCcoord_t;
using muGrid::Index_t;
using muGrid::Real;
using muGrid::threeD;
using muGrid::twoD;
using muGrid::NumpyProxy;
using muFFT::FFTEngineBase;
using pybind11::literals::operator""_a;
namespace py = pybind11;

/**
 * "Trampoline" class for handling the pure virtual methods, see
 * [http://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python]
 * for details
 */
class PyDerivativeBase : public DerivativeBase {
 public:
  //! base class
  using Parent = DerivativeBase;
  //! coordinate field
  using Vector = typename Parent::Vector;

  explicit PyDerivativeBase(Index_t spatial_dimension)
      : DerivativeBase(spatial_dimension) {}

  virtual Complex fourier(const Vector & wavevec) const {
    PYBIND11_OVERLOAD_PURE(Complex, DerivativeBase, fourier, wavevec);
  }
};

void add_derivative_base(py::module & mod, std::string name) {
  py::class_<DerivativeBase,                   // class
             std::shared_ptr<DerivativeBase>,  // holder
             PyDerivativeBase                  // trampoline base
             >(mod, name.c_str())
      .def(py::init<Index_t>())
      .def("fourier", &DerivativeBase::fourier, "wavevec"_a,
           "return Fourier representation of the derivative operator for a "
           "certain wavevector")
      .def(
          "fourier",
          [](DerivativeBase & derivative,
             py::array_t<Real, py::array::f_style> wavevectors) {
            py::buffer_info wavevectors_buffer = wavevectors.request();
            std::vector<ssize_t> output_shape;
            // The first dimension contains the components of The wavevector.
            // This is equal to the dimension of space.
            ssize_t nb_components{wavevectors_buffer.shape[0]};
            // The next dimensions simply hold entries and we don't care about
            // the shape. The return array should have the same shape, minus
            // the first dimension.
            ssize_t nb_entries = 1;
            for (int i = 1; i < wavevectors_buffer.ndim; ++i) {
              output_shape.push_back(wavevectors_buffer.shape[i]);
              nb_entries *= wavevectors_buffer.shape[i];
            }
            // Create output array with appropriate shape.
            py::array_t<Complex, py::array::f_style> factors(output_shape);
            py::buffer_info factors_buffer = factors.request();
            // Loop over all entries and call fourier method.
            auto wavevector = static_cast<const Real *>(wavevectors_buffer.ptr);
            for (ssize_t i = 0; i < nb_entries; ++i) {
              static_cast<Complex *>(factors_buffer.ptr)[i] =
                  derivative.fourier(Eigen::Map<const DerivativeBase::Vector>(
                      wavevector, nb_components));
              wavevector += nb_components;
            }
            return factors;
          },
          "wavevectors"_a,
          "return Fourier representation of the derivative operator for a "
          "certain wavevector");
}

void add_fourier_derivative(py::module & mod, std::string name) {
  py::class_<FourierDerivative,                   // class
             std::shared_ptr<FourierDerivative>,  // holder
             DerivativeBase                       // base class
             >(mod, name.c_str())
      .def(py::init<Index_t, Index_t>(), "spatial_dimension"_a, "direction"_a)
      .def(py::init([](Index_t spatial_dimension, Index_t direction,
                       const Eigen::ArrayXd & shift) {
             // Default: shift = vector (of correct dimension) filled with
             // zeros
             if ((shift.size() == 1) and (shift(0, 0) == 0)) {
               Eigen::VectorXd default_shift{
                   Eigen::ArrayXd::Zero(spatial_dimension)};
               return new FourierDerivative(spatial_dimension, direction,
                                            default_shift);
             }
             // is the shift vector correctly given?
             if (shift.size() != spatial_dimension) {
               std::stringstream s;
               s << "The real space shift has " << shift.size() << " entries, "
                 << "but the Fourier derivative is " << spatial_dimension
                 << "D.";
               throw muGrid::RuntimeError(s.str());
             }
             return new FourierDerivative(spatial_dimension, direction, shift);
           }),
           "spatial_dimension"_a, "direction"_a, "shift"_a = 0);
}

void add_discrete_derivative(py::module & mod, std::string name) {
  py::class_<DiscreteDerivative,                   // class
             std::shared_ptr<DiscreteDerivative>,  // holder
             DerivativeBase                        // base class
             >(mod, name.c_str())
      .def(py::init([](const DynCcoord_t & lbounds,
                       py::array_t<Real, py::array::f_style> stencil) {
             const py::buffer_info & info = stencil.request();
             if (info.ndim != lbounds.get_dim()) {
               std::stringstream s;
               s << "Stencil bounds have " << lbounds.get_dim() << " entries, "
                 << "but stencil itself is " << info.ndim << "-dimensional.";
               throw muGrid::RuntimeError(s.str());
             }
             DynCcoord_t nb_pts(info.ndim);
             for (int i = 0; i < info.ndim; ++i) {
               nb_pts[i] = info.shape[i];
             }
             return new DiscreteDerivative(
                 nb_pts, lbounds,
                 Eigen::Map<Eigen::ArrayXd>(static_cast<double *>(info.ptr),
                                            info.size));
           }),
           "lbounds"_a, "stencil"_a,
           "Constructor with raw stencil information\n"
           "nb_pts: stencil size\n"
           "lbounds: relative starting point of stencil, e.g. (-2,) means\n"
           "         that the stencil start two pixels to the left of where\n"
           "         the derivative should be computed\n"
           "stencil: stencil coefficients")
      .def("rollaxes", &DiscreteDerivative::rollaxes, "distance"_a = 1)
      .def("apply", &DiscreteDerivative::apply<Real>, "in_field"_a, "in_dof"_a,
           "out_field"_a, "out_dof"_a, "fac"_a = 1.0)
    .def_property_readonly("lbounds", &DiscreteDerivative::get_lbounds)
    .def(
         "apply",
         [](const DiscreteDerivative & self,
            py::array_t<Real, py::array::f_style> & input_array) {
           const py::buffer_info & info = input_array.request();
           if (info.ndim != self.get_dim()) {
             std::stringstream s;
             s << "Stencil is " << self.get_dim() << "-dimensional, "
               << "but the input array is " << input_array.ndim()
               << "-dimensional.";
             throw muGrid::RuntimeError(s.str());
           }
           py::array_t<double, py::array::f_style> output_array(info.shape);
           DynCcoord_t nb_domain_grid_pts{info.shape};
           DynCcoord_t subdomain_locations(info.ndim);
           NumpyProxy<Real, py::array::f_style> input_proxy(
                nb_domain_grid_pts, nb_domain_grid_pts,
                subdomain_locations, 1, input_array);
           NumpyProxy<Real, py::array::f_style> output_proxy(
               nb_domain_grid_pts, nb_domain_grid_pts,
               subdomain_locations, 1, output_array);

           self.apply(input_proxy.get_field(), 0, output_proxy.get_field(), 0);
           return output_array;
         },
         "input_array"_a,
         "Apply the discrete derivative stencil to the input array.")

      .def_property_readonly("stencil", [](const DiscreteDerivative & self) {
        const Eigen::ArrayXd & stencil = self.get_stencil();
        return py::array_t<double, py::array::f_style>(self.get_nb_pts(),
                                                       stencil.data());
      });
}

void add_derivatives(py::module & mod) {
  add_derivative_base(mod, "DerivativeBase");
  add_fourier_derivative(mod, "FourierDerivative");
  add_discrete_derivative(mod, "DiscreteDerivative");
}
