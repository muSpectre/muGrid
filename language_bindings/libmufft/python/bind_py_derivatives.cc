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

#include <memory>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "libmufft/derivative.hh"

using muGrid::Real;
using muGrid::Complex;
using muGrid::Dim_t;
using muGrid::twoD;
using muGrid::threeD;
using muGrid::DynCcoord_t;
using muFFT::DerivativeBase;
using muFFT::FourierDerivative;
using muFFT::DiscreteDerivative;
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

  explicit PyDerivativeBase(Dim_t spatial_dimension)
      : DerivativeBase(spatial_dimension) {}

  virtual Complex fourier(const Vector & wavevec) const {
    PYBIND11_OVERLOAD_PURE(Complex, DerivativeBase, fourier, wavevec);
  }
};

void add_derivative_base(py::module & mod, std::string name) {
  py::class_<DerivativeBase,                   // class
             std::shared_ptr<DerivativeBase>,  // holder
             PyDerivativeBase                  // trampoline
             >(mod, name.c_str())
      .def(py::init<Dim_t>())
      .def("fourier", &DerivativeBase::fourier,
           "wavevec"_a,
           "return Fourier representation of the derivative operator for a "
           "certain wavevector");
}

void add_fourier_derivative(py::module & mod, std::string name) {
  using ArrayXXd = Eigen::Array<Real, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>;
  using ArrayXc = Eigen::Array<Complex, Eigen::Dynamic, 1>;

  py::class_<FourierDerivative,                   // class
             std::shared_ptr<FourierDerivative>,  // holder
             DerivativeBase                       // base class
             >(mod, name.c_str())
      .def(py::init<Dim_t, Dim_t>(),
           "spatial_dimension"_a, "direction"_a)
      .def("fourier", &FourierDerivative::fourier,
           "wavevec"_a,
           "return Fourier representation of the derivative operator for a "
           "certain wavevector")
      .def("fourier",
           [](FourierDerivative & derivative,
              const Eigen::Ref<ArrayXXd> & wavevectors) {
             ArrayXc factors(wavevectors.cols());
             for (int i = 0; i < wavevectors.cols(); ++i) {
               factors[i] = derivative.fourier(wavevectors.col(i));
             }
             return factors;
           },
           "wavevectors"_a,
           "return Fourier representation of the derivative operator for a "
           "certain wavevector");
}

void add_discrete_derivative(py::module & mod, std::string name) {
  using ArrayXXd = Eigen::Array<Real, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>;
  using ArrayXc = Eigen::Array<Complex, Eigen::Dynamic, 1>;

  py::class_<DiscreteDerivative,                   // class
             std::shared_ptr<DiscreteDerivative>,  // holder
             DerivativeBase                        // base class
             >(mod, name.c_str())
      .def(py::init([](std::vector<Dim_t> nb_pts, std::vector<Dim_t> lbounds,
                       const Eigen::ArrayXd &stencil) {
             return new DiscreteDerivative(DynCcoord_t(nb_pts),
                                           DynCcoord_t(lbounds), stencil);
           }),
           "nb_pts"_a, "lbounds"_a, "stencil"_a)
      .def("fourier", &DiscreteDerivative::fourier,
           "wavevector"_a,
           "return Fourier representation of the derivative operator for a "
           "certain wavevector")
      .def("fourier",
           [](DiscreteDerivative & derivative,
              const Eigen::Ref<ArrayXXd> & wavevectors) {
             ArrayXc factors(wavevectors.cols());
             for (int i = 0; i < wavevectors.cols(); ++i) {
               factors[i] = derivative.fourier(wavevectors.col(i));
             }
             return factors;
           },
           "wavevectors"_a,
           "return Fourier representation of the derivative operator for a "
           "certain wavevector")
      .def("rollaxes", &DiscreteDerivative::rollaxes,
           "distance"_a = 1);
}

void add_derivatives(py::module & mod) {
  add_derivative_base(mod, "DerivativeBase");
  add_fourier_derivative(mod, "FourierDerivative");
  add_discrete_derivative(mod, "DiscreteDerivative");
}
