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

#include "projection/derivative.hh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <sstream>
#include <memory>

using muGrid::Real;
using muGrid::Complex;
using muGrid::Dim_t;
using muGrid::twoD;
using muGrid::threeD;
using muSpectre::DerivativeBase;
using muSpectre::FourierDerivative;
using muSpectre::DiscreteDerivative;
using pybind11::literals::operator""_a;
namespace py = pybind11;

/**
 * "Trampoline" class for handling the pure virtual methods, see
 * [http://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python]
 * for details
 */
template <Dim_t DimS>
class PyDerivativeBase : public DerivativeBase<DimS> {
 public:
  //! base class
  using Parent = DerivativeBase<DimS>;
  //! coordinate field
  using Vector = typename Parent::Vector;

  virtual Complex fourier(const Vector & wavevec) const {
    PYBIND11_OVERLOAD_PURE(Complex, DerivativeBase<DimS>, fourier, wavevec);
  }
};

template <Dim_t DimS>
void add_derivative_base(py::module & mod, std::string name_start) {
  std::stringstream name{};
  name << name_start << '_' << DimS << 'd';

  py::class_<DerivativeBase<DimS>,                   // class
             std::shared_ptr<DerivativeBase<DimS>>,  // holder
             PyDerivativeBase<DimS>                  // trampoline
             >(mod, name.str().c_str())
      .def(py::init<>())
      .def("fourier", &DerivativeBase<DimS>::fourier,
           "wavevec"_a,
           "return Fourier representation of the derivative operator for a "
           "certain wavevector");
}

template <Dim_t DimS>
void add_fourier_derivative(py::module & mod, std::string name_start) {
  using ArrayXDd = Eigen::Array<Real, DimS, Eigen::Dynamic, Eigen::RowMajor>;
  using ArrayXc = Eigen::Array<Complex, Eigen::Dynamic, 1>;

  std::stringstream name{};
  name << name_start << '_' << DimS << 'd';

  py::class_<FourierDerivative<DimS>,                   // class
             std::shared_ptr<FourierDerivative<DimS>>,  // holder
             DerivativeBase<DimS>                       // base class
             >(mod, name.str().c_str())
      .def(py::init<Dim_t>(),
           "direction"_a)
      .def("fourier", &FourierDerivative<DimS>::fourier,
           "wavevec"_a,
           "return Fourier representation of the derivative operator for a "
           "certain wavevector")
      .def("fourier",
           [](FourierDerivative<DimS> & derivative,
              const Eigen::Ref<ArrayXDd> & wavevectors) {
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

template <Dim_t DimS>
void add_discrete_derivative(py::module & mod, std::string name_start) {
  using Ccoord = typename DiscreteDerivative<DimS>::Ccoord;
  /* TODO: Belong to the second "fourier" implementation below
  using ArrayDXd = Eigen::Array<Real, Eigen::Dynamic, DimS, Eigen::RowMajor>; */
  using ArrayXDd = Eigen::Array<Real, DimS, Eigen::Dynamic, Eigen::RowMajor>;
  using ArrayXc = Eigen::Array<Complex, Eigen::Dynamic, 1>;

  std::stringstream name{};
  name << name_start << '_' << DimS << 'd';

  py::class_<DiscreteDerivative<DimS>,                   // class
             std::shared_ptr<DiscreteDerivative<DimS>>,  // holder
             DerivativeBase<DimS>                        // base class
             >(mod, name.str().c_str())
      .def(py::init<Ccoord, Ccoord, const Eigen::ArrayXd &>())
      .def("fourier", &DiscreteDerivative<DimS>::fourier,
           "wavevector"_a,
           "return Fourier representation of the derivative operator for a "
           "certain wavevector")
      /* TODO: Decide if we need both functions below. There can be confusion for
         symmetric matrices.
      .def("fourier",
           [](DiscreteDerivative<DimS> & derivative,
              const Eigen::Ref<ArrayDXd> & wavevectors) {
             ArrayXc factors(wavevectors.rows());
             for (int i = 0; i < wavevectors.rows(); ++i) {
               factors[i] = derivative.fourier(wavevectors.row(i));
             }
             return factors;
           },
           "wavevectors"_a,
           "return Fourier representation of the derivative operator for a "
           "certain wavevector")
      */
      .def("fourier",
           [](DiscreteDerivative<DimS> & derivative,
              const Eigen::Ref<ArrayXDd> & wavevectors) {
             ArrayXc factors(wavevectors.cols());
             for (int i = 0; i < wavevectors.cols(); ++i) {
               factors[i] = derivative.fourier(wavevectors.col(i));
             }
             return factors;
           },
           "wavevectors"_a,
           "return Fourier representation of the derivative operator for a "
           "certain wavevector")
      .def("rollaxes", &DiscreteDerivative<DimS>::rollaxes,
           "distance"_a = 1);
}

void add_derivatives(py::module & mod) {
  add_derivative_base<twoD>(mod, "DerivativeBase");
  add_derivative_base<threeD>(mod, "DerivativeBase");

  add_fourier_derivative<twoD>(mod, "FourierDerivative");
  add_fourier_derivative<threeD>(mod, "FourierDerivative");

  add_discrete_derivative<twoD>(mod, "DiscreteDerivative");
  add_discrete_derivative<threeD>(mod, "DiscreteDerivative");
}
