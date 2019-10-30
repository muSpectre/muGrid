/**
 * @file   bind_py_communicator.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   22 May 2019
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

#include <libmufft/communicator.hh>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Eigen/Dense>

using pybind11::literals::operator""_a;
using muFFT::Int;
using muFFT::Real;
using muFFT::Uint;
using muFFT::Complex;
namespace py = pybind11;

void add_communicator(py::module & mod) {
  py::class_<muFFT::Communicator>(mod, "Communicator")
#ifdef WITH_MPI
      .def(py::init([](size_t comm) {
             return new muFFT::Communicator(MPI_Comm(comm));
           }),
           "communicator"_a = size_t(MPI_COMM_SELF))
      .def_property_readonly("mpi_comm",
                             [](muFFT::Communicator & comm) {
                               return size_t(comm.get_mpi_comm());
                             })
#else
      .def(py::init())
#endif
      .def_property_readonly_static(
          "has_mpi", [](py::object) { return muFFT::Communicator::has_mpi(); })
      .def_property_readonly("rank", &muFFT::Communicator::rank)
      .def_property_readonly("size", &muFFT::Communicator::size)
      .def("sum", &muFFT::Communicator::sum<Int>)
      .def("sum", &muFFT::Communicator::sum<Real>)
      .def("sum", &muFFT::Communicator::sum_mat<Real>)
      .def("sum", &muFFT::Communicator::sum_mat<Int>)
      .def("sum", &muFFT::Communicator::sum_mat<Uint>)
      .def("sum", &muFFT::Communicator::sum_mat<Complex>)
      .def("gather", &muFFT::Communicator::gather<Real>)
      .def("gather", &muFFT::Communicator::gather<Int>)
      .def("gather", &muFFT::Communicator::gather<Uint>)
      .def("gather", &muFFT::Communicator::gather<Complex>);
}
