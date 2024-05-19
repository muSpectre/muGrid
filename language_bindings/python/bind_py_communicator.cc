/**
 * @file   bind_py_communicator.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   22 May 2019
 *
 * @brief  Python bindings for the muGrid Communicator
 *
 * Copyright © 2018 Till Junge
 *
 * µGrid is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µGrid is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µGrid; see the file COPYING. If not, write to the
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

#include "bind_py_declarations.hh"

#include "libmugrid/communicator.hh"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

using pybind11::literals::operator""_a;
using muGrid::Int;
using muGrid::Real;
using muGrid::Uint;
using muGrid::Complex;
using muGrid::Index_t;
namespace py = pybind11;

void add_communicator(py::module & mod) {
  py::class_<muGrid::Communicator>(mod, "Communicator")
#ifdef WITH_MPI
      .def(py::init([](size_t comm) {
             return new muGrid::Communicator(MPI_Comm(comm));
           }),
           "communicator"_a = size_t(MPI_COMM_SELF))
      .def_property_readonly("mpi_comm",
                             [](muGrid::Communicator & comm) {
                               return size_t(comm.get_mpi_comm());
                             })
#else
      .def(py::init())
#endif
      .def_property_readonly_static(
          "has_mpi", [](py::object) { return muGrid::Communicator::has_mpi(); })
      .def_property_readonly("rank", &muGrid::Communicator::rank)
      .def_property_readonly("size", &muGrid::Communicator::size)
      .def("barrier", &muGrid::Communicator::barrier)
      .def("sum", [](muGrid::Communicator & comm,
                     const int & arg) { return comm.sum(arg); })
      .def("sum", [](muGrid::Communicator & comm,
                     const Real & arg) { return comm.sum(arg); })
      .def("sum",
           [](muGrid::Communicator & comm,
              const Eigen::Ref<muGrid::DynMatrix_t<Real>> & arg) {
             return comm.sum(arg);
           })
      .def("sum",
           [](muGrid::Communicator & comm,
              const Eigen::Ref<muGrid::DynMatrix_t<Int>> & arg) {
             return comm.sum(arg);
           })
      .def("sum",
           [](muGrid::Communicator & comm,
              const Eigen::Ref<muGrid::DynMatrix_t<Uint>> & arg) {
             return comm.sum(arg);
           })
      .def("sum",
           [](muGrid::Communicator & comm,
              const Eigen::Ref<muGrid::DynMatrix_t<Index_t>> & arg) {
             return comm.sum(arg);
           })
      .def("sum",
           [](muGrid::Communicator & comm,
              const Eigen::Ref<muGrid::DynMatrix_t<Complex>> & arg) {
             return comm.sum(arg);
           })
      .def("cumulative_sum", &muGrid::Communicator::cumulative_sum<Int>)
      .def("cumulative_sum", &muGrid::Communicator::cumulative_sum<Real>)
      .def("cumulative_sum", &muGrid::Communicator::cumulative_sum<Uint>)
      .def("cumulative_sum", &muGrid::Communicator::cumulative_sum<Index_t>)
      .def("cumulative_sum", &muGrid::Communicator::cumulative_sum<Complex>)
      .def("gather",
           [](muGrid::Communicator & comm,
              const Eigen::Ref<muGrid::DynMatrix_t<Real>> & arg) {
             return comm.gather(arg);
           })
      .def("gather",
           [](muGrid::Communicator & comm,
              const Eigen::Ref<muGrid::DynMatrix_t<Int>> & arg) {
             return comm.gather(arg);
           })
      .def("gather",
           [](muGrid::Communicator & comm,
              const Eigen::Ref<muGrid::DynMatrix_t<Uint>> & arg) {
             return comm.gather(arg);
           })
      .def("gather",
           [](muGrid::Communicator & comm,
              const Eigen::Ref<muGrid::DynMatrix_t<Index_t>> & arg) {
             return comm.gather(arg);
           })
      .def("gather",
           [](muGrid::Communicator & comm,
              const Eigen::Ref<muGrid::DynMatrix_t<Complex>> & arg) {
             return comm.gather(arg);
           })
      .def(
          "bcast",
          [](muGrid::Communicator & comm, Real & scalar_arg, const Int & root) {
            return comm.bcast<Real>(scalar_arg, root);
          },
          "scalar_arg"_a, "root"_a)
      .def(
          "bcast",
          [](muGrid::Communicator & comm, Int & scalar_arg, const Int & root) {
            return comm.bcast<Int>(scalar_arg, root);
          },
          "scalar_arg"_a, "root"_a)
      .def(
          "bcast",
          [](muGrid::Communicator & comm, Uint & scalar_arg, const Int & root) {
            return comm.bcast<Uint>(scalar_arg, root);
          },
          "scalar_arg"_a, "root"_a)
      .def(
          "bcast",
          [](muGrid::Communicator & comm, Index_t & scalar_arg,
             const Int & root) {
            return comm.bcast<Index_t>(scalar_arg, root);
          },
          "scalar_arg"_a, "root"_a)
      .def(
          "bcast",
          [](muGrid::Communicator & comm, Complex & scalar_arg,
             const Int & root) {
            return comm.bcast<Complex>(scalar_arg, root);
          },
          "scalar_arg"_a, "root"_a);
}
