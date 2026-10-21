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

#include "mpi/communicator.hh"
#include "mpi/gpu_aware_mpi.hh"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <cstdint>
#include <memory>
#include <string>

using pybind11::literals::operator""_a;
using muGrid::Int;
using muGrid::Real;
using muGrid::Uint;
using muGrid::Complex;
using muGrid::Index_t;
using muGrid::BufferLocation;
using muGrid::TypeDescriptor;
namespace py = pybind11;

namespace {
  /**
   * A NumPy or CuPy array as the buffer exchanges of Communicator see it: a
   * C-contiguous copy (or the array itself if it already is one), read
   * through the array interface its library provides -- CuPy's
   * __cuda_array_interface__ (CUDA and ROCm builds alike) or NumPy's
   * __array_interface__.
   */
  struct ArrayBuffer {
    py::object contiguous;  //!< keeps the buffer alive
    py::object shape;
    py::object dtype;
    void * data;
    Index_t count;
    TypeDescriptor type;
    BufferLocation location;
  };

  py::module_ array_module(BufferLocation location) {
    return py::module_::import(location == BufferLocation::Device ? "cupy"
                                                                  : "numpy");
  }

  TypeDescriptor type_of(const std::string & typestr) {
    // e.g. "<f8": byte order, kind, item size in bytes
    if (typestr.size() >= 3 && typestr[0] != '>') {
      const std::string kind_size{typestr.substr(1)};
      if (kind_size == "f8") return TypeDescriptor::Real;
      if (kind_size == "f4") return TypeDescriptor::Real32;
      if (kind_size == "c16") return TypeDescriptor::Complex;
      if (kind_size == "c8") return TypeDescriptor::Complex32;
      if (kind_size == "i4") return TypeDescriptor::Int;
      if (kind_size == "u4") return TypeDescriptor::Uint;
      if (kind_size == "i" + std::to_string(sizeof(Index_t))) {
        return TypeDescriptor::Index;
      }
    }
    throw py::type_error("no MPI exchange for arrays of dtype " + typestr +
                         "; use float32/64, complex64/128, int32/64 or "
                         "uint32");
  }

  py::dict interface_of(const py::object & array, BufferLocation location) {
    return array.attr(location == BufferLocation::Device
                          ? "__cuda_array_interface__"
                          : "__array_interface__");
  }

  void * data_of(const py::dict & interface) {
    return reinterpret_cast<void *>(
        interface["data"].cast<py::tuple>()[0].cast<std::uintptr_t>());
  }

  ArrayBuffer buffer_of(const py::object & a) {
    const BufferLocation location{py::hasattr(a, "__cuda_array_interface__")
                                      ? BufferLocation::Device
                                      : BufferLocation::Host};
    auto xp{array_module(location)};
    py::object array{xp.attr("asarray")(a)};
    py::object contiguous{xp.attr("ascontiguousarray")(array)};
    py::dict interface{interface_of(contiguous, location)};
    Index_t count{1};
    for (auto extent : array.attr("shape").cast<py::tuple>()) {
      count *= extent.cast<Index_t>();
    }
    return ArrayBuffer{contiguous,
                       array.attr("shape"),
                       array.attr("dtype"),
                       data_of(interface),
                       count,
                       type_of(interface["typestr"].cast<std::string>()),
                       location};
  }

  //! A new array of `shape` and the dtype and library of `like`, with the
  //! buffer to write into
  std::pair<py::object, void *> empty_like(const ArrayBuffer & like,
                                           const py::object & shape) {
    py::object result{
        array_module(like.location).attr("empty")(shape, "dtype"_a = like.dtype)};
    return {result, data_of(interface_of(result, like.location))};
  }

  //! Let the CuPy stream that wrote a device array finish: the exchanges
  //! drain the default stream, which need not be the one CuPy uses.
  void wait_for_writer(const ArrayBuffer & buffer) {
    if (buffer.location == BufferLocation::Device) {
      py::module_::import("cupy")
          .attr("cuda")
          .attr("get_current_stream")()
          .attr("synchronize")();
    }
  }

  //! A non-blocking allreduce with the arrays it reads and writes. The
  //! request is declared last so that it is destroyed -- and so waited
  //! for -- first, while the arrays are still alive.
  struct PyAllreduceRequest {
    py::object send;
    py::object result;
    muGrid::AllreduceRequest request;
  };

  muGrid::ReduceOp reduce_op(const std::string & op) {
    if (op == "sum") return muGrid::ReduceOp::Sum;
    if (op == "max") return muGrid::ReduceOp::Max;
    if (op == "min") return muGrid::ReduceOp::Min;
    throw py::value_error("allreduce: op must be 'sum', 'max' or 'min', not '" +
                          op + "'");
  }
}  // namespace

void add_communicator(py::module & mod) {
  py::class_<PyAllreduceRequest>(mod, "AllreduceRequest",
                                 "A reduction started by "
                                 "Communicator.iallreduce.")
      .def(
          "wait",
          [](PyAllreduceRequest & self) {
            {
              py::gil_scoped_release release{};
              self.request.wait();
            }
            return self.result;
          },
          "Complete the reduction and return its result (again on later "
          "calls).");

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
      // Whether MPI may be handed device pointers (MUGRID_GPU_AWARE_MPI, else
      // the MPI library's own report through whichever GPU-support query it
      // provides). For information only: the exchanges below decide by
      // themselves and stage through the host otherwise.
      .def_property_readonly_static(
          "gpu_aware_mpi",
          [](py::object) { return muGrid::mpi_is_gpu_aware(); })
      // Array exchanges for NumPy and CuPy arrays alike. Each returns a new
      // array of the input's library and dtype; device arrays go to MPI
      // directly or through the host, as the communicator decides.
      .def(
          "allreduce",
          [](const muGrid::Communicator & comm, const py::object & a,
             const std::string & op) {
            const auto send{buffer_of(a)};
            auto [result, recv]{empty_like(send, send.shape)};
            const auto reduction{reduce_op(op)};
            wait_for_writer(send);
            {
              py::gil_scoped_release release{};
              comm.allreduce(send.data, recv, send.count, send.type,
                             reduction, send.location);
            }
            return result;
          },
          "a"_a, "op"_a = "sum",
          "Elementwise reduction (op: 'sum', 'max' or 'min') of an array "
          "over all ranks, returned on every rank.")
      .def(
          "iallreduce",
          [](const muGrid::Communicator & comm, const py::object & a,
             const std::string & op) {
            const auto send{buffer_of(a)};
            auto [result, recv]{empty_like(send, send.shape)};
            const auto reduction{reduce_op(op)};
            wait_for_writer(send);
            auto request{std::make_unique<PyAllreduceRequest>()};
            request->send = send.contiguous;
            request->result = result;
            request->request =
                comm.iallreduce(send.data, recv, send.count, send.type,
                                reduction, send.location);
            return request;
          },
          "a"_a, "op"_a = "sum",
          "Start the reduction `allreduce` performs and return at once; the "
          "returned request's wait() completes it and returns the result. "
          "Work that does not need the result can run in between.")
      .def(
          "allgather",
          [](const muGrid::Communicator & comm, const py::object & a) {
            const auto send{buffer_of(a)};
            py::list shape{};
            shape.append(comm.size());
            for (auto extent : send.shape.cast<py::tuple>()) {
              shape.append(extent);
            }
            auto [result, recv]{empty_like(send, py::tuple(shape))};
            wait_for_writer(send);
            {
              py::gil_scoped_release release{};
              comm.allgather(send.data, recv, send.count, send.type,
                             send.location);
            }
            return result;
          },
          "a"_a,
          "Every rank's array, of one shape and dtype on all ranks, stacked "
          "along a new leading axis in rank order, on every rank.")
      .def(
          "sendrecv",
          [](const muGrid::Communicator & comm, const py::object & a,
             int dest, int source, int tag) {
            const auto send{buffer_of(a)};
            auto [result, recv]{empty_like(send, send.shape)};
            wait_for_writer(send);
            {
              py::gil_scoped_release release{};
              comm.sendrecv(send.data, send.count, dest, recv, send.count,
                            source, send.type, send.location, tag);
            }
            return result;
          },
          "a"_a, "dest"_a, "source"_a, "tag"_a = 0,
          "Send an array to rank `dest` and return the one rank `source` "
          "sends, which must have the same shape and dtype.")
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
      .def("max", [](muGrid::Communicator & comm,
                     const int & arg) { return comm.max(arg); })
      .def("max", [](muGrid::Communicator & comm,
                     const Real & arg) { return comm.max(arg); })
      .def("max", [](muGrid::Communicator & comm,
                     const Uint & arg) { return comm.max(arg); })
      .def("max", [](muGrid::Communicator & comm,
                     const Index_t & arg) { return comm.max(arg); })
      .def("min", [](muGrid::Communicator & comm,
                     const int & arg) { return comm.min(arg); })
      .def("min", [](muGrid::Communicator & comm,
                     const Real & arg) { return comm.min(arg); })
      .def("min", [](muGrid::Communicator & comm,
                     const Uint & arg) { return comm.min(arg); })
      .def("min", [](muGrid::Communicator & comm,
                     const Index_t & arg) { return comm.min(arg); })
      // Scalar reductions with the MPI datatype fixed by the method name.
      // The overloaded `sum`/`min`/`max` above are chosen by pybind11 from
      // the *value*: an int beyond 32 bits falls through to the double
      // overload (rounded above 2^53), a complex is cast to its real part,
      // and two ranks whose values pick different overloads disagree on the
      // datatype of one collective. The Python wrapper calls these with the
      // type taken from the dtype instead.
      .def("_sum_float64", [](muGrid::Communicator & comm,
                              const Real & arg) { return comm.sum(arg); })
      .def("_sum_int64", [](muGrid::Communicator & comm,
                            const Index_t & arg) { return comm.sum(arg); })
      .def("_sum_complex128",
           [](muGrid::Communicator & comm, const Complex & arg) {
             return comm.sum(arg);
           })
      .def("_max_float64", [](muGrid::Communicator & comm,
                              const Real & arg) { return comm.max(arg); })
      .def("_max_int64", [](muGrid::Communicator & comm,
                            const Index_t & arg) { return comm.max(arg); })
      .def("_min_float64", [](muGrid::Communicator & comm,
                              const Real & arg) { return comm.min(arg); })
      .def("_min_int64", [](muGrid::Communicator & comm,
                            const Index_t & arg) { return comm.min(arg); })
      .def("all", [](muGrid::Communicator & comm,
                     const bool & arg) { return comm.all(arg); })
      .def("any", [](muGrid::Communicator & comm,
                     const bool & arg) { return comm.any(arg); })
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
