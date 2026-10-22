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

#include "array_exchange.hh"

#include "memory/gpu_runtime.hh"
#include "mpi/communicator.hh"
#include "mpi/gpu_aware_mpi.hh"
#include "mpi/mpi_environment.hh"

#include <pybind11/pybind11.h>
#include <pybind11/complex.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

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
  using muGrid::python::ExchangeBuffer;
  using muGrid::python::ImportedArray;

  //! What an exchange does with the values: reductions compute with them
  //! and need an arithmetic MPI type; the other exchanges only move them,
  //! which works for any fixed-size dtype, sent as its bytes.
  enum class Use { Arithmetic, Movement };

  std::string dtype_name(const DLDataType & t) {
    static const char * kinds[]{"int", "uint", "float", "opaque", "bfloat",
                                "complex", "bool"};
    std::stringstream out{};
    out << (t.code < 7 ? kinds[t.code] : "code" + std::to_string(t.code))
        << int(t.bits);
    if (t.lanes != 1) {
      out << "x" << t.lanes;
    }
    return out.str();
  }

  /**
   * An array ready to be sent: imported (array_exchange.hh), in C order --
   * its own memory, or a packed copy for any other strides -- and counted
   * in elements of its MPI type. Construct it with the array's device
   * current (DeviceGuard), so that a packed copy lands there.
   */
  struct SendBuffer {
    ImportedArray array;
    std::shared_ptr<ExchangeBuffer> scratch{};
    const void * data{nullptr};
    Index_t count{0};
    TypeDescriptor type{TypeDescriptor::Uint8};

    SendBuffer(ImportedArray imported, Use use) : array{std::move(imported)} {
      this->count = this->array.size();
      auto type{this->array.arithmetic_type()};
      if (!type) {
        if (use == Use::Arithmetic) {
          throw py::type_error("no MPI reduction for arrays of dtype " +
                               dtype_name(this->array.dtype) +
                               "; use float32/64, complex64/128, int32/64, "
                               "uint32 or uint8");
        }
        // Moved, not computed with: send the bytes
        type = TypeDescriptor::Uint8;
        this->count *= this->array.itemsize;
      }
      this->type = *type;
      this->data = muGrid::python::contiguous_data(this->array, this->scratch);
    }
  };

  //! A non-blocking allreduce with the buffers it reads and writes. The
  //! request is declared last so that it is destroyed -- and so waited
  //! for -- first, while the buffers are still alive.
  struct PyAllreduceRequest {
    std::shared_ptr<SendBuffer> send;
    py::object result;
    muGrid::AllreduceRequest request;
  };

  //! Per-rank header of allgatherv: element count, dtype (DLPack code,
  //! bits and lanes packed), item size, ndim, then the shape (with the
  //! gathered axis first), padded
  constexpr Index_t max_dims{64};
  constexpr Index_t header_size{4 + max_dims};

  std::string shape_string(const Index_t * dims, Index_t ndim) {
    std::stringstream out{};
    out << "(";
    for (Index_t d{0}; d < ndim; ++d) {
      out << (d ? ", " : "") << dims[d];
    }
    out << (ndim == 1 ? ",)" : ")");
    return out.str();
  }

  int narrow_count(Index_t value) {
    if (value > std::numeric_limits<int>::max()) {
      throw py::value_error("allgatherv: the gathered array is too large "
                            "for one MPI message");
    }
    return static_cast<int>(value);
  }

  /**
   * The plan of a concatenation across ranks: every rank's array joined
   * along `axis`, in rank order. The extents along `axis` may differ; dtype
   * and all other extents must agree. Ranks contributing no elements are
   * left out (their shape does not matter). Every rank sees every rank's
   * header, so a mismatch raises on all ranks alike.
   */
  struct ConcatenationPlan {
    std::unique_ptr<muGrid::python::DeviceGuard> guard;
    std::unique_ptr<SendBuffer> send;  //!< the local array, gathered axis first
    std::vector<Index_t> result_shape;  //!< gathered axis first
    std::vector<int> counts, displs;
    Index_t result_axis;
  };

  ConcatenationPlan plan_concatenation(const muGrid::Communicator & comm,
                                       const py::object & a, Index_t axis) {
    auto array{muGrid::python::import_array(a)};
    const auto ndim{static_cast<Index_t>(array.shape.size())};
    if (ndim == 0 || ndim > max_dims) {
      throw py::value_error("needs an array of 1 to " +
                            std::to_string(max_dims) + " dimensions");
    }
    if (axis < -ndim || axis >= ndim) {
      throw py::value_error("axis " + std::to_string(axis) +
                            " is out of range for an array of " +
                            std::to_string(ndim) + " dimensions");
    }
    array.move_axis_to_front(axis < 0 ? axis + ndim : axis);
    ConcatenationPlan plan{};
    plan.guard = std::make_unique<muGrid::python::DeviceGuard>(array);
    std::vector<Index_t> header(header_size, 0);
    header[0] = array.size();
    header[1] = (Index_t{array.dtype.code} << 24) |
                (Index_t{array.dtype.bits} << 16) | array.dtype.lanes;
    header[2] = array.itemsize;
    header[3] = ndim;
    for (Index_t d{0}; d < ndim; ++d) {
      header[4 + d] = array.shape[d];
    }
    plan.send = std::make_unique<SendBuffer>(std::move(array), Use::Movement);

    const int nb_ranks{comm.size()};
    std::vector<Index_t> headers(header_size * nb_ranks);
    comm.allgather(header.data(), headers.data(), header_size,
                   TypeDescriptor::Index, BufferLocation::Host);
    auto row{[&headers](int r) { return headers.data() + r * header_size; }};

    int ref{-1};  // the first rank that contributes elements
    for (int r{0}; r < nb_ranks; ++r) {
      if (row(r)[1] != row(0)[1] || row(r)[2] != row(0)[2]) {
        throw py::type_error("the arrays must have one dtype on all ranks, "
                             "but ranks 0 and " +
                             std::to_string(r) + " differ");
      }
      if (ref < 0 && row(r)[0] > 0) {
        ref = r;
      }
    }
    const Index_t result_ndim{ref < 0 ? ndim : row(ref)[3]};
    const Index_t * rest{ref < 0 ? header.data() + 5 : row(ref) + 5};
    Index_t length{0};
    plan.counts.assign(nb_ranks, 0);
    plan.displs.assign(nb_ranks, 0);
    const Index_t scale{header[0] > 0 ? plan.send->count / header[0]
                                      : (plan.send->type ==
                                                 TypeDescriptor::Uint8
                                             ? header[2]
                                             : 1)};
    Index_t offset{0};
    for (int r{0}; r < nb_ranks; ++r) {
      const Index_t * h{row(r)};
      if (h[0] == 0) {
        plan.displs[r] = narrow_count(offset);
        continue;
      }
      bool same{h[3] == result_ndim};
      for (Index_t d{1}; same && d < result_ndim; ++d) {
        same = h[4 + d] == rest[d - 1];
      }
      if (!same) {
        throw py::value_error(
            "arrays may differ only along the gathered axis, but rank " +
            std::to_string(ref) + " has shape " +
            shape_string(row(ref) + 4, row(ref)[3]) + " and rank " +
            std::to_string(r) + " " + shape_string(h + 4, h[3]) +
            " (gathered axis first)");
      }
      length += h[4];
      plan.counts[r] = narrow_count(h[0] * scale);
      plan.displs[r] = narrow_count(offset);
      offset += h[0] * scale;
    }
    narrow_count(offset);

    plan.result_shape.push_back(length);
    for (Index_t d{1}; d < result_ndim; ++d) {
      plan.result_shape.push_back(rest[d - 1]);
    }
    plan.result_axis = axis < 0 ? axis + result_ndim : axis;
    return plan;
  }

  //! np.concatenate across ranks, on every rank
  py::object allgatherv_array(const muGrid::Communicator & comm,
                              const py::object & a, Index_t axis) {
    auto plan{plan_concatenation(comm, a, axis)};
    const auto & send{*plan.send};
    auto result{muGrid::python::allocate_like(send.array, plan.result_shape)};
    {
      py::gil_scoped_release release{};
      std::lock_guard<std::mutex> lock{muGrid::mpi_call_mutex()};
      comm.allgatherv(send.data, send.count, result->data, plan.counts,
                      plan.displs, send.type, send.array.location);
    }
    result->move_front_axis_to(plan.result_axis);
    return muGrid::python::hand_back(send.array, std::move(result));
  }

  //! np.concatenate across ranks, on the root; None elsewhere
  py::object gatherv_array(const muGrid::Communicator & comm,
                           const py::object & a, int root, Index_t axis) {
    auto plan{plan_concatenation(comm, a, axis)};
    const auto & send{*plan.send};
    const bool is_root{comm.rank() == root};
    std::shared_ptr<ExchangeBuffer> result{};
    if (is_root) {
      result = muGrid::python::allocate_like(send.array, plan.result_shape);
    }
    {
      py::gil_scoped_release release{};
      std::lock_guard<std::mutex> lock{muGrid::mpi_call_mutex()};
      comm.gatherv(send.data, send.count, is_root ? result->data : nullptr,
                   plan.counts, plan.displs, root, send.type,
                   send.array.location);
    }
    if (!is_root) {
      return py::none();
    }
    result->move_front_axis_to(plan.result_axis);
    return muGrid::python::hand_back(send.array, std::move(result));
  }

  //! Rank root's array into every rank's, in place: all ranks pass an
  //! array of one shape and dtype, which must be writable and contiguous
  //! (a non-contiguous one would be broadcast into a copy and lost)
  void bcast_array(const muGrid::Communicator & comm, const py::object & a,
                   int root) {
    auto array{muGrid::python::import_array(a)};
    if (!array.c_contiguous() || array.read_only) {
      throw py::value_error("bcast fills the array in place, so it must be "
                            "C-contiguous and writeable");
    }
    muGrid::python::DeviceGuard guard{array};
    const SendBuffer buffer{std::move(array), Use::Movement};
    {
      py::gil_scoped_release release{};
      std::lock_guard<std::mutex> lock{muGrid::mpi_call_mutex()};
      comm.bcast(const_cast<void *>(buffer.data), buffer.count, buffer.type,
                 root, buffer.array.location);
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
      // The array's own library reads it next, on whatever stream it uses
      if (buffer.array.location == BufferLocation::Device) {
        GPU_STREAM_SYNCHRONIZE_DEFAULT();
      }
#endif
    }
  }

  //! 'sum', 'max' or 'min', or mpi4py's MPI.SUM/MAX/MIN if mpi4py is
  //! loaded (compared by identity; muGrid does not import it)
  muGrid::ReduceOp reduce_op(const py::object & op) {
    if (py::isinstance<py::str>(op)) {
      const auto name{op.cast<std::string>()};
      if (name == "sum") return muGrid::ReduceOp::Sum;
      if (name == "max") return muGrid::ReduceOp::Max;
      if (name == "min") return muGrid::ReduceOp::Min;
    } else {
      py::dict modules{py::module_::import("sys").attr("modules")};
      if (modules.contains("mpi4py.MPI")) {
        py::object MPI{modules["mpi4py.MPI"]};
        if (op.is(MPI.attr("SUM"))) return muGrid::ReduceOp::Sum;
        if (op.is(MPI.attr("MAX"))) return muGrid::ReduceOp::Max;
        if (op.is(MPI.attr("MIN"))) return muGrid::ReduceOp::Min;
      }
    }
    throw py::value_error("op must be 'sum', 'max' or 'min' (or mpi4py's "
                          "MPI.SUM, MPI.MAX or MPI.MIN), not " +
                          py::repr(op).cast<std::string>());
  }
}  // namespace

void add_communicator(py::module & mod) {
  // Starting and stopping MPI (mpi/mpi_environment.hh). muGrid starts MPI
  // only when Communicator.world() asks for it and stops it at interpreter
  // exit only if it started it; see muGrid/__init__.py.
  mod.def("mpi_initialized", &muGrid::mpi_initialized,
          "Whether MPI has been initialised, by anyone.");
  mod.def("mpi_finalized", &muGrid::mpi_finalized,
          "Whether MPI has been finalised, by anyone.");
  mod.def("mpi_initialized_by_mugrid", &muGrid::mpi_initialized_by_mugrid,
          "Whether it was muGrid that initialised MPI.");
  mod.def(
      "finalize_mpi",
      [](bool force) {
        py::gil_scoped_release release{};
        muGrid::finalize_mpi(force);
      },
      "force"_a = false,
      "Finalise MPI if muGrid initialised it (with force, whoever did). "
      "Owned communicators and open parallel files are left to the OS "
      "afterwards. Registered at interpreter exit unless "
      "MUGRID_MPI_FINALIZE=0.");
  mod.def("mpi_thread_level", &muGrid::mpi_thread_level,
          "The thread level MPI granted: 'single', 'funneled', 'serialized' "
          "or 'multiple'; 'none' while MPI is not running.");
  mod.def("mpi_library_version", &muGrid::mpi_library_version,
          "The MPI library's own version string ('none' without MPI).");
  mod.def("mpi_comm_world_handle", &muGrid::mpi_comm_world_handle,
          "MPI_COMM_WORLD as an integer, as mpi4py's MPI._handleof gives it; "
          "equal between muGrid and mpi4py only if they share one MPI "
          "library.");

  // Arrays of any library for the exchanges (array_exchange.hh), and the
  // few array operations the Python communicator needs without knowing
  // the array's library.
  muGrid::python::add_exchange_buffer(mod);
  mod.def(
      "_exchange_contiguous",
      [](const py::object & a) -> py::object {
        auto array{muGrid::python::import_array(a)};
        if (array.c_contiguous()) {
          return a;
        }
        muGrid::python::DeviceGuard guard{array};
        std::shared_ptr<ExchangeBuffer> copy{};
        muGrid::python::contiguous_data(array, copy);  // packs into `copy`
        return muGrid::python::hand_back(array, std::move(copy));
      },
      "a"_a,
      "`a` if it is C-contiguous, else a C-contiguous copy of the same "
      "library and device.");
  mod.def(
      "_exchange_describe",
      [](const py::object & a) {
        const auto array{muGrid::python::import_array(a)};
        const auto & t{array.dtype};
        const auto module{array.library.is_none()
                              ? py::object(py::none())
                              : array.library.attr("__name__")};
        return py::make_tuple(py::tuple(py::cast(array.shape)),
                              py::make_tuple(int(t.code), int(t.bits),
                                             int(t.lanes)),
                              array.device_type, module);
      },
      "a"_a,
      "(shape, (DLPack dtype code, bits, lanes), DLPack device type, "
      "library) of an array: what another rank needs to allocate its "
      "counterpart with _exchange_empty.");
  mod.def(
      "_exchange_empty",
      [](const py::tuple & shape, const py::tuple & dtype, int device_type,
         const py::object & library) {
        ImportedArray like{};
        for (auto extent : shape) {
          like.shape.push_back(extent.cast<Index_t>());
        }
        like.dtype = DLDataType{static_cast<uint8_t>(dtype[0].cast<int>()),
                                static_cast<uint8_t>(dtype[1].cast<int>()),
                                static_cast<uint16_t>(dtype[2].cast<int>())};
        like.itemsize = (dtype[1].cast<Index_t>() + 7) / 8;
        like.device_type = device_type;
        like.location = (device_type == kDLCPU || device_type == kDLCUDAHost ||
                         device_type == kDLROCMHost)
                            ? BufferLocation::Host
                            : BufferLocation::Device;
#if defined(MUGRID_ENABLE_CUDA)
        if (like.location == BufferLocation::Device) {
          GPU_CHECK(cudaGetDevice(&like.device_id));
        }
#elif defined(MUGRID_ENABLE_HIP)
        if (like.location == BufferLocation::Device) {
          GPU_CHECK(hipGetDevice(&like.device_id));
        }
#endif
        like.library = library.is_none()
                           ? py::object(py::none())
                           : py::object(py::module_::import(
                                 library.cast<std::string>().c_str()));
        return muGrid::python::hand_back(
            like, muGrid::python::allocate_like(like));
      },
      "shape"_a, "dtype"_a, "device_type"_a, "library"_a,
      "An uninitialised array as _exchange_describe describes it, on the "
      "current device, of the named library (an ExchangeBuffer if None).");
  mod.def(
      "_exchange_full_like",
      [](const py::object & a, const py::object & value) {
        const auto array{muGrid::python::import_array(a)};
        // The value in the array's dtype, through NumPy on the host
        const auto host{muGrid::python::import_array(
            py::module_::import("numpy").attr("asarray")(
                value,
                "dtype"_a = muGrid::python::numpy_typestr(array.dtype)))};
        muGrid::python::DeviceGuard guard{array};
        ImportedArray scalar_like{array};
        scalar_like.shape.clear();
        scalar_like.strides.clear();
        scalar_like.python_scalar = false;
        auto result{muGrid::python::allocate_like(scalar_like)};
        if (array.location == BufferLocation::Device) {
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
          GPU_MEMCPY_H2D(result->data, host.data, array.itemsize);
#endif
        } else {
          std::memcpy(result->data, host.data, array.itemsize);
        }
        return muGrid::python::hand_back(scalar_like, std::move(result));
      },
      "a"_a, "value"_a,
      "A 0-d array holding `value`, of `a`'s dtype, library and device.");

  py::class_<PyAllreduceRequest>(mod, "AllreduceRequest",
                                 "A reduction started by "
                                 "Communicator.iallreduce.")
      .def(
          "wait",
          [](PyAllreduceRequest & self) {
            {
              py::gil_scoped_release release{};
              std::lock_guard<std::mutex> lock{muGrid::mpi_call_mutex()};
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
      .def_static(
          "world",
          []() {
            py::gil_scoped_release release{};
            return muGrid::Communicator::world();
          },
          "MPI_COMM_WORLD, starting MPI first if nobody has. Collective.")
      .def(
          "split",
          [](const muGrid::Communicator & comm, int color, int key) {
            py::gil_scoped_release release{};
            std::lock_guard<std::mutex> lock{muGrid::mpi_call_mutex()};
            return comm.split(color, key);
          },
          "color"_a, "key"_a = 0,
          "A communicator over the ranks passing the same color, ordered by "
          "key; it owns its handle. Collective.")
      .def(
          "dup",
          [](const muGrid::Communicator & comm) {
            py::gil_scoped_release release{};
            std::lock_guard<std::mutex> lock{muGrid::mpi_call_mutex()};
            return comm.dup();
          },
          "A new handle to the same group. Collective.")
      .def("compare", &muGrid::Communicator::compare, "other"_a,
           "'identical', 'congruent' (same ranks, same order), 'similar' "
           "(same ranks) or 'unequal'.")
      .def(
          "discover_node",
          [](muGrid::Communicator & comm) {
            py::gil_scoped_release release{};
            std::lock_guard<std::mutex> lock{muGrid::mpi_call_mutex()};
            comm.discover_node();
          },
          "Find out node_rank and node_size. Collective; world(), split() "
          "and dup() have done it already.")
      .def_property_readonly("node_rank", &muGrid::Communicator::node_rank,
                             "Rank among the ranks on this node (for one "
                             "GPU per rank).")
      .def_property_readonly("node_size", &muGrid::Communicator::node_size,
                             "Number of ranks on this node.")
      // Whether MPI may be handed device pointers (MUGRID_GPU_AWARE_MPI, else
      // the MPI library's own report through whichever GPU-support query it
      // provides). For information only: the exchanges below decide by
      // themselves and stage through the host otherwise.
      .def_property_readonly_static(
          "gpu_aware_mpi",
          [](py::object) { return muGrid::mpi_is_gpu_aware(); })
      // Array exchanges for arrays of any library that speaks DLPack (NumPy,
      // CuPy, PyTorch, JAX, ...; see array_exchange.hh). Each returns a new
      // array of the input's library and dtype, on the input's device;
      // device arrays go to MPI directly or through the host, as the
      // communicator decides.
      .def(
          "allreduce",
          [](const muGrid::Communicator & comm, const py::object & a,
             const py::object & op) {
            auto array{muGrid::python::import_array(a)};
            const auto reduction{reduce_op(op)};
            muGrid::python::DeviceGuard guard{array};
            const SendBuffer send{std::move(array), Use::Arithmetic};
            auto result{muGrid::python::allocate_like(send.array)};
            {
              py::gil_scoped_release release{};
              std::lock_guard<std::mutex> lock{muGrid::mpi_call_mutex()};
              comm.allreduce(send.data, result->data, send.count, send.type,
                             reduction, send.array.location);
            }
            // A Python number comes back as one, as from mpi4py
            return muGrid::python::hand_back(send.array, std::move(result));
          },
          "a"_a, "op"_a = "sum",
          "Elementwise reduction of an array over all ranks, returned on "
          "every rank; a Python number comes back as a Python number. op: "
          "'sum', 'max' or 'min', or mpi4py's MPI.SUM/MAX/MIN.")
      .def(
          "iallreduce",
          [](const muGrid::Communicator & comm, const py::object & a,
             const py::object & op) {
            auto array{muGrid::python::import_array(a)};
            const auto reduction{reduce_op(op)};
            muGrid::python::DeviceGuard guard{array};
            auto send{std::make_shared<SendBuffer>(std::move(array),
                                                   Use::Arithmetic)};
            auto result{muGrid::python::allocate_like(send->array)};
            void * recv{result->data};
            auto request{std::make_unique<PyAllreduceRequest>()};
            request->send = send;
            // Always an array: a Python number cannot be filled in later
            send->array.python_scalar = false;
            request->result =
                muGrid::python::hand_back(send->array, std::move(result));
            request->request =
                comm.iallreduce(send->data, recv, send->count, send->type,
                                reduction, send->array.location);
            return request;
          },
          "a"_a, "op"_a = "sum",
          "Start the reduction `allreduce` performs and return at once; the "
          "returned request's wait() completes it and returns the result. "
          "Work that does not need the result can run in between.")
      .def(
          "allgather",
          [](const muGrid::Communicator & comm, const py::object & a) {
            auto array{muGrid::python::import_array(a)};
            muGrid::python::DeviceGuard guard{array};
            const SendBuffer send{std::move(array), Use::Movement};
            std::vector<Index_t> shape{comm.size()};
            shape.insert(shape.end(), send.array.shape.begin(),
                         send.array.shape.end());
            auto result{muGrid::python::allocate_like(send.array, shape)};
            {
              py::gil_scoped_release release{};
              std::lock_guard<std::mutex> lock{muGrid::mpi_call_mutex()};
              comm.allgather(send.data, result->data, send.count, send.type,
                             send.array.location);
            }
            return muGrid::python::hand_back(send.array, std::move(result));
          },
          "a"_a,
          "Every rank's array, of one shape and dtype on all ranks, stacked "
          "along a new leading axis in rank order, on every rank. For "
          "arrays whose sizes differ between ranks, see allgatherv.")
      .def(
          "sendrecv",
          [](const muGrid::Communicator & comm, const py::object & a,
             int dest, int source, int tag) {
            auto array{muGrid::python::import_array(a)};
            muGrid::python::DeviceGuard guard{array};
            const SendBuffer send{std::move(array), Use::Movement};
            auto result{muGrid::python::allocate_like(send.array)};
            {
              py::gil_scoped_release release{};
              std::lock_guard<std::mutex> lock{muGrid::mpi_call_mutex()};
              comm.sendrecv(send.data, send.count, dest, result->data,
                            send.count, source, send.type,
                            send.array.location, tag);
            }
            return muGrid::python::hand_back(send.array, std::move(result));
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
      .def("gatherv", &gatherv_array, "a"_a, "root"_a = 0, "axis"_a = 0,
           "As allgatherv, but the result lands on rank `root` only; the "
           "other ranks get None.")
      .def("bcast", &bcast_array, "a"_a, "root"_a = 0,
           "Rank root's array into every rank's, in place: every rank passes "
           "a C-contiguous, writeable array of one shape and dtype, host or "
           "device, and leaves with the root's values.")
      .def("allgatherv", &allgatherv_array, "a"_a, "axis"_a = 0,
           "np.concatenate across ranks: every rank's array joined along "
           "`axis`, in rank order, on every rank. The extent along `axis` "
           "may differ between ranks; dtype and all other extents must "
           "agree, and ranks with no elements are left out. Host or device "
           "arrays, of any fixed-size dtype. For arrays of one shape stacked "
           "along a new axis, see allgather.")
      .def(
          "bcast_scalar",
          [](muGrid::Communicator & comm, Real & scalar_arg, const Int & root) {
            return comm.bcast<Real>(scalar_arg, root);
          },
          "scalar_arg"_a, "root"_a)
      .def(
          "bcast_scalar",
          [](muGrid::Communicator & comm, Int & scalar_arg, const Int & root) {
            return comm.bcast<Int>(scalar_arg, root);
          },
          "scalar_arg"_a, "root"_a)
      .def(
          "bcast_scalar",
          [](muGrid::Communicator & comm, Complex & scalar_arg,
             const Int & root) {
            return comm.bcast<Complex>(scalar_arg, root);
          },
          "scalar_arg"_a, "root"_a);
}
