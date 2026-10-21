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
#include "mpi/mpi_environment.hh"

#include <pybind11/pybind11.h>
#include <pybind11/complex.h>

#include <cstdint>
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
    Index_t count;  //!< in elements of `type`
    TypeDescriptor type;
    BufferLocation location;
  };

  //! What an exchange does with the values: reductions compute with them
  //! and need an arithmetic MPI type; the other exchanges only move them,
  //! which works for any fixed-size dtype, sent as its bytes.
  enum class Use { Arithmetic, Movement };

  py::module_ array_module(BufferLocation location) {
    return py::module_::import(location == BufferLocation::Device ? "cupy"
                                                                  : "numpy");
  }

  std::optional<TypeDescriptor> arithmetic_type(const std::string & typestr) {
    // e.g. "<f8": byte order, kind, item size in bytes
    if (typestr.size() >= 3 && typestr[0] != '>') {
      const std::string kind_size{typestr.substr(1)};
      if (kind_size == "f8") return TypeDescriptor::Real;
      if (kind_size == "f4") return TypeDescriptor::Real32;
      if (kind_size == "c16") return TypeDescriptor::Complex;
      if (kind_size == "c8") return TypeDescriptor::Complex32;
      if (kind_size == "i4") return TypeDescriptor::Int;
      if (kind_size == "u4") return TypeDescriptor::Uint;
      if (kind_size == "u1") return TypeDescriptor::Uint8;
      if (kind_size == "i" + std::to_string(sizeof(Index_t))) {
        return TypeDescriptor::Index;
      }
    }
    return std::nullopt;
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

  ArrayBuffer buffer_of(const py::object & a, Use use = Use::Arithmetic) {
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
    const auto typestr{interface["typestr"].cast<std::string>()};
    auto type{arithmetic_type(typestr)};
    if (!type) {
      if (use == Use::Arithmetic) {
        throw py::type_error("no MPI reduction for arrays of dtype " +
                             typestr +
                             "; use float32/64, complex64/128, int32/64, "
                             "uint32 or uint8");
      }
      if (array.attr("dtype").attr("hasobject").cast<bool>()) {
        throw py::type_error("arrays of Python objects cannot be sent over "
                             "MPI; serialise them to a uint8 array first");
      }
      // Moved, not computed with: send the bytes
      type = TypeDescriptor::Uint8;
      count *= array.attr("dtype").attr("itemsize").cast<Index_t>();
    }
    return ArrayBuffer{contiguous,         array.attr("shape"),
                       array.attr("dtype"), data_of(interface),
                       count,              *type,
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

  //! Per-rank header of allgatherv: element count, dtype number, item
  //! size, ndim, then the shape (with the gathered axis first), padded
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
    py::module_ xp;
    ArrayBuffer send;       //!< the local array, gathered axis first
    py::object dtype;
    py::tuple result_shape;  //!< gathered axis first
    std::vector<int> counts, displs;
    Index_t result_axis;
    BufferLocation location;
  };

  ConcatenationPlan plan_concatenation(const muGrid::Communicator & comm,
                                       const py::object & a, Index_t axis) {
    const BufferLocation location{py::hasattr(a, "__cuda_array_interface__")
                                      ? BufferLocation::Device
                                      : BufferLocation::Host};
    auto xp{array_module(location)};
    py::object array{xp.attr("asarray")(a)};
    const auto ndim{array.attr("ndim").cast<Index_t>()};
    if (ndim == 0 || ndim > max_dims) {
      throw py::value_error("needs an array of 1 to " +
                            std::to_string(max_dims) + " dimensions");
    }
    if (axis < -ndim || axis >= ndim) {
      throw py::value_error("axis " + std::to_string(axis) +
                            " is out of range for an array of " +
                            std::to_string(ndim) + " dimensions");
    }
    const Index_t own_axis{axis < 0 ? axis + ndim : axis};
    py::object moved{xp.attr("moveaxis")(array, own_axis, 0)};
    auto send{buffer_of(moved, Use::Movement)};
    py::object dtype{array.attr("dtype")};

    std::vector<Index_t> header(header_size, 0);
    header[0] = moved.attr("size").cast<Index_t>();
    header[1] = dtype.attr("num").cast<Index_t>();
    header[2] = dtype.attr("itemsize").cast<Index_t>();
    header[3] = ndim;
    const auto shape{moved.attr("shape").cast<py::tuple>()};
    for (Index_t d{0}; d < ndim; ++d) {
      header[4 + d] = shape[d].cast<Index_t>();
    }
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
    std::vector<int> counts(nb_ranks, 0), displs(nb_ranks, 0);
    const Index_t scale{header[0] > 0 ? send.count / header[0]
                                      : (send.type == TypeDescriptor::Uint8
                                             ? header[2]
                                             : 1)};
    Index_t offset{0};
    for (int r{0}; r < nb_ranks; ++r) {
      const Index_t * h{row(r)};
      if (h[0] == 0) {
        displs[r] = narrow_count(offset);
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
      counts[r] = narrow_count(h[0] * scale);
      displs[r] = narrow_count(offset);
      offset += h[0] * scale;
    }
    narrow_count(offset);

    py::list result_shape{};
    result_shape.append(length);
    for (Index_t d{1}; d < result_ndim; ++d) {
      result_shape.append(rest[d - 1]);
    }
    return ConcatenationPlan{xp,
                             std::move(send),
                             dtype,
                             py::tuple(result_shape),
                             std::move(counts),
                             std::move(displs),
                             axis < 0 ? axis + result_ndim : axis,
                             location};
  }

  //! np.concatenate across ranks, on every rank
  py::object allgatherv_array(const muGrid::Communicator & comm,
                              const py::object & a, Index_t axis) {
    auto plan{plan_concatenation(comm, a, axis)};
    py::object result{
        plan.xp.attr("empty")(plan.result_shape, "dtype"_a = plan.dtype)};
    void * recv{data_of(interface_of(result, plan.location))};
    wait_for_writer(plan.send);
    {
      py::gil_scoped_release release{};
      std::lock_guard<std::mutex> lock{muGrid::mpi_call_mutex()};
      comm.allgatherv(plan.send.data, plan.send.count, recv, plan.counts,
                      plan.displs, plan.send.type, plan.location);
    }
    return plan.xp.attr("moveaxis")(result, 0, plan.result_axis);
  }

  //! np.concatenate across ranks, on the root; None elsewhere
  py::object gatherv_array(const muGrid::Communicator & comm,
                           const py::object & a, int root, Index_t axis) {
    auto plan{plan_concatenation(comm, a, axis)};
    const bool is_root{comm.rank() == root};
    py::object result{py::none()};
    void * recv{nullptr};
    if (is_root) {
      result = plan.xp.attr("empty")(plan.result_shape, "dtype"_a = plan.dtype);
      recv = data_of(interface_of(result, plan.location));
    }
    wait_for_writer(plan.send);
    {
      py::gil_scoped_release release{};
      std::lock_guard<std::mutex> lock{muGrid::mpi_call_mutex()};
      comm.gatherv(plan.send.data, plan.send.count, recv, plan.counts,
                   plan.displs, root, plan.send.type, plan.location);
    }
    return is_root ? plan.xp.attr("moveaxis")(result, 0, plan.result_axis)
                   : result;
  }

  //! Rank root's array into every rank's, in place: all ranks pass an
  //! array of one shape and dtype, which must be writable and contiguous
  //! (a non-contiguous one would be broadcast into a copy and lost)
  void bcast_array(const muGrid::Communicator & comm, const py::object & a,
                   int root) {
    const BufferLocation location{py::hasattr(a, "__cuda_array_interface__")
                                      ? BufferLocation::Device
                                      : BufferLocation::Host};
    // NumPy's flags is a flagsobj, CuPy's lacks "writeable" (always true)
    const py::object flags{a.attr("flags")};
    if (!flags.attr("c_contiguous").cast<bool>() ||
        !py::getattr(flags, "writeable", py::bool_(true)).cast<bool>()) {
      throw py::value_error("bcast fills the array in place, so it must be "
                            "C-contiguous and writeable");
    }
    const auto buffer{buffer_of(a, Use::Movement)};
    wait_for_writer(buffer);
    {
      py::gil_scoped_release release{};
      std::lock_guard<std::mutex> lock{muGrid::mpi_call_mutex()};
      comm.bcast(buffer.data, buffer.count, buffer.type, root, location);
    }
  }

  //! Whether `a` is a Python number (not a NumPy scalar: np.float64 is a
  //! float subclass but carries an array interface, and comes back as a
  //! 0-d array)
  bool is_python_scalar(const py::object & a) {
    return !py::hasattr(a, "__array_interface__") &&
           (py::isinstance<py::bool_>(a) || py::isinstance<py::int_>(a) ||
            py::isinstance<py::float_>(a) || PyComplex_Check(a.ptr()));
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
      // Array exchanges for NumPy and CuPy arrays alike. Each returns a new
      // array of the input's library and dtype; device arrays go to MPI
      // directly or through the host, as the communicator decides.
      .def(
          "allreduce",
          [](const muGrid::Communicator & comm, const py::object & a,
             const py::object & op) {
            const auto send{buffer_of(a)};
            auto [result, recv]{empty_like(send, send.shape)};
            const auto reduction{reduce_op(op)};
            wait_for_writer(send);
            {
              py::gil_scoped_release release{};
              std::lock_guard<std::mutex> lock{muGrid::mpi_call_mutex()};
              comm.allreduce(send.data, recv, send.count, send.type,
                             reduction, send.location);
            }
            // A Python number comes back as one, as from mpi4py
            return is_python_scalar(a) ? result.attr("item")() : result;
          },
          "a"_a, "op"_a = "sum",
          "Elementwise reduction of an array over all ranks, returned on "
          "every rank; a Python number comes back as a Python number. op: "
          "'sum', 'max' or 'min', or mpi4py's MPI.SUM/MAX/MIN.")
      .def(
          "iallreduce",
          [](const muGrid::Communicator & comm, const py::object & a,
             const py::object & op) {
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
            const auto send{buffer_of(a, Use::Movement)};
            py::list shape{};
            shape.append(comm.size());
            for (auto extent : send.shape.cast<py::tuple>()) {
              shape.append(extent);
            }
            auto [result, recv]{empty_like(send, py::tuple(shape))};
            wait_for_writer(send);
            {
              py::gil_scoped_release release{};
              std::lock_guard<std::mutex> lock{muGrid::mpi_call_mutex()};
              comm.allgather(send.data, recv, send.count, send.type,
                             send.location);
            }
            return result;
          },
          "a"_a,
          "Every rank's array, of one shape and dtype on all ranks, stacked "
          "along a new leading axis in rank order, on every rank. For "
          "arrays whose sizes differ between ranks, see allgatherv.")
      .def(
          "sendrecv",
          [](const muGrid::Communicator & comm, const py::object & a,
             int dest, int source, int tag) {
            const auto send{buffer_of(a, Use::Movement)};
            auto [result, recv]{empty_like(send, send.shape)};
            wait_for_writer(send);
            {
              py::gil_scoped_release release{};
              std::lock_guard<std::mutex> lock{muGrid::mpi_call_mutex()};
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
