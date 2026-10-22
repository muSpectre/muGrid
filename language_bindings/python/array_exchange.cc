/**
 * @file   array_exchange.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   22 Oct 2026
 *
 * @brief  Arrays of any library as the buffer exchanges of Communicator see
 *         them: imported through DLPack, results handed back through DLPack
 *
 * Copyright © 2026 Lars Pastewka
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

#include "array_exchange.hh"

#include "memory/device_alloc.hh"
#include "memory/gpu_runtime.hh"

#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <new>
#include <string>

using pybind11::literals::operator""_a;

namespace muGrid {
  namespace python {

    namespace {
      //! The device type of this build's GPU memory, if any
      constexpr int gpu_device_type() {
#if defined(MUGRID_ENABLE_CUDA)
        return kDLCUDA;
#elif defined(MUGRID_ENABLE_HIP)
        return kDLROCM;
#else
        return -1;
#endif
      }

      /**
       * Where a DLPack device keeps its memory as far as MPI is concerned:
       * on the host (plain or pinned) or on this build's GPU (device or
       * managed memory). Anything else cannot be exchanged by this build.
       */
      BufferLocation location_of(int device_type) {
        switch (device_type) {
        case kDLCPU:
        case kDLCUDAHost:
        case kDLROCMHost:
          return BufferLocation::Host;
        case kDLCUDA:
        case kDLCUDAManaged:
          if (gpu_device_type() == kDLCUDA) {
            return BufferLocation::Device;
          }
          break;
        case kDLROCM:
          if (gpu_device_type() == kDLROCM) {
            return BufferLocation::Device;
          }
          break;
        default:
          break;
        }
        throw py::type_error(
            "the communicator cannot exchange arrays on DLPack device type " +
            std::to_string(device_type) + (gpu_device_type() < 0
                                               ? " in a build without GPU "
                                                 "support"
                                               : " in this build"));
      }

      Index_t itemsize_of(const DLDataType & dtype) {
        // Multi-lane types are moved as their bytes (arithmetic_type()
        // refuses them)
        return (static_cast<Index_t>(dtype.bits) * dtype.lanes + 7) / 8;
      }

      //! numpy-style typestr ("<f8") of a CUDA array interface -> DLPack
      DLDataType dtype_of_typestr(const std::string & typestr) {
        if (typestr.size() < 3 || typestr[0] == '>') {
          throw py::type_error("unsupported array typestr " + typestr);
        }
        const char kind{typestr[1]};
        const auto bits{static_cast<uint8_t>(8 * std::stoi(typestr.substr(2)))};
        switch (kind) {
        case 'f':
          return DLDataType{kDLFloat, bits, 1};
        case 'c':
          return DLDataType{kDLComplex, bits, 1};
        case 'i':
          return DLDataType{kDLInt, bits, 1};
        case 'u':
          return DLDataType{kDLUInt, bits, 1};
        case 'b':
          return DLDataType{kDLBool, bits, 1};
        default:
          throw py::type_error("unsupported array typestr " + typestr);
        }
      }

      //! The module a result goes back to: the top-level package of the
      //! input's type (numpy, cupy, torch, ...)
      py::object library_of(const py::object & a) {
        const auto module{py::type::of(a).attr("__module__").cast<std::string>()};
        try {
          return py::module_::import(module.substr(0, module.find('.')).c_str());
        } catch (py::error_already_set &) {
          return py::none();
        }
      }

      //! DLPack device of a pointer from CUDA's array interface (which does
      //! not carry one), from the runtime
      int device_of_pointer(const void * ptr) {
#if defined(MUGRID_ENABLE_CUDA)
        cudaPointerAttributes attr{};
        if (cudaPointerGetAttributes(&attr, ptr) == cudaSuccess) {
          return attr.device;
        }
        (void)cudaGetLastError();
#elif defined(MUGRID_ENABLE_HIP)
        hipPointerAttribute_t attr{};
        if (hipPointerGetAttributes(&attr, ptr) == hipSuccess) {
          return attr.device;
        }
        (void)hipGetLastError();
#else
        (void)ptr;
#endif
        int current{0};
#if defined(MUGRID_ENABLE_CUDA)
        (void)cudaGetDevice(&current);
#elif defined(MUGRID_ENABLE_HIP)
        (void)hipGetDevice(&current);
#endif
        return current;
      }

      //! Fill the fields of `out` from a DLPack tensor
      void read_tensor(const DLTensor & t, ImportedArray & out) {
        out.dtype = t.dtype;
        out.itemsize = itemsize_of(t.dtype);
        out.data = static_cast<char *>(t.data) + t.byte_offset;
        out.shape.assign(t.shape, t.shape + t.ndim);
        out.strides.resize(t.ndim);
        if (t.strides == nullptr) {  // C-contiguous
          Index_t stride{out.itemsize};
          for (int d{t.ndim - 1}; d >= 0; --d) {
            out.strides[d] = stride;
            stride *= out.shape[d];
          }
        } else {
          for (int d{0}; d < t.ndim; ++d) {
            out.strides[d] = t.strides[d] * out.itemsize;
          }
        }
      }

      /**
       * The stream argument of `__dlpack__` that makes a producer order its
       * pending work before the legacy default stream, which the C++
       * exchanges synchronise before MPI reads the buffer: 1 on CUDA (the
       * DLPack value for the legacy default stream), None on ROCm (where
       * the standard has no legacy-stream value and None means the default
       * stream), None for host memory.
       */
      py::object consumer_stream(int device_type) {
        if (device_type == kDLCUDA || device_type == kDLCUDAManaged) {
          return py::int_(1);
        }
        return py::none();
      }

      ImportedArray import_dlpack(const py::object & a) {
        ImportedArray out{};
        out.owner = a;
        const auto device{a.attr("__dlpack_device__")().cast<py::tuple>()};
        out.device_type = device[0].cast<int>();
        out.device_id = device[1].cast<int>();
        out.location = location_of(out.device_type);
        const auto stream{consumer_stream(out.device_type)};

        py::object capsule{};
        try {
          capsule = a.attr("__dlpack__")("stream"_a = stream,
                                         "max_version"_a = py::make_tuple(1, 0));
        } catch (py::error_already_set & e) {
          if (!e.matches(PyExc_TypeError)) {
            throw;
          }
          // A producer predating DLPack 1.0 (no max_version)
          capsule = a.attr("__dlpack__")("stream"_a = stream);
        }
        PyObject * cap{capsule.ptr()};
        const char * name{PyCapsule_GetName(cap)};
        if (name != nullptr && std::strcmp(name, "dltensor_versioned") == 0) {
          auto * m{static_cast<DLManagedTensorVersioned *>(
              PyCapsule_GetPointer(cap, name))};
          read_tensor(m->dl_tensor, out);
          out.read_only = (m->flags & DLPACK_FLAG_BITMASK_READ_ONLY) != 0;
          PyCapsule_SetName(cap, "used_dltensor_versioned");
          out.tensor = std::shared_ptr<void>(m, [](void * p) {
            auto * t{static_cast<DLManagedTensorVersioned *>(p)};
            if (t->deleter != nullptr) {
              py::gil_scoped_acquire gil{};
              t->deleter(t);
            }
          });
        } else if (name != nullptr && std::strcmp(name, "dltensor") == 0) {
          auto * m{static_cast<DLManagedTensor *>(
              PyCapsule_GetPointer(cap, name))};
          read_tensor(m->dl_tensor, out);
          PyCapsule_SetName(cap, "used_dltensor");
          out.tensor = std::shared_ptr<void>(m, [](void * p) {
            auto * t{static_cast<DLManagedTensor *>(p)};
            if (t->deleter != nullptr) {
              py::gil_scoped_acquire gil{};
              t->deleter(t);
            }
          });
        } else {
          throw py::type_error("__dlpack__ did not return a DLPack capsule");
        }
        out.library = library_of(a);
        return out;
      }

      /**
       * An object with only CUDA's array interface (v3). Its `stream` key
       * names the stream the producer wrote on: synchronise it; None means
       * the legacy default stream, which the exchange synchronises anyway.
       * Without a `stream` key there is no telling, so drain the device --
       * acceptable at an exchange.
       */
      ImportedArray import_cuda_array_interface(const py::object & a) {
        ImportedArray out{};
        out.owner = a;
        const py::dict cai{a.attr("__cuda_array_interface__")};
        const auto data{cai["data"].cast<py::tuple>()};
        out.data = reinterpret_cast<void *>(data[0].cast<std::uintptr_t>());
        out.read_only = data[1].cast<bool>();
        out.dtype = dtype_of_typestr(cai["typestr"].cast<std::string>());
        out.itemsize = itemsize_of(out.dtype);
        for (auto extent : cai["shape"].cast<py::tuple>()) {
          out.shape.push_back(extent.cast<Index_t>());
        }
        out.strides.resize(out.shape.size());
        if (!cai.contains("strides") || cai["strides"].is_none()) {
          Index_t stride{out.itemsize};
          for (auto d{static_cast<Index_t>(out.shape.size()) - 1}; d >= 0;
               --d) {
            out.strides[d] = stride;
            stride *= out.shape[d];
          }
        } else {
          Index_t d{0};
          for (auto stride : cai["strides"].cast<py::tuple>()) {
            out.strides[d++] = stride.cast<Index_t>();
          }
        }
        out.device_type = gpu_device_type();
        out.location = location_of(out.device_type);
        out.device_id = device_of_pointer(out.data);
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        if (!cai.contains("stream")) {
          GPU_DEVICE_SYNCHRONIZE();
        } else if (!cai["stream"].is_none()) {
          const auto handle{cai["stream"].cast<std::uintptr_t>()};
          if (handle != 1) {  // 1: the legacy default stream
            GPU_CHECK(
#if defined(MUGRID_ENABLE_CUDA)
                cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(handle))
#else
                hipStreamSynchronize(reinterpret_cast<hipStream_t>(handle))
#endif
            );
          }
        }
#endif
        out.library = library_of(a);
        return out;
      }

      /**
       * A NumPy array whose dtype DLPack has no code for (byte strings,
       * structured dtypes), through NumPy's array interface: moved as
       * opaque bytes, given back as a NumPy array of its dtype.
       */
      ImportedArray import_array_interface(const py::object & a) {
        const py::object dtype{a.attr("dtype")};
        if (dtype.attr("hasobject").cast<bool>()) {
          throw py::type_error("arrays of Python objects cannot be sent over "
                               "MPI; serialise them to a uint8 array first");
        }
        ImportedArray out{};
        out.owner = a;
        const py::dict ai{a.attr("__array_interface__")};
        const auto data{ai["data"].cast<py::tuple>()};
        out.data = reinterpret_cast<void *>(data[0].cast<std::uintptr_t>());
        out.read_only = data[1].cast<bool>();
        out.itemsize = dtype.attr("itemsize").cast<Index_t>();
        out.dtype = DLDataType{kDLOpaqueHandle, 8,
                               static_cast<uint16_t>(out.itemsize)};
        for (auto extent : ai["shape"].cast<py::tuple>()) {
          out.shape.push_back(extent.cast<Index_t>());
        }
        out.strides.resize(out.shape.size());
        if (!ai.contains("strides") || ai["strides"].is_none()) {
          Index_t stride{out.itemsize};
          for (auto d{static_cast<Index_t>(out.shape.size()) - 1}; d >= 0;
               --d) {
            out.strides[d] = stride;
            stride *= out.shape[d];
          }
        } else {
          Index_t d{0};
          for (auto stride : ai["strides"].cast<py::tuple>()) {
            out.strides[d++] = stride.cast<Index_t>();
          }
        }
        out.library = py::module_::import("numpy");
        out.numpy_dtype = dtype;
        return out;
      }

      //! Copy `height` runs of `width` bytes, `spitch`/`dpitch` apart
      void copy_2d(void * dst, std::size_t dpitch, const void * src,
                   Index_t spitch, std::size_t width, std::size_t height,
                   BufferLocation location) {
        const bool pitched{spitch > 0 &&
                           static_cast<std::size_t>(spitch) >= width};
        if (location == BufferLocation::Device) {
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
          if (pitched) {
            GPU_MEMCPY_2D_D2D(dst, dpitch, src,
                              static_cast<std::size_t>(spitch), width, height);
          } else {  // zero or negative strides (broadcasts, flips)
            for (std::size_t r{0}; r < height; ++r) {
              GPU_MEMCPY_D2D(static_cast<char *>(dst) + r * dpitch,
                             static_cast<const char *>(src) +
                                 static_cast<Index_t>(r) * spitch,
                             width);
            }
          }
          return;
#else
          throw py::type_error("device arrays need a build with GPU support");
#endif
        }
        for (std::size_t r{0}; r < height; ++r) {
          std::memcpy(static_cast<char *>(dst) + r * dpitch,
                      static_cast<const char *>(src) +
                          static_cast<Index_t>(r) * spitch,
                      width);
        }
      }

      //! Pack `a` (any strides) into the C-ordered `dst`
      void pack(const ImportedArray & a, void * dst) {
        const auto ndim{static_cast<Index_t>(a.shape.size())};
        if (a.size() == 0) {
          return;
        }
        if (ndim == 0) {
          copy_2d(dst, a.itemsize, a.data, a.itemsize, a.itemsize, 1,
                  a.location);
          return;
        }
        // Innermost block copied in one 2D copy: rows of the last axis if
        // its elements are adjacent (then rows along the axis before it),
        // single elements otherwise.
        const bool inner_run{a.strides[ndim - 1] == a.itemsize};
        const std::size_t width{static_cast<std::size_t>(
            inner_run ? a.shape[ndim - 1] * a.itemsize : a.itemsize)};
        Index_t row_axis{inner_run ? ndim - 2 : ndim - 1};
        const std::size_t height{static_cast<std::size_t>(
            row_axis >= 0 ? a.shape[row_axis] : 1)};
        const Index_t spitch{row_axis >= 0 ? a.strides[row_axis] : 0};
        const Index_t nb_outer_axes{row_axis >= 0 ? row_axis : 0};
        Index_t nb_blocks{1};
        for (Index_t d{0}; d < nb_outer_axes; ++d) {
          nb_blocks *= a.shape[d];
        }
        std::vector<Index_t> index(nb_outer_axes, 0);
        char * out{static_cast<char *>(dst)};
        for (Index_t block{0}; block < nb_blocks; ++block) {
          Index_t offset{0};
          for (Index_t d{0}; d < nb_outer_axes; ++d) {
            offset += index[d] * a.strides[d];
          }
          copy_2d(out, width, static_cast<const char *>(a.data) + offset,
                  spitch, width, height, a.location);
          out += width * height;
          for (Index_t d{nb_outer_axes - 1}; d >= 0; --d) {
            if (++index[d] < a.shape[d]) {
              break;
            }
            index[d] = 0;
          }
        }
      }

      //! Keeps an ExchangeBuffer alive for as long as a DLPack consumer
      //! holds the exported tensor
      struct ExportContext {
        std::vector<int64_t> shape, strides;
        py::object owner;
      };

      template <class Managed>
      void export_deleter(Managed * managed) {
        {
          py::gil_scoped_acquire gil{};
          delete static_cast<ExportContext *>(managed->manager_ctx);
        }
        delete managed;
      }

      template <class Managed>
      Managed * fill_managed(const ExchangeBuffer & buffer,
                             const py::object & owner) {
        auto * ctx{new ExportContext{buffer.shape, buffer.strides, owner}};
        auto * managed{new Managed{}};
        managed->manager_ctx = ctx;
        managed->deleter = export_deleter<Managed>;
        auto & t{managed->dl_tensor};
        t.data = buffer.data;
        t.device = DLDevice{static_cast<DLDeviceType>(buffer.device_type),
                            buffer.device_id};
        t.ndim = static_cast<int32_t>(buffer.shape.size());
        t.dtype = buffer.dtype;
        t.shape = ctx->shape.data();
        t.strides = ctx->strides.data();
        t.byte_offset = 0;
        return managed;
      }

      //! A DLPack capsule of `buffer`, versioned if the consumer can take it
      py::capsule export_capsule(const ExchangeBuffer & buffer,
                                 const py::object & owner, bool versioned) {
        if (versioned) {
          auto * managed{fill_managed<DLManagedTensorVersioned>(buffer, owner)};
          managed->version.major = DLPACK_MAJOR_VERSION;
          managed->version.minor = DLPACK_MINOR_VERSION;
          managed->flags = 0;  // writeable, and not a copy of anything
          py::capsule capsule{managed, "dltensor_versioned"};
          PyCapsule_SetDestructor(capsule.ptr(), [](PyObject * obj) {
            if (PyCapsule_IsValid(obj, "dltensor_versioned")) {  // unconsumed
              auto * t{static_cast<DLManagedTensorVersioned *>(
                  PyCapsule_GetPointer(obj, "dltensor_versioned"))};
              t->deleter(t);
            }
          });
          return capsule;
        }
        auto * managed{fill_managed<DLManagedTensor>(buffer, owner)};
        py::capsule capsule{managed, "dltensor"};
        PyCapsule_SetDestructor(capsule.ptr(), [](PyObject * obj) {
          if (PyCapsule_IsValid(obj, "dltensor")) {  // unconsumed
            auto * t{static_cast<DLManagedTensor *>(
                PyCapsule_GetPointer(obj, "dltensor"))};
            t->deleter(t);
          }
        });
        return capsule;
      }
    }  // namespace

    /* ---------------------------------------------------------------------- */
    std::string numpy_typestr(const DLDataType & dtype) {
      char kind{};
      switch (dtype.code) {
      case kDLInt:
        kind = 'i';
        break;
      case kDLUInt:
        kind = 'u';
        break;
      case kDLFloat:
        kind = 'f';
        break;
      case kDLComplex:
        kind = 'c';
        break;
      case kDLBool:
        kind = 'b';
        break;
      default:
        kind = 'V';  // opaque bytes
        break;
      }
      const auto itemsize{itemsize_of(dtype)};
      return std::string{itemsize == 1 ? '|' : '<'} + kind +
             std::to_string(itemsize);
    }

    /* ---------------------------------------------------------------------- */
    Index_t ImportedArray::size() const {
      Index_t n{1};
      for (auto extent : this->shape) {
        n *= extent;
      }
      return n;
    }

    /* ---------------------------------------------------------------------- */
    bool ImportedArray::c_contiguous() const {
      Index_t expected{this->itemsize};
      for (auto d{static_cast<Index_t>(this->shape.size()) - 1}; d >= 0; --d) {
        if (this->shape[d] == 0) {
          return true;
        }
        if (this->shape[d] != 1 && this->strides[d] != expected) {
          return false;
        }
        expected *= this->shape[d];
      }
      return true;
    }

    /* ---------------------------------------------------------------------- */
    void ImportedArray::move_axis_to_front(Index_t axis) {
      if (axis == 0) {
        return;
      }
      const auto extent{this->shape[axis]};
      const auto stride{this->strides[axis]};
      this->shape.erase(this->shape.begin() + axis);
      this->strides.erase(this->strides.begin() + axis);
      this->shape.insert(this->shape.begin(), extent);
      this->strides.insert(this->strides.begin(), stride);
    }

    /* ---------------------------------------------------------------------- */
    std::optional<TypeDescriptor> ImportedArray::arithmetic_type() const {
      const auto & t{this->dtype};
      if (t.lanes != 1) {
        return std::nullopt;
      }
      switch (t.code) {
      case kDLFloat:
        if (t.bits == 64) return TypeDescriptor::Real;
        if (t.bits == 32) return TypeDescriptor::Real32;
        break;
      case kDLComplex:
        if (t.bits == 128) return TypeDescriptor::Complex;
        if (t.bits == 64) return TypeDescriptor::Complex32;
        break;
      case kDLInt:
        if (t.bits == 32) return TypeDescriptor::Int;
        if (t.bits == 8 * sizeof(Index_t)) return TypeDescriptor::Index;
        break;
      case kDLUInt:
        if (t.bits == 32) return TypeDescriptor::Uint;
        if (t.bits == 8) return TypeDescriptor::Uint8;
        break;
      default:
        break;
      }
      return std::nullopt;
    }

    /* ---------------------------------------------------------------------- */
    ImportedArray import_array(const py::object & a) {
      // A Python number comes back as one, as from mpi4py. NumPy scalars do
      // not: np.float64 is a float subclass but has an array interface, and
      // comes back as a 0-d array.
      const bool scalar{!py::hasattr(a, "__dlpack__") &&
                        !py::hasattr(a, "__array_interface__") &&
                        !py::hasattr(a, "__cuda_array_interface__") &&
                        (py::isinstance<py::bool_>(a) ||
                         py::isinstance<py::int_>(a) ||
                         py::isinstance<py::float_>(a) ||
                         PyComplex_Check(a.ptr()))};
      py::object array{a};
      if (!py::hasattr(a, "__dlpack__") &&
          !py::hasattr(a, "__cuda_array_interface__")) {
        // Python numbers, lists, objects with only NumPy's array interface
        array = py::module_::import("numpy").attr("asarray")(a);
      }
      try {
        if (py::hasattr(array, "__dlpack__") &&
            py::hasattr(array, "__dlpack_device__")) {
          auto out{import_dlpack(array)};
          out.python_scalar = scalar;
          return out;
        }
        if (py::hasattr(array, "__cuda_array_interface__")) {
          return import_cuda_array_interface(array);
        }
      } catch (py::error_already_set & e) {
        // NumPy refuses to export dtypes DLPack has no code for (byte
        // strings, structured and object dtypes) with BufferError
        if (!e.matches(PyExc_BufferError) ||
            !py::hasattr(array, "__array_interface__")) {
          throw;
        }
      }
      return import_array_interface(array);
    }

    /* ---------------------------------------------------------------------- */
    ExchangeBuffer::ExchangeBuffer(const std::vector<Index_t> & shape,
                                   DLDataType dtype, BufferLocation location,
                                   int device_type, int device_id)
        : location{location}, device_type{device_type}, device_id{device_id},
          dtype{dtype} {
      const auto itemsize{itemsize_of(dtype)};
      Index_t count{1};
      for (auto extent : shape) {
        count *= extent;
        this->shape.push_back(extent);
      }
      this->strides.resize(shape.size());
      int64_t stride{1};
      for (auto d{static_cast<Index_t>(shape.size()) - 1}; d >= 0; --d) {
        this->strides[d] = stride;
        stride *= shape[d];
      }
      this->bytes = static_cast<std::size_t>(count * itemsize);
      // Never a null pointer, not even for an empty array
      const std::size_t allocation{this->bytes > 0 ? this->bytes : 1};
      if (location == BufferLocation::Device) {
        this->data = device_allocate(allocation, "communicator-exchange");
      } else {
        this->data = ::operator new(allocation);
      }
    }

    /* ---------------------------------------------------------------------- */
    ExchangeBuffer::~ExchangeBuffer() {
      if (this->location == BufferLocation::Device) {
        device_deallocate(this->data);
      } else {
        ::operator delete(this->data);
      }
    }

    /* ---------------------------------------------------------------------- */
    void ExchangeBuffer::move_front_axis_to(Index_t axis) {
      if (axis == 0) {
        return;
      }
      const auto extent{this->shape.front()};
      const auto stride{this->strides.front()};
      this->shape.erase(this->shape.begin());
      this->strides.erase(this->strides.begin());
      this->shape.insert(this->shape.begin() + axis, extent);
      this->strides.insert(this->strides.begin() + axis, stride);
    }

    /* ---------------------------------------------------------------------- */
    std::shared_ptr<ExchangeBuffer>
    allocate_like(const ImportedArray & like,
                  const std::optional<std::vector<Index_t>> & shape) {
      auto buffer{std::make_shared<ExchangeBuffer>(
          shape ? *shape : like.shape, like.dtype, like.location,
          like.device_type, like.device_id)};
      buffer->numpy_dtype = like.numpy_dtype;
      return buffer;
    }

    /* ---------------------------------------------------------------------- */
    const void * contiguous_data(const ImportedArray & a,
                                 std::shared_ptr<ExchangeBuffer> & scratch) {
      if (a.c_contiguous()) {
        return a.data;
      }
      scratch = allocate_like(a);
      pack(a, scratch->data);
      return scratch->data;
    }

    /* ---------------------------------------------------------------------- */
    py::object hand_back(const ImportedArray & like,
                         std::shared_ptr<ExchangeBuffer> buffer) {
      const bool numpy_dtype{!buffer->numpy_dtype.is_none()};
      py::object exported{py::cast(std::move(buffer))};
      if (numpy_dtype) {  // through the array interface, which knows dtypes
        return py::module_::import("numpy").attr("asarray")(exported);
      }
      if (like.python_scalar) {
        return py::module_::import("numpy")
            .attr("from_dlpack")(exported)
            .attr("item")();
      }
      if (!like.library.is_none() && py::hasattr(like.library, "from_dlpack")) {
        return like.library.attr("from_dlpack")(exported);
      }
      return exported;
    }

    /* ---------------------------------------------------------------------- */
    DeviceGuard::DeviceGuard(const ImportedArray & a) {
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
      if (a.location != BufferLocation::Device) {
        return;
      }
      int current{0};
#if defined(MUGRID_ENABLE_CUDA)
      GPU_CHECK(cudaGetDevice(&current));
      if (current != a.device_id) {
        GPU_CHECK(cudaSetDevice(a.device_id));
        this->previous = current;
      }
#else
      GPU_CHECK(hipGetDevice(&current));
      if (current != a.device_id) {
        GPU_CHECK(hipSetDevice(a.device_id));
        this->previous = current;
      }
#endif
#else
      (void)a;
#endif
    }

    /* ---------------------------------------------------------------------- */
    DeviceGuard::~DeviceGuard() {
#if defined(MUGRID_ENABLE_CUDA)
      if (this->previous >= 0) {
        (void)cudaSetDevice(this->previous);
      }
#elif defined(MUGRID_ENABLE_HIP)
      if (this->previous >= 0) {
        (void)hipSetDevice(this->previous);
      }
#endif
    }

    /* ---------------------------------------------------------------------- */
    void add_exchange_buffer(py::module & mod) {
      py::class_<ExchangeBuffer, std::shared_ptr<ExchangeBuffer>>(
          mod, "ExchangeBuffer",
          "The result of a communicator exchange, in memory muGrid allocated: "
          "an array that any library can take over through DLPack (e.g. "
          "numpy.from_dlpack, cupy.from_dlpack). The exchanges return it "
          "as it is only for inputs whose library has no from_dlpack.")
          .def(
              "__dlpack__",
              [](py::object self, const py::object & stream,
                 const py::object & max_version, const py::object & dl_device,
                 const py::object & copy) {
                const auto & buffer{self.cast<const ExchangeBuffer &>()};
                if (!dl_device.is_none()) {
                  const auto device{dl_device.cast<py::tuple>()};
                  if (device[0].cast<int>() != buffer.device_type ||
                      device[1].cast<int>() != buffer.device_id) {
                    throw py::buffer_error("an ExchangeBuffer is exported on "
                                           "its own device only");
                  }
                }
                if (!copy.is_none() && copy.cast<bool>()) {
                  throw py::buffer_error("an ExchangeBuffer is exported "
                                         "without copying");
                }
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
                // The buffer was written on the legacy default stream; a
                // consumer on another stream must not run ahead of that.
                if (buffer.location == BufferLocation::Device &&
                    !(!stream.is_none() && stream.cast<long long>() == 1)) {
                  GPU_STREAM_SYNCHRONIZE_DEFAULT();
                }
#else
                (void)stream;
#endif
                const bool versioned{
                    !max_version.is_none() &&
                    max_version.cast<py::tuple>()[0].cast<int>() >= 1};
                return export_capsule(buffer, self, versioned);
              },
              "stream"_a = py::none(), "max_version"_a = py::none(),
              "dl_device"_a = py::none(), "copy"_a = py::none())
          .def_property_readonly(
              "__array_interface__",
              [](const ExchangeBuffer & buffer) -> py::object {
                if (buffer.location != BufferLocation::Host) {
                  throw py::attribute_error("__array_interface__");
                }
                const py::object dtype{
                    buffer.numpy_dtype.is_none()
                        ? py::module_::import("numpy").attr("dtype")(
                              numpy_typestr(buffer.dtype))
                        : buffer.numpy_dtype};
                const auto itemsize{dtype.attr("itemsize").cast<int64_t>()};
                py::list strides{};
                for (auto stride : buffer.strides) {
                  strides.append(stride * itemsize);
                }
                py::dict ai{};
                ai["version"] = 3;
                ai["shape"] = py::tuple(py::cast(buffer.shape));
                ai["strides"] = py::tuple(strides);
                ai["typestr"] = dtype.attr("str");
                ai["descr"] = dtype.attr("descr");
                ai["data"] = py::make_tuple(
                    reinterpret_cast<std::uintptr_t>(buffer.data), false);
                return std::move(ai);
              })
          .def("__dlpack_device__",
               [](const ExchangeBuffer & buffer) {
                 return py::make_tuple(buffer.device_type, buffer.device_id);
               })
          .def_property_readonly(
              "shape",
              [](const ExchangeBuffer & buffer) {
                return py::tuple(py::cast(buffer.shape));
              })
          .def_property_readonly("nbytes", [](const ExchangeBuffer & buffer) {
            return buffer.bytes;
          });
    }
  }  // namespace python
}  // namespace muGrid
