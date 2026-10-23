/**
 * @file   array_exchange.hh
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

#ifndef LANGUAGE_BINDINGS_PYTHON_ARRAY_EXCHANGE_HH_
#define LANGUAGE_BINDINGS_PYTHON_ARRAY_EXCHANGE_HH_

#include "core/types.hh"
#include "mpi/communicator.hh"

#include <dlpack/dlpack.h>
#include <pybind11/pybind11.h>

#include <memory>
#include <optional>
#include <vector>

namespace muGrid {
  namespace python {
    namespace py = pybind11;

    /**
     * A Python array of any library as the communicator's exchanges see it.
     *
     * Read through DLPack (`__dlpack__` / `__dlpack_device__`), which gives
     * pointer, shape, strides, dtype, device type and device id without
     * knowing the library. Objects that only provide CUDA's array interface
     * (v3) are read through it; anything else goes through NumPy. Device
     * arrays are asked to order their pending work before the legacy default
     * stream, which the C++ exchanges synchronise before handing buffers to
     * MPI (see gpu_runtime.hh). The DLPack tensor is released when the last
     * copy of this object goes.
     */
    struct ImportedArray {
      py::object owner;  //!< the array; keeps the memory alive
      std::shared_ptr<void> tensor;  //!< releases the DLPack tensor
      void * data{nullptr};
      std::vector<Index_t> shape{};
      std::vector<Index_t> strides{};  //!< in bytes
      DLDataType dtype{};
      Index_t itemsize{0};
      BufferLocation location{BufferLocation::Host};
      int device_type{kDLCPU};  //!< DLDeviceType
      int device_id{0};
      bool read_only{false};
      bool python_scalar{false};  //!< a Python number, given back as one
      //! module whose from_dlpack takes the results (None: no library)
      py::object library{py::none()};
      //! For NumPy arrays of dtypes DLPack has no code for (byte strings,
      //! structured dtypes), read through NumPy's array interface and moved
      //! as opaque bytes: the NumPy dtype the results get back
      py::object numpy_dtype{py::none()};

      Index_t size() const;
      bool c_contiguous() const;
      //! Move `axis` to the front (a view: only shape and strides change)
      void move_axis_to_front(Index_t axis);
      //! The arithmetic MPI type of the elements, if there is one
      std::optional<TypeDescriptor> arithmetic_type() const;
    };

    //! NumPy's typestr ("<f8") of a DLPack dtype with a NumPy equivalent
    std::string numpy_typestr(const DLDataType & dtype);

    //! Read `a` (see ImportedArray)
    ImportedArray import_array(const py::object & a);

    /**
     * Memory muGrid allocated for an exchange's result, exported through
     * DLPack (and the array interfaces) so that the input's own library can
     * take it over. Host memory is allocated with operator new, device
     * memory through muGrid's device allocator, on the device it was asked
     * for. Shape and strides describe the view the consumer sees, which
     * need not be the order the buffer was filled in (allgatherv).
     */
    struct ExchangeBuffer {
      void * data{nullptr};
      std::size_t bytes{0};
      BufferLocation location{BufferLocation::Host};
      int device_type{kDLCPU};
      int device_id{0};
      DLDataType dtype{};
      std::vector<int64_t> shape{};
      std::vector<int64_t> strides{};  //!< in elements
      py::object numpy_dtype{py::none()};  //!< see ImportedArray::numpy_dtype

      ExchangeBuffer(const std::vector<Index_t> & shape, DLDataType dtype,
                     BufferLocation location, int device_type,
                     int device_id);
      ExchangeBuffer(const ExchangeBuffer &) = delete;
      ExchangeBuffer & operator=(const ExchangeBuffer &) = delete;
      ~ExchangeBuffer();

      //! Present axis 0 of the (C-ordered) buffer at position `axis`
      void move_front_axis_to(Index_t axis);
    };

    //! A C-contiguous buffer shaped like `like` (or `shape`), on its device
    std::shared_ptr<ExchangeBuffer>
    allocate_like(const ImportedArray & like,
                  const std::optional<std::vector<Index_t>> & shape = {});

    /**
     * Pointer to `a`'s elements in C order: its own memory if it is
     * C-contiguous, otherwise a packed copy (kept alive by `scratch`), made
     * with the host or device copies muGrid uses for strided halo faces.
     */
    const void * contiguous_data(const ImportedArray & a,
                                 std::shared_ptr<ExchangeBuffer> & scratch);

    //! `buffer` as an array of `like`'s library (a Python number for a
    //! number), or the buffer itself if that library has no from_dlpack
    py::object hand_back(const ImportedArray & like,
                         std::shared_ptr<ExchangeBuffer> buffer);

    /**
     * Make `a`'s device current for the lifetime of this object, so that
     * staging buffers and results land on the input's device and not on
     * whichever device happens to be current.
     */
    class DeviceGuard {
     public:
      explicit DeviceGuard(const ImportedArray & a);
      DeviceGuard(const DeviceGuard &) = delete;
      DeviceGuard & operator=(const DeviceGuard &) = delete;
      ~DeviceGuard();

     private:
      int previous{-1};
    };

    //! Register ExchangeBuffer with the module
    void add_exchange_buffer(py::module & mod);
  }  // namespace python
}  // namespace muGrid

#endif  // LANGUAGE_BINDINGS_PYTHON_ARRAY_EXCHANGE_HH_
