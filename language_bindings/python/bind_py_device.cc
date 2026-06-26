/**
 * @file   bind_py_device.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   29 Dec 2025
 *
 * @brief  Python bindings for Device and DeviceType
 *
 * Copyright © 2024 Lars Pastewka
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

#include <cstdint>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "memory/allocation_profiler.hh"
#include "memory/device.hh"
#include "memory/device_alloc.hh"
#include "memory/memory_info.hh"
#include "memory/unified_memory.hh"

using muGrid::Device;
using muGrid::DeviceType;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace {

    // Python-callable device allocator hook. The C++ hook is a plain
    // function pointer, so the Python callables live in statics and the
    // trampolines re-acquire the GIL (muGrid may allocate from code that
    // runs without it). The holders are heap-allocated and intentionally
    // leaked so that no py::object destructor runs after interpreter
    // finalization.
    py::object & py_device_allocate() {
        static py::object * holder{new py::object{}};
        return *holder;
    }

    py::object & py_device_deallocate() {
        static py::object * holder{new py::object{}};
        return *holder;
    }

    void * device_allocate_trampoline(std::size_t bytes) {
        if (!Py_IsInitialized()) {
            return nullptr;
        }
        py::gil_scoped_acquire gil{};
        try {
            return reinterpret_cast<void *>(
                py_device_allocate()(bytes).cast<std::uintptr_t>());
        } catch (py::error_already_set &) {
            // Out of memory (or any other failure) in the external
            // allocator; device_allocate() turns nullptr into a
            // RuntimeError.
            return nullptr;
        }
    }

    void device_deallocate_trampoline(void * ptr) {
        if (!Py_IsInitialized()) {
            // Interpreter is gone; the pool it would return to no longer
            // exists. Intentional leak at shutdown.
            return;
        }
        py::gil_scoped_acquire gil{};
        try {
            py_device_deallocate()(reinterpret_cast<std::uintptr_t>(ptr));
        } catch (py::error_already_set & e) {
            e.discard_as_unraisable("muGrid device deallocate hook");
        }
    }

}  // namespace

void add_device_classes(py::module & mod) {
    mod.def(
        "set_device_allocator",
        [](py::object allocate, py::object deallocate) {
            py_device_allocate() = std::move(allocate);
            py_device_deallocate() = std::move(deallocate);
            muGrid::set_device_allocator(device_allocate_trampoline,
                                         device_deallocate_trampoline);
        },
        "allocate"_a, "deallocate"_a,
        R"pbdoc(
        Register an external device allocator for all muGrid device memory.

        ``allocate(nbytes) -> int`` must return a device pointer (as an
        integer) to at least ``nbytes`` bytes and keep the underlying
        allocation alive until ``deallocate(ptr)`` is called with the same
        integer. Use :func:`muGrid.use_cupy_allocator` for the common case
        of routing through cupy's memory pool, so that one allocator owns
        the GPU and muGrid allocations cannot be starved by pool caching.
        )pbdoc");
    mod.def(
        "clear_device_allocator",
        []() {
            muGrid::set_device_allocator(nullptr, nullptr);
            py_device_allocate() = py::object{};
            py_device_deallocate() = py::object{};
        },
        R"pbdoc(
        Restore the default (raw cudaMalloc/hipMalloc) device allocator.

        Allocations made through a previously registered allocator are
        still freed through it; only new allocations use the default.
        )pbdoc");

    mod.def("device_has_unified_memory", &muGrid::device_has_unified_memory,
            "device_id"_a = -1,
            R"pbdoc(
        True if device memory on ``device_id`` is physically unified with the
        host (integrated GPU / APU such as MI300A), so device pointers are
        directly host-addressable with no copy. ``device_id=-1`` queries the
        currently selected device. Detects physically unified memory only,
        NOT NVIDIA managed/UVM memory. Override with the
        ``MUGRID_UNIFIED_MEMORY`` environment variable.
        )pbdoc");

    // --- Field allocation profiling --------------------------------------
    using muGrid::AllocationProfiler;
    mod.def("enable_allocation_profiling",
            []() { AllocationProfiler::instance().enable(); },
            "Start recording Field buffer allocations (requires building with "
            "-DMUGRID_PROFILE_ALLOCATIONS=ON to record any data).");
    mod.def("disable_allocation_profiling",
            []() { AllocationProfiler::instance().disable(); },
            "Stop recording Field buffer allocations.");
    mod.def("reset_allocation_profiling",
            []() { AllocationProfiler::instance().reset(); },
            "Discard accumulated statistics and restart the measurement "
            "window.");
    mod.def("allocation_profiling_enabled",
            []() { return AllocationProfiler::instance().is_enabled(); },
            "True if Field allocation recording is currently active.");
    mod.def("format_allocation_profile",
            []() { return AllocationProfiler::instance().format_report(); },
            "Return a human-readable multi-line summary of the recorded "
            "Field memory usage per memory pool.");
    mod.def(
        "allocation_profile",
        []() {
            const auto rep{AllocationProfiler::instance().report()};
            py::dict out{};
            out["elapsed_seconds"] = rep.elapsed_seconds;
            py::dict pools{};
            for (const auto & pool : rep.pools) {
                py::dict p{};
                p["unified"] = pool.unified;
                p["current_bytes"] = pool.current;
                p["peak_bytes"] = pool.peak;
                p["average_bytes"] = pool.average;
                if (pool.capacity.valid) {
                    p["total_bytes"] = pool.capacity.total;
                    p["available_bytes"] = pool.capacity.available;
                }
                py::list buffers{};
                for (const auto & buf : pool.buffers) {
                    py::dict b{};
                    b["label"] = buf.label;
                    b["space"] = buf.space;
                    b["current_bytes"] = buf.current_bytes;
                    b["peak_bytes"] = buf.peak_bytes;
                    buffers.append(std::move(b));
                }
                p["buffers"] = std::move(buffers);
                pools[pool.name.c_str()] = std::move(p);
            }
            out["pools"] = std::move(pools);
            return out;
        },
        R"pbdoc(
        Return the recorded Field memory usage as a nested dict.

        The top level has ``elapsed_seconds`` and ``pools`` (keyed by pool
        name, e.g. ``"host"``, ``"cuda:0"`` or ``"unified"``). Each pool gives
        ``peak_bytes``, ``average_bytes`` (time-weighted), ``current_bytes``,
        the physical ``total_bytes``/``available_bytes`` of the pool when
        known, and a ``buffers`` list of per-field ``{label, space,
        current_bytes, peak_bytes}`` records.
        )pbdoc");

    // Device type enumeration
    py::enum_<DeviceType>(mod, "DeviceType",
        R"pbdoc(
        Enumeration of supported device types.

        Values match DLPack's DLDeviceType for interoperability.
        )pbdoc")
        .value("CPU", DeviceType::CPU, "CPU/host memory")
        .value("CUDA", DeviceType::CUDA, "CUDA GPU memory")
        .value("CUDAHost", DeviceType::CUDAHost, "CUDA pinned host memory")
        .value("ROCm", DeviceType::ROCm, "ROCm/HIP GPU memory")
        .value("ROCmHost", DeviceType::ROCmHost, "ROCm pinned host memory")
        .export_values();

    // Device class for specifying where fields are allocated
    py::class_<Device>(mod, "Device",
        R"pbdoc(
        Device specification for field memory allocation.

        A Device specifies where field memory should be allocated. Use the
        static factory methods to create Device instances:

        - ``Device.cpu()`` - CPU/host memory
        - ``Device.cuda(id)`` - CUDA GPU memory (with optional device ID)
        - ``Device.rocm(id)`` - ROCm/HIP GPU memory (with optional device ID)
        - ``Device.gpu(id)`` - Auto-detect GPU backend (CUDA or ROCm)

        Attributes
        ----------
        is_device : bool
            True if this is a GPU device
        is_host : bool
            True if this is a host (CPU) device
        device_type : DeviceType
            The device type enumeration value
        device_id : int
            The device ID (for multi-GPU systems)
        device_string : str
            Device string representation (e.g., 'cpu', 'cuda:0')
        type_name : str
            Human-readable device type name (e.g., 'CPU', 'CUDA')

        Examples
        --------
        >>> fc = GlobalFieldCollection([64, 64], device=Device.cpu())
        >>> fc_gpu = GlobalFieldCollection([64, 64], device=Device.cuda(0))
        >>> fc_auto = GlobalFieldCollection([64, 64], device=Device.gpu())
        )pbdoc")
        .def(py::init<>(), "Create a default CPU device")
        .def(py::init<DeviceType, int>(), "device_type"_a, "device_id"_a = 0,
             "Create a device with explicit type and ID")
        .def_static("cpu", &Device::cpu, "Create a CPU device")
        .def_static("cuda", &Device::cuda, "device_id"_a = 0,
                    "Create a CUDA device with optional device ID")
        .def_static("rocm", &Device::rocm, "device_id"_a = 0,
                    "Create a ROCm device with optional device ID")
        .def_static("gpu", &Device::gpu, "device_id"_a = 0,
                    R"pbdoc(
            Create a GPU device using the default GPU backend.

            Automatically selects the available GPU backend:
            - Returns CUDA device if CUDA is available
            - Returns ROCm device if ROCm is available (and CUDA is not)
            - Returns CPU device if no GPU backend is available

            Parameters
            ----------
            device_id : int, optional
                GPU device ID (default: 0)

            Returns
            -------
            Device
                Device instance for the default GPU backend
            )pbdoc")
        // Read-only properties
        .def_property_readonly("is_device", &Device::is_device,
             "True if this is a GPU device")
        .def_property_readonly("is_host", &Device::is_host,
             "True if this is a host (CPU) device")
        .def_property_readonly("device_type", &Device::get_type,
             "The device type")
        .def_property_readonly("device_id", &Device::get_device_id,
             "The device ID")
        .def_property_readonly("device_string", &Device::get_device_string,
             "Device string (e.g., 'cpu', 'cuda:0')")
        .def_property_readonly("type_name", &Device::get_type_name,
             "Device type name (e.g., 'CPU', 'CUDA')")
        .def_property_readonly("is_unified", &Device::is_unified,
             "True if this device's memory is physically unified with the "
             "host (integrated GPU / APU), so its buffers are directly "
             "host-addressable without a staging copy")
        // Special methods
        .def("__repr__", [](const Device & d) {
            return "<Device: " + d.get_device_string() + ">";
        })
        .def("__eq__", &Device::operator==)
        .def("__ne__", &Device::operator!=)
        .def("__hash__", [](const Device & d) {
            // Hash based on device type and id
            return std::hash<int>{}(static_cast<int>(d.get_type()) * 1000 +
                                    d.get_device_id());
        });
}
