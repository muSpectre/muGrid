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

#include <pybind11/pybind11.h>

#include "memory/device.hh"

using muGrid::Device;
using muGrid::DeviceType;
using pybind11::literals::operator""_a;

namespace py = pybind11;

void add_device_classes(py::module & mod) {
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
