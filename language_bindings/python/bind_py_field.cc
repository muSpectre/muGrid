/**
 * @file   bind_py_field.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   16 Oct 2019
 *
 * @brief  Python bindings for fields
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

#include "core/types.hh"
#include "core/exception.hh"
#include "field/field.hh"
#include "field/field_map.hh"
#include "field/field_typed.hh"
#include "collection/field_collection.hh"
#include "collection/field_collection_global.hh"
#include "field/mapped_field.hh"

#include <dlpack/dlpack.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <sstream>
#include <vector>
#include <memory>
#include <complex>
#include <cstring>

using muGrid::Field;
using muGrid::FieldCollection;
using muGrid::GlobalFieldCollection;
using muGrid::Index_t;
using muGrid::RuntimeError;
using muGrid::Shape_t;
using muGrid::TypedField;
using muGrid::TypedFieldBase;
using muGrid::operator<<;
using pybind11::literals::operator""_a;

using muGrid::Real;
using muGrid::Int;
using muGrid::Complex;
using muGrid::Uint;
namespace py = pybind11;

/**
 * Context for DLPack managed tensor - holds the field reference to prevent
 * deallocation while the tensor is in use.
 * Templated on MemorySpace to support both Host and Device fields.
 */
template<typename T, typename MemorySpace = muGrid::HostSpace>
struct DLPackContext {
    TypedFieldBase<T, MemorySpace>* field;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;

    DLPackContext(TypedFieldBase<T, MemorySpace>* f, const Shape_t& s, const Shape_t& st)
        : field(f), shape(s.begin(), s.end()) {
        // DLPack spec says strides are in elements, not bytes
        strides.reserve(st.size());
        for (auto stride : st) {
            strides.push_back(static_cast<int64_t>(stride));
        }
    }
};

/**
 * Deleter function for DLManagedTensorVersioned
 */
template<typename T, typename MemorySpace = muGrid::HostSpace>
void dlpack_deleter(DLManagedTensorVersioned* tensor) {
    auto* ctx = static_cast<DLPackContext<T, MemorySpace>*>(tensor->manager_ctx);
    delete ctx;
    delete tensor;
}

/**
 * Get DLDataType for C++ types
 */
template<typename T>
DLDataType get_dlpack_dtype();

template<>
DLDataType get_dlpack_dtype<float>() {
    return DLDataType{kDLFloat, 32, 1};
}

template<>
DLDataType get_dlpack_dtype<double>() {
    return DLDataType{kDLFloat, 64, 1};
}

template<>
DLDataType get_dlpack_dtype<int32_t>() {
    return DLDataType{kDLInt, 32, 1};
}

template<>
DLDataType get_dlpack_dtype<int64_t>() {
    return DLDataType{kDLInt, 64, 1};
}

template<>
DLDataType get_dlpack_dtype<uint32_t>() {
    return DLDataType{kDLUInt, 32, 1};
}

template<>
DLDataType get_dlpack_dtype<uint64_t>() {
    return DLDataType{kDLUInt, 64, 1};
}

template<>
DLDataType get_dlpack_dtype<std::complex<float>>() {
    return DLDataType{kDLComplex, 64, 1};
}

template<>
DLDataType get_dlpack_dtype<std::complex<double>>() {
    return DLDataType{kDLComplex, 128, 1};
}

/**
 * Create a DLPack capsule for a typed field using the versioned protocol.
 * Works for both Host and Device (GPU) fields via get_void_data_ptr(false).
 * Uses DLManagedTensorVersioned with flags for proper writeable support.
 * Requires numpy >= 2.1 or other consumers that support versioned dlpack.
 */
template<typename T, typename MemorySpace = muGrid::HostSpace>
py::capsule create_dlpack_capsule(TypedFieldBase<T, MemorySpace>& field) {
    auto& coll = field.get_collection();
    if (!coll.is_initialised()) {
        throw RuntimeError("Field collection isn't initialised yet");
    }

    // Get shape and strides for SubPt iteration (full buffer with ghosts)
    auto iter_unit = muGrid::IterUnit::SubPt;
    Shape_t shape = field.get_shape(iter_unit);
    Shape_t strides = field.get_strides(iter_unit, 1);  // strides in elements

    // Create context to hold shape/stride data
    auto* ctx = new DLPackContext<T, MemorySpace>(&field, shape, strides);

    // Create DLManagedTensorVersioned (the modern DLPack protocol)
    auto* managed = new DLManagedTensorVersioned();

    // Set version
    managed->version.major = DLPACK_MAJOR_VERSION;
    managed->version.minor = DLPACK_MINOR_VERSION;

    // Set manager context and deleter
    managed->manager_ctx = ctx;
    managed->deleter = dlpack_deleter<T, MemorySpace>;

    // Set flags - 0 means writable, not a copy
    managed->flags = 0;

    // Fill in tensor info
    auto& tensor = managed->dl_tensor;

    // Get data pointer - pass false to skip host assertion (works for any memory space)
    tensor.data = field.get_void_data_ptr(false);

    // Set device based on field's memory space using virtual methods
    tensor.device = DLDevice{
        static_cast<DLDeviceType>(field.get_dlpack_device_type()),
        field.get_device_id()
    };

    tensor.ndim = static_cast<int32_t>(shape.size());
    tensor.dtype = get_dlpack_dtype<T>();
    tensor.shape = ctx->shape.data();
    tensor.strides = ctx->strides.data();
    tensor.byte_offset = 0;

    // Create PyCapsule with the versioned name for DLPack protocol.
    auto capsule = py::capsule(managed, "dltensor_versioned");

    // Set up a destructor that only cleans up if the capsule was NOT consumed
    PyCapsule_SetDestructor(capsule.ptr(), [](PyObject* obj) {
        const char* name = PyCapsule_GetName(obj);
        // Only delete if capsule was never consumed
        if (name != nullptr && std::strcmp(name, "dltensor_versioned") == 0) {
            auto* tensor = static_cast<DLManagedTensorVersioned*>(PyCapsule_GetPointer(obj, name));
            if (tensor && tensor->deleter) {
                tensor->deleter(tensor);
            }
        }
    });

    return capsule;
}

void add_field(py::module &mod) {
    py::class_<Field>(mod, "Field")
            .def("set_zero", &Field::set_zero)
            .def("stride", &Field::get_stride)
            .def_property_readonly("buffer_size", &Field::get_buffer_size)
            .def_property_readonly("element_size_in_bytes", &Field::get_element_size_in_bytes)
            // Shape with ghosts (SubPt layout) - this is what __dlpack__ exports
            .def_property_readonly("shape",
                                   [](Field &field) {
                                       return field.get_shape(muGrid::IterUnit::SubPt);
                                   })
            // Shape without ghosts (SubPt layout)
            .def_property_readonly("shape_s",
                                   [](Field &field) {
                                       return field.get_shape_without_ghosts(muGrid::IterUnit::SubPt);
                                   })
            // Offsets for slicing out ghosts (SubPt layout)
            .def_property_readonly("offsets_s",
                                   [](Field &field) {
                                       return field.get_offsets_without_ghosts(muGrid::IterUnit::SubPt);
                                   })
            // Shape with ghosts (Pixel layout)
            .def_property_readonly("shape_pg",
                                   [](Field &field) {
                                       return field.get_shape(muGrid::IterUnit::Pixel);
                                   })
            // Shape without ghosts (Pixel layout)
            .def_property_readonly("shape_p",
                                   [](Field &field) {
                                       return field.get_shape_without_ghosts(muGrid::IterUnit::Pixel);
                                   })
            // Offsets for slicing out ghosts (Pixel layout)
            .def_property_readonly("offsets_p",
                                   [](Field &field) {
                                       return field.get_offsets_without_ghosts(muGrid::IterUnit::Pixel);
                                   })
            // Strides (SubPt layout, in elements)
            .def_property_readonly("strides",
                                   [](Field &field) {
                                       return field.get_strides(muGrid::IterUnit::SubPt, 1);
                                   })
            // Strides (Pixel layout, in elements)
            .def_property_readonly("strides_p",
                                   [](Field &field) {
                                       return field.get_strides(muGrid::IterUnit::Pixel, 1);
                                   })
            .def_property_readonly("name", &Field::get_name)
            .def_property_readonly("collection", &Field::get_collection)
            .def_property_readonly("nb_components", &Field::get_nb_components)
            .def_property_readonly("components_shape", &Field::get_components_shape)
            .def_property_readonly("nb_entries", &Field::get_nb_entries)
            .def_property_readonly("nb_buffer_entries", &Field::get_nb_buffer_entries)
            .def_property_readonly("is_global", &Field::is_global)
            .def_property_readonly("sub_division", &Field::get_sub_division_tag)
            .def_property_readonly("spatial_dim", &Field::get_spatial_dim)
            // Device introspection for GPU-aware interop
            .def_property_readonly(
                "device",
                [](const Field &field) {
                    return field.get_device_string();
                },
                "Returns the device where the field data resides ('cpu' or 'cuda:N' or 'rocm:N')")
            .def_property_readonly(
                "is_on_gpu",
                [](const Field &field) {
                    return field.is_on_device();
                },
                "Returns True if the field data resides on a GPU");
}

template<class T>
void add_typed_field(py::module &mod, std::string name) {
    py::class_<TypedFieldBase<T>, Field>(mod, (name + "Base").c_str())
            .def(
                "get_pixel_map",
                [](TypedFieldBase<T> &field, const Index_t &nb_rows) {
                    return field.get_pixel_map(nb_rows);
                },
                "nb_rows"_a = muGrid::Unknown,
                py::return_value_policy::reference_internal)
            .def(
                "get_sub_pt_map",
                [](TypedFieldBase<T> &field, const Index_t &nb_rows) {
                    return field.get_sub_pt_map(nb_rows);
                },
                "nb_rows"_a = muGrid::Unknown,
                py::return_value_policy::reference_internal)
            // DLPack support: versioned protocol for numpy >= 2.1
            .def(
                "__dlpack__",
                [](TypedFieldBase<T> &self, py::object stream) {
                    (void)stream;  // Stream parameter not used for CPU tensors
                    return create_dlpack_capsule(self);
                },
                "stream"_a = py::none(),
                "Export field data via DLPack for zero-copy interop with NumPy, PyTorch, JAX, CuPy, etc.")
            .def(
                "__dlpack_device__",
                [](TypedFieldBase<T> &self) {
                    // Return (device_type, device_id) using field's actual device info
                    return py::make_tuple(
                        self.get_dlpack_device_type(),
                        self.get_device_id()
                    );
                },
                "Return DLPack device tuple (device_type, device_id)");

    py::class_<TypedField<T>, TypedFieldBase<T> >(mod, name.c_str())
            .def("clone", &TypedField<T>::clone, "new_name"_a, "allow_overwrite"_a,
                 py::return_value_policy::reference_internal);
}

/**
 * Register device-space (GPU) typed field bindings.
 * These inherit from the base Field class and provide DLPack support for GPU arrays.
 * Only compiled when CUDA or HIP backends are enabled.
 */
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
template<class T>
void add_typed_field_device(py::module &mod, std::string name) {
    using DeviceSpace = muGrid::DefaultDeviceSpace;

    // Device-space TypedFieldBase - inherits from Field
    // Note: get_pixel_map and get_sub_pt_map are host-only, so not exposed here
    py::class_<TypedFieldBase<T, DeviceSpace>, Field>(mod, (name + "DeviceBase").c_str())
            // DLPack support: versioned protocol for numpy >= 2.1
            .def(
                "__dlpack__",
                [](TypedFieldBase<T, DeviceSpace> &self, py::object stream) {
                    (void)stream;  // TODO: support CUDA stream for async transfers
                    return create_dlpack_capsule(self);
                },
                "stream"_a = py::none(),
                "Export GPU field data via DLPack for zero-copy interop with CuPy, PyTorch, JAX, etc.")
            .def(
                "__dlpack_device__",
                [](TypedFieldBase<T, DeviceSpace> &self) {
                    return py::make_tuple(
                        self.get_dlpack_device_type(),
                        self.get_device_id()
                    );
                },
                "Return DLPack device tuple (device_type, device_id)");

    // Device-space TypedField
    py::class_<TypedField<T, DeviceSpace>, TypedFieldBase<T, DeviceSpace>>(
            mod, (name + "Device").c_str())
            .def("clone", &TypedField<T, DeviceSpace>::clone, "new_name"_a, "allow_overwrite"_a,
                 py::return_value_policy::reference_internal);
}
#endif

template<typename T, muGrid::Mapping Mutability>
decltype(auto) add_field_map_const(py::module &mod, const std::string &name) {
    std::string full_name{
        name +
        (Mutability == muGrid::Mapping::Mut ? "Mut" : "Const")
    };
    using Map_t = muGrid::FieldMap<T, Mutability>;

    py::class_<Map_t> pyclass(mod, full_name.c_str());
    pyclass.def("mean", [](const Map_t &map) { return map.mean(); });
    return pyclass;
}

template<typename T>
void add_field_map(py::module &mod, const std::string &name) {
    add_field_map_const<T, muGrid::Mapping::Const>(mod, name);
    add_field_map_const<T, muGrid::Mapping::Mut>(mod, name).def(
        "set_uniform",
        [](muGrid::FieldMap<T, muGrid::Mapping::Mut> &map,
           py::EigenDRef<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > val) {
            map = val;
        },
        "value"_a);
}

template<typename T>
void add_mutable_mapped_field(py::module &mod, const std::string &name) {
    using MappedField_t =
            muGrid::MappedField<muGrid::FieldMap<T, muGrid::Mapping::Mut> >;

    py::class_<MappedField_t>(mod, name.c_str())
            .def_property_readonly("field",
                                   [](MappedField_t &mf) -> muGrid::TypedField<T> & {
                                       return mf.get_field();
                                   })
            .def_property_readonly(
                "map",
                [](MappedField_t &mf)
            -> muGrid::FieldMap<T, muGrid::Mapping::Mut> & {
                    return mf.get_map();
                });
}

void add_field_classes(py::module &mod) {
    add_field(mod);

    // Host-space typed fields
    add_typed_field<muGrid::Real>(mod, "RealField");
    add_typed_field<muGrid::Complex>(mod, "ComplexField");
    add_typed_field<muGrid::Int>(mod, "IntField");
    add_typed_field<muGrid::Uint>(mod, "UintField");

    // Device-space (GPU) typed fields - only when CUDA/HIP is enabled
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
    add_typed_field_device<muGrid::Real>(mod, "RealField");
    add_typed_field_device<muGrid::Complex>(mod, "ComplexField");
    add_typed_field_device<muGrid::Int>(mod, "IntField");
    add_typed_field_device<muGrid::Uint>(mod, "UintField");
#endif

    add_field_map<muGrid::Real>(mod, "RealFieldMap");
    add_field_map<muGrid::Complex>(mod, "ComplexFieldMap");
    add_field_map<muGrid::Int>(mod, "IntFieldMap");
    add_field_map<muGrid::Uint>(mod, "UintFieldMap");

    add_mutable_mapped_field<muGrid::Real>(mod, "RealMappedField");
    add_mutable_mapped_field<muGrid::Complex>(mod, "ComplexMappedField");
    add_mutable_mapped_field<muGrid::Int>(mod, "IntMappedField");
    add_mutable_mapped_field<muGrid::Uint>(mod, "UintMappedField");
}
