# Session Context - December 9, 2025

## Branch: `25_refactor`

## Completed Work

### 1. ConvolutionOperator Refactoring (DONE)
Changes made to `src/libmugrid/convolution_operator.hh`:

- **Moved GPU methods from public to private:**
  - `apply_on_device<DeviceSpace>()`
  - `transpose_on_device<DeviceSpace>()`
  - `get_device_apply_operator<DeviceSpace>()`
  - `get_device_transpose_operator<DeviceSpace>()`

- **Moved protected data members to private:**
  - `pixel_offset`, `pixel_operator`, `conv_pts_shape`
  - `nb_pixelnodal_pts`, `nb_quad_pts`, `nb_operators`
  - `spatial_dim`, `nb_conv_pts`

Changes made to `tests/test_convolution_operator.cc`:
- Simplified GPU tests from direct kernel correctness tests to consistency tests
- Tests now use public API (`apply()`, `transpose()`) rather than private methods
- Renamed tests: `device_kernel_correctness` -> `device_apply_consistency`
- Renamed tests: `device_transpose_kernel_correctness` -> `device_transpose_consistency`

**Status:** Code compiles and C++ tests pass.

### 2. nanobind Migration (DONE)
Migrated all Python bindings from pybind11 to nanobind for better DLPack support.

**Files migrated:**
- `bind_py_declarations.hh` - Header with function declarations
- `bind_py_module.cc` - Module entry point
- `bind_py_common_mugrid.cc` - Common types and enums
- `bind_py_communicator.cc` - MPI communicator bindings
- `bind_py_field.cc` - Field bindings with DLPack support
- `bind_py_state_field.cc` - State field bindings
- `bind_py_field_collection.cc` - Field collection bindings
- `bind_py_decomposition.cc` - Domain decomposition bindings
- `bind_py_convolution_operator.cc` - Convolution operator bindings
- `bind_py_file_io.cc` - File I/O bindings
- `bind_py_options_dictionary.cc` - Dictionary bindings
- `numpy_tools.hh` - NumPy array wrapping utilities
- `python_helpers.hh` - Python helper functions

**Build system changes:**
- `CMakeLists.txt` - Updated to fetch and use nanobind v2.4.0 instead of pybind11
- `language_bindings/python/CMakeLists.txt` - Changed `pybind11_add_module` to `nanobind_add_module`

### 3. GPU-Aware Python Bindings (DONE)

**Device Introspection (Field base class):**
```python
field.device      # Returns 'cpu', 'cuda:N', or 'rocm:N'
field.is_on_gpu   # Returns True if data is on GPU
```

**DLPack Support (TypedFieldBase classes):**
```python
# Zero-copy interop with PyTorch, JAX, CuPy, etc.
torch_tensor = torch.from_dlpack(field)
jax_array = jax.dlpack.from_dlpack(field)
cupy_array = cupy.from_dlpack(field)

# DLPack protocol methods
field.__dlpack__(stream=None)     # Export as DLPack capsule
field.__dlpack_device__()         # Returns (device_type, device_id)
```

**Device Types (DLPack standard):**
- `kDLCPU = 1` - CPU memory
- `kDLCUDA = 2` - CUDA GPU memory
- `kDLROCM = 10` - ROCm/HIP GPU memory

## Key API Changes (pybind11 -> nanobind)

| pybind11 | nanobind |
|----------|----------|
| `py::module` | `nb::module_` |
| `py::class_` | `nb::class_` |
| `py::init<>()` | `nb::init<>()` |
| `def_property_readonly` | `def_prop_ro` |
| `def_property` | `def_prop_rw` |
| `def_property_readonly_static` | `def_prop_ro_static` |
| `py::return_value_policy::reference_internal` | `nb::rv_policy::reference_internal` |
| `PYBIND11_MODULE` | `NB_MODULE` |
| `PYBIND11_OVERRIDE_PURE` | `NB_OVERRIDE_PURE` |
| `py::buffer_protocol()` | (handled via ndarray) |
| `py::array_t<T>` | `nb::ndarray<nb::numpy, T>` |
| `py::make_iterator` | `nb::make_iterator(type, name, begin, end)` |

## Current Architecture

**Field Memory Model:**
- All fields are currently `TypedFieldBase<T, HostSpace>` (host memory)
- GPU operations copy to device, execute, copy back (via Kokkos deep_copy)
- DLPack enables zero-copy from host fields to GPU frameworks

**Future Extensions:**
When device-space fields are added to the FieldCollection:
1. `field.device` will return the actual device ('cuda:0', 'rocm:0', etc.)
2. `field.is_on_gpu` will return True for device fields
3. `__dlpack_device__` will return the correct device tuple
4. Accessors (`.s`, `.p`, etc.) will return cupy arrays implicitly when on GPU

## Pending Work

**Testing:**
Build and test the migration:
```bash
mkdir build && cd build
cmake .. -DMUGRID_ENABLE_PYTHON=ON
make -j
ctest -V
```

Python test:
```python
import muGrid
fc = muGrid.GlobalFieldCollection([10, 10])
field = fc.register_real_field("test", 1)
field.set_zero()

# Check device introspection
print(field.device)      # 'cpu'
print(field.is_on_gpu)   # False

# DLPack interop (requires torch/jax/cupy installed)
import torch
t = torch.from_dlpack(field)
print(t.shape, t.device)
```

## Files Modified (Uncommitted)

**C++ library:**
```
src/libmugrid/convolution_operator.hh  | Modified
src/libmugrid/numpy_tools.hh           | Migrated to nanobind
src/libmugrid/python_helpers.hh        | Migrated to nanobind
```

**Python bindings:**
```
language_bindings/python/bind_py_declarations.hh
language_bindings/python/bind_py_module.cc
language_bindings/python/bind_py_common_mugrid.cc
language_bindings/python/bind_py_communicator.cc
language_bindings/python/bind_py_field.cc
language_bindings/python/bind_py_state_field.cc
language_bindings/python/bind_py_field_collection.cc
language_bindings/python/bind_py_decomposition.cc
language_bindings/python/bind_py_convolution_operator.cc
language_bindings/python/bind_py_file_io.cc
language_bindings/python/bind_py_options_dictionary.cc
```

**Build system:**
```
CMakeLists.txt
language_bindings/python/CMakeLists.txt
```

## Environment Notes
- MPI tests with np_8 stall (environment issue, not code)
- User is installing CUDA-aware MPI

## Relevant Documentation Links
- nanobind: https://nanobind.readthedocs.io/
- nanobind ndarray: https://nanobind.readthedocs.io/en/latest/ndarray.html
- DLPack: https://dmlc.github.io/dlpack/latest/
- Kokkos: https://kokkos.github.io/kokkos-core-wiki/
