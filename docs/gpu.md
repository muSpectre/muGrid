# GPU

µGrid supports GPU acceleration through CUDA (NVIDIA) and HIP (AMD ROCm) backends.
Fields can be allocated on either host (CPU) or device (GPU) memory, and operations
on GPU fields are executed on the GPU.

## Building with GPU support

GPU support is enabled at configure time with `MUGRID_ENABLE_CUDA=ON` (NVIDIA)
or `MUGRID_ENABLE_HIP=ON` (AMD); Python access to device arrays needs
[CuPy](https://cupy.dev/). Build options, architecture selection and the CuPy
packages are described under [Installation](installation.md#gpu-support).

## Checking GPU availability

The module flags report whether µGrid was compiled with GPU support:

```python
import muGrid

print(f"CUDA available: {muGrid.has_cuda}")
print(f"ROCm available: {muGrid.has_rocm}")
print(f"Any GPU backend: {muGrid.has_gpu}")
```

`has_gpu` is `True` if either backend was compiled in. Whether a GPU can
actually be used at run time is reported by `muGrid.is_gpu_available()`, which
attempts a small device allocation.

## Device selection

Field collections can be created on a specific device using the `device` parameter.
The device can be specified as a simple string or as a `Device` object.

The accepted strings (`"cpu"`, `"gpu"`, `"gpu:N"`, `"cuda:N"`, `"rocm:N"`, …)
are listed in the [Python API](python.md#device-strings). `"gpu"` selects the
compiled backend and keeps scripts portable between NVIDIA and AMD systems.

Here is an example of creating a field collection on the GPU:

```python
import muGrid

# Create a GPU field collection on the compiled backend
fc = muGrid.GlobalFieldCollection([64, 64], device="gpu")

# Or explicitly specify CUDA
fc = muGrid.GlobalFieldCollection([64, 64], device="cuda")

# Or using Device object for auto-detection
fc = muGrid.GlobalFieldCollection([64, 64], device=muGrid.Device.gpu())

# Create a field on the GPU
field = fc.real_field("my_field")
print(f"Field is on GPU: {field.is_on_gpu}")
print(f"Device: {field.device}")
```

## Working with GPU arrays

When accessing GPU field data, the array views (`s`, `p`, `sg`, `pg`) return
[CuPy](https://cupy.dev/) arrays instead of numpy arrays. CuPy provides a
numpy-compatible API for GPU arrays:

```python
import muGrid

# Check GPU is available
if not muGrid.has_gpu:
    raise RuntimeError("GPU support not available")

import cupy as cp

# Create GPU field collection
fc = muGrid.GlobalFieldCollection([64, 64], device="cuda")

# Create fields
field_a = fc.real_field("a")
field_b = fc.real_field("b")

# Initialize with CuPy (GPU) operations
field_a.p[...] = cp.random.randn(*field_a.p.shape)
field_b.p[...] = cp.sin(field_a.p) + cp.cos(field_a.p)

# Compute on GPU
result = cp.sum(field_b.p ** 2)
print(f"Sum of squares: {result}")
```

For writing device-agnostic code, avoid importing `numpy` or `cupy` directly.
In the above example, execute the sum with:

```python
result = (field_b.p ** 2).sum()
```

This may not always be possible. In this case, it may be useful to import
`numpy` or `cupy` under the same module alias:

```python
if device == "cpu":
    import numpy as xp
else:
    import cupy as xp
```

## Device memory allocation

By default µGrid allocates device memory with raw `cudaMalloc` / `hipMalloc`,
while Python array libraries such as CuPy serve their allocations from a caching
pool that does not return freed blocks to the driver. Two independent allocators
on one device can starve each other: near capacity µGrid's raw allocation fails
even though free blocks sit cached inside CuPy's pool. Route both through a
single owner to remove this failure mode by construction:

```python
import muGrid

muGrid.use_cupy_allocator()   # µGrid draws from CuPy's memory pool
fc = muGrid.GlobalFieldCollection([512, 512, 512], device="gpu")
```

Call `use_cupy_allocator()` **once, before** creating any GPU field collection.
The inverse, `muGrid.route_cupy_through_mugrid()`, instead makes µGrid the single
owner of every device byte (and records CuPy's allocations in the allocation
profiler alongside Fields); `muGrid.clear_device_allocator()` restores the raw
default.

### Unified-memory APUs and the full HBM

On a unified-memory accelerator — an APU such as the **AMD Instinct MI300A**,
where the CPU and GPU share one physical HBM stack — the default device
allocator only reaches the *coarse-grained device-local window*, which is a
fraction of the physical memory. On MI300A that window is ~62.8 GiB of the
128 GB package. The remaining memory is **not** physically partitioned away: it
is visible to the CPU and reachable from the GPU through managed memory (the
same physical HBM, at native bandwidth). A plain `hipMalloc` nonetheless runs
out of memory once allocations exceed that window.

*Managed* (unified) memory — `hipMallocManaged` / `cudaMallocManaged` — draws
from the whole shared pool instead. Point CuPy's default pool at managed memory
and then route µGrid through it, so **fields, solver scratch and the
preconditioner all use the full HBM**:

```python
import cupy, muGrid

# CuPy's default pool -> managed (unified) memory ...
cupy.cuda.set_allocator(cupy.cuda.MemoryPool(cupy.cuda.malloc_managed).malloc)
# ... and µGrid draws from that same pool.
muGrid.use_cupy_allocator()

# A large double-precision grid that overflows the coarse-grained window
# allocates from the full unified HBM:
fc = muGrid.GlobalFieldCollection([512, 512, 512], device="gpu")
```

!!! note "When this matters"
    Only when a workload's device footprint exceeds the coarse-grained window
    — e.g. a 512³ double-precision FFT-preconditioned solve on MI300A needs
    ~70 GiB > 62.8 GiB, so it OOMs with the default allocator but fits with
    managed memory. Grids that fit the window need no change. On a true APU
    managed pages live in the same physical HBM as the default pool, so there is
    no PCIe migration; a modest first-touch / page-fault overhead is still
    possible and worth benchmarking for latency-sensitive runs.

!!! note "Install the allocator before the first device field"
    `use_cupy_allocator()` and the managed pool only govern allocations made
    after they are set, so set them before any GPU field is created. Switching
    or clearing the allocator later is safe: each buffer is freed through the
    allocator that produced it, which is kept alive until its last buffer is
    gone.

## Zero-copy data exchange

µGrid uses the [DLPack](https://github.com/dmlc/dlpack) protocol for zero-copy
data exchange between the C++ library and Python. This means:

- No data is copied when accessing field arrays
- Modifications to the array directly modify the underlying field data
- GPU data stays on the GPU

This is particularly important for performance when working with large arrays on GPUs.

## CartesianDecomposition with GPU

When using domain decomposition with `CartesianDecomposition`, you can also
specify the device:

```python
import muGrid

# Create communicator (serial or MPI)
comm = muGrid.Communicator()

# Create decomposition on GPU
decomp = muGrid.CartesianDecomposition(
    comm,
    nb_domain_grid_pts=[128, 128],
    nb_subdivisions=[1, 1],
    nb_ghosts_left=[1, 1],
    nb_ghosts_right=[1, 1],
    device="cuda"
)

# Create GPU field
field = decomp.real_field("gpu_field")

# Access coordinates (always returned as host-side NumPy arrays,
# regardless of the field's device)
x, y = decomp.coords
```

## Convolution operators on GPU

Convolution operators can also operate on GPU fields. The convolution is performed
on the GPU when both input and output fields are on the GPU:

```python
import numpy as np
import muGrid

if not muGrid.has_gpu:
    raise RuntimeError("GPU support not available")

import cupy as cp

# Create GPU field collection
fc = muGrid.GlobalFieldCollection([64, 64], device="cuda")

# Create Laplacian stencil (defined on CPU as numpy array)
stencil = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
laplace = muGrid.GenericLinearOperator([-1, -1], stencil)

# Create fields
input_field = fc.real_field("input")
output_field = fc.real_field("output")

# Initialize input (using CuPy)
input_field.p[...] = cp.random.randn(*input_field.p.shape)

# Apply convolution (executed on GPU) - fields are passed directly
laplace.apply(input_field, output_field)
```

## FFT on GPU

µGrid provides GPU-accelerated FFT operations, but there are important differences
between the NVIDIA (CUDA/cuFFT) and AMD (ROCm/rocFFT) backends.

### NVIDIA GPUs (cuFFT)

On NVIDIA GPUs µGrid uses cuFFT. A serial 2D or 3D real-to-complex /
complex-to-real transform is issued as one planned multidimensional transform
(`cufftPlanMany`) rather than axis by axis, which avoids intermediate work
buffers and per-plane kernel launches.

cuFFT has one documented restriction:

> *"Strides on the real part of real-to-complex and complex-to-real transforms
> are not supported."*

µGrid therefore lays out device buffers with the real (innermost) axis at unit
stride: device fields use structure-of-arrays storage, in which each component
is contiguous (see [Architecture](architecture.md#storage-order)). With that
arrangement:

- **Serial** 2D and 3D transforms run as a single planned N-D cuFFT transform.
- **MPI-parallel** transforms also run on cuFFT. In a *slab* decomposition the
  two locally complete axes are transformed together as one planned 2D
  transform; a *pencil* decomposition transforms one axis at a time between
  transposes (see [FFT](fft.md#choosing-the-decomposition)).

### AMD GPUs (rocFFT)

On AMD GPUs µGrid calls rocFFT directly (not through hipFFT). rocFFT accepts
arbitrary input and output strides for all transform types, including R2C and
C2R (`rocfft_plan_description_set_data_layout()`). µGrid nevertheless uses the
same structure-of-arrays layout as on NVIDIA hardware, so both backends share
one code path and no storage-order conversion is needed. rocFFT cannot plan
some ghost-padded strided N-D layouts (e.g. a $32 \times 32$ grid with one
ghost layer); for those the engine detects the planning failure before any data
is written, remembers it, and uses the axis-by-axis path instead.

### Example: GPU FFT

A GPU FFT engine is created like a CPU one; 2D and 3D, serial and MPI-parallel,
all run on the device:

```python
import muGrid

nb_grid_pts = [64, 64, 64]
device = "gpu" if muGrid.has_gpu else "cpu"
engine = muGrid.FFTEngine(nb_grid_pts, device=device)
```

## GPU data and MPI

### Technical background

In MPI-parallel runs, µGrid has to communicate data that lives in GPU memory:
ghost (halo) layers of the domain decomposition and the global transposes of the
distributed FFT. Two properties of MPI implementations shape how this is done:

1. **Strided datatypes are slow on device memory.** MPI derived datatypes
   (`MPI_Type_vector`, subarrays) describe non-contiguous data declaratively, and
   on host memory implementations handle them well. On device memory, however,
   common MPI stacks pack such datatypes block by block with one small device
   copy each — for a halo slice this can mean tens of thousands of 8-byte copies
   and a slowdown of several orders of magnitude. µGrid therefore never hands
   strided device data to MPI: halos and transpose blocks are first gathered into
   contiguous device staging buffers, sent as flat messages, and scattered on the
   receiving side.

2. **Not every MPI can read device pointers at all.** Passing a GPU pointer to MPI
   requires a *GPU-aware* build of the MPI library (e.g. Open MPI/UCX with CUDA
   support, HPC-X, MVAPICH2-GDR, Cray MPICH with GTL). A plain MPI build —
   including the stock packages of most Linux distributions — dereferences the
   pointer on the host and crashes.

### Runtime detection and host fallback

µGrid detects GPU-aware MPI at runtime. With Open MPI (and derivatives such as
HPC-X), the official `MPIX_Query_cuda_support()` / `MPIX_Query_rocm_support()`
extensions are queried. For MPI stacks that cannot be queried, the conservative
answer is *not GPU-aware*.

When MPI is not GPU-aware, the contiguous staging buffers are bounced through host
memory: pack on the device, copy once to a host buffer, communicate, copy back,
unpack on the device. This is correct with any MPI library and, because only
flat buffers are copied, costs one device-to-host and one host-to-device copy
per message.

### The `MUGRID_GPU_AWARE_MPI` environment variable

The environment variable `MUGRID_GPU_AWARE_MPI` overrides the detection in both
directions:

- `MUGRID_GPU_AWARE_MPI=1` forces direct device pointers. Use this for MPI stacks
  that are GPU-aware but cannot be queried (e.g. some MPICH derivatives), where
  detection would needlessly fall back to host staging.

- `MUGRID_GPU_AWARE_MPI=0` forces the host bounce — a kill switch for *broken*
  GPU-aware stacks. GPU support in MPI is a complex, layered feature (UCX
  transports, IPC, GPUDirect), and real installations have been observed to
  *silently corrupt* device messages in specific size windows (e.g. UCX's
  `cuda_ipc` transport between consumer GPUs). Forcing the host path trades some
  bandwidth for transfers that only use plain `cudaMemcpy`/`hipMemcpy`. The
  distributed-FFT transpose posts several device-to-device transfers concurrently
  (non-blocking all-to-all), which is more likely to expose such a transport bug
  than a serialized exchange would; if an FFT-based solve produces wrong results
  only with GPU-aware MPI, this kill switch (or `UCX_TLS=^cuda_ipc`) is the first
  thing to try.

```bash
# Force the host-staging fallback (kill switch)
MUGRID_GPU_AWARE_MPI=0 mpiexec -n 4 python3 my_solver.py

# Assert that the MPI stack is GPU-aware
MUGRID_GPU_AWARE_MPI=1 mpiexec -n 4 python3 my_solver.py
```

### Validating an installation

If you suspect the MPI/GPU stack, run the MPI test suite on the GPU device:

  ```bash
  mpiexec -n 2 python3 -m pytest tests/python_mpi_fft_tests.py -k gpu
  ```

  with and without `MUGRID_GPU_AWARE_MPI=0`. If results differ, the GPU-aware
  path of your MPI installation is broken; keep the kill switch on and report the
  issue to your MPI/UCX vendor.

## Performance considerations

GPU acceleration is most beneficial when:

- Working with large grids (the GPU parallelism outweighs data transfer overhead)
- Performing many operations on the same data (data stays on GPU)
- Using operations that parallelize well (element-wise operations, FFTs, convolutions)

For small grids or infrequent operations, the overhead of CPU-GPU data transfer may
outweigh the benefits of GPU computation.
