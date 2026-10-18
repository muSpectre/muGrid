# Installation

µGrid builds with [CMake](https://cmake.org/) (≥ 3.18) and uses C++20 language
and standard-library features (`<ranges>`, `<span>`), so it needs a recent
compiler. It produces a C++ core library and an optional Python extension.

## Python quick start

```bash
pip install muGrid
```

On most platforms this installs a binary wheel built with a minimal
configuration. To compile for your specific platform (autodetecting MPI and
NetCDF) instead:

```bash
pip install -v --force-reinstall --no-cache --no-binary muGrid muGrid
```

Wheel and source builds through `pip` use
[scikit-build-core](https://scikit-build-core.readthedocs.io/) and always switch
off the C++ tests and examples (`MUGRID_ENABLE_TESTS=OFF`,
`MUGRID_ENABLE_EXAMPLES=OFF`).

µGrid autodetects [MPI](https://www.mpi-forum.org/). For I/O it uses
[Unidata NetCDF](https://www.unidata.ucar.edu/software/netcdf/) for serial
builds and [PnetCDF](https://parallel-netcdf.github.io/) for MPI-parallel
builds. The build prints a configuration summary showing what was detected, for
example:

```text
--------------------
muGrid configuration
  Version        : v1.1.0
  Eigen          : 5.0.1 (fetched)
  CUDA           : OFF
  ROCm/HIP       : OFF
  MPI            : ON - MPI 3.1
  Parallel NetCDF: ON - PnetCDF 1.13.0
  Python bindings: ON - Python 3.12
  pybind11       : 2.13.6 (fetched)
  Tests          : ON
  Examples       : ON
  Coverage       : OFF
--------------------
```

In a serial build the NetCDF line reads `Unidata NetCDF : ON - NetCDF <version>`
instead of `Parallel NetCDF`.

## Requirements

Required:

- [CMake](https://cmake.org/) ≥ 3.18 and [git](https://git-scm.com/).
- A C++20 compiler with C++20 standard-library support. Known-good minimum
  versions are GCC 12, Clang/libc++ 15, Apple Clang 15 (Xcode 15) and MSVC 19.34
  (Visual Studio 2022 17.4); configuration stops with an error if a test
  program using `<ranges>`/`<span>` does not compile.
- [Python](https://www.python.org/) ≥ 3.10 with development headers, and
  [NumPy](https://numpy.org/) ≥ 2.1 (for the extension; the array views use
  `numpy.from_dlpack`).

Fetched automatically if not found:

- [pybind11](https://pybind11.readthedocs.io/) (≥ 2.11),
  [Eigen](https://eigen.tuxfamily.org/) (≥ 5.0.1), and
  [DLPack](https://github.com/dmlc/dlpack) (for zero-copy array exchange).

Optional:

- [Boost.Test](https://www.boost.org/doc/libs/release/libs/test/) (C++ unit
  tests), [Unidata NetCDF](https://www.unidata.ucar.edu/software/netcdf/)
  (serial I/O), [PnetCDF](https://parallel-netcdf.github.io/) (parallel I/O),
  [mpi4py](https://mpi4py.readthedocs.io/) (required at import time by an
  MPI-enabled Python extension),
  [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (NVIDIA GPUs), and
  [ROCm/HIP](https://rocm.docs.amd.com/) with rocFFT (AMD GPUs).

## Building from source

Clone the repository:

```bash
git clone https://github.com/muSpectre/muGrid.git
cd muGrid
```

Configure a *development* build (debug symbols, assertions):

```bash
cmake -DCMAKE_BUILD_TYPE=Debug -B build-debug
cmake --build build-debug
```

or a *production* build (optimised; `Release` is also the default when no
build type is given), optionally with the [Ninja](https://ninja-build.org/)
backend:

```bash
cmake -DCMAKE_BUILD_TYPE=Release -G Ninja -B build-release \
      -DPython_EXECUTABLE=$(which python3)
cmake --build build-release
```

Pass `-DPython_EXECUTABLE` pointing at the Python environment you intend to
use; otherwise CMake may pick an interpreter without NumPy or pytest and the
Python tests are not registered.

To use the package from the build tree without installing it, put the built
extension (`_muGrid`) and the pure-Python package on `PYTHONPATH`:

```bash
export PYTHONPATH=$PWD/build-release/language_bindings/python:$PWD/language_bindings/python
python -c "import muGrid; print(muGrid.version_string())"
```

## Enabling and disabling features

CMake autodetects MPI and NetCDF; you can override every feature explicitly.

```bash
# Core features (default ON; MPI and NetCDF switch themselves off when the
# library is not found)
cmake -DMUGRID_ENABLE_MPI=OFF ..       # serial build
cmake -DMUGRID_ENABLE_NETCDF=OFF ..    # no NetCDF I/O
cmake -DMUGRID_ENABLE_PYTHON=OFF ..    # no Python bindings
cmake -DMUGRID_ENABLE_TESTS=OFF ..     # no tests
cmake -DMUGRID_ENABLE_EXAMPLES=OFF ..  # no examples

# Optional (default OFF)
cmake -DMUGRID_NATIVE_ARCH=ON ..       # -march=native / -mcpu=native
cmake -DMUGRID_ENABLE_COVERAGE=ON ..   # coverage instrumentation
```

`MUGRID_NATIVE_ARCH` lets the auto-vectorised kernels use the full SIMD width of
the build machine (e.g. AVX2/AVX-512 on x86-64, whose baseline ISA is SSE2). The
resulting binary does not run on CPUs lacking those instructions, which is why
the option is off by default and never used for wheels.

You can combine options, e.g. a minimal serial build:

```bash
cmake -DMUGRID_ENABLE_MPI=OFF -DMUGRID_ENABLE_NETCDF=OFF \
      -DMUGRID_ENABLE_TESTS=OFF -DCMAKE_BUILD_TYPE=Release ..
```

## GPU support

GPU support is off by default and is not autodetected. Enable exactly one
backend — CUDA (NVIDIA) or HIP (AMD); the two options are mutually exclusive.
See the [GPU](gpu.md) guide for using fields and solvers on the device.

```bash
# NVIDIA CUDA
cmake -DMUGRID_ENABLE_CUDA=ON ..
cmake -DMUGRID_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80;90" ..

# AMD ROCm/HIP (requires rocFFT)
cmake -DMUGRID_ENABLE_HIP=ON ..
cmake -DMUGRID_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx942" ..
```

If `CMAKE_CUDA_ARCHITECTURES` (resp. `CMAKE_HIP_ARCHITECTURES`) is not set, it
defaults to `native` — the GPUs present on the build machine — on CMake ≥ 3.24,
and to `70;80;90` (resp. `gfx906;gfx908;gfx90a`) on older CMake. Set an explicit
list for a portable or multi-architecture binary.

!!! warning "Match the architecture list to your GPU"
    The list must include your device's architecture (CUDA compute capability
    `70` = V100, `80` = A100, `90` = H100, `120` = Blackwell; HIP `gfx906` =
    MI50/MI60, `gfx908` = MI100, `gfx90a` = MI200, `gfx942` = MI300). A kernel
    compiled only for architectures the device cannot run fails to launch. µGrid
    checks the runtime error after every kernel launch and raises a
    `RuntimeError` on the first one; without this check, runtime-API calls such
    as memory copies would keep succeeding while the kernels did nothing.

To drive the GPU path from Python, install a [CuPy](https://cupy.dev/) build
matching your CUDA/ROCm version (see the
[CuPy installation guide](https://docs.cupy.dev/en/stable/install.html)):

```bash
pip install cupy-cuda12x    # CUDA 12.x
pip install cupy-cuda11x    # CUDA 11.x
pip install cupy-rocm-5-0   # ROCm (pick the package matching your ROCm)
```

## Running the tests

```bash
# Everything registered with CTest: C++ Boost.Test suites (if Boost.Test was
# found), Python tests, MPI runs on several rank counts
ctest --test-dir build-release --output-on-failure

# Python tests only, against whatever `import muGrid` resolves to
python3 -m pytest tests/ -v
```

GPU tests run only when µGrid was built with GPU support, and are skipped when
no GPU is available at run time or CuPy is not installed.

## Getting help and contributing

If you run into trouble, open an
[issue](https://github.com/muSpectre/muGrid/issues). Contributions are welcome
— new features must be documented and have unit tests.
