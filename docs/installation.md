# Installation

µGrid builds with [CMake](https://cmake.org/) (≥ 3.18) and makes heavy use of
C++20, so it needs a relatively modern compiler. It produces a C++ core and an
optional Python extension.

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

µGrid autodetects [MPI](https://www.mpi-forum.org/). For I/O it uses
[Unidata NetCDF](https://www.unidata.ucar.edu/software/netcdf/) for serial
builds and [PnetCDF](https://parallel-netcdf.github.io/) for MPI-parallel
builds. The build prints a configuration summary showing what was detected:

```text
--------------------
muGrid configuration
  Version        : v0.96.0
  Eigen          : 5.0.1
  CUDA           : OFF
  HIP            : OFF
  MPI            : ON - MPI 3.1
  Parallel NetCDF: ON - PnetCDF 1.13.0
  Python bindings: ON - Python 3.11
  pybind11       : 2.13.6
  Tests          : ON
  Examples       : ON
--------------------
```

## Requirements

Required:

- [CMake](https://cmake.org/) ≥ 3.18 and [git](https://git-scm.com/).
- A C++20 compiler.
- [Python](https://www.python.org/) ≥ 3.10 with development headers, and
  [NumPy](https://numpy.org/) (for the extension).

Fetched automatically if not found:

- [pybind11](https://pybind11.readthedocs.io/) (≥ 2.11),
  [Eigen](https://eigen.tuxfamily.org/) (≥ 5.0.1), and
  [DLPack](https://github.com/dmlc/dlpack) (for GPU array exchange).

Optional:

- [Boost.Test](https://www.boost.org/doc/libs/release/libs/test/) (C++ unit
  tests), [Unidata NetCDF](https://www.unidata.ucar.edu/software/netcdf/)
  (serial I/O), [PnetCDF](https://parallel-netcdf.github.io/) (parallel I/O),
  [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (NVIDIA GPUs), and
  [ROCm/HIP](https://rocm.docs.amd.com/) (AMD GPUs).

## Building from source

Clone the repository:

```bash
git clone https://github.com/muSpectre/muGrid.git
```

Configure a *development* build (debug symbols, assertions):

```bash
cmake -DCMAKE_BUILD_TYPE=Debug -B build-debug
cmake --build build-debug
```

or a *production* build (optimised), optionally with the
[Ninja](https://ninja-build.org/) backend for faster compilation:

```bash
cmake -DCMAKE_BUILD_TYPE=Release -G Ninja -B build-release
cmake --build build-release
```

To use the package from the build tree, put the built extension and the
pure-Python wrapper on `PYTHONPATH`:

```bash
export PYTHONPATH=$PWD/build-release:$PWD/language_bindings/python
python -c "import muGrid; print('ok')"
```

## Enabling and disabling features

CMake autodetects features by default; you can override them explicitly.

```bash
# Core features (all default ON when their dependencies are present)
cmake -DMUGRID_ENABLE_MPI=OFF ..       # serial build
cmake -DMUGRID_ENABLE_NETCDF=OFF ..    # no NetCDF I/O
cmake -DMUGRID_ENABLE_PYTHON=OFF ..    # no Python bindings
cmake -DMUGRID_ENABLE_TESTS=OFF ..     # no tests
cmake -DMUGRID_ENABLE_EXAMPLES=OFF ..  # no examples
```

You can combine options, e.g. a minimal serial build:

```bash
cmake -DMUGRID_ENABLE_MPI=OFF -DMUGRID_ENABLE_NETCDF=OFF \
      -DMUGRID_ENABLE_TESTS=OFF -DCMAKE_BUILD_TYPE=Release ..
```

## GPU support

GPU support is off by default and must be enabled explicitly. Enable exactly
one backend — CUDA (NVIDIA) or HIP (AMD). See the [GPU](gpu.md) guide for
usage.

```bash
# NVIDIA CUDA — set the architectures to match your device(s)
cmake -DMUGRID_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80;90" ..

# AMD ROCm/HIP
cmake -DMUGRID_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx942" ..
```

!!! warning "Match `CMAKE_CUDA_ARCHITECTURES` to your GPU"
    The architecture list must include your device's compute capability (e.g.
    `70`=V100, `80`=A100, `90`=H100, `120`=Blackwell). On CMake ≥ 3.24 you can
    use `-DCMAKE_CUDA_ARCHITECTURES=native` to auto-detect the build host's
    GPU. A binary built only for an architecture the device does not support
    will fail to launch its kernels — see the [GPU](gpu.md) page.

To drive the GPU path from Python, install a [CuPy](https://cupy.dev/) build
matching your CUDA/ROCm version.

## Running the tests

```bash
ctest --test-dir build-release --output-on-failure
```

GPU tests skip automatically when no device is present.

## Getting help and contributing

µGrid is under active development. If you run into trouble, open an
[issue](https://github.com/muSpectre/muGrid/issues). Contributions are welcome
— new features must be documented and have unit tests.
