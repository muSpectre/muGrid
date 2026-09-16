# µGrid

µGrid is a C++ library for discrete representation of fields on structured
grids, with Python bindings. It provides efficient data structures and
algorithms for solving partial differential equations on regular grids,
with support for MPI parallelization and GPU acceleration.

## Features

- **Field collections**: Manage scalar, vector, and tensor fields on structured
  grids with flexible memory layouts
- **Domain decomposition**: Cartesian decomposition with ghost cell communication
  for stencil operations
- **Linear operators**: Stencil convolutions, a hard-coded Laplacian, and FEM
  gradient/divergence operators for linear simplex (P1) and multilinear (Q1)
  elements
- **Fused matrix-free operators**: Problem-specific kernels that apply an
  operator, assemble its diagonal, or contract a sensitivity without ever
  forming a matrix or storing an intermediate field — for example isotropic
  linear elasticity on a regular grid
- **Iterative solvers**: Preconditioned conjugate gradients, in a standard and
  a pipelined variant that overlaps its reductions
- **Preconditioners**: Jacobi, Fourier (reference-stiffness/Green), its
  per-mode block form, and the combined Green-Jacobi preconditioner
- **Vector algebra**: BLAS-like kernels on host and device (dot products,
  norms, axpy, scaling), including fused variants that return a reduction and
  update a vector in one pass
- **FFT engine**: Built-in Fast Fourier Transform with MPI-parallel support
  (auto-selected slab or pencil decomposition) and native cuFFT/rocFFT N-D
  transforms on the GPU
- **GPU support**: Optional CUDA and HIP backends. Fields, operators, solvers
  and preconditioners all run on device, so a solve need not return to the
  host; on unified-memory accelerators the allocator can be routed through
  managed memory
- **NetCDF I/O**: Serial and parallel file I/O for checkpointing and analysis

µGrid is written in C++20 and has language bindings for
[Python](https://www.python.org/) via pybind11.

This README contains only a small quick start guide. Please refer to the
[full documentation](https://muspectre.github.io/muGrid/) for more help.

## Quick start

To install µGrid, run

    pip install muGrid

Note that on most platforms this will install a binary wheel, that was
compiled with a minimal configuration. To compile for your specific platform
use

    pip install -v --no-binary muGrid muGrid

which will compile the code. µGrid will autodetect
[MPI](https://www.mpi-forum.org/).
GPU support ([CUDA](https://developer.nvidia.com/cuda)/[ROCm](https://www.amd.com/en/developer/resources/rocm-hub.html))
is off by default and must be enabled explicitly at build time (see the
documentation); it is not autodetected.
For I/O, it will try to use
[Unidata NetCDF](https://www.unidata.ucar.edu/software/netcdf/)
for serial builds and
[PnetCDF](https://parallel-netcdf.github.io/) for MPI-parallel builds.
Monitor output to see which of these options were automatically detected.

## Funding

This development has received funding from the
[Swiss National Science Foundation](https://www.snf.ch/en),
the
[European Research Council](https://erc.europa.eu),
and the
[Deutsche Forschungsgemeinschaft](https://www.dfg.de/).
