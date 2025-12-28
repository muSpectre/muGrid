# µGrid

µGrid is a C++ library for discrete representation of fields on structured
grids, with Python bindings. It provides efficient data structures and
algorithms for solving partial differential equations on regular grids,
with support for MPI parallelization and GPU acceleration.

## Features

- **Field collections**: Manage scalar, vector, and tensor fields on structured
  grids with flexible memory layouts
- **FFT engine**: Built-in Fast Fourier Transform with MPI-parallel support
  using pencil decomposition
- **Convolution operators**: Discrete differential operators including Laplacian
  and FEM gradient operators for spectral methods
- **Domain decomposition**: Cartesian decomposition with ghost cell communication
  for stencil operations
- **GPU support**: Optional CUDA and HIP backends for GPU-accelerated computation
- **NetCDF I/O**: Serial and parallel file I/O for checkpointing and analysis

µGrid is written in C++17 and has language bindings for
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
For I/O, it will try to use
[Unidata NetCDF](https://www.unidata.ucar.edu/software/netcdf/)
for serial builds and
[PnetCDF](https://parallel-netcdf.github.io/) for MPI-parallel builds.
Monitor output to see which of these options were automatically detected.

## Funding

This development has received funding from the
[Swiss National Science Foundation](https://www.snf.ch/en)
within an Ambizione Project and by the
[European Research Council](https://erc.europa.eu) within
[Starting Grant 757343](https://cordis.europa.eu/project/id/757343).
