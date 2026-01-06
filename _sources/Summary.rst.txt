.. _Summary:

Summary
-------

*µ*\Grid is a C++20 library for discrete representation of fields on structured
grids, with Python bindings. It provides efficient data structures and algorithms
for solving partial differential equations on regular grids, with support for
MPI parallelization and GPU acceleration.

Features
********

- **Field collections**: Manage scalar, vector, and tensor fields on structured
  grids
- **Domain decomposition**: Cartesian decomposition with ghost cell communication
  for stencil operations
- **Linear operators**: Discrete differential operators including Laplacian
  and FEM gradient operators for spectral methods
- **FFT engine**: Built-in Fast Fourier Transform using `PocketFFT <https://github.com/mreineck/pocketfft>`_
  on CPUs with MPI-parallel support using pencil decomposition
- **GPU support**: Optional CUDA and ROCm backends for GPU-accelerated computation
- **NetCDF I/O**: Serial (Unidata NetCDF) and parallel (PnetCDF) file I/O for
  checkpointing and analysis

*µ*\Grid has language bindings for Python via `pybind11 <https://pybind11.readthedocs.io/>`_,
exposing fields through `NumPy <https://numpy.org/>`_ arrays and supporting
interoperability with `CuPy <https://cupy.dev/>`_ for GPU arrays.

History
*******

*µ*\Grid is part of the *µ*\Spectre project, which provides an open-source
platform for efficient FFT-based continuum mesoscale modelling. It was originally
developed as the grid infrastructure for *µ*\Spectre but is now a standalone
library that can be used independently.

Funding
*******

Development has been funded by the
`Swiss National Science Foundation <https://snf.ch>`_, the
`European Research Council <https://erc.europa.eu>`_ and the
`Deutsche Foschungsgemeinschaft <https://www.dfg.de/>`.
