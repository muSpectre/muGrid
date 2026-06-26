# µGrid

µGrid is a C++20 library for the discrete representation of fields on
structured grids, with Python bindings. It provides efficient data structures
and algorithms for solving partial differential equations on regular grids,
with MPI parallelization and GPU acceleration.

## Highlights

- **Field collections.** Manage scalar, vector, and tensor fields on a shared
  structured grid, in host (CPU) or device (GPU) memory, exposed to Python as
  zero-copy [NumPy](https://numpy.org/)/[CuPy](https://cupy.dev/) arrays via
  DLPack.
- **Domain decomposition.** Cartesian decomposition with ghost-cell
  communication for stencil operations, parallelised with MPI.
- **FFT engine.** A built-in FFT using [PocketFFT](https://github.com/mreineck/pocketfft)
  on the CPU and native cuFFT/rocFFT N-D transforms on the GPU, with
  MPI-parallel slab or pencil decomposition (auto-selected).
- **Operators.** Discrete differential operators — convolution stencils, a
  hard-coded Laplacian, and FEM gradient operators for spectral methods.
- **Linear algebra.** Ghost-aware reductions and vector updates (`vecdot`,
  `norm_sq`, `axpy`, …) that run on the same CPU/GPU fields.
- **NetCDF I/O.** Serial (Unidata NetCDF) and parallel (PnetCDF) file I/O for
  checkpointing and analysis.

## A first example

```python
import muGrid

# A 64 x 64 grid with a single scalar field
fc = muGrid.GlobalFieldCollection([64, 64])
field = fc.real_field("temperature")

# field.p is a NumPy view of the pixel data (no copy)
field.p[...] = 0.0
```

## Where to go next

- [Installation](installation.md) — install the wheel or build the C++ core and
  the optional MPI/GPU backends.
- [Python API](python.md) — field collections, array views, decomposition, the
  FFT engine, and device selection.
- [C++ API](cpp.md) — the Python-free core in namespace `muGrid`.
- [FFT](fft.md), [Operators](operators.md), [Linear algebra](linalg.md),
  [GPU](gpu.md) — the topic guides.
- [Examples](examples.md) and [Benchmark](benchmark.md) — a Poisson solver and
  its CPU/GPU scaling across grid sizes.
- [Architecture](architecture.md) — the design of the field/collection model.

## History

µGrid is part of the µSpectre project, an open-source platform for FFT-based
continuum mesoscale modelling. It was originally developed as the grid
infrastructure for µSpectre and is now a standalone library that can be used
independently.

## Funding

Development has been funded by the
[Swiss National Science Foundation](https://snf.ch), the
[European Research Council](https://erc.europa.eu) and the
[Deutsche Forschungsgemeinschaft](https://www.dfg.de/).

## License

µGrid is free software; you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, either version 3, or (at your option) any later version.
