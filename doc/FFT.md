# FFT Engine Design

This document describes the design and implementation decisions for the muGrid FFT engine.

## Overview

The FFT engine provides distributed Fast Fourier Transform operations on structured grids with MPI parallelization. It uses pencil decomposition for efficient scaling to large numbers of MPI ranks.

## Storage Order

### Array of Structures (AoS) vs Structure of Arrays (SoA)

muGrid supports two storage orders for multi-component fields:

- **AoS (Array of Structures)**: Components are interleaved per pixel
  - Memory layout: `[c0p0, c1p0, c2p0, c0p1, c1p1, c2p1, ...]`
  - Used on **CPU** (default for `HostSpace`)
  - Stride between consecutive X elements = `nb_components`
  - Component offset = `comp` (component index)

- **SoA (Structure of Arrays)**: Components are stored in separate contiguous blocks
  - Memory layout: `[c0p0, c0p1, c0p2, ..., c1p0, c1p1, c1p2, ...]`
  - Used on **GPU** (default for `CUDASpace` and `ROCmSpace`)
  - Stride between consecutive X elements = 1
  - Component offset = `comp * nb_pixels`

### Why Different Storage Orders?

- **GPU (SoA)**: Coalesced memory access is critical for GPU performance. When threads access consecutive memory locations, the hardware can combine multiple accesses into a single transaction. SoA ensures that threads processing different pixels of the same component access contiguous memory.

- **CPU (AoS)**: Cache locality for multi-component operations. When processing a single pixel, all its components are in the same cache line.

### Design Decision: Work Buffers Match Memory Space

**Key Design Decision**: FFT work buffers use the same storage order as the memory space they reside in (SoA on GPU, AoS on CPU). This avoids expensive storage order conversions during FFT operations.

The FFT engine detects the storage order of input/output fields using `field.get_storage_order()` and computes the appropriate strides for each buffer.

## Stride Calculations

### 2D Case

For a 2D field with dimensions `[Nx, Ny]` and `nb_components` components:

**SoA strides:**
```
comp_offset_factor = nb_pixels        // Component i starts at i * nb_pixels
x_stride = 1                          // Consecutive X elements are contiguous
row_dist = row_width                  // Distance between rows (Y direction)
```

**AoS strides:**
```
comp_offset_factor = 1                // Component i starts at offset i
x_stride = nb_components              // Skip over all components between X elements
row_dist = row_width * nb_components  // Distance between rows includes all components
```

### 3D Case

For a 3D field with dimensions `[Nx, Ny, Nz]`:

**SoA strides:**
```
comp_offset_factor = nb_pixels
x_stride = 1
y_dist = row_x
z_dist = row_x * rows_y
```

**AoS strides:**
```
comp_offset_factor = 1
x_stride = nb_components
y_dist = row_x * nb_components
z_dist = row_x * rows_y * nb_components
```

## GPU Backend Limitations

### cuFFT Strided R2C/C2R Limitation

cuFFT has a documented limitation: **strided real-to-complex (R2C) and complex-to-real (C2R) transforms are not supported**. The stride on the real data side must be 1.

This affects multi-component FFTs because with AoS storage, the stride between consecutive real values would be `nb_components` (not 1).

**Solution**: With SoA storage order on GPU, each component's data is contiguous (stride = 1), which satisfies cuFFT's requirement. The FFT engine loops over components, executing one batched FFT per component.

### rocFFT Native API

Unlike cuFFT, rocFFT's native API (`rocfft_plan_description_set_data_layout()`) supports arbitrary strides for both input and output. However, for consistency and to avoid storage order conversions, we use the same SoA approach on AMD GPUs.

## MPI Parallel FFT Design

### Pencil Decomposition

For 3D distributed FFT with a `P1 x P2` process grid (Z distributed across
P1, Y across P2 in real space), the engine uses a true pencil decomposition:

1. **Z-pencil** `[Fx, Ny/P2, Nz/P1]`: FFT along X (r2c)
2. **Y-pencil** `[Fx/P2, Ny, Nz/P1]`: Transpose X<->Y within the row
   communicator (scatter X across P2, gather Y), then FFT along Y (c2c)
3. **X-pencil** `[Fx/P2, Ny/P1, Nz]`: Transpose Y<->Z within the column
   communicator (scatter Y across P1, gather Z), then FFT along Z (c2c)

Each rank holds O(N³/P) data at every stage; no dimension is ever
replicated across ranks. The X-pencil is the final Fourier-space layout: X
distributed across P2, Y across P1, Z full.

### Transpose Operations

The `Transpose` class handles MPI all-to-all communication for
redistributing data between pencil orientations. Every transpose is a
genuine scatter-gather: each rank sends a disjoint block to every peer and
receives into a disjoint block, as required by the MPI standard for
`MPI_Alltoallw`. Both AoS (host) and SoA (device) storage orders are
supported; the engine threads the storage order of the fields through to
the datatype construction.

### Communication Efficiency

**Design Decision**: MPI communication does not require explicit packing
and unpacking. Non-contiguous blocks are described with MPI derived
datatypes (`MPI_Type_create_subarray` for the spatial block,
`MPI_Type_contiguous`/`MPI_Type_create_hvector` for AoS/SoA components) and
communicated with a single `MPI_Alltoallw` per transpose. Because only MPI
intrinsics touch the data buffers, the same code path operates directly on
device memory with a GPU-aware MPI implementation — responsibility for
packing lands on the MPI library, which can use specialized device kernels
or staging as appropriate.

## Algorithm: Forward 2D FFT

```
1. r2c FFT along X for each component
   - Input: real field (with ghosts)
   - Output: work buffer (half-complex, no ghosts)
   - Loop over components, batched over Y rows

2. [MPI only] Transpose X<->Y
   - Redistributes data across ranks
   - Changes from Y-local to X-local

3. c2c FFT along Y for each component
   - In-place on work buffer (serial) or output (MPI)
   - Batched over X values

4. [Serial only] Copy work to output
   - Same storage order, direct copy
```

## Algorithm: Forward 3D FFT

```
1. r2c FFT along X for each component
   - Input: real field
   - Output: Z-pencil work buffer (half-complex in X)
   - Batched over Y rows for each Z plane

2a. [MPI, P2 > 1] Transpose X<->Y (row communicator)
    - Z-pencil [Fx, Ny/P2, Nz/P1] to Y-pencil [Fx/P2, Ny, Nz/P1]
    - Scatter X across P2, gather Y
    - (P2 == 1: plain copy, shapes are identical)

2b. c2c FFT along Y for each component
    - On Y-pencil buffer
    - Batched over local X values per Z plane

3. [MPI, P1 > 1] Transpose Y<->Z (column communicator)
   - Y-pencil [Fx/P2, Ny, Nz/P1] to X-pencil [Fx/P2, Ny/P1, Nz] (output)
   - Scatter Y across P1, gather Z
   - (P1 == 1: plain copy, shapes are identical)

4. c2c FFT along Z for each component
   - On output buffer
   - Batched over local (X, Y) positions

5. [Serial only] Copy work to output
```

## Per-Component Looping

Instead of attempting to batch across components (which would require non-unit strides on GPU), the FFT engine loops over components:

```cpp
for (Index_t comp = 0; comp < nb_components; ++comp) {
    Index_t in_comp_offset = comp * in_comp_factor;
    Index_t out_comp_offset = comp * work_comp_factor;
    backend->r2c(Nx, batch_size,
                 input_ptr + in_base + in_comp_offset,
                 in_x_stride, in_row_dist,
                 work_ptr + out_comp_offset,
                 work_x_stride, work_row_dist);
}
```

This approach:
- Satisfies cuFFT's unit-stride requirement for R2C/C2R
- Allows efficient batching within each component
- Works correctly for both SoA and AoS storage orders
- Has minimal overhead (one kernel launch per component)

## Normalization

Like FFTW and cuFFT, the transforms are **unnormalized**. A forward FFT followed by an inverse FFT multiplies the result by N (the transform size). Users must explicitly normalize if needed.

## File Structure

- `fft_engine.hh`: Template class `FFTEngine<MemorySpace>` with all transform implementations
- `fft_engine_base.hh/cc`: Base class with field collections and transpose management
- `fft_backend.hh`: Abstract interface for FFT backends (1D primitives plus optional N-dimensional transforms)
- `pocketfft_backend.cc`: CPU backend using pocketfft
- `cufft_backend.cc`: NVIDIA GPU backend using cuFFT
- `rocfft_backend.cc`: AMD GPU backend using rocFFT
- `transpose.hh/cc`: MPI transpose operations

## Future Improvements

1. **Batched multi-component GPU FFT**: For c2c transforms (which don't have the stride limitation), explore batching across all components in a single kernel launch.

2. **deep_copy with storage order conversion**: Extend the `deep_copy` utility to optionally convert between AoS and SoA during the copy, enabling more flexible data movement.

3. **GPU derived-datatype performance**: GPU-aware MPI implementations vary in how efficiently they handle derived datatypes on device memory (some use specialized pack kernels, others stage through host). If profiling shows the `MPI_Alltoallw` transposes dominate on a given system, consider a vendor multi-GPU FFT backend (cuFFTMp/heFFTe) as an alternative.
