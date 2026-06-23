# FFT

µGrid includes a built-in Fast Fourier Transform (FFT) engine that operates on
µGrid fields. The engine uses [pocketfft](https://github.com/mreineck/pocketfft)
for efficient FFT computation.

## The FFTEngine class

Instantiating an FFT engine is straightforward:

```python
import muGrid

# Create a 1D FFT engine
engine = muGrid.FFTEngine([nx])

# Create a 2D FFT engine
engine = muGrid.FFTEngine([nx, ny])

# Create a 3D FFT engine
engine = muGrid.FFTEngine([nx, ny, nz])
```

The engine supports 1D, 2D, and 3D grids. 1D FFT is serial-only (no MPI
parallelization), while 2D and 3D support MPI-parallel execution.

The engine provides properties for querying the grid dimensions:

```python
engine = muGrid.FFTEngine([16, 20])

print(f"Real-space grid: {engine.nb_domain_grid_pts}")      # (16, 20)
print(f"Fourier-space grid: {engine.nb_fourier_grid_pts}")  # (9, 20)
print(f"Backend: {engine.backend_name}")                    # "pocketfft"
```

The Fourier-space grid has reduced dimensions due to the real-to-complex (r2c)
transform exploiting Hermitian symmetry.

## Creating FFT fields

The engine manages two field collections: one for real-space fields and one for
Fourier-space fields. Use the engine's methods to create fields:

```python
import muGrid

engine = muGrid.FFTEngine([16, 20])

# Create real-space field (real-valued)
real_field = engine.real_space_field("real")
print(f"Real field shape: {real_field.s.shape}")  # (1, 16, 20)

# Create Fourier-space field (complex-valued)
fourier_field = engine.fourier_space_field("fourier")
print(f"Fourier field shape: {fourier_field.s.shape}")  # (1, 9, 20)
```

## Forward and inverse transforms

The `fft` method performs the forward transform (real to Fourier space) and
`ifft` performs the inverse transform (Fourier to real space). See
`examples/fft_roundtrip.py`.

## 1D FFT

For 1D problems (e.g., time series analysis, 1D signal processing), create a 1D
FFT engine by passing a single-element list:

```python
import numpy as np
import muGrid

# Create 1D FFT engine
N = 64
engine = muGrid.FFTEngine([N])

# Create fields
real_field = engine.real_space_field("signal")
fourier_field = engine.fourier_space_field("spectrum")

# Initialize with a test signal (e.g., sine wave)
x = np.linspace(0, 1, N, endpoint=False)
real_field.p[:] = np.sin(2 * np.pi * 3 * x)  # 3 Hz sine wave

# Forward FFT
engine.fft(real_field, fourier_field)

# The Fourier field has N/2+1 complex values due to Hermitian symmetry
print(f"Fourier shape: {fourier_field.p.shape}")  # (33,) for N=64

# Inverse FFT and normalize
engine.ifft(fourier_field, real_field)
real_field.p[:] *= engine.normalisation
```

The 1D FFT output matches NumPy's `fft.rfft`:

```python
# Compare with NumPy
data = np.random.randn(64)
real_field.p[:] = data
engine.fft(real_field, fourier_field)

numpy_result = np.fft.rfft(data)
np.testing.assert_allclose(fourier_field.p.flatten(), numpy_result)
```

!!! note

    1D FFT is **serial-only** and will raise an error if used with multiple MPI
    ranks.

## Normalization

µGrid FFT transforms are **not normalized** by default. A forward-inverse
roundtrip picks up a global factor equal to the total number of grid points.
This factor is available as the `normalisation` property:

```python
engine = muGrid.FFTEngine([8, 10])
print(f"Normalization factor: {engine.normalisation}")  # 1/80 = 0.0125
```

To recover the original values after a roundtrip, multiply by the normalization
factor after the inverse transform.

Normalization is left to the user because there are multiple valid conventions
(normalize forward, normalize inverse, or split between both), and different
applications may prefer different choices.

## Frequency and coordinate arrays

The engine provides properties for accessing frequency and coordinate arrays
that are correctly shaped for the local subdomain. This is particularly useful
for MPI-parallel computations where each rank only handles a portion of the
grid.

### Frequency arrays

The `fftfreq` and `ifftfreq` properties provide FFT frequency arrays:

```python
import muGrid
import numpy as np

engine = muGrid.FFTEngine([7, 4])

# Normalized frequencies (range [-0.5, 0.5))
qx, qy = engine.fftfreq
print(f"fftfreq shape: {engine.fftfreq.shape}")  # (2, 4, 4) for r2c transform

# Integer frequency indices
iqx, iqy = engine.ifftfreq
print(f"ifftfreq dtype: {iqx.dtype}")  # integer type

# These match numpy's frequency arrays (sliced for r2c transform)
freq_ref = np.array(np.meshgrid(
    *(np.fft.fftfreq(n) for n in [7, 4]), indexing="ij"
))
freq_ref = freq_ref[:, :4, :]  # Slice for half-complex
np.testing.assert_allclose(engine.fftfreq, freq_ref)
```

The frequency arrays have shape `[dim, local_fx, local_fy, ...]` where the first
axis indexes the spatial dimension and the remaining axes match the local
Fourier subdomain dimensions.

### Coordinate arrays

The `coords` and `icoords` properties provide real-space coordinate arrays:

```python
import muGrid

engine = muGrid.FFTEngine([7, 4])

# Normalized coordinates (range [0, 1))
x, y = engine.coords
print(f"coords shape: {engine.coords.shape}")  # (2, 7, 4)

# Integer coordinate indices
ix, iy = engine.icoords
print(f"icoords dtype: {ix.dtype}")  # integer type

# Verify: integer coords = fractional * n
np.testing.assert_allclose(ix, x * 7)
np.testing.assert_allclose(iy, y * 4)
```

For fields with ghost regions, use `coordsg` and `icoordsg` to get coordinates
including the ghost cells:

```python
engine = muGrid.FFTEngine(
    [64, 64],
    nb_ghosts_left=[1, 1],
    nb_ghosts_right=[1, 1]
)

# Coordinates without ghosts
x, y = engine.coords

# Coordinates with ghosts (larger array)
xg, yg = engine.coordsg
print(f"Without ghosts: {engine.coords.shape}")   # (2, local_nx, local_ny)
print(f"With ghosts: {engine.coordsg.shape}")     # (2, local_nx+2, local_ny+2)
```

### MPI-parallel frequency arrays

In MPI-parallel runs, frequency and coordinate arrays return only the data for
the local subdomain owned by each rank:

```python
from mpi4py import MPI
import muGrid

comm = muGrid.Communicator(MPI.COMM_WORLD)
engine = muGrid.FFTEngine([128, 128], comm)

# Each rank gets frequencies for its local Fourier subdomain
qx, qy = engine.fftfreq
print(f"Rank {MPI.COMM_WORLD.rank}: fftfreq shape = {engine.fftfreq.shape}")

# Coordinates for local real-space subdomain
x, y = engine.coords
print(f"Rank {MPI.COMM_WORLD.rank}: coords shape = {engine.coords.shape}")
```

### Example: Fourier-space operations

A common pattern is using frequency arrays to construct Fourier-space operators:

```python
import numpy as np
import muGrid

engine = muGrid.FFTEngine([64, 64])
qx, qy = engine.fftfreq

real_field = engine.real_space_field("u")
fourier_field = engine.fourier_space_field("u_hat")

# Initialize with a cosine wave
x, y = engine.coords
real_field.p[0] = np.cos(2 * np.pi * x)

# Forward FFT
engine.fft(real_field, fourier_field)

# Compute Laplacian in Fourier space: -4π²(qx² + qy²) * u_hat
laplacian_hat = -4 * np.pi**2 * (qx**2 + qy**2) * fourier_field.p[0]

# Or use integer frequencies for identifying specific modes
iqx, iqy = engine.ifftfreq
# Find the (1, 0) mode
mode_mask = (np.abs(iqx) == 1) & (iqy == 0)
```

## Hermitian grid dimensions

For real-to-complex transforms, the output has reduced dimensions due to
Hermitian symmetry. Use `get_hermitian_grid_pts` to compute the Fourier-space
grid size:

```python
import muGrid

# 1D example
fourier_1d = muGrid.get_hermitian_grid_pts([64])
print(f"1D Fourier grid: {fourier_1d}")  # [33]

# 2D example
fourier_2d = muGrid.get_hermitian_grid_pts([64, 64])
print(f"2D Fourier grid: {fourier_2d}")  # [33, 64]

# 3D example
fourier_3d = muGrid.get_hermitian_grid_pts([64, 64, 64])
print(f"3D Fourier grid: {fourier_3d}")  # [33, 64, 64]
```

The first dimension is reduced to `n//2 + 1`.

## FFT normalization factor

The `fft_normalization` function returns the normalization factor for a given
grid size:

```python
import muGrid

norm = muGrid.fft_normalization([16, 20])
print(f"Normalization: {norm}")  # 1/320
```

## Example: Fourier derivative

A common use of FFTs is computing derivatives in Fourier space. See
`examples/fourier_derivative.py` for an example computing the gradient of a 2D
field.

## Multi-component fields

FFT fields can have multiple components:

```python
import muGrid

engine = muGrid.FFTEngine([16, 20])

# Create field with 3 components (e.g., velocity vector)
velocity = engine.real_space_field("velocity", components=(3,))
velocity_hat = engine.fourier_space_field("velocity_hat", components=(3,))

print(f"Velocity shape: {velocity.s.shape}")          # (3, 1, 16, 20)
print(f"Velocity_hat shape: {velocity_hat.s.shape}")  # (3, 1, 9, 20)

# FFT transforms all components - fields are passed directly
engine.fft(velocity, velocity_hat)
engine.ifft(velocity_hat, velocity)
```

## MPI-parallel FFT

The `FFTEngine` class supports MPI parallelization in 2D and 3D. The grid is
split across the ranks either as a **slab** (one distributed axis) or a **pencil**
(a 2D process grid), selected with the `decomposition` argument (see
[Choosing the decomposition](#choosing-the-decomposition) below); the default
picks one automatically.

### Basic parallel usage

```python
import numpy as np
from mpi4py import MPI
import muGrid
from muGrid import Communicator

# Create parallel FFT engine
nb_grid_pts = [128, 128, 128]
comm = Communicator(MPI.COMM_WORLD)
engine = muGrid.FFTEngine(nb_grid_pts, comm)

# Each rank has a subdomain
print(f"Rank {MPI.COMM_WORLD.rank}:")
print(f"  Global grid: {engine.nb_domain_grid_pts}")
print(f"  Local subdomain: {engine.nb_subdomain_grid_pts}")
print(f"  Process grid: {engine.process_grid}")
print(f"  Process coords: {engine.process_coords}")
```

### Choosing the decomposition

The `decomposition` argument selects how the grid is distributed:

```python
engine = muGrid.FFTEngine(nb_grid_pts, comm, decomposition="auto")
```

- `"slab"` — split only the last axis (process grid `P1 × 1`). Each rank holds
  full `X` and `Y` planes, so the forward and inverse transforms do the two local
  axes as **one planned N-D transform** and need only a single `Y↔Z` transpose
  (instead of two). This is faster — markedly so on the GPU, where it avoids a
  transpose round trip through device memory — but cannot use more ranks than the
  last grid dimension (`num_ranks ≤ Nz`).
- `"pencil"` — split two axes over a balanced 2D process grid. It transforms one
  axis at a time between two transposes, so it scales to many more ranks than slab
  but moves more data.
- `"auto"` (default) — use slab when it fits (`num_ranks ≤ Nz`) and fall back to
  pencil otherwise. This gives the faster slab path on typical single-node and
  moderate-rank runs while still scaling to large rank counts.

The chosen process grid is reported by `engine.process_grid` (`(P1, P2)`; slab
has `P2 == 1`). The [homogenization preconditioner
benchmark](benchmark_homogenization_preconditioner.md) exercises the FFT under
MPI; on the GPU the slab N-D path is several times faster than pencil for the
same rank count.

### FFT with ghost regions

For computations requiring ghost cells (e.g., stencil operations), specify the
ghost buffer sizes:

```python
import muGrid
from muGrid import Communicator
from mpi4py import MPI

comm = Communicator(MPI.COMM_WORLD)

# Create FFT engine with ghost regions
engine = muGrid.FFTEngine(
    nb_domain_grid_pts=[64, 64, 64],
    comm=comm,
    nb_ghosts_left=[1, 1, 1],
    nb_ghosts_right=[1, 1, 1]
)

# Real-space fields include ghost regions
real_field = engine.real_space_field("displacement", components=(3,))

# Fourier-space fields have no ghosts (hard assumption)
fourier_field = engine.fourier_space_field("displacement_k", components=(3,))
```

Run with MPI:

```bash
mpirun -np 4 python parallel_fft.py
```

## GPU support

When µGrid is compiled with CUDA or HIP support, FFT operations can run on GPU.
See the [GPU documentation](gpu.md) for details on building with GPU support and
working with GPU fields.

The FFT engine also runs natively on GPU memory: pass `device=` to the
`FFTEngine` wrapper (or use the `FFTEngineCUDA` / `FFTEngineROCm` engines), which
dispatch to the cuFFT / rocFFT backends. See the "FFT on GPU" section of the
[GPU documentation](gpu.md).
