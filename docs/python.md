# Python API

The µGrid Python module exposes the core library through a Pythonic interface.
This page documents the main public API grouped by topic. It pairs with the
narrative guide and the topic pages on [FFT](fft.md), [Operators](operators.md),
[GPU](gpu.md), and [Linear algebra](linalg.md), which go into more depth.

µGrid handles *field* quantities (scalars, vectors, tensors) that vary over a
uniform Cartesian grid. The discrete coordinates are **pixels** (called voxels
in 3D, but we use *pixel* throughout). Each pixel can be subdivided into logical
**sub-points** (e.g. quadrature points or elements), and each sub-point carries
a field **component**. Fields are grouped into **field collections** that share
the same spatial discretization.

```python
import muGrid
```

## Field collections

Field collections manage groups of fields on a structured grid. The collection
knows the spatial discretization; individual fields may differ in number of
sub-points and components. A **global** collection has values at every pixel; a
**local** collection has values only on a subset of pixels (e.g. material data
that exists only in part of the domain).

Fields are *named*, and names must be unique. The accessor methods
(`real_field`, `complex_field`, `int_field`, `uint_field`) register a field if
it does not yet exist and return the existing field if it does. The matching
`register_*_field` methods instead fail if the field already exists, and
`get_field` raises if the field is missing.

### GlobalFieldCollection

```python
muGrid.GlobalFieldCollection(nb_grid_pts, nb_sub_pts=None,
                             nb_ghosts_left=None, nb_ghosts_right=None,
                             device=None)
```

Manages a set of fields sharing the same global grid structure, allocated in
host (CPU) or device (GPU) memory.

| Parameter | Description |
|-----------|-------------|
| `nb_grid_pts` | Grid dimensions, e.g. `[64, 64]` (2D) or `[32, 32, 32]` (3D). |
| `nb_sub_pts` | Dict of sub-points per pixel for each sub-point type. Default `{}`. |
| `nb_ghosts_left` | Ghost cells on the low-index side. Default: no ghosts. |
| `nb_ghosts_right` | Ghost cells on the high-index side. Default: no ghosts. |
| `device` | `"cpu"`, `"cuda"`, `"cuda:N"`, `"rocm"`, `"rocm:N"`, or a `Device`. Default `"cpu"`. |

```python
fc = muGrid.GlobalFieldCollection([64, 64])
field = fc.real_field("temperature")
field.p[:] = 300.0                       # set temperature to 300 K

fc_gpu = muGrid.GlobalFieldCollection([64, 64], device="cuda")
```

### LocalFieldCollection

```python
muGrid.LocalFieldCollection(spatial_dim, name="", nb_sub_pts=None, device=None)
```

Manages fields on a subset of pixels, typically for material-specific data in
heterogeneous simulations. Unlike `GlobalFieldCollection`, the first argument is
the spatial dimension (2 or 3). The `device` parameter behaves as above.

### Field creation methods

Both collection types expose the same field-creation methods:

```python
real_field(name, components=(), sub_pt="pixel")
complex_field(name, components=(), sub_pt="pixel")
int_field(name, components=(), sub_pt="pixel")
uint_field(name, components=(), sub_pt="pixel")
```

| Parameter | Description |
|-----------|-------------|
| `name` | Unique field name. |
| `components` | Shape of the field components. Default `()` (scalar). |
| `sub_pt` | Sub-point type name. Default `"pixel"`. |

Each returns a [`Field`](#fields-array-views) with `.s`, `.p`, `.sg`, `.pg`
accessors. Pass `components` as a tuple to make vector- or tensor-valued fields,
e.g. `(3,)` for a vector or `(2, 2)` for a 2×2 matrix.

!!! note
    A scalar field (`components=()`) and a single-component field
    (`components=(1,)`) differ: the single component appears as a leading
    dimension of size 1 in the field shape.

## Fields & array views

A `Field` wraps a C++ µGrid field and provides zero-copy array views into the
underlying data. On the host the views are NumPy arrays; on the GPU they are
CuPy arrays (see [GPU](gpu.md)). Because the views alias the field storage,
writing to them writes the field directly.

Each field has the default multidimensional shape

```python
(components, sub-points, pixels)
```

For example, a 3×3 second-rank tensor on two quadrature points on an 11×12 grid
has shape `(3, 3, 2, 11, 12)`. Components are dropped when there is a single
(scalar) component, but sub-points are always present even when there is only
one.

### Field

```python
muGrid.Field(cpp_field)
```

Each field provides four array accessors. They differ along two axes — **layout**
(SubPt vs Pixel) and **ghost regions** (excluded vs included):

| Accessor | Layout | Ghosts | Shape |
|----------|--------|--------|-------|
| `s`  | SubPt | excluded | `(*components, nb_sub_pts, *spatial_no_ghosts)` |
| `sg` | SubPt | included | `(*components, nb_sub_pts, *spatial_with_ghosts)` |
| `p`  | Pixel | excluded | `(nb_components * nb_sub_pts, *spatial_no_ghosts)` |
| `pg` | Pixel | included | `(nb_components * nb_sub_pts, *spatial_with_ghosts)` |

The **SubPt** layout (`s`, `sg`) exposes the sub-points as an explicit dimension;
the **Pixel** layout (`p`, `pg`) folds the sub-points into the last dimension of
the components. Use SubPt when operating on sub-points separately, Pixel when
treating all sub-point values uniformly. The `g` variants include the ghost
cells used with domain decomposition; for fields without ghosts, `s`/`sg` (and
`p`/`pg`) return views of the same shape.

Field quantities come first in the index order because numerical codes
typically vectorize over the spatial domain. NumPy broadcasting then makes
per-pixel operations natural:

```python
ux, uy, uz = displacement_field.s            # each shape (sub_pts, *spatial)
displacement_field.s[...] /= np.sqrt(ux**2 + uy**2 + uz**2)
```

The default storage order is column-major, so components are stored next to each
other in memory.

Additional properties:

| Property | Description |
|----------|-------------|
| `is_on_gpu` | `True` if the field resides in GPU memory (`bool`). |
| `device` | The device the field lives on (`'cpu'` or `'cuda:N'`). |

### wrap_field

```python
muGrid.wrap_field(field)
```

Wrap a C++ field in a Python `Field` object, giving it the NumPy/CuPy array
accessors.

## Cartesian decomposition & ghosts

For MPI-parallel computations on structured grids, `CartesianDecomposition`
splits the global domain across ranks and manages the **ghost** buffer regions
needed by stencil operations.

### CartesianDecomposition

```python
muGrid.CartesianDecomposition(communicator, nb_domain_grid_pts,
                              nb_subdivisions=None,
                              nb_ghosts_left=None, nb_ghosts_right=None,
                              nb_sub_pts=None, device=None)
```

| Parameter | Description |
|-----------|-------------|
| `communicator` | MPI `Communicator` for parallel execution. |
| `nb_domain_grid_pts` | Global domain grid dimensions. |
| `nb_subdivisions` | Subdivisions in each dimension. Default: automatic. |
| `nb_ghosts_left` | Ghost cells on the low-index side. Default: no ghosts. |
| `nb_ghosts_right` | Ghost cells on the high-index side. Default: no ghosts. |
| `nb_sub_pts` | Dict of sub-points per pixel for each sub-point type. |
| `device` | `"cpu"`, `"cuda"`, `"cuda:N"`, `"rocm"`, `"rocm:N"`, or a `Device`. Default `"cpu"`. |

It also exposes the field-creation methods (`real_field`, etc.) and the
property `nb_grid_pts`, the local subdomain dimensions (alias for
`nb_subdomain_grid_pts`).

```python
from muGrid import Communicator, CartesianDecomposition

comm = Communicator()
decomp = CartesianDecomposition(
    comm,
    nb_domain_grid_pts=[128, 128],
    nb_subdivisions=[1, 1],
    nb_ghosts_left=[1, 1],
    nb_ghosts_right=[1, 1],
)
field = decomp.real_field("displacement", components=(3,))
```

Ghost-buffer operations:

```python
set_nb_sub_pts(sub_pt_type, nb_sub_pts)   # set sub-point count for a type
communicate_ghosts(field)                  # fill ghost buffers from neighbors
reduce_ghosts(field)                       # accumulate ghosts back to interior
```

`reduce_ghosts` is the adjoint of `communicate_ghosts`, needed for transpose
operations (e.g. divergence) with periodic boundary conditions; it zeros the
ghost buffers afterwards.

## Communicator

```python
muGrid.Communicator(communicator=None)
```

Factory for the µGrid communicator. Pass a bare MPI communicator (mpi4py or a
µGrid communicator object); the default is a serial communicator containing just
the current process.

```python
from muGrid import Communicator
comm = Communicator()                 # serial

from mpi4py import MPI
comm = Communicator(MPI.COMM_WORLD)   # MPI-parallel
```

## Operators

Discrete operators apply stencil-based convolutions to fields — gradients,
Laplacians, and other differential operators. A convolution turns a field on
nodal points into a field on quadrature points; the transpose does the reverse.
In index notation the convolution and its transpose are

$$ f_{c,o,q,p} = \sum_{n} \sum_{k} s_{o,q,n,k}\, g_{c,n,p+k} $$

$$ g_{c,n,p} = \sum_{o} \sum_{q} \sum_{k} s_{o,q,n,k}\, f_{c,o,q,p-k} $$

where \(f\) is the output, \(g\) the input, \(s\) the stencil, \(o\) the
operators, \(c\) the components, \(q\) the quadrature points, \(n\) the nodal
points, \(p\) the pixels, and \(k\) runs over the stencil width. See
[Operators](operators.md) for the full treatment.

### GenericLinearOperator

```python
muGrid.GenericLinearOperator(offset, stencil)
```

Applies a convolution (stencil) operation to fields.

| Parameter | Description |
|-----------|-------------|
| `offset` | Offset of the stencil origin relative to the current pixel (one entry per spatial dimension). |
| `stencil` | Stencil coefficients; the shape sets the stencil size. |

The stencil may be given in a lower-dimensional shape, in which case missing
dimensions are taken to be unity (first nodal points, then quadrature points,
then operators).

```python
import numpy as np

# 2D Laplacian stencil
stencil = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
laplace = muGrid.GenericLinearOperator([-1, -1], stencil)
laplace.apply(input_field, output_field)
```

Methods:

```python
apply(nodal_field, quadrature_point_field)
transpose(quadrature_point_field, nodal_field, weights=None)
```

### LaplaceOperator

```python
muGrid.LaplaceOperator(spatial_dim, scale=1.0)
```

A hard-coded, optimized Laplacian using the standard 5-point (2D) or 7-point
(3D) finite-difference stencil. `spatial_dim` is 2 or 3; `scale` scales the
result (default `1.0`).

```python
laplace = muGrid.LaplaceOperator(2, scale=-1.0)
laplace.apply(input_field, output_field)
```

Methods:

```python
apply(input_field, output_field)
apply_increment(input_field, alpha, output_field)   # output += alpha * L(input)
transpose(input_field, output_field, weights=None)  # same as apply
```

### FEMGradientOperator

```python
muGrid.FEMGradientOperator(spatial_dim, grid_spacing=None)
```

A hard-coded gradient operator using linear finite-element shape functions on
triangles (2D) or tetrahedra (3D). `grid_spacing` defaults to unit spacing in
each direction.

```python
grad = muGrid.FEMGradientOperator(2)
grad.apply(nodal_field, quadrature_point_gradient_field)
```

Methods:

```python
apply(nodal_field, quadrature_point_field)
apply_increment(nodal_field, alpha, quadrature_point_field)
transpose(quadrature_point_field, nodal_field, weights=None)            # divergence
transpose_increment(quadrature_point_field, alpha, nodal_field, weights=None)
```

### IsotropicStiffnessOperator2D / 3D

```python
muGrid.IsotropicStiffnessOperator2D(grid_spacing)
muGrid.IsotropicStiffnessOperator3D(grid_spacing)
```

Fused stiffness operators for isotropic linear-elastic materials. They compute
\(K u = B^T C B u\) for linear finite elements without storing the full
stiffness matrix, exploiting the isotropic structure

$$ K = 2\mu G + \lambda V $$

where \(G\) and \(V\) are geometry-only matrices (shared by all voxels) and the
Lamé parameters \(\lambda, \mu\) may vary spatially. This reduces memory from
\(O(N \times 24^2)\) for full \(K\) storage to \(O(N \times 2)\) for the
spatially-varying parameters plus \(O(1)\) for the shared geometry.

`grid_spacing` is `[hx, hy]` (2D) or `[hx, hy, hz]` (3D).

```python
import muGrid

stiffness = muGrid.IsotropicStiffnessOperator3D([0.1, 0.1, 0.1])

fc = muGrid.CartesianDecomposition(
    muGrid.Communicator(),
    nb_domain_grid_pts=[32, 32, 32],
    nb_ghosts_left=[1, 1, 1],
    nb_ghosts_right=[1, 1, 1],
)
u = fc.real_field("displacement", (3,))
f = fc.real_field("force", (3,))
lam = fc.real_field("lambda")      # Lamé first parameter
mu = fc.real_field("mu")           # shear modulus
lam.p[...] = 1.0
mu.p[...] = 1.0

fc.communicate_ghosts(u)
stiffness.apply(u, lam, mu, f)     # f = K @ u
```

Methods:

```python
apply(displacement, lam, mu, force)
apply_increment(displacement, lam, mu, alpha, force)   # force += alpha * K @ u
```

## FFT engine

`FFTEngine` provides distributed FFTs on structured grids, MPI-parallelized with
an auto-selected slab or pencil decomposition (`decomposition=` argument). See
[FFT](fft.md) for details.

### FFTEngine

```python
muGrid.FFTEngine(nb_domain_grid_pts, communicator=None,
                 nb_ghosts_left=None, nb_ghosts_right=None, nb_sub_pts=None)
```

| Parameter | Description |
|-----------|-------------|
| `nb_domain_grid_pts` | Global grid dimensions `[Nx, Ny]` or `[Nx, Ny, Nz]`. |
| `communicator` | MPI `Communicator`. Default: serial. |
| `nb_ghosts_left` | Ghost cells on the low-index side of each dimension. |
| `nb_ghosts_right` | Ghost cells on the high-index side of each dimension. |
| `nb_sub_pts` | Dict of sub-points per pixel. |

```python
engine = muGrid.FFTEngine([64, 64])
real_field = engine.real_space_field("displacement", components=(3,))
fourier_field = engine.fourier_space_field("displacement_k", components=(3,))
engine.fft(real_field, fourier_field)
engine.ifft(fourier_field, real_field)
real_field.s[:] *= engine.normalisation
```

!!! note
    The transforms are **unnormalized**. To recover the original data after
    `ifft(fft(x))`, multiply by `normalisation`.

Transform methods:

```python
fft(input_field, output_field)    # real space -> Fourier space
ifft(input_field, output_field)   # Fourier space -> real space
```

Field registration (the `register_*` variants raise `RuntimeError` if the name
already exists; the others get-or-create):

```python
register_real_space_field(name, components=())
register_fourier_space_field(name, components=())
real_space_field(name, components=())       # returns a real-valued Field
fourier_space_field(name, components=())    # returns a complex-valued Field
```

Attributes:

| Attribute | Description |
|-----------|-------------|
| `normalisation` | Normalization factor (`float`); multiply after `ifft(fft(x))`. |
| `spatial_dim` | Spatial dimension, 2 or 3 (`int`). |
| `fftfreq` | Normalized FFT frequencies for the local Fourier subdomain, shape `[dim, local_fx, ...]`, values in `[-0.5, 0.5)`. |
| `ifftfreq` | Integer FFT frequency indices for the local Fourier subdomain. |
| `coords` | Normalized real-space coordinates of the local subdomain (no ghosts), values in `[0, 1)`. |
| `icoords` | Integer real-space coordinate indices of the local subdomain (no ghosts). |
| `coordsg` | Like `coords` but including ghost cells. |
| `icoordsg` | Like `icoords` but including ghost cells. |

For MPI-parallel runs, the frequency and coordinate arrays return only the
portion owned by the local rank.

### FFT utilities

```python
muGrid.fft_normalization(nb_grid_pts)      # -> float
muGrid.get_hermitian_grid_pts(nb_grid_pts) # -> tuple of int (rfft representation)
```

## File I/O

Fields can be written to disk in [NetCDF](https://en.wikipedia.org/wiki/NetCDF)
format. µGrid uses Unidata NetCDF for serial builds and PnetCDF when built with
MPI.

### FileIONetCDF

```python
muGrid.FileIONetCDF(file_name, open_mode="read", communicator=None)
```

| Parameter | Description |
|-----------|-------------|
| `file_name` | Path to the NetCDF file. |
| `open_mode` | `"read"`, `"write"`, `"overwrite"`, or `"append"`. Default `"read"`. |
| `communicator` | MPI `Communicator` for parallel I/O. Default: serial. |

```python
file = muGrid.FileIONetCDF("output.nc", open_mode="overwrite")
file.register_field_collection(field_collection)
file.append_frame().write()
```

Methods and attributes:

```python
register_field_collection(collection)   # GlobalFieldCollection, LocalFieldCollection,
                                         # or CartesianDecomposition
append_frame()                           # -> frame object for writing
```

`OpenMode` is an enum with values `Read`, `Write`, `Overwrite`, `Append`.

## Linear algebra

The `muGrid.Solvers` module provides simple parallel iterative solvers. See
[Linear algebra](linalg.md) for more.

### conjugate_gradients

```python
muGrid.Solvers.conjugate_gradients(comm, fc, b, x, hessp, prec=None,
                                   tol=1e-6, maxiter=1000,
                                   callback=None, timer=None)
```

Matrix-free conjugate-gradient solution of `Ax = b`, where `A` is represented by
`hessp` (which computes the product of `A` with a vector). The solution `x` is
refined in place until `||Ax - b|| < tol` or `maxiter` iterations are reached.

| Parameter | Description |
|-----------|-------------|
| `comm` | `muGrid.Communicator` for parallel processing. |
| `fc` | Collection for temporary fields (`GlobalFieldCollection`, `LocalFieldCollection`, or `CartesianDecomposition`). |
| `b` | Right-hand-side field. |
| `x` | Initial guess; modified in place. |
| `hessp` | Callable `hessp(input_field, output_field)` computing `A @ x`. |
| `prec` | Optional preconditioner `prec(input_field, output_field)`. Default `None`. |
| `tol` | Convergence tolerance. Default `1e-6`. |
| `maxiter` | Maximum iterations. Default `1000`. |
| `callback` | Called as `callback(iteration, state_dict)`, where `state_dict` has keys `"x"`, `"r"`, `"p"`, `"rr"` (squared residual norm). |
| `timer` | Optional `muTimer.Timer` for profiling. |

Returns the solution field `x`. Raises `RuntimeError` if it fails to converge
within `maxiter`, or if the Hessian is not positive definite.

## Device selection

The `Device` class and `DeviceType` enum specify where field data is allocated.
See [GPU](gpu.md) for the full picture.

### Device

```python
muGrid.Device
```

Represents a compute device (CPU or GPU). Use the factory static methods to
construct one:

```python
Device.cpu()                  # CPU device
Device.cuda(device_id=0)      # CUDA GPU
Device.rocm(device_id=0)      # ROCm GPU
Device.gpu(device_id=0)       # default available GPU backend (CUDA, else ROCm, else CPU)
```

`Device.gpu()` is the recommended way to request GPU execution without
hard-coding a backend.

Read-only properties:

| Property | Description |
|----------|-------------|
| `is_host` | `True` for a host (CPU) device. |
| `is_device` | `True` for a device (GPU) memory location. |
| `device_type` | The `DeviceType` enum value. |
| `device_id` | Device ID for multi-GPU systems (0 for single-GPU or CPU). |

```python
import muGrid

cpu = muGrid.Device.cpu()
gpu0 = muGrid.Device.cuda()
gpu1 = muGrid.Device.cuda(1)

cpu.is_host        # True   (properties, not methods)
gpu0.is_device     # True

fc = muGrid.GlobalFieldCollection([64, 64], device=muGrid.Device.cuda())
```

### DeviceType

Enumeration for device types; values follow DLPack conventions.

| Value | Description |
|-------|-------------|
| `CPU` | CPU device (1). |
| `CUDA` | NVIDIA CUDA GPU (2). |
| `CUDAHost` | CUDA pinned host memory (3). |
| `ROCm` | AMD ROCm GPU (10). |
| `ROCmHost` | ROCm pinned host memory (11). |

## Enumerations

### IterUnit

Iteration unit type: `Pixel` (iterate over pixels) or `SubPt` (iterate over
sub-points).

### StorageOrder

Array storage order:

| Value | Description |
|-------|-------------|
| `ArrayOfStructures` | Components consecutive in memory (default host layout). |
| `StructureOfArrays` | Pixels consecutive in memory (device layout). |
| `Automatic` | Inherit the storage order from the `FieldCollection`. |

## Module constants

Compile-time configuration flags:

| Constant | Meaning |
|----------|---------|
| `muGrid.has_mpi` | MPI support is enabled. |
| `muGrid.has_cuda` | CUDA GPU support is compiled in. |
| `muGrid.has_rocm` | ROCm/HIP GPU support is compiled in. |
| `muGrid.has_gpu` | Any GPU support is available. |
| `muGrid.has_netcdf` | NetCDF I/O support is available. |

## Timer

```python
muTimer.Timer()
```

A hierarchical timer for performance measurement, usable as a context manager.

!!! note
    `Timer` is **not** part of µGrid. It is provided by the separate `muTimer`
    package (`import muTimer`), which the examples depend on.

```python
import muTimer
timer = muTimer.Timer()
with timer("outer"):
    with timer("inner"):
        ...        # some computation
timer.print_summary()
```

Methods:

```python
timer(name)            # context manager timing a named block
get_time(name)         # total elapsed seconds ("outer/inner" for nested)
get_calls(name)        # number of times the region was entered
print_summary()        # print a formatted summary
to_dict()              # export timing data as a dict (JSON-friendly)
```
