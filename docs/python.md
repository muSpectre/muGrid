# Python API

This page is the reference for the public Python API of µGrid, grouped by
topic. Background and worked usage live on the topic pages —
[FFT](fft.md), [Operators](operators.md), [Linear algebra](linalg.md),
[GPU](gpu.md) — and on the [Examples](examples.md) page.

µGrid handles *field* quantities (scalars, vectors, tensors) that vary over a
uniform Cartesian grid of $\prod_\alpha N_\alpha$ points in $d = 2$ or $3$
dimensions. The discrete coordinates are **pixels** (voxels in 3D; *pixel* is
used throughout). Each pixel can be subdivided into logical **sub-points**
(e.g. quadrature points or elements), and each sub-point carries the field's
**components**. Fields are grouped into **field collections** that share the
same spatial discretization.

```python
import muGrid
```

## Field collections

A field collection holds the grid geometry (grid points, sub-points per pixel,
ghost layers) and the fields defined on it; individual fields may differ in
their number of sub-points and components. A **global** collection has values
at every pixel; a **local** collection has values only on a subset of pixels
(e.g. material data that exists only in part of the domain).

Fields are *named*, and names are unique within a collection. The accessor
methods (`real_field`, `complex_field`, `int_field`, `uint_field`) return the
existing field of that name or create it. The `register_*_field` methods of the
underlying C++ collection fail if the field already exists, and `get_field`
raises if it is missing.

### GlobalFieldCollection

```python
muGrid.GlobalFieldCollection(nb_grid_pts, nb_sub_pts=None,
                             nb_ghosts_left=None, nb_ghosts_right=None,
                             device=None, ghosts=None,
                             nb_subdomain_grid_pts=None, storage_order=None)
```

All arguments after `nb_grid_pts` are keyword arguments.

| Parameter | Description |
|-----------|-------------|
| `nb_grid_pts` | Grid dimensions, e.g. `[64, 64]` (2D) or `[32, 32, 32]` (3D). |
| `nb_sub_pts` | Dict mapping sub-point type name to sub-points per pixel. Default `{}`. Alias `sub_pts`. |
| `nb_ghosts_left` | Ghost layers on the low-index side, per dimension. Default: none. |
| `nb_ghosts_right` | Ghost layers on the high-index side, per dimension. Default: none. |
| `ghosts` | Ghost specification, mutually exclusive with `nb_ghosts_left`/`nb_ghosts_right` (see [Ghost specification](#ghost-specification)). |
| `device` | Device string or `Device` (see [Device selection](#device-selection)). Default `"cpu"`. |
| `nb_subdomain_grid_pts` | Local subdomain size, if the collection holds only part of the grid. |
| `storage_order` | A `StorageOrder`; default is the memory space's native order. |

```python
fc = muGrid.GlobalFieldCollection([64, 64])
field = fc.real_field("temperature")
field.p[...] = 300.0

fc_gpu = muGrid.GlobalFieldCollection([64, 64], device="gpu")
```

### LocalFieldCollection

```python
muGrid.LocalFieldCollection(spatial_dim, name="", nb_sub_pts=None, device=None)
```

Manages fields on a subset of pixels, typically material-specific data in
heterogeneous simulations. The first argument is the spatial dimension
$d \in \{2, 3\}$; `device` behaves as above.

### Field creation methods

All collection types (`GlobalFieldCollection`, `LocalFieldCollection`,
`CartesianDecomposition`) expose:

```python
real_field(name, components=(), sub_pt="pixel", dtype=np.float64)
complex_field(name, components=(), sub_pt="pixel", dtype=np.complex128)
int_field(name, components=(), sub_pt="pixel")
uint_field(name, components=(), sub_pt="pixel")
```

| Parameter | Description |
|-----------|-------------|
| `name` | Unique field name. |
| `components` | Shape of the per-sub-point components. Default `()` (scalar). |
| `sub_pt` | Sub-point type name; must be declared in `nb_sub_pts` unless it is `"pixel"`. Default `"pixel"`. |
| `dtype` | `np.float64`/`np.float32` (real) or `np.complex128`/`np.complex64` (complex). Single precision halves the memory footprint. |

Each returns a [`Field`](#field). Pass `components` as a tuple for vector- or
tensor-valued fields, e.g. `(3,)` for a vector or `(2, 2)` for a 2×2 matrix.

!!! note
    A scalar field (`components=()`) and a single-component field
    (`components=(1,)`) differ: the single component appears as a leading
    axis of length 1 in the array views.

### Ghost specification

`GlobalFieldCollection`, `CartesianDecomposition` and `FFTEngine` accept a
`ghosts` argument that sizes the ghost layers:

- an operator (any object with a `ghost_requirement` attribute, which includes
  all µGrid stencil operators): the layers that operator needs for all of its
  operations;
- a sequence of operators: the elementwise maximum of their requirements;
- an int $n$: $n$ layers on both sides of every dimension;
- a pair `(left, right)`, each an int or a per-dimension sequence.

## Fields & array views

### Field

```python
muGrid.Field(cpp_field)
muGrid.wrap_field(cpp_field)   # same, for a raw C++ field
```

A `Field` wraps a C++ field and provides zero-copy array views of its data via
DLPack: NumPy arrays on the host, CuPy arrays on the GPU (see [GPU](gpu.md)).
Writing to a view writes the field. Attributes not defined by the wrapper are
forwarded to the C++ field (e.g. `name`, `components_shape`, `nb_sub_pts`,
`collection`, `set_zero()`).

The four views differ in **layout** (sub-points as a separate axis or folded
into the components) and in whether **ghost layers** are included. With
components shape $(c_1, \dots, c_m)$, $S$ sub-points and the spatial shape
$(n_1, \dots, n_d)$:

| View | Layout | Ghosts | Shape |
|------|--------|--------|-------|
| `s`  | SubPt | excluded | $(c_1, \dots, c_m, S, n_1, \dots, n_d)$ |
| `sg` | SubPt | included | same, with ghost-padded spatial axes |
| `p`  | Pixel | excluded | $(c_1, \dots, c_{m-1}, c_m S, n_1, \dots, n_d)$ |
| `pg` | Pixel | included | same, with ghost-padded spatial axes |

The SubPt views always carry the sub-point axis, even for $S = 1$. The Pixel
views fold the sub-points into the last component axis; a scalar field with
$S = 1$ has no leading axis in `p`/`pg`. For example, a 3×3 tensor on two
quadrature points of an 11×12 grid has `s.shape == (3, 3, 2, 11, 12)` and
`p.shape == (3, 6, 11, 12)`. For fields without ghosts, `s`/`sg` (and `p`/`pg`)
have the same shape.

The order of the folded Pixel axis depends on the storage order: component
fastest in array-of-structures storage (the host default), sub-point fastest in
structure-of-arrays storage (the device default). Each is a zero-copy view only
in its own storage order, so when the two orders disagree — more than one
sub-point *and* a last component axis longer than one — `p`/`pg` raise
`ValueError` on structure-of-arrays storage. Use `s`/`sg` for such fields; they
index components and sub-points separately and are identical in both storage
orders.

Field quantities come first in the index order, so NumPy broadcasting over the
spatial axes makes per-pixel operations natural:

```python
ux, uy, uz = displacement_field.s            # each shape (sub_pts, *spatial)
displacement_field.s[...] /= np.sqrt(ux**2 + uy**2 + uz**2)
```

Additional properties:

| Property | Description |
|----------|-------------|
| `is_on_gpu` | `True` if the field resides in GPU memory. |
| `device` | Device string of the field's memory, e.g. `'cpu'` or `'cuda:0'`. |
| `dtype` | NumPy dtype of the entries (`float64`, `float32`, `complex128`, …). |

## Domain decomposition

### CartesianDecomposition

```python
muGrid.CartesianDecomposition(communicator, nb_domain_grid_pts,
                              nb_subdivisions=None,
                              nb_ghosts_left=None, nb_ghosts_right=None,
                              nb_sub_pts=None, device=None, ghosts=None)
```

Splits the global grid across the $P$ ranks of `communicator` and manages the
ghost layers needed by stencil operations.

| Parameter | Description |
|-----------|-------------|
| `communicator` | `Communicator` (or an mpi4py communicator, which is wrapped). |
| `nb_domain_grid_pts` | Global grid dimensions. |
| `nb_subdivisions` | Ranks per dimension; their product must be $P$. Default: the prime factors of $P$, largest first, are assigned one at a time to the axis with the most grid points per rank, which keeps subdomains close to cubic. |
| `nb_ghosts_left` | Ghost layers on the low-index side. Default: none. |
| `nb_ghosts_right` | Ghost layers on the high-index side. Default: none. |
| `nb_sub_pts` | Dict of sub-points per pixel for each sub-point type. |
| `device` | Device string or `Device`. Default `"cpu"`. |
| `ghosts` | [Ghost specification](#ghost-specification), alternative to `nb_ghosts_*`. |

The decomposition exposes the [field creation methods](#field-creation-methods),
the wrapped `collection`, and `nb_grid_pts` (the local subdomain size, alias
for `nb_subdomain_grid_pts`; `nb_domain_grid_pts` is the global size).

```python
from muGrid import Communicator, CartesianDecomposition, LaplaceOperator

laplace = LaplaceOperator(2)
decomp = CartesianDecomposition(Communicator(), [128, 128], ghosts=laplace)
field = decomp.real_field("u")
```

Ghost-layer methods:

```python
set_nb_sub_pts(sub_pt_type, nb_sub_pts)   # declare a sub-point type
communicate_ghosts(field)                  # fill ghost layers from the owners
reduce_ghosts(field)                       # add ghost values to their owners
```

`communicate_ghosts` copies each owned value into the ghost layers that mirror
it (periodic wrap-around included). `reduce_ghosts` is its exact adjoint: it
adds every ghost value to the owned value it mirrors and then zeros the ghost
layers. It is needed only for code that scatters into ghost layers; the
transposes of µGrid's own operators gather from communicated input ghosts and
write owned points only, so their results must not be reduced. Both work for
ghost layers of any width: when a layer is wider than a neighbouring rank's
interior (or a rank owns no grid points), data is relayed through intermediate
ranks, so the ghost width can be chosen from the stencil alone.

### Communicator

```python
muGrid.Communicator(communicator=None)
```

The default is a serial communicator containing only the current process;
constructing it under a multi-rank MPI launcher emits a `RuntimeWarning`.
`Communicator.world()` is `MPI_COMM_WORLD` and starts MPI (with
`MPI_THREAD_SERIALIZED`) if nothing has yet; opening a NetCDF file in a
PnetCDF build does the same, since PnetCDF needs MPI running even for one
rank. Nothing else does: importing µGrid does not start MPI, and mpi4py is
not needed. µGrid finalises MPI at interpreter exit if it started it
(`muGrid.finalize_mpi()` does so earlier; `MUGRID_MPI_FINALIZE=0` leaves it to
the OS, for MPI stacks that misbehave once the GPU context is gone).

A communicator provides `rank`, `size`, `barrier()`, the reductions `sum`,
`max`, `min`, `all`, `any` and `cumulative_sum`, the array exchanges
`allreduce`, `allgather`, `allgatherv` and `sendrecv` (NumPy or CuPy arrays),
`split(color, key=0)`, `dup()`, `compare(other)`, and `node_rank`/`node_size`
(its rank among the ranks on this node, for choosing one GPU per rank).

`bcast(obj, root=0)`, `gather(obj, root=0)` and `allgather(obj)` follow
mpi4py's lower-case collectives: any Python object travels pickled, `gather`
returns a list on the root and `None` elsewhere, `allgather` a list on every
rank. Arrays are the exception and go as buffers: `bcast` of a NumPy or CuPy
array arrives as an array of the same kind, and `allgather` of an array (or a
NumPy scalar) comes back stacked along a new leading axis. `allreduce(a, op)`
takes `"sum"`, `"max"`, `"min"` or mpi4py's `MPI.SUM`/`MAX`/`MIN`, and returns a
Python number for a Python number.

```python
from muGrid import Communicator
comm = Communicator()          # serial
comm = Communicator.world()    # MPI-parallel: mpirun -np 4 python script.py
```

An mpi4py communicator is accepted too (`Communicator(MPI.COMM_WORLD)`), and
`comm.mpi4py_comm` hands a µGrid communicator to mpi4py when it is imported.
µGrid checks that mpi4py is linked against the same MPI library and raises
otherwise.

`muGrid.parprint(*args, comm=None, **kwargs)` prints on rank 0 only.

## Operators

Stencil operators assume nothing about boundary conditions: periodicity and
parallelism come from ghost layers filled by `communicate_ghosts` beforehand.
Every operator reports the ghost layers it needs as `(left, right)` tuples via
`apply_ghost_requirement`, `transpose_ghost_requirement` and their elementwise
maximum `ghost_requirement`, and validates them once per call before the
kernel runs. Definitions, discretisations and usage are on the
[Operators](operators.md) page.

### GenericLinearOperator

```python
muGrid.GenericLinearOperator(offset, stencil)
```

| Parameter | Description |
|-----------|-------------|
| `offset` | Position of the stencil origin relative to the current pixel, one entry per dimension. |
| `stencil` | Stencil coefficients of shape `(nb_output, nb_quad, nb_input, *stencil_shape)`. Non-spatial axes may be left off from the right: $d$ axes mean one output, $d+1$ axes `(nb_output, *stencil_shape)`, $d+2$ axes `(nb_output, nb_quad, *stencil_shape)`; missing counts are 1. |

```python
apply(nodal_field, quadrature_point_field)
transpose(quadrature_point_field, nodal_field, weights=None)
```

Without `weights` the transpose is the plain matrix transpose
$\mathbf{B}^\top$; pass one weight per quadrature point (e.g.
`fem.quadrature_weights`) to obtain $\mathbf{B}^\top W$.

```python
import numpy as np

stencil = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
laplace = muGrid.GenericLinearOperator([-1, -1], stencil)
laplace.apply(input_field, output_field)
```

### LaplaceOperator

```python
muGrid.LaplaceOperator(spatial_dim, scale=1.0, grid_spacing=None)
```

The $(2d+1)$-point finite-difference Laplacian, multiplied by `scale`. With
`grid_spacing` $= (h_1, \dots, h_d)$ the neighbours along axis $\alpha$ are
weighted by $\mathrm{scale}/h_\alpha^2$; without it every axis uses unit
spacing.

```python
apply(input_field, output_field)
apply_increment(input_field, alpha, output_field)   # output += alpha * L input
transpose(input_field, output_field, weights=None)  # equal to apply
```

### FEMGradientOperator

```python
muGrid.FEMGradientOperator(spatial_dim, grid_spacing=None,
                           element=muGrid.FEMElement.q1, dtype=np.float64)
```

Finite-element gradient $\mathbf{B}$ from nodal values to quadrature points.
`element` is `FEMElement.q1` (bilinear/trilinear, one element per pixel,
$2^d$ Gauss points) or `FEMElement.p1` (linear simplices: 2 triangles per pixel
or 5 tetrahedra per voxel, one quadrature point each). `grid_spacing` defaults
to 1 along every axis; `dtype` (`np.float64` or `np.float32`) is the precision
of the fields it is applied to. Properties: `nb_quad_pts`,
`quadrature_weights`, `coefficients`, `grid_spacing`.

```python
apply(nodal_field, quadrature_point_field)
apply_increment(nodal_field, alpha, quadrature_point_field)
transpose(quadrature_point_field, nodal_field, weights=None)
transpose_increment(quadrature_point_field, alpha, nodal_field, weights=None)
```

With `weights=None` the transpose uses the physical quadrature weights $W$, so
`transpose(apply(u))` is $\mathbf{B}^\top W \mathbf{B} \mathbf{u}$.

### IsotropicStiffnessOperator

```python
muGrid.IsotropicStiffnessOperator(spatial_dim, grid_spacing,
                                  element=muGrid.FEMElement.q1)
muGrid.IsotropicStiffnessOperator2D(grid_spacing, element=muGrid.FEMElement.q1)
muGrid.IsotropicStiffnessOperator3D(grid_spacing, element=muGrid.FEMElement.q1)
```

Fused matrix-free stiffness $\mathbf{K}\mathbf{u} = \mathbf{B}^\top \mathbf{C}
\mathbf{B} \mathbf{u}$ for isotropic linear elasticity with per-pixel Lamé
fields $\lambda, \mu$, using $K_e = 2\mu G + \lambda V$ (see
[Operators](operators.md#isotropicstiffnessoperator)). `grid_spacing` is
$(h_1, \dots, h_d)$. Displacement/force and the Lamé fields live on the same
collection, with at least one ghost layer on every side; communicate the
ghosts of the displacement and of the Lamé fields before applying.

```python
apply(displacement, lambda_field, mu_field, force)              # f = K u
apply_increment(displacement, lambda_field, mu_field, alpha, force)
apply_uniform(displacement, lam, mu, force)                     # scalar λ, μ
apply_uniform_increment(displacement, lam, mu, alpha, force)
apply_macro_rhs(lambda_field, mu_field, E_macro, force)         # f = Bᵀ C E
average_stress(displacement, lambda_field, mu_field, E_macro)   # local ∫ σ dV
assemble_diagonal(lambda_field, mu_field, diagonal)             # diag(K)
compute_sensitivity(forward_disp, forward_macro, costate_disp,
                    costate_macro, g_shear, g_vol)
```

`E_macro` is the macroscopic strain flattened row-major ($d^2$ entries).
`average_stress` returns the un-reduced local volume integral of
$\sigma = \mathbf{C} : (E + \operatorname{sym}\nabla u)$; sum it across ranks
and divide by the cell volume. `compute_sensitivity` writes the per-pixel
contractions $a_e^\top G b_e$ and $a_e^\top V b_e$ of the total forward and
costate strains.

```python
stiffness = muGrid.IsotropicStiffnessOperator3D([0.1, 0.1, 0.1])
fc = muGrid.CartesianDecomposition(muGrid.Communicator(), [32, 32, 32],
                                   ghosts=stiffness)
u = fc.real_field("displacement", (3,))
f = fc.real_field("force", (3,))
lam = fc.real_field("lambda")
mu = fc.real_field("mu")
lam.p[...] = 1.0
mu.p[...] = 1.0
for field in (u, lam, mu):
    fc.communicate_ghosts(field)
stiffness.apply(u, lam, mu, f)
```

### NodalMomentOperator

```python
muGrid.NodalMomentOperator(dim, grid_spacing, element=muGrid.FEMElement.q1)
muGrid.NodalMomentOperator2D(grid_spacing, element=muGrid.FEMElement.q1)
muGrid.NodalMomentOperator3D(grid_spacing, element=muGrid.FEMElement.q1)
compute(rho, moments, moment_gradients)
```

For each Q1 cell $e$, computes the moments $M_k = \int_e \rho(\mathbf{x})^k\,
\mathrm{d}\mathbf{x}$, $k = 2, 3, 4$, of the finite-element interpolant of a
nodal scalar field $\rho$, and the derivatives of their sum with respect to the
nodal values, using 3-point Gauss–Legendre quadrature per axis (exact for the
quartic integrand). `rho` is a one-component field with communicated ghosts;
`moments` and `moment_gradients` are three-component fields on the same
collection. Properties: `cell_volume`, `nb_quad`.

### GridTransfer

```python
muGrid.GridTransfer(spatial_dim)
muGrid.GridTransfer2D()
muGrid.GridTransfer3D()
prolong(coarse, fine)
restrict(fine, coarse)
```

Transfer between nested nodal grids where the fine grid has twice the extent
of the coarse one. `prolong` is multilinear interpolation $P$; `restrict` is
its exact adjoint $P^\top$, the tensor product of the stencil
$[\tfrac12, 1, \tfrac12]$. Both act component-wise. Each rank's fine subdomain
must have twice the extent and twice the global offset of its coarse subdomain
(use the same power-of-two `nb_subdivisions` on every level). Communicate the
ghosts of the input field first; neither method writes ghosts.

## FFT engine

### FFTEngine

```python
muGrid.FFTEngine(nb_domain_grid_pts, communicator=None,
                 nb_ghosts_left=None, nb_ghosts_right=None, nb_sub_pts=None,
                 device=None, ghosts=None, decomposition="auto")
```

Distributed real-to-complex FFT on a $d$-dimensional grid. The engine's
real-space collection is also a ghosted domain decomposition that other code
can build on. See [FFT](fft.md) for background.

| Parameter | Description |
|-----------|-------------|
| `nb_domain_grid_pts` | Global grid dimensions $(N_1, \dots, N_d)$. |
| `communicator` | `Communicator` or mpi4py communicator. Default: serial. |
| `nb_ghosts_left` | Real-space ghost layers on the low-index side. |
| `nb_ghosts_right` | Real-space ghost layers on the high-index side. |
| `nb_sub_pts` | Dict of sub-points per pixel. |
| `device` | `"cpu"` (PocketFFT), or a GPU device (cuFFT/rocFFT). Default `"cpu"`. |
| `ghosts` | [Ghost specification](#ghost-specification) for the real-space fields. |
| `decomposition` | `"auto"`, `"slab"` or `"pencil"` (see [FFT](fft.md#choosing-the-decomposition)). |

```python
engine = muGrid.FFTEngine([64, 64])
real_field = engine.real_space_field("displacement", components=(3,))
fourier_field = engine.fourier_space_field("displacement_k", components=(3,))
engine.fft(real_field, fourier_field)
engine.ifft(fourier_field, real_field)
real_field.s[...] *= engine.normalisation
```

!!! note
    The transforms are unnormalised: `ifft(fft(x))` returns
    $\prod_\alpha N_\alpha \, x$. Multiply by `normalisation`
    $= 1/\prod_\alpha N_\alpha$ to recover `x`.

Methods:

```python
fft(input_field, output_field)    # real space -> Fourier space
ifft(input_field, output_field)   # Fourier space -> real space
communicate_ghosts(field)         # real-space fields only
reduce_ghosts(field)

register_real_space_field(name, components=(), dtype=np.float64, sub_pt="pixel")
register_fourier_space_field(name, components=(), dtype=np.complex128, sub_pt="pixel")
real_space_field(name, components=(), dtype=np.float64, sub_pt="pixel")
fourier_space_field(name, components=(), dtype=np.complex128, sub_pt="pixel")
```

The `register_*` variants raise `RuntimeError` if the name exists; the others
get or create. A `float32`/`complex64` field pair selects a single-precision
transform. Input and output must have the same components and sub-points.

Attributes:

| Attribute | Description |
|-----------|-------------|
| `normalisation` | $1/\prod_\alpha N_\alpha$ (`float`). |
| `spatial_dim` | $d$ (`int`). |
| `nb_domain_grid_pts`, `nb_subdomain_grid_pts` | Global and local real-space grid size. |
| `nb_fourier_grid_pts`, `nb_fourier_subdomain_grid_pts` | Global and local Fourier grid size; axis 0 has $\lfloor N_1/2 \rfloor + 1$ points. |
| `process_grid` | Process grid $(P_1, P_2)$; slab has $P_2 = 1$. |
| `communicator` | The wrapped communicator the engine was built with. |
| `real_space_collection`, `fourier_space_collection` | The two field collections. |
| `fftfreq` | Frequencies $q$ in cycles per grid point of the local Fourier subdomain, shape `[dim, *local]`; axis 0 holds $[0, \tfrac12]$ (an even-$N$ Nyquist mode is $+\tfrac12$), the other axes $[-\tfrac12, \tfrac12)$. |
| `ifftfreq` | Integer frequency indices $q N_\alpha$ of the local Fourier subdomain. |
| `coords` | Fractional real-space coordinates $x_\alpha / (N_\alpha h_\alpha) \in [0, 1)$ of the local subdomain, no ghosts. |
| `icoords` | Integer real-space indices of the local subdomain, no ghosts. |
| `coordsg`, `icoordsg` | As above including ghost layers, wrapped periodically (the left ghost of index 0 reports $N - 1$, not $-1$). |

In MPI runs the frequency and coordinate arrays cover only the local rank's
portion. Multiply fractional coordinates by the domain length
$N_\alpha h_\alpha$ to obtain physical positions; the wavevector is
$k_\alpha = 2\pi q_\alpha / h_\alpha$.

### FFT utilities

```python
muGrid.fft_normalization(nb_grid_pts)      # -> 1 / prod(nb_grid_pts)
muGrid.get_hermitian_grid_pts(nb_grid_pts) # -> half-complex grid size
```

## File I/O

### FileIONetCDF

```python
muGrid.FileIONetCDF(file_name, open_mode="read", communicator=None)
```

Reads and writes fields in [NetCDF](https://en.wikipedia.org/wiki/NetCDF)
format: Unidata NetCDF in serial builds, PnetCDF in MPI builds. PnetCDF writes
CDF-5 files, which must be read back through µGrid or another CDF-5-capable
reader.

| Parameter | Description |
|-----------|-------------|
| `file_name` | Path to the NetCDF file. |
| `open_mode` | `"read"`, `"write"`, `"overwrite"`, `"append"`, or an `OpenMode`. Default `"read"`. |
| `communicator` | `Communicator` for parallel I/O. Default: serial. |

```python
register_field_collection(collection, field_names=..., state_field_unique_prefixes=...)
register_frame_variable(name, shape, dtype, fill_value=None)
append_frame()               # -> FileFrame; call .write() on it
file[i]                      # FileFrame i; .read() / .write()
len(file)                    # number of frames
write_global_attribute(name, value)
read_global_attribute(name)
close()
```

`register_field_collection` accepts a `GlobalFieldCollection`,
`LocalFieldCollection` or `CartesianDecomposition`; by default all fields of
the collection are registered. `OpenMode` is an enum with values `Read`,
`Write`, `Overwrite`, `Append`.

```python
file = muGrid.FileIONetCDF("output.nc", open_mode="overwrite")
file.register_field_collection(field_collection)
file.append_frame().write()
file.close()
```

#### Per-frame quantities

Besides grid fields, a frame can carry small **grid-less** quantities — a
scalar, vector or tensor with one value for the whole domain per frame (e.g.
an applied deformation gradient, the physical time, or a load parameter).
These are stored as NetCDF variables with the unlimited `frame` dimension plus
fixed component dimensions, and are replicated (identical on every rank).

| Parameter | Description |
|-----------|-------------|
| `name` | Unique variable name. |
| `shape` | Shape of one frame's value, e.g. `[3, 3]` for a tensor (`[]`/`[n]` for a scalar/vector). |
| `dtype` | NumPy dtype of the elements (e.g. `np.float64`). |
| `fill_value` | Optional NetCDF fill value, a single value castable to `dtype`. |

Call `register_frame_variable` before the first frame is written. It returns a
NumPy array that **views** the variable's buffer: write the current frame's
value into it, then `write` it together with any fields of that frame. The view
pins the file object, so call `file.close()` to flush and release it (a plain
`del file` does not close the file while the view is alive).

```python
file = muGrid.FileIONetCDF("output.nc", open_mode="overwrite")
file.register_field_collection(field_collection, field_names=["density"])
F = file.register_frame_variable("applied_deformation_gradient", [3, 3], np.float64)
for step in path:
    F[...] = deformation_gradient(step)
    field_collection.get_field("density").p[...] = density(step)
    file.append_frame().write(["density", "applied_deformation_gradient"])
file.close()
```

Reading mirrors writing: register the same variable, `read` a frame and inspect
the view. Under MPI, rank 0 writes the value (the other ranks take part in the
collective call with an empty request) and every rank reads the full value, so
the result is identical in serial and parallel.

For a value that does not vary per frame (e.g. the physical cell size), use a
global attribute (`write_global_attribute`/`read_global_attribute`), stored
once per file.

## Solvers

Matrix-free solvers for $\mathbf{A}\mathbf{x} = \mathbf{b}$ with $\mathbf{A}$
symmetric positive definite, built on the [Linear algebra](linalg.md) kernels.
They work for host and device fields and in MPI runs.

```python
from muGrid.Solvers import (conjugate_gradients,
                            conjugate_gradients_pipelined, ConvergenceError)
```

### conjugate_gradients

```python
conjugate_gradients(comm, fc, b, x, hessp, prec=None, tol=None, maxiter=1000,
                    callback=None, timer=None, rtol=None, atol=0.0,
                    residual=None)
```

Preconditioned conjugate gradients. Convergence is declared on the true
residual $\mathbf{r} = \mathbf{b} - \mathbf{A}\mathbf{x}$ when
$\|\mathbf{r}\|_2 \le \max(\mathrm{rtol}\,\|\mathbf{b}\|_2, \mathrm{atol})$.

| Parameter | Description |
|-----------|-------------|
| `comm` | `Communicator` used for the global reductions. |
| `fc` | Collection (`GlobalFieldCollection`, `LocalFieldCollection` or `CartesianDecomposition`) holding the solver's work fields. They stay registered after the solve, so the next solve on the same collection reuses them instead of allocating anew; their names start with `cg-work`. |
| `b` | Right-hand side field. Its dtype sets the working precision. |
| `x` | Initial guess; overwritten with the solution. |
| `hessp` | Callable `hessp(x, Ax)` writing $\mathbf{A}\mathbf{x}$ into `Ax`. |
| `prec` | Optional callable `prec(r, z)` computing $\mathbf{z} = \mathbf{M}\mathbf{r}$, $\mathbf{M}$ symmetric positive definite. Default: none. |
| `rtol` | Relative tolerance. Default `1e-6`. |
| `atol` | Absolute tolerance. Default `0.0`. |
| `tol` | Alias for `atol` that also sets `rtol = 0` unless `rtol` is given; emits a `DeprecationWarning`. |
| `maxiter` | Maximum number of iterations. Default `1000`. |
| `callback` | Called as `callback(iteration, state)` before the first and after every iteration; `state` has keys `"x"`, `"r"`, `"p"`, `"z"`, `"rr"` ($\mathbf{r}^\top\mathbf{r}$) and `"rz"` ($\mathbf{r}^\top\mathbf{z}$). |
| `timer` | Optional `muTimer.Timer` recording the time spent per operation. |
| `residual` | Optional field that receives the final residual $\mathbf{b} - \mathbf{A}\mathbf{x}$. |

Returns `x`. Raises `ConvergenceError` (a subclass of `RuntimeError`) if
`maxiter` is reached, if the true residual stagnates above the tolerance (the
attainable accuracy of the working precision), if the operator or
preconditioner is found not to be positive definite, or if a residual becomes
NaN or infinite. With a `float32` right-hand side, an `rtol` below
`muGrid.Solvers.FLOAT32_RTOL_FLOOR` (`1e-6`) triggers a `RuntimeWarning`.

Since the work fields remain in `fc`, a `register_field_collection` of that
collection without `field_names` would write them to the file as well; name
the fields you want written.

### conjugate_gradients_pipelined

```python
conjugate_gradients_pipelined(comm, fc, b, x, hessp, prec=None, tol=None,
                              maxiter=1000, callback=None, timer=None,
                              rtol=None, atol=0.0, replacement_interval=None,
                              residual=None)
```

Pipelined CG (Ghysels–Vanroose) with one non-blocking global reduction per
iteration, overlapped with the operator and preconditioner applications, and
periodic residual replacement every `replacement_interval` iterations.
Parameters, return value and exceptions are as for `conjugate_gradients`.

## Preconditioners

`muGrid.Preconditioners` provides preconditioners $\mathbf{M} \approx
\mathbf{A}^{-1}$ for the solvers. Each is a callable `prec(r, z)` (subclass of
`Preconditioner` with an `apply(r, z)` method) that can be passed as `prec`.
Below, $\mathcal{F}$ is the unnormalised forward FFT, $\mathcal{F}^{-1}$ its
inverse including the factor $1/\prod_\alpha N_\alpha$, and $q$ the frequency.

```python
from muGrid.Preconditioners import (
    IdentityPreconditioner, JacobiPreconditioner, FourierPreconditioner,
    BlockFourierPreconditioner, GreenJacobiPreconditioner,
    make_reference_stiffness_preconditioner, make_green_jacobi_preconditioner,
)
```

| Class / factory | Signature | Action |
|-----------------|-----------|--------|
| `IdentityPreconditioner` | `()` | $\mathbf{z} = \mathbf{r}$ (same as `prec=None`). |
| `JacobiPreconditioner` | `(diagonal, name="jacobi-preconditioner", communicator=None)` | $\mathbf{z} = D^{-1}\mathbf{r}$, $D = \operatorname{diag}(\mathbf{A})$. |
| `FourierPreconditioner` | `(engine, kernel, name="fourier-preconditioner", timer=None, dtype=np.float64)` | $\mathbf{z} = \mathcal{F}^{-1}[k(q)\,\mathcal{F}\mathbf{r}]$ with a scalar kernel $k$ per mode. |
| `BlockFourierPreconditioner` | `(engine, blocks, name="block-fourier-preconditioner", timer=None, dtype=np.float64)` | $\mathbf{z} = \mathcal{F}^{-1}[\hat K(q)^{-1}\,\mathcal{F}\mathbf{r}]$ with a $n \times n$ block per mode for an $n$-component field. |
| `GreenJacobiPreconditioner` | `(green, diagonal, void_tol=0.0, name="green-jacobi-preconditioner", timer=None, communicator=None)` | $\mathbf{z} = J^{1/2} G J^{1/2}\mathbf{r}$, combining a Green preconditioner $G$ with a Jacobi scaling $J$. |
| `make_reference_stiffness_preconditioner` | `(engine, apply_reference_stiffness=None, nb_components=None, ..., element=None, grid_spacing=None, lambda_ref=None, mu_ref=None, evaluate_symbol=None)` | Inverse of the uniform reference stiffness $\mathbf{K}^\mathrm{ref}$ by FFT (Green's-function preconditioner). |
| `make_green_jacobi_preconditioner` | `(engine, stiffness_op, lambda_field, mu_field, nb_components, reference_lambda=None, reference_mu=None, element=None, grid_spacing=None, void_tol=0.0, ...)` | Green-Jacobi preconditioner for the fused `IsotropicStiffnessOperator`. |

The module also contains `MultigridReferencePreconditioner`,
`HybridFourierTridiagonalPreconditioner` and `AnalyticReferencePreconditioner`,
alternative approximations of $(\mathbf{K}^\mathrm{ref})^{-1}$; their
docstrings give the full parameter lists. Usage is shown on the
[Examples](linalg.md#preconditioners) page.

## Linear algebra

`muGrid.linalg` provides ghost-aware BLAS-1 kernels on fields (`vecdot`,
`norm_sq`, `axpy`, `scal`, `axpby`, `copy`, `axpy_norm_sq`,
`pipelined_cg_dots`, `cross`, `leray_project`); see
[Linear algebra](linalg.md).

## Device selection

### Device strings

Every `device` argument accepts a `Device` or one of the strings

| String | Device |
|--------|--------|
| `"cpu"`, `"host"`, `None` | Host memory. |
| `"gpu"`, `"device"`, `"gpu:N"` | The compiled GPU backend (CUDA, else ROCm), device `N` (default 0). |
| `"cuda"`, `"cuda:N"` | CUDA device `N`. |
| `"rocm"`, `"rocm:N"` | ROCm device `N`. |

A GPU string raises `RuntimeError` if µGrid was built without GPU support.

### Device

```python
Device.cpu()                  # host
Device.cuda(device_id=0)      # CUDA GPU
Device.rocm(device_id=0)      # ROCm GPU
Device.gpu(device_id=0)       # compiled GPU backend: CUDA, else ROCm, else CPU
```

Read-only properties:

| Property | Description |
|----------|-------------|
| `is_host` | `True` for host memory. |
| `is_device` | `True` for device (GPU) memory. |
| `device_type` | The `DeviceType` value. |
| `device_id` | Device index (0 for the host). |

### DeviceType

Values follow the DLPack conventions:

| Value | Description |
|-------|-------------|
| `CPU` | Host (1). |
| `CUDA` | NVIDIA CUDA GPU (2). |
| `CUDAHost` | CUDA pinned host memory (3). |
| `ROCm` | AMD ROCm GPU (10). |
| `ROCmHost` | ROCm pinned host memory (11). |

### Device memory allocators

`use_cupy_allocator()`, `route_cupy_through_mugrid(pooled=True,
label="<cupy>")`, `restore_default_cupy_allocator()`,
`set_device_allocator(...)` and `clear_device_allocator()` control which
allocator serves device memory; see [GPU](gpu.md#device-memory-allocation).

## Enumerations

### IterUnit

Iteration unit: `Pixel` (iterate over pixels) or `SubPt` (iterate over
sub-points).

### StorageOrder

| Value | Description |
|-------|-------------|
| `ArrayOfStructures` | Components of a pixel consecutive in memory (host default). |
| `StructureOfArrays` | Pixels of a component consecutive in memory (device default). |
| `Automatic` | Inherit the storage order from the field collection. |

### FEMElement

`FEMElement.q1` (bilinear/trilinear) or `FEMElement.p1` (linear simplices),
selecting the element of the FE operators.

## Module constants and diagnostics

| Name | Meaning |
|------|---------|
| `muGrid.has_mpi` | Built with MPI. |
| `muGrid.has_cuda` | Built with CUDA. |
| `muGrid.has_rocm` | Built with ROCm/HIP. |
| `muGrid.has_gpu` | Built with any GPU backend. |
| `muGrid.has_netcdf` | Built with NetCDF I/O. |
| `muGrid.netcdf_backend` | `"PnetCDF"` (MPI) or `"NetCDF"` (serial); `None` without NetCDF. |
| `muGrid.netcdf_version` | NetCDF/PnetCDF library version string; `None` without NetCDF. |
| `muGrid.host_arch`, `muGrid.device_arch` | CPU and GPU architectures the library was compiled for. |
| `muGrid.__version__` | µGrid version string. |
| `muGrid.is_gpu_available()` | `True` if a GPU can actually be allocated on at run time (cached). |

```python
muGrid.version_string(communicator=None, device=None)
```

Returns a one-line summary of the build and the current run, e.g.

```text
muGrid 1.1.0 (MPI: ON, rank 0/4; CUDA: ON, device cuda:2; PnetCDF 1.14.1)
```

Pass the run's `communicator` to report rank and size, and the `Device` in use
to report the GPU device. Disabled features are shown as `OFF` (`MPI: OFF`,
`GPU: OFF`, `NetCDF: OFF`).

## Timer

The solvers and preconditioners accept a `timer` from the separate
[muTimer](https://pypi.org/project/muTimer/) package (`import muTimer`), which
the examples depend on. It is not part of µGrid.

```python
import muTimer
timer = muTimer.Timer()
with timer("outer"):
    with timer("inner"):
        ...
timer.print_summary()
```
