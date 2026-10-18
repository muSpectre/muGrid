# C++ API

The µGrid core lives in `src/libmugrid/` under the namespace `muGrid` and has
no dependency on Python. The source tree is organised by concern (`core/`,
`grid/`, `field/`, `collection/`, `memory/`, `mpi/`, `operators/`, `fft/`,
`linalg/`, `io/`); every subdirectory is on the public include path, so headers
are included without a directory prefix (`#include "field_collection_global.hh"`).

This page is a tour of the main classes and how they fit together. Member-level
documentation (every signature) is in the Doxygen comments of the headers. See
[Installation](installation.md) for build instructions and the
[Python API](python.md) for the bindings.

## Common type aliases

Numerical code uses the project's type aliases (`core/types.hh`,
`core/coordinates.hh`) rather than raw C++ types; these are the types every
data structure is instantiated and tested with.

- `muGrid::Real`, `muGrid::Complex`, `muGrid::Int`, `muGrid::Uint` — the scalar
  types fields store (`Real` is `double`, `Complex` is `std::complex<double>`).
  `Real32`/`Complex32` are the single-precision variants used by
  single-precision fields and operators.
- `muGrid::Index_t` (`std::ptrdiff_t`) — sizes and offsets. It is signed and
  matches Eigen's index type, and it is wide enough for arrays with more than
  $2^{31}$ entries (e.g. $65536 \times 65536$).
- `muGrid::Dim_t` (`int`) — a *signed* integer for dimensions; it can hold
  $-1$, which Eigen uses to denote a dynamic size.
- `muGrid::GridIndex<Dim>` and `muGrid::GridPoint<Dim>` — integer pixel
  coordinates / grid sizes and real-valued coordinates with compile-time
  dimension (`std::array<Index_t, Dim>`, `std::array<Real, Dim>`).
- `muGrid::DynGridIndex` (alias `GridShape`) — integer coordinates with a
  dimension chosen at run time (up to 4), used where templating on the
  dimension is impossible, e.g. in the Python bindings.
- `muGrid::IterUnit` and `muGrid::StorageOrder` — enumerations controlling
  iteration units and in-memory storage order.

## The field abstraction

A *field* is the discretisation of a mathematical field on grid points:
numerical data associated with all pixels/voxels of a grid (or a subset
thereof). Field values can be scalar, vectorial, matricial, tensorial or a
generic array of integer, real or complex values per pixel.

Fields defined on every pixel/voxel are **global fields**; fields defined on a
subset of pixels/voxels are **local fields**. For example, the strain field is
global (it exists in the whole domain), whereas an internal state variable of a
material in a composite is local (only defined on the pixels of that material).

The same field is often viewed in different ways by different parts of a
problem. Consider a global strain field in a 3D finite-strain problem with
$255 \times 255 \times 255$ voxels: the solver treats it as a vector of length
$3^2 \cdot 255^3$, the FFT sees a four-dimensional array of shape
$255 \times 255 \times 255 \times 3^2$, and the constitutive laws see a
sequence of second-rank tensors (shape $255^3 \times 3 \times 3$). To reconcile
these interpretations without copying data, µGrid splits a field into three
parts:

- **Storage — the field.** Owns the memory. It knows the scalar type, the
  number of pixels/voxels, the number of sub-points per pixel and the number of
  scalar components per sub-point (e.g. 9 for an asymmetric second-rank tensor
  in 3D).
- **Representation — the field map.** Defines how to interpret per-pixel data
  (vector, matrix, …), which mathematical operations are available, and allows
  iterating pixel by pixel or sub-point by sub-point.
- **Access / management — the field collection.** Maps a pixel/voxel coordinate
  or index to the position of the associated data. Because this mapping is
  identical for every field on the same domain, it is held once by the
  collection.

## Field collections

Field collections own a group of fields on a structured grid; all fields in a
collection share the same spatial discretisation and sub-point declarations.

- `muGrid::FieldCollection` — common base class.
- `muGrid::GlobalFieldCollection` — defined on the whole (local subdomain of
  the) grid; constructed from the global grid size `nb_domain_grid_pts`,
  optionally with the subdomain size and location, the sub-point map, storage
  order, ghost layers and `Device`.
- `muGrid::LocalFieldCollection` — defined on a subset of pixels, added one at
  a time with `add_pixel(global_index)` and finalised with `initialise()`.

Fields are identified by a unique name. `register_real_field`,
`register_complex_field`, `register_int_field`, `register_uint_field` (and
`register_real_state_field` etc. for state fields) create a field and fail if
the name exists; `real_field(...)` etc. get or create; `get_field(name)` and
`get_state_field(prefix)` retrieve existing ones, and `field_exists(name)`
tests for them.

```cpp
#include "field_collection_global.hh"
#include "field_typed.hh"

// A 2D collection on a 255 x 255 grid, host memory
muGrid::GlobalFieldCollection collection{muGrid::DynGridIndex{255, 255}};

// A real-valued field with 4 components per pixel (a 2x2 tensor)
muGrid::Field & strain{collection.register_real_field("strain", 4)};
```

## Fields

Fields store data, so they are distinguished by their scalar type (`Int`,
`Uint`, `Real`, `Complex`, and the 32-bit variants), their memory space and
their component shape.

Fields have protected constructors: they are created only through a
collection's `register_*` methods, which return a reference while the
collection keeps the field in a `std::unique_ptr`. Fields therefore cannot be
copied or freed while field maps refer to them.

- `muGrid::Field` — abstract base for all fields (name, shape, sub-points,
  storage order, device).
- `muGrid::TypedFieldBase<T, MemorySpace>` and
  `muGrid::TypedField<T, MemorySpace>` — fields of a concrete scalar type in a
  memory space (`HostSpace`, `CUDASpace`, `ROCmSpace`). Aliases: `RealField`,
  `ComplexField`, `IntField`, `UintField` (host) and `RealFieldDevice` etc.
  (default device space).
- `muGrid::StateField` / `muGrid::TypedStateField<T>` — a current value plus a
  fixed number of old values (see [State fields](#state-fields)).

Host fields expose their memory as Eigen maps (`eigen_vec()`, `eigen_mat()`,
`eigen_sub_pt()`, `eigen_pixel()`), which is convenient for solver-level
vector operations.

## Field maps

Field maps are light-weight handles (cheap to create and destroy) that are
iterable and give per-pixel or per-sub-point access to a field's data. The map
type defines what dereferencing an iterator or `operator[]` returns.

Statically sized maps (`field_map_static.hh`) know the iterate size at compile
time, so their iterates are fixed-size Eigen objects and linear algebra on them
is vectorised:

- `muGrid::ScalarFieldMap<T, Mutability, IterUnit>`
- `muGrid::ArrayFieldMap<T, Mutability, NbRow, NbCol, IterUnit>` and
  `muGrid::MatrixFieldMap<T, Mutability, NbRow, NbCol, IterUnit>`
- `muGrid::T1FieldMap`, `T2FieldMap`, `T4FieldMap` — first-, second- and
  fourth-rank tensor maps of a given spatial dimension.

`Mutability` is `muGrid::Mapping::Const` or `muGrid::Mapping::Mut`; `IterUnit`
is `IterUnit::Pixel` or `IterUnit::SubPt`. The dynamically sized
`muGrid::FieldMap<T, Mutability>` supports the same operations with run-time
iterate size; linear algebra on its iterates cannot be specialised for a fixed
size and is therefore slower.

```cpp
#include "field_map_static.hh"

muGrid::MatrixFieldMap<muGrid::Real, muGrid::Mapping::Mut, 2, 2,
                       muGrid::IterUnit::Pixel>
    strain_map{strain};
for (auto && tensor : strain_map) {
    // `tensor` is an Eigen 2x2 map referencing this pixel's data
}
```

## State fields

Internal variables of inelastic materials (damage, plastic strain, …) need a
current value plus one or more old values. A **state field** holds this history
in one object: `current()` gives read/write access to the current value and
`old(n)` read-only access to the value $n$ steps ago; `cycle()` advances the
history. **State field maps** (`MatrixStateFieldMap`, `T2StateFieldMap`, …)
iterate over current and old values together. State fields are created with
`register_state_field<T>(prefix, nb_memory, nb_components)` or the typed
`register_real_state_field` etc.

## Mapped fields

When a field is used by a single owner (e.g. a material's internal variable),
a **mapped field** bundles the field and its map into one object
(`MappedScalarField`, `MappedMatrixField`, `MappedT2Field`, `MappedT4Field`, …,
and `MappedStateField` variants), removing the bookkeeping of keeping the two
in sync.

## Domain decomposition and ghosts

- `muGrid::Decomposition` — abstract base for a domain decomposition.
- `muGrid::CartesianDecomposition` — splits the global grid across MPI ranks
  into a Cartesian process grid and owns the ghosted `GlobalFieldCollection` of
  the local subdomain. `communicate_ghosts(field)` fills ghost layers from
  their owners (periodic wrap-around included); `reduce_ghosts(field)` is its
  exact adjoint (adds ghost values to their owners, then zeros the ghosts).
- `muGrid::Communicator` — wrapper over an MPI communicator for collective and
  point-to-point operations; without `WITH_MPI` it compiles to a serial
  implementation with the same interface.

## FFT engine

- `muGrid::FFTEngineBase` — non-templated base holding the slab/pencil
  decomposition, the real- and Fourier-space collections and the transposes.
  It derives from `CartesianDecomposition`, so the real-space collection is the
  ghosted domain decomposition.
- `muGrid::FFTEngine<MemorySpace>` — performs the forward (`fft`) and inverse
  (`ifft`) transforms through an `FFT1DBackend` (PocketFFT on the host,
  cuFFT/rocFFT on the device).

Frequency helpers (`fft_utils.hh`): `muGrid::fft_freq`, `muGrid::fft_freqind`,
`muGrid::rfft_freq`, `muGrid::rfft_freqind`, `muGrid::fft_normalization`,
`muGrid::get_hermitian_grid_pts`. See [FFT](fft.md) for usage and
[Architecture](architecture.md#fft-engine-design) for the internal design.

## Operators

Stencil operators derive from `muGrid::LinearOperator` (`apply`,
`apply_increment`, `transpose`, `transpose_increment`, and the ghost
requirements `get_apply_ghost_requirement()`,
`get_transpose_ghost_requirement()`, `get_ghost_requirement()`):

- `muGrid::GenericLinearOperator` — arbitrary convolution stencil.
- `muGrid::LaplaceOperator<Dim>` (aliases `LaplaceOperator2D`,
  `LaplaceOperator3D`) — $(2d+1)$-point finite-difference Laplacian.
- `muGrid::FEMGradientOperator<Element, T>` — finite-element gradient; aliases
  `FEMGradientOperatorQ1_2D`, `FEMGradientOperatorQ1_3D` (Q1) and
  `FEMGradientOperator2D`, `FEMGradientOperator3D` (P1), with `_32` suffixes for
  single precision. Element tables live in `fem_element.hh`.
- `muGrid::IsotropicStiffnessOperator<Dim>` (aliases
  `IsotropicStiffnessOperator2D`, `IsotropicStiffnessOperator3D`) — fused
  isotropic linear-elastic stiffness, a `MaterialOperator` taking per-pixel
  Lamé fields.

See [Operators](operators.md) for definitions and usage and
[Architecture](architecture.md#operator-kernel-design) for the kernel design.

## File I/O

- `muGrid::FileIOBase` — abstract base for file I/O backends (frames, open
  modes).
- `muGrid::FileIONetCDF` — NetCDF reader/writer for field collections and
  per-frame variables (Unidata NetCDF in serial builds, PnetCDF under MPI).

## Grid utilities

Conversions between pixel coordinates and linear indices are in
`grid/index_ops.hh` (namespace `muGrid::CcoordOps`), e.g.
`get_coord(nb_grid_pts, locations, index)` and
`get_index(nb_grid_pts, locations, coord)`. Pixel iteration and strides are in
`grid/pixels.hh` and `grid/strides.hh`.

## Device / memory spaces

`memory/` provides `muGrid::Device`, the memory spaces `HostSpace`,
`CUDASpace` and `ROCmSpace`, and the device allocators. See [GPU](gpu.md) for
the device-side workflow.
