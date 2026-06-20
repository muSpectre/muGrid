# Architecture

This page describes the internal design of key µGrid components: the distributed
FFT engine and the discrete differential operators.

## FFT Engine Design

The FFT engine provides distributed Fast Fourier Transform operations on
structured grids with MPI parallelization. It uses pencil decomposition for
efficient scaling to large numbers of MPI ranks.

### Storage Order

µGrid supports two storage orders for multi-component fields:

- **AoS (Array of Structures)** — components interleaved per pixel,
  `[c0p0, c1p0, c2p0, c0p1, c1p1, c2p1, ...]`. Default on **CPU**
  (`HostSpace`). X-element stride = `nb_components`; component offset = `comp`.
- **SoA (Structure of Arrays)** — components in separate contiguous blocks,
  `[c0p0, c0p1, c0p2, ..., c1p0, c1p1, c1p2, ...]`. Default on **GPU**
  (`CUDASpace`, `ROCmSpace`). X-element stride = 1; component offset =
  `comp * nb_pixels`.

**Why two orders?** On the GPU, coalesced memory access is critical: SoA lets
threads processing different pixels of the same component touch contiguous
memory, so the hardware combines accesses into one transaction. On the CPU, AoS
gives cache locality for multi-component operations — all components of a pixel
share a cache line.

!!! note
    FFT work buffers use the same storage order as the memory space they live
    in (SoA on GPU, AoS on CPU). This avoids expensive storage-order conversions
    during FFT operations. The engine detects the order via
    `field.get_storage_order()` and computes the appropriate strides per buffer.

### Stride Calculations

For a 2D field `[Nx, Ny]` with `nb_components` components:

```text
SoA:                                  AoS:
  comp_offset_factor = nb_pixels        comp_offset_factor = 1
  x_stride           = 1                x_stride           = nb_components
  row_dist           = row_width        row_dist           = row_width * nb_components
```

For a 3D field `[Nx, Ny, Nz]`:

```text
SoA:                                  AoS:
  comp_offset_factor = nb_pixels        comp_offset_factor = 1
  x_stride           = 1                x_stride           = nb_components
  y_dist             = row_x            y_dist             = row_x * nb_components
  z_dist             = row_x * rows_y   z_dist             = row_x * rows_y * nb_components
```

### GPU Backend Limitations

!!! warning
    cuFFT does not support **strided real-to-complex (R2C) and complex-to-real
    (C2R) transforms** — the stride on the real-data side must be 1. With AoS
    storage the stride between consecutive real values would be `nb_components`.

**Solution**: With SoA on GPU each component is contiguous (stride = 1),
satisfying cuFFT's requirement. The engine loops over components, running one
batched FFT per component.

rocFFT's native API (`rocfft_plan_description_set_data_layout()`) does support
arbitrary input and output strides, but for consistency and to avoid storage
order conversions, µGrid uses the same SoA approach on AMD GPUs.

### MPI Parallel FFT Design

For 3D distributed FFT the engine uses **pencil decomposition**:

1. **Z-pencil**: data distributed in Y and Z, FFT along X (r2c)
2. **Y-pencil**: transpose Y↔Z, FFT along Y (c2c)
3. **X-pencil**: transpose X↔Z, FFT along Z (c2c)

The `Transpose` class handles the MPI all-to-all communication that
redistributes data between pencil orientations.

!!! warning
    Transpose operations currently assume AoS layout. With GPU memory this may
    require a storage-order conversion or an update to the `Transpose` class to
    handle SoA.

!!! note
    The FFT module is designed so MPI communication needs no explicit packing
    and unpacking — data is laid out so that `MPI_Alltoall` operates directly on
    contiguous memory regions.

### Forward FFT Algorithms

**2D:**

```text
1. r2c FFT along X for each component
   - Input: real field (with ghosts) -> work buffer (half-complex, no ghosts)
   - Loop over components, batched over Y rows
2. [MPI only] Transpose X<->Y (Y-local to X-local)
3. c2c FFT along Y for each component
   - In-place on work buffer (serial) or output (MPI), batched over X
4. [Serial only] Copy work to output (same storage order, direct copy)
```

**3D:**

```text
1. r2c FFT along X for each component
   - Input: real field -> Z-pencil work buffer (half-complex in X)
   - Batched over Y rows for each Z plane
2a. [MPI] Transpose Y<->Z (Z-pencil to Y-pencil)
2b. c2c FFT along Y for each component (Y-pencil; transforms per (X,Z))
2c. [MPI] Transpose Z<->Y (Y-pencil back to Z-pencil)
3.  [MPI] Transpose X<->Z, or copy (Z-pencil to X-pencil output)
4.  c2c FFT along Z for each component (output buffer; transforms per (X,Y))
5.  [Serial only] Copy work to output
```

### Per-Component Looping

Rather than batching across components (which would need non-unit strides on
GPU), the engine loops over components:

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

This satisfies cuFFT's unit-stride R2C/C2R requirement, allows efficient
batching within each component, works for both SoA and AoS, and adds minimal
overhead (one kernel launch per component).

!!! note
    Like FFTW and cuFFT, the transforms are **unnormalized**: a forward FFT
    followed by an inverse FFT multiplies by N (the transform size). Users must
    normalize explicitly if needed.

### File Structure

```text
src/libmugrid/fft/
├── fft_engine.hh         # Template class FFTEngine<MemorySpace>
├── fft_engine_base.hh/cc # Base class: field collections and transpose management
├── fft_backend.hh        # Abstract backend interface (1D primitives + ND)
├── pocketfft_backend.cc  # CPU backend using pocketfft
├── cufft_backend.cc      # NVIDIA GPU backend using cuFFT
├── rocfft_backend.cc     # AMD GPU backend using rocFFT
└── transpose.hh/cc       # MPI transpose operations
```

## Operator Kernel Design

µGrid provides several discrete differential operators optimized for structured
grids, with common design principles for ghost-region handling and periodicity.

| Operator | Description | Stencil Size | Input → Output |
|----------|-------------|--------------|----------------|
| `GenericLinearOperator` | Flexible stencil-based convolution | Configurable | nodal → quad or nodal → nodal |
| `FEMGradientOperator` | FEM gradient | 8 nodes (3D), 4 nodes (2D) | nodal → quad |
| `LaplaceOperator` | 7-point (3D) / 5-point (2D) Laplacian | 7 (3D), 5 (2D) | nodal → nodal |
| `IsotropicStiffnessOperator` | Fused elasticity kernel | 8 nodes (3D), 4 nodes (2D) | nodal → nodal |

### Common Design Principles

**Ghost region handling.** Operators make no assumption about boundary
conditions and operate only on an interior region. Distributed-memory
parallelization and periodic BCs are realized through ghost-cell communication:

- Ghost regions are filled by `CartesianDecomposition::communicate_ghosts()`
  before the operator runs.
- **Periodic BC**: ghost regions hold copies from the opposite boundary.
- **Non-periodic BC with Dirichlet**: boundary node values are constrained;
  their forces are not computed by the fused kernels (they would be overwritten
  anyway).
- Operators compute forces for **interior nodes only** (e.g. indices 1 to n-2
  for a ghost buffer of width 1), never boundary nodes (0 and n-1).

**Stencil offset convention.** Each operator uses a stencil offset to define
where it "centers", determining which element's quadrature points correspond to
which nodal region:

```python
# FEMGradientOperator with offset [0, 0, 0]:
# Output element (0,0,0) uses input nodes at (0,0,0), (1,0,0), (0,1,0), etc.
grad_op = muGrid.FEMGradientOperator(3, grid_spacing)

# GenericLinearOperator with custom offset
conv_op = muGrid.GenericLinearOperator([0, 0, 0], coefficients)
```

**Upfront validation.** Ghost-region requirements are validated **once** before
kernel execution: operators check sufficient ghost cells exist, invalid
configurations throw clear errors, and there is no bounds checking inside the
hot loop.

**Memory layout.** CPU operators use AoS, GPU operators use SoA:

- **CPU**: DOF components contiguous, `[ux0, uy0, uz0, ux1, uy1, uz1, ...]`
- **GPU**: spatial indices contiguous, `[ux0, ux1, ..., uy0, uy1, ..., uz0, uz1, ...]`

Kernel implementations may differ between CPU and GPU to suit each architecture.

### GenericLinearOperator

A flexible stencil-based convolution that can represent any linear differential
operator as a sparse stencil. It takes a list of coefficient arrays (one per
output component per stencil point), with the stencil defined by relative
offsets from the current position, and supports both forward (apply) and adjoint
(transpose) operations.

The forward and transpose operations have different ghost requirements because
the stencil direction is reversed:

- **Forward (apply)** reads at `p + s` (stencil offset s):
  - Left ghosts: `max(-offset, 0)` per dimension
  - Right ghosts: `max(stencil_shape - 1 + offset, 0)` per dimension
- **Transpose** reads at `p - s` (reversed direction):
  - Left ghosts: `max(stencil_shape - 1 + offset, 0)` per dimension (swapped)
  - Right ghosts: `max(-offset, 0)` per dimension (swapped)

For a stencil with offset `[0, 0, 0]` and shape `[2, 2, 2]`: apply needs
left=0, right=1 (reads ahead); transpose needs left=1, right=0 (gathers from
behind).

```python
# Create from FEMGradientOperator coefficients
grad_op = muGrid.FEMGradientOperator(3, grid_spacing)
conv_op = muGrid.GenericLinearOperator([0, 0, 0], grad_op.coefficients)

# Apply gradient
conv_op.apply(nodal_field, quadrature_field)

# Apply divergence (transpose)
conv_op.transpose(quadrature_field, nodal_field, quad_weights)
```

### FEMGradientOperator

Computes gradients at quadrature points from nodal displacements using linear
finite element (P1) shape functions. In 2D each pixel is split into 2 triangles
(2 quadrature points); in 3D each voxel into 5 tetrahedra (Kuhn triangulation,
5 quadrature points).

The gradient at quadrature point q is:

$$ \frac{\partial u}{\partial x_i} = \sum_I B_q[i,I] \times u_I / h_i $$

where \(B_q\) contains shape-function gradients for the element containing q.

**Shape-function gradients:**

```text
2D, B_2D[triangle][dim][node]:        3D, B_3D[tet][dim][node]:
  Triangle 0: nodes 0,1,2 (lower-left)  Tet 0: central tetrahedron (weight 1/3)
  Triangle 1: nodes 1,2,3 (upper-right) Tets 1-4: corner tetrahedra (weight 1/6 each)
```

**Ghost requirements:**

- **Forward (gradient, nodal → quadrature)**: input nodal field needs 1 ghost
  cell on the **right** in each dimension; output quadrature field needs none.
  Iterates over elements [0, n-2], reading nodal values at element corners
  [ix, ix+1] × [iy, iy+1] × [iz, iz+1].
- **Transpose (divergence, quadrature → nodal)**: input quadrature/stress field
  needs 1 ghost cell on the **left** in each dimension (periodic BC); output
  nodal field has no specific requirement. Iterates over elements [0, n-2],
  accumulating into nodal values at element corners.

The swap arises because forward reads "ahead" (node ix+1 at element ix → right
ghosts), while transpose gathers from elements "behind" (element ix-1
contributes to node ix → left ghosts for periodic wraparound).

### LaplaceOperator

An optimized discrete Laplacian for Poisson-type problems, using the standard
finite-difference form:

$$ \nabla^2 u \approx \frac{u[i+1] + u[i-1] - 2u[i]}{h^2} + \ldots $$

```text
2D (5-point):        3D (7-point):
     1                 Center: -6
   1 -4 1              Neighbors: +1 (x6)
     1
```

It needs 1 ghost cell in each direction (to access neighbors). The operator is
self-adjoint (transpose equals forward apply), negative semi-definite (positive
semi-definite with `-scale`), and has a configurable scale factor for grid
spacing and sign conventions.

```python
# Laplacian with scale = -1/h^2 for positive-definite form
laplace = muGrid.LaplaceOperator3D(scale=-1.0 / h**2)
laplace.apply(u, laplacian_u)
```

### IsotropicStiffnessOperator

A fused kernel computing `force = K @ displacement` for isotropic linear elastic
materials, avoiding explicit assembly of the stiffness matrix K. The element
stiffness matrix decomposes as:

$$ K_e = 2\mu G + \lambda V $$

where:

- \(G = \sum_q w_q B_q^T I' B_q\) — geometry matrix (shear stiffness)
- \(V = \sum_q w_q (B_q^T m)(m^T B_q)\) — volumetric coupling matrix
- \(I' = \text{diag}(1, 1, 1, 0.5, 0.5, 0.5)\) — Voigt scaling for strain energy
- \(m = [1, 1, 1, 0, 0, 0]^T\) — trace selector vector

This reduces memory from O(N × DOF²) for full K to O(N × 2) for spatially
varying materials.

**Element decomposition:** 2D uses 2 triangles per pixel (quadrature weights
`[0.5, 0.5]`); 3D uses 5 tetrahedra per voxel (Kuhn triangulation, weights
`[1/3, 1/6, 1/6, 1/6, 1/6]`).

All fields use **node-based indexing** with the same grid dimensions:

| Field | Dimensions | Left Ghosts | Right Ghosts |
|-------|------------|-------------|--------------|
| Displacement | (nx, ny, nz) | 1 | 1 |
| Force | (nx, ny, nz) | 1 | 1 |
| Material (lam, mu) | (nx, ny, nz) | 1 | 1 |

!!! note
    The kernel does not distinguish periodic from non-periodic BCs. Ghost-cell
    communication (`CartesianDecomposition::communicate_ghosts()`) handles
    periodicity: periodic BCs put copies from the opposite boundary into the
    ghost cells, non-periodic BCs fill them with appropriate boundary values.
    This unified approach simplifies the kernel and yields consistent CPU/GPU
    behavior.

**Gather pattern.** The kernel iterates over interior nodes and gathers
contributions from all neighboring elements:

```text
2D (4 elements per node):           3D (8 elements per node):
  (-1,-1) -> local node 3             (-1,-1,-1) -> 7   (0,-1,-1) -> 6
  ( 0,-1) -> local node 2             (-1, 0,-1) -> 5   (0, 0,-1) -> 4
  (-1, 0) -> local node 1             (-1,-1, 0) -> 3   (0,-1, 0) -> 2
  ( 0, 0) -> local node 0             (-1, 0, 0) -> 1   (0, 0, 0) -> 0
```

The kernel iterates over all nodes (indices 0 to n-1), reading from ghost cells
at positions -1 (left) and n (right), which must be populated beforehand:

```cpp
// Iterate over all nodes - ghost communication handles periodicity
for (ix = 0; ix < nnx; ix++):
    for (iy = 0; iy < nny; iy++):
        // Compute force at node (ix, iy)
```

**Kernel pseudocode:**

```cpp
for each node (ix, iy, iz) in [0, nnx) x [0, nny) x [0, nnz):
    f = [0, 0, 0]  // force accumulator

    for each neighboring element (8 in 3D):
        // Element indices can be -1 (left ghost) or nx (right ghost)
        ex, ey, ez = element indices relative to node
        local_node = which corner of element is this node

        // Material fields are node-indexed, read from element position
        lam = material_lambda[ex, ey, ez]
        mu  = material_mu[ex, ey, ez]

        // Gather displacement from all element nodes (may read ghosts)
        u = [displacement at each element corner node]

        // Compute stiffness contribution
        for each DOF d:
            row = local_node * NB_DOFS + d
            f[d] += sum_j (2*mu * G[row,j] + lam * V[row,j]) * u[j]

    force[ix, iy, iz] = f
```

### Memory Layout Details

**CPU (AoS)** — DOF components contiguous:

```text
Memory: [ux0, uy0, uz0, ux1, uy1, uz1, ux2, uy2, uz2, ...]
```

```cpp
disp_stride_d = 1;                    // DOF components contiguous
disp_stride_x = NB_DOFS;              // = 3 for 3D
disp_stride_y = NB_DOFS * nx;
disp_stride_z = NB_DOFS * nx * ny;
```

**GPU (SoA)** — spatial indices contiguous for coalesced access:

```text
Memory: [ux0, ux1, ux2, ..., uy0, uy1, uy2, ..., uz0, uz1, uz2, ...]
```

```cpp
disp_stride_x = 1;                    // Spatial x contiguous
disp_stride_y = nx;
disp_stride_z = nx * ny;
disp_stride_d = nx * ny * nz;         // DOF components separated
```

Pointers are offset to account for ghost cells:

```cpp
const Real* disp_data = displacement.data() + ghost_offset;
// Now index 0 = first interior node, index -1 = left ghost
```

### File Structure

```text
src/libmugrid/operators/
├── linear.hh              # Base LinearOperator class
├── generic.hh/.cc         # GenericLinearOperator
├── laplace_2d.hh/.cc      # LaplaceOperator2D
├── laplace_3d.hh/.cc      # LaplaceOperator3D
├── fem_gradient_2d.hh/.cc # FEMGradientOperator2D
├── fem_gradient_3d.hh/.cc # FEMGradientOperator3D
└── solids/
    ├── isotropic_stiffness_2d.hh/.cc  # IsotropicStiffnessOperator2D
    ├── isotropic_stiffness_3d.hh/.cc  # IsotropicStiffnessOperator3D
    └── isotropic_stiffness_gpu.cc     # GPU kernels
```

### Testing

Tests live in `tests/python_isotropic_stiffness_operator_tests.py`:

- `test_compare_with_generic_operator`: fused kernel vs. explicit Bᵀ C B
- `test_unit_impulse_*`: response to unit displacement at specific nodes
- `test_symmetry`: verifies K is symmetric
- `ValidationGuardTest*`: error handling for invalid configurations
- `GPUUnitImpulseTest`: GPU kernel matches CPU output

With node-based indexing, all tests compare the full output (all nodes), not
just interior nodes.
