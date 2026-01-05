# Operator Kernel Design

This document describes the design of the discrete differential operators in µGrid, including their common design principles for ghost region handling and periodicity.

## Operator Overview

µGrid provides several discrete differential operators optimized for structured grids:

| Operator | Description | Stencil Size | Input → Output |
|----------|-------------|--------------|----------------|
| `GenericLinearOperator` | Flexible stencil-based convolution | Configurable | nodal → quad or nodal → nodal |
| `FEMGradientOperator` | FEM gradient (nodal → quadrature) | 8 nodes (3D), 4 nodes (2D) | nodal → quad |
| `LaplaceOperator` | 7-point (3D) or 5-point (2D) Laplacian | 7 (3D), 5 (2D) | nodal → nodal |
| `IsotropicStiffnessOperator` | Fused elasticity kernel | 8 nodes (3D), 4 nodes (2D) | nodal → nodal |

## Common Design Principles

All operators follow these design principles for consistency and correctness:

### 1. Ghost Region Handling

All operators make no assumption over boundary conditions and operator on a interior region only. Distributed-memory parallelization and periodic boundary conditions are implemented through ghost cell communication:

- **Ghost regions** are filled by `CartesianDecomposition::communicate_ghosts()` before the operator is called
- For **periodic BC**: ghost regions contain copies from the opposite boundary
- For **non-periodic BC** with Dirichlet conditions: boundary node values are constrained, and their forces are not computed by the fused kernels (they would be overwritten anyway)
- **Interior node computation only**: For all BCs, operators compute forces only for interior nodes (e.g. indices 1 to n-2, assuming a ghost buffer of with 1), not boundary nodes (indices 0 and n-1)

### 2. Stencil Offset Convention

All operators use a stencil offset to define where the operator "centers":

```python
# Example: FEMGradientOperator with offset [0, 0, 0]
# Output element (0,0,0) uses input nodes at (0,0,0), (1,0,0), (0,1,0), etc.
grad_op = muGrid.FEMGradientOperator(3, grid_spacing)

# Example: GenericLinearOperator with custom offset
conv_op = muGrid.GenericLinearOperator([0, 0, 0], coefficients)
```

The offset determines which element's quadrature points correspond to which nodal region.

### 3. Upfront Validation

Ghost region requirements are validated **once** before kernel execution, not during iteration:

- Operators check that sufficient ghost cells exist in the field collection
- Invalid configurations throw clear error messages
- No bounds checking inside the hot loop

### 4. Memory Layout

CPU operators use AoS, and GPU operators use SoA memory layouts:

- CPU: DOF components contiguous `[ux0, uy0, uz0, ux1, uy1, uz1, ...]`
- GPU: Spatial indices contiguous `[ux0, ux1, ..., uy0, uy1, ..., uz0, uz1, ...]`

The kernel implementations may differ between CPU and GPU as they may be optimized for the respective target architecture.

---

## GenericLinearOperator

### Purpose
Flexible stencil-based convolution operator that can represent any linear differential operator as a sparse stencil.

### Design
- Takes a list of coefficient arrays, one per output component per stencil point
- Stencil defined by relative offsets from the current position
- Supports both forward (apply) and adjoint (transpose) operations

### Ghost Requirements

The forward and transpose operations have different ghost requirements because the stencil direction is reversed:

**Forward (`apply`)**: reads at position `p + s` (stencil offset s)
- Left ghosts: `max(-offset, 0)` in each dimension
- Right ghosts: `max(stencil_shape - 1 + offset, 0)` in each dimension

**Transpose**: reads at position `p - s` (reversed stencil direction)
- Left ghosts: `max(stencil_shape - 1 + offset, 0)` in each dimension (swapped from apply's right)
- Right ghosts: `max(-offset, 0)` in each dimension (swapped from apply's left)

**Example**: For a stencil with offset `[0, 0, 0]` and shape `[2, 2, 2]`:
- Apply: left=0, right=1 (reads ahead)
- Transpose: left=1, right=0 (gathers from behind)

### Usage
```python
# Create from FEMGradientOperator coefficients
grad_op = muGrid.FEMGradientOperator(3, grid_spacing)
conv_op = muGrid.GenericLinearOperator([0, 0, 0], grad_op.coefficients)

# Apply gradient
conv_op.apply(nodal_field, quadrature_field)

# Apply divergence (transpose)
conv_op.transpose(quadrature_field, nodal_field, quad_weights)
```

---

## FEMGradientOperator

### Purpose
Computes gradients at quadrature points from nodal displacements using linear finite element (P1) shape functions.

### Mathematical Formulation

For linear FEM on structured grids:
- **2D**: Each pixel decomposed into 2 triangles, 2 quadrature points
- **3D**: Each voxel decomposed into 5 tetrahedra (Kuhn triangulation), 5 quadrature points

The gradient at quadrature point q is:
```
∂u/∂x_i = Σ_I B_q[i,I] × u_I / h_i
```
where `B_q` contains shape function gradients for the element containing q.

### Shape Function Gradients

**2D (2 triangles per pixel):**
```
B_2D[triangle][dim][node]
Triangle 0: nodes 0,1,2 (lower-left)
Triangle 1: nodes 1,2,3 (upper-right)
```

**3D (5 tetrahedra per voxel):**
```
B_3D[tet][dim][node]
Tet 0: central tetrahedron (weight 1/3)
Tets 1-4: corner tetrahedra (weight 1/6 each)
```

### Ghost Requirements

The forward and transpose operations have different ghost requirements because the stencil direction is reversed:

**Forward (`apply`): Gradient - nodal → quadrature**
- Input (nodal field): 1 ghost cell on **right** in each dimension
- Output (quadrature field): no ghosts needed
- Iteration: over elements [0, n-2], reads nodal values at element corners [ix, ix+1] × [iy, iy+1] × [iz, iz+1]

**Transpose (`transpose`): Divergence - quadrature → nodal**
- Input (quadrature/stress field): 1 ghost cell on **left** in each dimension (for periodic BC)
- Output (nodal field): no specific ghost requirement
- Iteration: over elements [0, n-2], accumulates contributions to nodal values at element corners

The ghost requirement swap occurs because:
- Forward reads "ahead" (node ix+1 when at element ix) → needs right ghosts
- Transpose gathers from elements "behind" (element ix-1 contributes to node ix) → needs left ghosts for periodic wraparound

### Methods

| Method | Description |
|--------|-------------|
| `apply(u, grad_u)` | Gradient: nodal → quadrature |
| `transpose(stress, force, weights)` | Divergence: quadrature → nodal |

The transpose applies quadrature weights and computes the discretized divergence.

---

## LaplaceOperator

### Purpose
Optimized discrete Laplacian for Poisson-type problems.

### Mathematical Formulation

Standard finite difference Laplacian:
```
∇²u ≈ (u[i+1] + u[i-1] - 2u[i])/h² + ...
```

**2D (5-point stencil):**
```
     1
   1 -4 1
     1
```

**3D (7-point stencil):**
```
Center: -6, Neighbors: +1 (×6)
```

### Ghost Requirements
- 1 ghost cell in each direction (for accessing neighbors)

### Properties
- Self-adjoint: transpose equals forward apply
- Negative semi-definite (positive semi-definite with `-scale`)
- Configurable scale factor for grid spacing and sign conventions

### Usage
```python
# Create Laplacian with scale = -1/h² for positive-definite form
laplace = muGrid.LaplaceOperator3D(scale=-1.0 / h**2)
laplace.apply(u, laplacian_u)
```

---

## IsotropicStiffnessOperator

### Purpose
Fused kernel computing `force = K @ displacement` for isotropic linear elastic materials, avoiding explicit assembly of the stiffness matrix K.

### Mathematical Formulation

For isotropic materials, the element stiffness matrix decomposes as:
```
K_elem = 2μ G + λ V
```
where:
- `G = Σ_q w_q B_q^T I' B_q` - geometry matrix (shear stiffness)
- `V = Σ_q w_q (B_q^T m)(m^T B_q)` - volumetric coupling matrix
- `I' = diag(1, 1, 1, 0.5, 0.5, 0.5)` - Voigt scaling for strain energy
- `m = [1, 1, 1, 0, 0, 0]^T` - trace selector vector

This reduces memory from O(N × DOF²) for full K to O(N × 2) for spatially-varying materials.

### Element Decomposition

**2D:** 2 triangles per pixel
- Quadrature weights: `[0.5, 0.5]`

**3D:** 5 tetrahedra per voxel (Kuhn triangulation)
- Quadrature weights: `[1/3, 1/6, 1/6, 1/6, 1/6]`

### Field Requirements

All fields use **node-based indexing** with the same grid dimensions:

| Field | Dimensions | Left Ghosts | Right Ghosts |
|-------|------------|-------------|--------------|
| Displacement | (nx, ny, nz) | 1 | 1 |
| Force | (nx, ny, nz) | 1 | 1 |
| Material (λ, μ) | (nx, ny, nz) | 1 | 1 |

**Key design principle**: The kernel does not distinguish between periodic and non-periodic boundary conditions. Ghost cell communication (via `CartesianDecomposition::communicate_ghosts()`) handles periodicity:
- For **periodic BC**: ghost cells contain copies from the opposite boundary
- For **non-periodic BC**: ghost cells are filled with appropriate boundary values

This unified approach simplifies the kernel implementation and enables consistent behavior across CPU and GPU.

### Gather Pattern

The kernel iterates over **interior nodes** and gathers contributions from all neighboring elements:

**2D (4 elements per node):**
```
Element offsets and local node indices:
  (-1,-1) → local node 3    (0,-1) → local node 2
  (-1, 0) → local node 1    (0, 0) → local node 0
```

**3D (8 elements per node):**
```
Element offsets and local node indices:
  (-1,-1,-1) → local node 7    (0,-1,-1) → local node 6
  (-1, 0,-1) → local node 5    (0, 0,-1) → local node 4
  (-1,-1, 0) → local node 3    (0,-1, 0) → local node 2
  (-1, 0, 0) → local node 1    (0, 0, 0) → local node 0
```

### Iteration Bounds

The kernel iterates over all interior nodes (indices 0 to n-1):
```cpp
// Iterate over all nodes - ghost communication handles periodicity
for (ix = 0; ix < nnx; ix++):
    for (iy = 0; iy < nny; iy++):
        // Compute force at node (ix, iy)
```

The kernel reads from ghost cells at positions -1 (left) and n (right), which must be populated before calling the operator. For periodic BC, these contain copies from the opposite boundary. For non-periodic BC, these contain boundary values (typically zero for homogeneous Dirichlet conditions).

### Kernel Pseudocode

```cpp
for each node (ix, iy, iz) in [0, nnx) × [0, nny) × [0, nnz):
    f = [0, 0, 0]  // force accumulator

    for each neighboring element (8 in 3D):
        // Element indices can be -1 (left ghost) or nx (right ghost)
        ex, ey, ez = element indices relative to node
        local_node = which corner of element is this node

        // Material fields are node-indexed, read from element position
        // (element at ex,ey,ez has material stored at node ex,ey,ez)
        λ = material_lambda[ex, ey, ez]
        μ = material_mu[ex, ey, ez]

        // Gather displacement from all element nodes (may read ghosts)
        u = [displacement at each element corner node]

        // Compute stiffness contribution
        for each DOF d:
            row = local_node * NB_DOFS + d
            f[d] += Σ_j (2μ G[row,j] + λ V[row,j]) × u[j]

    force[ix, iy, iz] = f
```

### Methods

| Method | Description |
|--------|-------------|
| `apply(u, λ, μ, f)` | f = K @ u |
| `apply_increment(u, λ, μ, α, f)` | f += α × K @ u |

---

## Memory Layout Details

### CPU Layout (Array of Structures)

DOF components are contiguous in memory:
```
Memory: [ux0, uy0, uz0, ux1, uy1, uz1, ux2, uy2, uz2, ...]
```

Strides:
```cpp
disp_stride_d = 1;                    // DOF components contiguous
disp_stride_x = NB_DOFS;              // = 3 for 3D
disp_stride_y = NB_DOFS * nx;
disp_stride_z = NB_DOFS * nx * ny;
```

### GPU Layout (Structure of Arrays)

Spatial indices are contiguous for coalesced memory access:
```
Memory: [ux0, ux1, ux2, ..., uy0, uy1, uy2, ..., uz0, uz1, uz2, ...]
```

Strides:
```cpp
disp_stride_x = 1;                    // Spatial x contiguous
disp_stride_y = nx;
disp_stride_z = nx * ny;
disp_stride_d = nx * ny * nz;         // DOF components separated
```

### Data Pointer Offset

Pointers are offset to account for ghost cells:
```cpp
const Real* disp_data = displacement.data() + ghost_offset;
// Now index 0 = first interior node, index -1 = left ghost
```

---

## File Structure

```
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

---

## Testing

Tests are in `tests/python_isotropic_stiffness_operator_tests.py`:

- `test_compare_with_generic_operator`: Compares fused kernel against explicit B^T C B
- `test_unit_impulse_*`: Verifies response to unit displacement at specific nodes
- `test_symmetry`: Verifies K is symmetric
- `ValidationGuardTest*`: Verifies proper error handling for invalid configurations
- `GPUUnitImpulseTest`: Verifies GPU kernel matches CPU kernel output

With node-based indexing, all tests compare the full output (all nodes), not just interior nodes.
