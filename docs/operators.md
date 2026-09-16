# Operators

µGrid provides a hierarchy of linear operators for discretizing partial
differential equations on structured grids. This page explains the different
operator types and when to use each one.

## Overview

Linear operators in µGrid fall into three categories:

1. **Generic operators**: flexible, user-defined stencils for arbitrary convolutions
2. **Gradient/divergence operators**: FEM-based operators for computing derivatives
3. **Fused operators**: highly optimized kernels for specific problems (Laplace,
   elasticity, nodal moments)

Prefer fused operators when available, as they provide the best performance.
Generic operators are useful for prototyping or implementing custom stencils.

| Operator | Use Case | Performance |
|----------|----------|-------------|
| `GenericLinearOperator` | Custom stencils, prototyping | Baseline |
| `FEMGradientOperator` | Gradient/divergence with FEM | 2-3× faster than generic |
| `LaplaceOperator` | Scalar Poisson problems | 2-4× faster than generic |
| `IsotropicStiffnessOperator` | Linear elasticity (isotropic) | 5-10× faster than unfused |
| `NodalMomentOperator` | Polynomial functionals of a nodal field | O(1) scratch instead of one grid copy per quadrature point |

## Generic Linear Operators

The `GenericLinearOperator` class (formerly `StencilGradientOperator`)
implements arbitrary convolution stencils. It is the most flexible operator
type but also the slowest due to indirect memory access patterns.

### Creating a generic operator

A generic operator requires:

1. **Stencil coefficients**: a numpy array of any shape containing the weights
2. **Offset**: the position of the stencil origin relative to array indices

Example: 5-point Laplacian in 2D:

```python
import numpy as np
import muGrid

# 5-point Laplacian stencil
h = 0.1  # Grid spacing
stencil = np.array([
    [0,  1, 0],
    [1, -4, 1],
    [0,  1, 0]
]) / h**2

# Offset: stencil[0,0] corresponds to neighbor at (-1, -1)
offset = [-1, -1]

laplace = muGrid.GenericLinearOperator(offset, stencil)
```

### Multi-component stencils

For operators with multiple input/output components (like gradients), use
higher-dimensional coefficient arrays:

```python
# 2D gradient stencil: 2 output components, 1 input, 2×2 stencil
# Shape: [nb_output, nb_quad, nb_input, stencil_x, stencil_y]
gradient_coeffs = np.zeros((2, 1, 1, 2, 2))

# ∂/∂x: forward difference
gradient_coeffs[0, 0, 0, 0, 0] = -1/h
gradient_coeffs[0, 0, 0, 1, 0] = +1/h

# ∂/∂y: forward difference
gradient_coeffs[1, 0, 0, 0, 0] = -1/h
gradient_coeffs[1, 0, 0, 0, 1] = +1/h

gradient_op = muGrid.GenericLinearOperator([0, 0], gradient_coeffs)
```

### Using generic operators

```python
# Create field collection with ghost regions
decomposition = muGrid.CartesianDecomposition(
    comm, (64, 64),
    nb_ghosts_left=(1, 1),
    nb_ghosts_right=(1, 1),
)

input_field = decomposition.real_field("input")
output_field = decomposition.real_field("output")

# Fill ghost regions before applying operator
decomposition.communicate_ghosts(input_field)

# Apply the operator
laplace.apply(input_field, output_field)
```

## FEM Gradient Operator

The `FEMGradientOperator` computes gradients using linear finite element
shape functions. Each pixel/voxel is subdivided into simplicial elements
(triangles in 2D, tetrahedra in 3D) with multiple quadrature points.

### Properties

| Dimension | Elements | Quadrature Points |
|-----------|----------|-------------------|
| 2D | 2 triangles per pixel | 2 (one per triangle) |
| 3D | 5 tetrahedra per voxel | 5 (one per tetrahedron) |

### Creating the operator

```python
# Grid spacing (can be non-uniform)
grid_spacing = (0.1, 0.1)  # 2D
# grid_spacing = (0.1, 0.1, 0.1)  # 3D

gradient_op = muGrid.FEMGradientOperator(spatial_dim=2, grid_spacing=grid_spacing)

# Query operator properties
print(f"Quadrature points: {gradient_op.nb_quad_pts}")
print(f"Quadrature weights: {gradient_op.quadrature_weights}")
```

### Gradient operation

The forward operation computes gradients at quadrature points:

```python
# Input: scalar or vector field at nodes
# Output: gradient tensor at quadrature points

# For a displacement field u with shape [dim, nx, ny]:
# gradient_op.apply(u, grad_u) produces grad_u with shape [dim, dim, quad, nx, ny]

decomposition.communicate_ghosts(u_field)
gradient_op.apply(u_field, gradient_field)
```

### Divergence (transpose) operation

The transpose operation computes the (negative) divergence, weighted by
quadrature weights:

```python
# Input: tensor field at quadrature points
# Output: vector field at nodes

gradient_op.transpose(stress_field, force_field, quad_weights)
```

The quadrature weights are typically the element volumes divided by the number
of quadrature points per element.

## Laplace Operator

The `LaplaceOperator` provides an optimized implementation of the discrete
Laplacian using the standard 5-point (2D) or 7-point (3D) stencil.

```python
# Create Laplacian with scaling factor
# Negative scale makes the operator positive-definite (for CG solver)
scale = -1.0 / h**2
laplace = muGrid.LaplaceOperator(spatial_dim=2, scale=scale)

# Apply to fields
decomposition.communicate_ghosts(u_field)
laplace.apply(u_field, result_field)
```

The hard-coded implementation is significantly faster than an equivalent
`GenericLinearOperator` because:

- Memory access patterns are predictable
- The compiler can vectorize the inner loops (SIMD)
- GPU kernels are highly optimized

## Isotropic Stiffness Operator

The `IsotropicStiffnessOperator2D` and `IsotropicStiffnessOperator3D`
classes implement fused stiffness operators for isotropic linear elastic
materials. They compute:

$$
\mathbf{f} = \mathbf{K} \mathbf{u} = \mathbf{B}^T \mathbf{C} \mathbf{B} \mathbf{u}
$$

where:

- \(\mathbf{u}\) is the displacement field
- \(\mathbf{B}\) is the strain-displacement matrix (gradient operator)
- \(\mathbf{C}\) is the material stiffness tensor
- \(\mathbf{f}\) is the internal force vector

### Mathematical formulation

For isotropic materials, the stiffness tensor \(\mathbf{C}\) depends only
on two Lamé parameters:

- \(\lambda\) (first Lamé parameter, related to bulk modulus)
- \(\mu\) (shear modulus)

The element stiffness matrix decomposes as:

$$
\mathbf{K}_e = 2\mu \mathbf{G} + \lambda \mathbf{V}
$$

where \(\mathbf{G}\) and \(\mathbf{V}\) are geometry-only matrices
computed once at construction time. This decomposition enables:

1. **Memory efficiency**: store only 2 scalars per element instead of a full stiffness matrix
2. **Computational efficiency**: avoid explicit matrix assembly and storage

### Creating the operator

```python
# 2D operator
grid_spacing = (0.1, 0.1)
stiffness_op_2d = muGrid.IsotropicStiffnessOperator2D(grid_spacing)

# 3D operator
grid_spacing = (0.1, 0.1, 0.1)
stiffness_op_3d = muGrid.IsotropicStiffnessOperator3D(grid_spacing)
```

### Material fields

The operator requires two material fields containing the Lamé parameters at
each element (pixel/voxel):

```python
# Create decomposition for element-based fields (one fewer grid point in each direction)
# because elements are defined between nodes
element_grid_pts = tuple(n - 1 for n in nb_grid_pts)

element_decomposition = muGrid.CartesianDecomposition(
    comm, element_grid_pts,
    nb_subdivisions=subdivisions,
    nb_ghosts_left=(1,) * dim,
    nb_ghosts_right=(1,) * dim,
)

# Create material fields with shape [nx-1, ny-1] (2D) or [nx-1, ny-1, nz-1] (3D)
lambda_field = element_decomposition.real_field("lambda")
mu_field = element_decomposition.real_field("mu")
```

### Computing Lamé parameters from engineering constants

```python
def lame_parameters(E, nu):
    """
    Compute Lamé parameters from Young's modulus E and Poisson's ratio nu.

    Parameters
    ----------
    E : float
        Young's modulus
    nu : float
        Poisson's ratio

    Returns
    -------
    lam : float
        First Lamé parameter (λ)
    mu : float
        Shear modulus (μ)
    """
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lam, mu

# Example: heterogeneous material with matrix and inclusions
E_matrix, E_inclusion = 1.0, 10.0
nu = 0.3

lam_matrix, mu_matrix = lame_parameters(E_matrix, nu)
lam_inclusion, mu_inclusion = lame_parameters(E_inclusion, nu)

# Set material properties based on phase field (0 = matrix, 1 = inclusion)
lambda_field.p[...] = lam_matrix * (1 - phase) + lam_inclusion * phase
mu_field.p[...] = mu_matrix * (1 - phase) + mu_inclusion * phase

# Fill ghost regions (only needs to be done once)
element_decomposition.communicate_ghosts(lambda_field)
element_decomposition.communicate_ghosts(mu_field)
```

### Applying the operator

```python
# Displacement and force fields on node-based grid
# Shape: [dim, nx, ny] for 2D, [dim, nx, ny, nz] for 3D
u_field = decomposition.real_field("displacement", (dim,))
f_field = decomposition.real_field("force", (dim,))

# Apply stiffness operator: f = K @ u
decomposition.communicate_ghosts(u_field)
stiffness_op.apply(u_field, lambda_field, mu_field, f_field)

# Increment form: f += alpha * K @ u
stiffness_op.apply_increment(u_field, lambda_field, mu_field, alpha, f_field)
```

### Complete linear elasticity example

```python
import numpy as np
import muGrid
from muGrid.Solvers import conjugate_gradients

# Parameters
nb_grid_pts = (32, 32)
dim = 2
E_matrix, E_inclusion = 1.0, 10.0
nu = 0.3

# Compute Lamé parameters
lam_matrix = E_matrix * nu / ((1 + nu) * (1 - 2 * nu))
mu_matrix = E_matrix / (2 * (1 + nu))
lam_inclusion = E_inclusion * nu / ((1 + nu) * (1 - 2 * nu))
mu_inclusion = E_inclusion / (2 * (1 + nu))

# Grid spacing
grid_spacing = tuple(1.0 / n for n in nb_grid_pts)

# Create operator
stiffness_op = muGrid.IsotropicStiffnessOperator2D(grid_spacing)

# Domain decomposition for nodal fields
comm = muGrid.Communicator()
decomposition = muGrid.CartesianDecomposition(
    comm, nb_grid_pts,
    nb_ghosts_left=(1, 1),
    nb_ghosts_right=(1, 1),
)

# Domain decomposition for element fields
element_decomposition = muGrid.CartesianDecomposition(
    comm, tuple(n - 1 for n in nb_grid_pts),
    nb_ghosts_left=(1, 1),
    nb_ghosts_right=(1, 1),
)

# Create fields
u_field = decomposition.real_field("displacement", (dim,))
f_field = decomposition.real_field("force", (dim,))
lambda_field = element_decomposition.real_field("lambda")
mu_field = element_decomposition.real_field("mu")

# Set up material (circular inclusion)
coords = element_decomposition.coords
X, Y = coords[0], coords[1]
phase = ((X - 0.5)**2 + (Y - 0.5)**2 < 0.25**2).astype(float)

lambda_field.p[...] = lam_matrix * (1 - phase) + lam_inclusion * phase
mu_field.p[...] = mu_matrix * (1 - phase) + mu_inclusion * phase
element_decomposition.communicate_ghosts(lambda_field)
element_decomposition.communicate_ghosts(mu_field)

# Stiffness operator wrapper for CG solver
def apply_stiffness(u, f):
    decomposition.communicate_ghosts(u)
    stiffness_op.apply(u, lambda_field, mu_field, f)

# Solve (with appropriate RHS setup for homogenization...)
# conjugate_gradients(comm, decomposition, rhs, u_field, hessp=apply_stiffness, ...)
```

## Nodal Moment Operator

`NodalMomentOperator{2,3}D` computes, for every cell, the moments of a nodal
scalar field's finite-element interpolant

$$ M_k(e) = \int_e \rho(x)^k \, dx , \qquad k = 2, 3, 4 $$

together with the derivative of their sum with respect to each nodal value.

### Why moments rather than an energy

A polynomial function of the interpolant is a fixed combination of these. The
phase-field double well of a topology optimisation,
$W(\rho) = \rho^2 (1-\rho)^2 = \rho^2 - 2\rho^3 + \rho^4$, has cell integral
`M2 - 2*M3 + M4` and nodal gradient the same combination of the moment
gradients. Keeping moments as the interface leaves the choice of energy to the
caller, so no material model is compiled into µGrid — the same split
`IsotropicStiffnessOperator.compute_sensitivity` uses.

### Why it is fused

Evaluated array-at-a-time, this computation materialises the interpolant at
every quadrature point of every cell at once: with the 3-point-per-axis rule
in 3D that is a 27-fold copy of the grid, plus a temporary per term of the
polynomial. The kernel consumes each quadrature point in registers instead, so
the pass is **O(1)** in scratch memory. One thread owns one node and the cell
of the same index, the block's tile plus its one-node halo is staged in shared
memory, and each thread writes only its own entries — so there are no atomics
and no scatter or ghost-reduction pass.

### Quadrature

| Element | Rule | Points per cell |
|---------|------|-----------------|
| Q1 (bilinear quad / trilinear hex) | 3-point-per-axis tensor Gauss | 9 (2D), 27 (3D) |
| P1 (triangles / tetrahedra) | the same rule placed on each sub-simplex | 18 (2D), 135 (3D) |

Both are exact for the quartic integrand. The simplex rules use **Gauss-Jacobi**
rather than Gauss-Legendre points: the collapsed (Duffy) map from the cube
carries a Jacobian — $(1-u)$ in 2D, $(1-u)^2(1-v)$ in 3D — and folding it into
the weight function keeps 3 points per axis sufficient. Leaving it in the
integrand instead inflates the degree by up to 2, and a 3-point
Gauss-Legendre rule is then exact for `M2` and `M3` but wrong for `M4`. Every
weight is positive, so a cell's double-well energy can never come out negative.

### Creating the operator

```python
import muGrid

# Q1 is the default; pass muGrid.FEMElement.p1 for simplices
moment_op = muGrid.NodalMomentOperator3D(grid_spacing=[h, h, h])
moment_op = muGrid.NodalMomentOperator3D([h, h, h], muGrid.FEMElement.p1)

moment_op.nb_quad      # quadrature points per cell (27 for Q1 in 3D)
moment_op.cell_volume  # h_x h_y h_z
```

### Applying the operator

`rho` is a scalar field whose ghosts have been communicated; `moments` (a cell
quantity) and `moment_gradients` (a nodal one) each carry
`NodalMomentOperator3D.nb_moments` components and must live on the same
collection. Only the interior (owned) region is written.

```python
rho = fc.real_field("rho")
moments = fc.real_field("moments", (3,))
moment_gradients = fc.real_field("moment_gradients", (3,))

decomposition.communicate_ghosts(rho)
moment_op.compute(rho, moments, moment_gradients)

# Double-well energy and its nodal gradient
import numpy as np
c = np.array([1.0, -2.0, 1.0])            # coefficients on M2, M3, M4
m = np.asarray(moments.p).reshape((3, -1))
g = np.asarray(moment_gradients.p).reshape((3, -1))
energy = comm.sum(float(c @ m.sum(axis=1)))
grad = (c @ g)
```

Component `j` holds the moment for `k = j + NodalMomentOperator3D.first_moment`,
i.e. `k = 2, 3, 4`. Both `float64` and `float32` fields are supported, on host
and on device.

## Performance Comparison

The fused `IsotropicStiffnessOperator` provides significant performance
advantages over manually computing \(\mathbf{B}^T \mathbf{C} \mathbf{B}\):

| Approach | Relative Speed | Memory per Element |
|----------|----------------|--------------------|
| Full stiffness matrix | 1× (baseline) | 576 floats (3D) |
| Separate B, C, B^T ops | 2-3× | Full C tensor + intermediates |
| Fused isotropic operator | 5-10× | 2 floats (λ, μ) |

The speedup comes from:

1. **Reduced memory traffic**: only read 2 material values per element instead of full tensors
2. **No intermediate storage**: strain and stress computed on-the-fly
3. **Optimized kernels**: hand-tuned CPU loops and GPU kernels
4. **Better cache utilization**: predictable access patterns

### GPU Performance

On GPUs, the fused operators show even greater advantages:

- Atomic-free implementation using gather patterns
- Shared memory optimization for cooperative node loading
- High occupancy due to low register pressure

Typical GPU speedups are 5-10× over unfused approaches on modern NVIDIA and AMD GPUs.

## When to Use Each Operator

| Operator | Recommended Use |
|----------|-----------------|
| `GenericLinearOperator` | Prototyping, custom stencils, uncommon PDEs |
| `FEMGradientOperator` | Computing gradients/divergence, anisotropic materials |
| `LaplaceOperator` | Scalar Poisson problems, diffusion equations |
| `IsotropicStiffnessOperator` | Linear elasticity with isotropic materials |
| `NodalMomentOperator` | Integrals of a polynomial in a nodal field, e.g. a phase-field double well |

For production code solving standard PDEs, always prefer the fused operators
when available. They provide the best performance while maintaining numerical
accuracy identical to the unfused approach.
