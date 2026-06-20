# Examples

This chapter provides step-by-step examples for building numerical solvers with
µGrid. We present two complete examples: a Poisson solver and a linear elasticity
solver for micromechanical homogenization.

For details on the different operator types available and when to use each one,
see the [Operators](operators.md) chapter.

## Poisson Solver

The Poisson equation is a fundamental PDE that appears in many physical contexts
(heat conduction, electrostatics, etc.). We solve:

$$
-\nabla^2 u = f
$$

with periodic boundary conditions on a unit domain.

### Setting up the grid

First, we import the necessary modules and set up a 2D grid with ghost regions
for the stencil operations:

```python
import numpy as np
import muGrid
from muGrid.Solvers import conjugate_gradients

# Create a communicator (serial execution)
comm = muGrid.Communicator()

# Grid parameters
nb_grid_pts = (64, 64)
dim = len(nb_grid_pts)

# Ghost layers: 1 cell on each side for the 5-point stencil
nb_ghosts_left = (1, 1)
nb_ghosts_right = (1, 1)

# Create the domain decomposition
# (works for both serial and MPI-parallel execution)
decomposition = muGrid.CartesianDecomposition(
    comm,
    nb_domain_grid_pts=nb_grid_pts,
    nb_ghosts_left=nb_ghosts_left,
    nb_ghosts_right=nb_ghosts_right,
)
```

### Creating the Laplacian operator

µGrid provides an optimized `LaplaceOperator` for the discrete Laplacian:

```python
# Grid spacing (assuming unit domain)
h = 1.0 / nb_grid_pts[0]

# Scale factor: negative because -∇² must be positive definite for CG
scale = -1.0 / h**2

# Hard-coded Laplacian operator (optimized implementation)
laplace = muGrid.LaplaceOperator(dim, scale)
```

The `LaplaceOperator` implements the standard 5-point stencil in 2D (7-point in
3D) with optimized memory access patterns for both CPU and GPU.

### Creating fields and setting up the RHS

```python
# Create fields using the decomposition
rhs = decomposition.real_field("rhs")
solution = decomposition.real_field("solution")

# Set up a smooth right-hand side
# Get coordinates for each pixel in the local domain
coords = decomposition.coords
X, Y = coords[0], coords[1]

rhs.p[...] = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)

# Remove mean (necessary for periodic Poisson with Neumann-like conditions)
rhs.p[...] -= np.mean(rhs.p)
```

### Solving with conjugate gradients

The conjugate gradient solver requires a function that applies the linear operator:

```python
def apply_laplacian(x, Ax):
    """Apply the Laplacian operator: Ax = L @ x"""
    # Fill ghost regions with periodic boundary values
    decomposition.communicate_ghosts(x)
    # Apply the stencil
    laplace.apply(x, Ax)

# Solve the system
conjugate_gradients(
    comm,
    decomposition,
    rhs,
    solution,
    hessp=apply_laplacian,
    tol=1e-6,
    maxiter=1000,
)

print(f"Solution range: [{solution.p.min():.6f}, {solution.p.max():.6f}]")
```

### Complete Poisson solver

Here is the complete, minimal Poisson solver (`examples/poisson.py`):

```python
import numpy as np
import muGrid
from muGrid.Solvers import conjugate_gradients

# Setup
comm = muGrid.Communicator()
nb_grid_pts = (64, 64)
h = 1.0 / nb_grid_pts[0]

decomposition = muGrid.CartesianDecomposition(
    comm,
    nb_domain_grid_pts=nb_grid_pts,
    nb_ghosts_left=(1, 1),
    nb_ghosts_right=(1, 1),
)

# Laplacian operator (negative for positive-definiteness)
laplace = muGrid.LaplaceOperator(2, -1.0 / h**2)

# Fields
rhs = decomposition.real_field("rhs")
solution = decomposition.real_field("solution")

# RHS: smooth function with zero mean
coords = decomposition.coords
X, Y = coords[0], coords[1]
rhs.p[...] = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
rhs.p[...] -= np.mean(rhs.p)

# Linear operator for CG
def apply_laplacian(x, Ax):
    decomposition.communicate_ghosts(x)
    laplace.apply(x, Ax)

# Solve
conjugate_gradients(comm, decomposition, rhs, solution,
                    hessp=apply_laplacian, tol=1e-6, maxiter=1000)

print(f"Solved! Solution range: [{solution.p.min():.4f}, {solution.p.max():.4f}]")
```

### Preconditioning

The conjugate-gradient solver accepts a preconditioner through its `prec`
argument: any callable `prec(r, z)` that overwrites `z` with `M⁻¹ r`. The
`muGrid.Preconditioners` module provides a small class hierarchy implementing
this contract:

- `IdentityPreconditioner` — no-op, equivalent to `prec=None`;
- `JacobiPreconditioner(diagonal)` — divides by the operator diagonal. Useful for
  strongly heterogeneous coefficients, where it equilibrates the spectrum (for a
  constant diagonal it merely rescales the system and does not change the
  iteration). The diagonal may be spatial-only (shared across field components)
  or per-component;
- `FourierPreconditioner(engine, kernel)` — applies a spectral kernel,
  `z = F⁻¹[k(q) · F r]`, using a muGrid `FFTEngine`. With the inverse symbol of
  (an approximation to) the operator as the kernel, this is the classic FFT
  preconditioner.

#### Spectral (FFT) preconditioning of the Poisson problem

The finite-difference Laplacian diagonalizes in Fourier space, so its exact
inverse symbol is available in closed form — with it, conjugate gradients
converges in a single iteration (it becomes a direct solver). For operators that
are only *approximately* diagonalized by the FFT (e.g. weakly heterogeneous
coefficients), the same kernel built from the homogeneous reference operator
still yields mesh-independent iteration counts.

One detail is essential for MPI runs: the solver fields and the FFT must share a
single domain decomposition. A stand-alone `CartesianDecomposition` would in
general split the domain differently than the FFT's pencil decomposition. Since
the `FFTEngine` *is* a `CartesianDecomposition` (it supports ghost buffers in
real space), the engine itself serves as the decomposition for everything — the
stencil operator, the solver work fields, and the transforms:

```python
import numpy as np
import muGrid
from muGrid.Preconditioners import FourierPreconditioner
from muGrid.Solvers import conjugate_gradients

comm = muGrid.Communicator()
nb_grid_pts = (64, 64)
h = 1.0 / nb_grid_pts[0]

# The FFT engine doubles as the (ghosted) domain decomposition
engine = muGrid.FFTEngine(nb_grid_pts, comm,
                          nb_ghosts_left=(1, 1), nb_ghosts_right=(1, 1))

laplace = muGrid.LaplaceOperator(len(nb_grid_pts), -1.0 / h**2)

rhs = engine.real_space_field("rhs")
solution = engine.real_space_field("solution")

X, Y = engine.coords
rhs.p[...] = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)

def hessp(x, Ax):
    engine.communicate_ghosts(x)
    laplace.apply(x, Ax)

# Exact inverse symbol of -FD-Laplacian/h²; the q = 0 mode of the
# periodic Laplacian is singular and is projected out by a zero entry
# (the right-hand side must be free of it, i.e. have zero mean).
def inverse_fd_laplacian(engine):
    q = engine.fftfreq  # shape [dim, *local_fourier_shape]
    denom = (4 * np.sin(np.pi * q) ** 2 / h**2).sum(axis=0)
    return np.where(denom > 0, 1 / np.where(denom > 0, denom, 1), 0.0)

prec = FourierPreconditioner(engine, inverse_fd_laplacian)

conjugate_gradients(comm, engine.real_space_collection, rhs, solution,
                    hessp=hessp, prec=prec, tol=1e-8, maxiter=10)
```

The kernel is evaluated once on the rank-local Fourier subdomain
(`engine.fftfreq` only exposes local frequencies), so the code is unchanged
between serial and MPI-parallel execution. The FFT normalisation is folded into
the kernel; the kernel broadcasts over field components, so vector- and
tensor-valued unknowns are preconditioned per component.

#### Jacobi preconditioning of heterogeneous problems

For a screened Poisson problem `(-∇²/h² + c(x)) u = b` with a coefficient `c(x)`
varying over many orders of magnitude, dividing by the operator diagonal restores
a well-clustered spectrum:

```python
from muGrid.Preconditioners import JacobiPreconditioner

diag = engine.real_space_field("diagonal")
diag.p[...] = 4 / h**2 + c  # stencil center plus screening coefficient

conjugate_gradients(comm, engine.real_space_collection, rhs, solution,
                    hessp=hessp_screened,
                    prec=JacobiPreconditioner(diag),
                    tol=1e-8, maxiter=1000)
```

In the test suite (`tests/python_preconditioner_tests.py`), this reduces the
iteration count on a 32×32 problem with `c` spanning six orders of magnitude from
164 to 21; the FFT-preconditioned Poisson solve converges in a single iteration.

Both preconditioners run wherever the solver fields live: the Fourier kernel and
the inverse diagonal are stored in fields on the engine's collections and applied
with muGrid's fused linear-algebra kernels (`linalg.scal` with a field-valued
`alpha`), so a solve on a GPU device stays on the device with no host transfers in
the iteration loop.

## Linear Elasticity Solver

This example computes the effective elastic properties of a heterogeneous
material using FEM-based homogenization. The governing equation is:

$$
\nabla \cdot \boldsymbol{\sigma} = 0
$$

where the stress is related to strain by Hooke's law:

$$
\boldsymbol{\sigma} = \mathbf{C} : \boldsymbol{\varepsilon}
$$

and strain is the symmetric gradient of displacement:

$$
\boldsymbol{\varepsilon} = \frac{1}{2}(\nabla \mathbf{u} + \nabla \mathbf{u}^T)
$$

For isotropic materials, µGrid provides the fused `IsotropicStiffnessOperator`
which computes the entire stiffness operation
\(\mathbf{K}\mathbf{u} = \mathbf{B}^T \mathbf{C} \mathbf{B} \mathbf{u}\)
efficiently without explicitly forming intermediate tensors.

### Material properties

Isotropic elastic materials are characterized by two Lamé parameters (λ, μ),
which can be computed from Young's modulus *E* and Poisson's ratio *ν*:

```python
import numpy as np
import muGrid
from muGrid.Solvers import conjugate_gradients

def lame_parameters(E, nu):
    """Compute Lamé parameters from Young's modulus and Poisson's ratio."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lam, mu

# Material parameters
E_matrix = 1.0      # Young's modulus of matrix
E_inclusion = 10.0  # Young's modulus of inclusion (10x stiffer)
nu = 0.3            # Poisson's ratio (same for both)

lam_matrix, mu_matrix = lame_parameters(E_matrix, nu)
lam_inclusion, mu_inclusion = lame_parameters(E_inclusion, nu)
```

### Setting up the grid and operator

The `IsotropicStiffnessOperator` operates on nodal displacement fields and
requires material properties (λ, μ) defined per element:

```python
nb_grid_pts = (32, 32)
dim = 2

# Grid spacing
grid_spacing = tuple(1.0 / n for n in nb_grid_pts)

# Create the fused stiffness operator
stiffness_op = muGrid.IsotropicStiffnessOperator2D(grid_spacing)

# For 3D problems:
# stiffness_op = muGrid.IsotropicStiffnessOperator3D(grid_spacing)
```

### Setting up the microstructure

We create a simple circular inclusion in the center. Material fields are defined
per element (one fewer grid point in each direction than nodal fields):

```python
comm = muGrid.Communicator()

# Domain decomposition for nodal fields (displacements, forces)
decomposition = muGrid.CartesianDecomposition(
    comm,
    nb_domain_grid_pts=nb_grid_pts,
    nb_ghosts_left=(1,) * dim,
    nb_ghosts_right=(1,) * dim,
)

# Domain decomposition for element fields (material properties)
# Elements are defined between nodes, so one fewer in each direction
element_grid_pts = tuple(n - 1 for n in nb_grid_pts)
element_decomposition = muGrid.CartesianDecomposition(
    comm,
    nb_domain_grid_pts=element_grid_pts,
    nb_ghosts_left=(1,) * dim,
    nb_ghosts_right=(1,) * dim,
)

# Create material fields
lambda_field = element_decomposition.real_field("lambda")
mu_field = element_decomposition.real_field("mu")

# Get element coordinates (centers of elements)
coords = element_decomposition.coords
X, Y = coords[0], coords[1]

# Circular inclusion at center with radius 0.25
radius = 0.25
distance = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
phase = (distance < radius).astype(float)  # 1 = inclusion, 0 = matrix

# Set material properties
lambda_field.p[...] = lam_matrix * (1 - phase) + lam_inclusion * phase
mu_field.p[...] = mu_matrix * (1 - phase) + mu_inclusion * phase

# Fill ghost regions (only needs to be done once)
element_decomposition.communicate_ghosts(lambda_field)
element_decomposition.communicate_ghosts(mu_field)

print(f"Inclusion volume fraction: {np.mean(phase):.4f}")
```

### Creating displacement and force fields

```python
# Displacement field: vector with dim components at nodes
u_field = decomposition.real_field("displacement", (dim,))

# Force field (RHS): vector with dim components at nodes
f_field = decomposition.real_field("force", (dim,))
```

### The stiffness operator

The fused operator combines gradient, constitutive, and divergence operations:

```python
def apply_stiffness(u_in, f_out):
    """
    Apply stiffness operator: f = K @ u = B^T C B u
    """
    decomposition.communicate_ghosts(u_in)
    stiffness_op.apply(u_in, lambda_field, mu_field, f_out)
```

This is significantly faster than manually computing the sequence
\(\varepsilon = \mathbf{B}\mathbf{u}\), \(\sigma = \mathbf{C}:\varepsilon\),
\(\mathbf{f} = \mathbf{B}^T\sigma\) because:

1. No intermediate storage for strain and stress tensors
2. Optimized memory access patterns
3. Material properties stored as just two scalars per element

See the [Operators](operators.md) chapter for detailed performance comparisons.

### Solving for effective properties

To compute effective properties, we apply unit macroscopic strains and measure
the resulting average stress. Here's a simplified approach:

```python
# For homogenization, we solve: K @ u = -K @ (E_macro · x)
# where E_macro is the applied macroscopic strain

# This requires computing the RHS by applying the stiffness operator
# to a linear displacement field u_linear = E_macro · x

# Initialize with the macroscopic strain contribution
# (Full implementation in examples/homogenization.py)

# Solve equilibrium
conjugate_gradients(
    comm,
    decomposition,
    f_field,
    u_field,
    hessp=apply_stiffness,
    tol=1e-6,
    maxiter=500,
)
```

### Complete homogenization example

A complete homogenization example that computes effective elastic properties is
provided in `examples/homogenization.py`. It includes:

- Full RHS computation for applied macroscopic strains
- Computation of all independent effective stiffness components
- Validation against analytical bounds (Voigt, Reuss, Hashin-Shtrikman)
- Support for both 2D and 3D problems
- MPI parallelization for large-scale computations
- GPU acceleration

Run the example:

```bash
# 2D, 64×64 grid
python examples/homogenization.py -n 64,64

# 3D, 32×32×32 grid
python examples/homogenization.py -n 32,32,32

# With MPI parallelization
mpiexec -n 4 python examples/homogenization.py -n 128,128

# On GPU
python examples/homogenization.py -n 256,256 -d gpu
```

### Scaling benchmarks

Two benchmark pages report CPU and GPU timings across a range of grid sizes,
together with the hardware they were run on:

- [Benchmark (Poisson)](benchmark.md) — the CG Poisson solve, generated by
  `examples/benchmark.py`.
- [Benchmark (homogenization)](benchmark_homogenization.md) — the 3D FEM
  homogenization solve, generated by `examples/benchmark_homogenization.py`.

Re-run either script to refresh the numbers for your own hardware.
