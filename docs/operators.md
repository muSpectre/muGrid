# Operators

µGrid's operators are linear maps between fields on a structured grid,
evaluated as stencils. This page defines each operator mathematically and
documents its Python interface. Kernel-level design notes (iteration bounds,
memory layout, gather patterns) are in the
[operator kernel design notes](kernel_design.md).

| Operator | Maps | Defined by |
|---|---|---|
| `GenericLinearOperator` | nodal → nodal or quadrature points | arbitrary stencil coefficients |
| `FEMGradientOperator` | nodal → quadrature points | P1 or Q1 finite-element gradient $\mathbf{B}$ |
| `LaplaceOperator` | nodal → nodal | 5-point (2D) / 7-point (3D) finite-difference Laplacian |
| `IsotropicStiffnessOperator2D/3D` | nodal → nodal | fused $\mathbf{K} = \mathbf{B}^\top W \mathbf{C}\mathbf{B}$ for isotropic elasticity |

Notation: dimension $d$, grid points $\mathbf{n}$ with $N_\alpha$ points and
spacing $h_\alpha$ along axis $\alpha$, unit vector $\mathbf{e}_\alpha$.

## Common conventions

### Boundary conditions and ghost layers

Operators make no assumption about boundary conditions. An operator reads input
values at stencil offsets around every output point, including values in the
ghost layers of the field collection. Periodicity and MPI parallelism come
entirely from filling the ghost layers beforehand with
`communicate_ghosts()` of the `CartesianDecomposition` (or `FFTEngine`) that
owns the fields; any other boundary condition is imposed by writing the
appropriate ghost values.

Each operator reports the ghost layers it needs as `((left...), (right...))`:

- `apply_ghost_requirement` — layers read by `apply`,
- `transpose_ghost_requirement` — layers read by `transpose`,
- `ghost_requirement` — the elementwise maximum of the two.

Passing an operator (or a list of operators) as the `ghosts=` argument of
`CartesianDecomposition`, `GlobalFieldCollection` or `FFTEngine` sizes the
ghost buffers from `ghost_requirement`. The requirement is validated once
before the kernel runs; too few ghost layers raise an error. The operator
writes every output point whose stencil lies inside the ghosted buffer, so
with more ghost layers than required it also writes part of the ghost region.

### Precision and devices

All operators accept `float64` and `float32` fields (for
`FEMGradientOperator` the precision is chosen at construction with `dtype=`)
and host or device fields; the kernel follows the memory space of the fields.

## GenericLinearOperator

A generic operator is defined by an offset $\mathbf{o} \in \mathbb{Z}^d$ and
coefficients $S_{k q i}(\mathbf{s})$ for output component $k$, quadrature point
(sub-point) $q$, input component $i$ and stencil position
$\mathbf{s} \in \prod_\alpha \{0, \ldots, S_\alpha - 1\}$. Its action is

$$
y_{kq}(\mathbf{n}) = \sum_{i}\sum_{\mathbf{s}}
  S_{kqi}(\mathbf{s})\; x_i(\mathbf{n} + \mathbf{o} + \mathbf{s}),
$$

and its transpose, with optional per-quadrature-point weights $w_q$ (default
$w_q = 1$, i.e. the bare matrix transpose), is

$$
x_i(\mathbf{n}) = \sum_{k,q} w_q \sum_{\mathbf{s}}
  S_{kqi}(\mathbf{s})\; y_{kq}(\mathbf{n} - \mathbf{o} - \mathbf{s}).
$$

`apply` needs $\max(0, -o_\alpha)$ ghost layers on the left and
$\max(0, o_\alpha + S_\alpha - 1)$ on the right of axis $\alpha$; `transpose`
needs the mirror image. An apply costs $O(\mathrm{nnz}\,N)$ operations for
$\mathrm{nnz}$ non-zero coefficients and $N$ grid points.

```python
muGrid.GenericLinearOperator(offset, stencil)
```

`stencil` is an array of shape `(nb_output, nb_quad, nb_input, *stencil_shape)`;
leading axes may be omitted and default to 1. The 5-point Laplacian with unit
spacing:

```python
import numpy as np
import muGrid

stencil = np.array([[0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]])
laplace = muGrid.GenericLinearOperator([-1, -1], stencil)  # stencil[0, 0] is at (-1, -1)
```

A forward-difference gradient with two output components:

```python
h = 0.1
coeffs = np.zeros((2, 1, 1, 2, 2))   # (nb_output, nb_quad, nb_input, 2, 2)
coeffs[0, 0, 0, 0, 0], coeffs[0, 0, 0, 1, 0] = -1 / h, 1 / h   # d/dx
coeffs[1, 0, 0, 0, 0], coeffs[1, 0, 0, 0, 1] = -1 / h, 1 / h   # d/dy
gradient = muGrid.GenericLinearOperator([0, 0], coeffs)
```

Applying it:

```python
comm = muGrid.Communicator()
decomposition = muGrid.CartesianDecomposition(comm, (64, 64), ghosts=laplace)

x = decomposition.real_field("x")
y = decomposition.real_field("y")
decomposition.communicate_ghosts(x)
laplace.apply(x, y)
laplace.transpose(y, x)            # bare transpose (unit weights)
```

Methods: `apply(nodal_field, quadrature_point_field)` and
`transpose(quadrature_point_field, nodal_field, weights=None)`. Properties: `offset`, `stencil_shape`, `coefficients`, `spatial_dim`,
`nb_quad_pts`, `nb_input_components`, `nb_output_components`.

### Fourier symbol

For a scalar operator (one output component, quadrature point and input
component), `fourier(phases)` evaluates the symbol

$$
\hat{S}(\mathbf{q}) = \sum_{\mathbf{s}} S(\mathbf{s})\,
  \exp\!\big(2\pi i\,\mathbf{q}\cdot(\mathbf{o} + \mathbf{s})\big)
$$

for frequencies $\mathbf{q}$ in cycles per grid point. `phases` has shape
`(d, ...)`; the result has the trailing shape. Operators with several
components have a matrix-valued symbol and raise.

## FEMGradientOperator

The FE gradient operator $\mathbf{B}$ maps nodal values to gradients at the
quadrature points of a finite-element discretisation. Nodes are the grid
points; the element of pixel $\mathbf{n}$ has the $2^d$ corner nodes
$\mathbf{n} + \mathbf{a}$, $\mathbf{a} \in \{0, 1\}^d$. With the reference
shape-function gradients $B_{q\alpha\mathbf{a}}$ of node $\mathbf{a}$ in
direction $\alpha$ at quadrature point $q$,

$$
(\mathbf{B}\mathbf{u})_{i\alpha q}(\mathbf{n}) =
  \frac{1}{h_\alpha} \sum_{\mathbf{a} \in \{0,1\}^d}
  B_{q\alpha\mathbf{a}}\; u_i(\mathbf{n} + \mathbf{a}).
$$

The physical quadrature weights are $W_q = \bar{w}_q \prod_\alpha h_\alpha$
with volume fractions $\sum_q \bar{w}_q = 1$.

| Element | 2D | 3D |
|---|---|---|
| Q1 (default) | bilinear quadrilateral, $2 \times 2$ Gauss points, $\bar{w}_q = 1/4$ | trilinear hexahedron, $2 \times 2 \times 2$ Gauss points, $\bar{w}_q = 1/8$ |
| P1 | 2 linear triangles, one point each, $\bar{w}_q = 1/2$ | 5 linear tetrahedra, one point each: central $\bar{w}_0 = 1/3$, four corner $\bar{w}_q = 1/6$ |

<!-- TODO: rationale for Q1 as the default element? -->

The Gauss points of Q1 lie at $\tfrac12 \mp \tfrac{1}{2\sqrt{3}}$ per axis, so
the rule integrates products of shape-function gradients exactly. The P1
tetrahedral split is a central tetrahedron on the corners $(1,0,0)$, $(0,1,0)$,
$(0,0,1)$, $(1,1,1)$ plus four corner tetrahedra; it has no preferred diagonal,
unlike the six-tetrahedron Kuhn triangulation, and the same split is used in
every voxel so that the operator is translation invariant. The reference
tables are in `src/libmugrid/operators/fem_element.hh`.

The transpose is the **quadrature-weighted** transpose $\mathbf{B}^\top W$, the
discrete (weak) divergence:

$$
\langle W\,\mathbf{B}\mathbf{u}, \mathbf{v}\rangle
  = \langle \mathbf{u}, \mathbf{B}^\top W \mathbf{v}\rangle ,
\qquad
\mathbf{B}^\top W \mathbf{B}\,\mathbf{u} = \text{FE stiffness action}.
$$

So `transpose(apply(u))` assembles $\mathbf{B}^\top W \mathbf{B}\mathbf{u}$;
do not multiply by the quadrature weights again. Passing `weights` replaces $W$
(e.g. to fold per-quadrature-point coefficients into the assembly);
`weights=[1] * nb_quad_pts` gives the bare $\mathbf{B}^\top$.

Ghost protocol: `apply` reads one right ghost layer of the nodal field per
axis; `transpose` writes only the owned nodes and reads one left ghost layer of
the quadrature-point field (the element below each node). Call
`communicate_ghosts` on the input before either operation, and do not call
`reduce_ghosts` on the output of `transpose` (its ghost nodes are not
written).

```python
muGrid.FEMGradientOperator(spatial_dim, grid_spacing=None,
                           element=muGrid.FEMElement.q1, dtype=np.float64)
```

`grid_spacing` defaults to 1 on every axis; `element` is `FEMElement.q1` or
`FEMElement.p1`; `dtype` is `np.float64` or `np.float32`.

```python
dim = 2
grad = muGrid.FEMGradientOperator(dim, grid_spacing=(0.1, 0.1))
print(grad.nb_quad_pts)          # 4 (Q1 in 2D)
print(grad.quadrature_weights)   # W_q = 0.25 * 0.1 * 0.1 each

comm = muGrid.Communicator()
decomposition = muGrid.CartesianDecomposition(
    comm, (64, 64), ghosts=grad, nb_sub_pts={"quad": grad.nb_quad_pts})

u = decomposition.real_field("u", (dim,))                   # nodal vector
grad_u = decomposition.real_field("grad_u", (dim, dim), "quad")
f = decomposition.real_field("f", (dim,))

decomposition.communicate_ghosts(u)
grad.apply(u, grad_u)                       # B u
decomposition.communicate_ghosts(grad_u)
grad.transpose(grad_u, f)                   # B^T W (B u)
```

A nodal field with $n_c$ components maps to a quadrature-point field with
$n_c\,d$ components per quadrature point. Methods:
`apply(nodal_field, quadrature_point_field)`,
`apply_increment(nodal_field, alpha, quadrature_point_field)`,
`transpose(quadrature_point_field, nodal_field, weights=None)` and
`transpose_increment(quadrature_point_field, alpha, nodal_field, weights=None)`.
Properties: `grid_spacing`, `quadrature_weights`,
`nb_quad_pts`, `spatial_dim`, and `offset`, `stencil_shape`, `coefficients`
describing the equivalent generic stencil (so
`GenericLinearOperator(grad.offset, grad.coefficients)` reproduces
$\mathbf{B}$).

## LaplaceOperator

The Laplace operator is the second-order central-difference Laplacian

$$
(L u)(\mathbf{n}) = s \sum_{\alpha=0}^{d-1}
  \frac{u(\mathbf{n} + \mathbf{e}_\alpha) - 2u(\mathbf{n}) +
        u(\mathbf{n} - \mathbf{e}_\alpha)}{h_\alpha^2}
$$

with scale factor $s$ — the 5-point stencil in 2D and the 7-point stencil in
3D. Without `grid_spacing`, $h_\alpha = 1$ on every axis and the stencil is
$s\,[0, 1, 0;\, 1, -4, 1;\, 0, 1, 0]$, so a cubic grid can equivalently use
$s = 1/h^2$. The truncation error is $O(h_\alpha^2)$ per axis.

On a periodic grid the symbol is

$$
\hat{L}(\mathbf{q}) = -s \sum_\alpha \frac{4 \sin^2(\pi q_\alpha)}{h_\alpha^2},
$$

so $L$ is symmetric and, for $s > 0$, negative semidefinite with the constant
mode as null space. Conjugate gradients needs a positive (semi)definite
operator, i.e. $s < 0$; the condition number of $-L$ on the non-constant modes
grows as $O(\max_\alpha N_\alpha^2)$. The operator is self-adjoint, so
`transpose` equals `apply` (quadrature weights are not supported and raise).
It needs one ghost layer on each side.

```python
muGrid.LaplaceOperator(spatial_dim, scale=1.0, grid_spacing=None)
```

```python
# -Laplacian (positive semidefinite) on a cubic grid with spacing h
laplace = muGrid.LaplaceOperator(2, scale=-1.0 / h**2)
# the same on an anisotropic grid: neighbours along axis d carry -1/h_d^2
laplace = muGrid.LaplaceOperator(2, scale=-1.0, grid_spacing=[hx, hy])

decomposition.communicate_ghosts(u)
laplace.apply(u, lu)
laplace.apply_increment(u, alpha, lu)      # lu += alpha * L u
```

Properties: `scale`, `grid_spacing`, `nb_stencil_pts`, `spatial_dim`.
`LaplaceOperator2D`/`LaplaceOperator3D` are the dimension-specific bindings
(`LaplaceOperator2D(scale=1.0)` or `LaplaceOperator2D(grid_spacing, scale=1.0)`).

The FE stiffness of `FEMGradientOperator` on a cubic grid,
`transpose(apply(u))`, discretises $-h^{d-2}\,\nabla^2$ and is positive
semidefinite; `LaplaceOperator` with $s = 1/h^2$ discretises $+\nabla^2$.

The dedicated kernel makes one unit-stride pass over input and output with the
$d$ weights in registers. The equivalent `GenericLinearOperator` sweeps the
output once per stencil non-zero ($2d+1$ passes) and resolves offsets at run
time.

## IsotropicStiffnessOperator

The fused stiffness operator computes the action of the FE stiffness matrix of
isotropic linear elasticity,

$$
\mathbf{f} = \mathbf{K}\mathbf{u}, \qquad
\mathbf{K} = \mathbf{B}^\top W \mathbf{C}\,\mathbf{B},
$$

for a nodal displacement $\mathbf{u}$ with $d$ components, without storing
strain or stress fields. The isotropic stiffness of element $e$ is
$\mathbf{C}_e = \lambda_e\,\mathbf{I}\otimes\mathbf{I} + 2\mu_e\,\mathbb{I}^\text{sym}$
with per-pixel Lamé parameters $\lambda_e, \mu_e$, so the strain energy density
is $\tfrac12\big(\lambda (\operatorname{tr}\boldsymbol{\varepsilon})^2 +
2\mu\,\boldsymbol{\varepsilon}:\boldsymbol{\varepsilon}\big)$ and the element
stiffness matrix decomposes into two geometry-only matrices:

$$
K_e = 2\mu_e\,G + \lambda_e\,V,
\qquad
G_{ai,bj} = \sum_q W_q\, \boldsymbol{\varepsilon}_q[\phi_a\mathbf{e}_i] :
                         \boldsymbol{\varepsilon}_q[\phi_b\mathbf{e}_j],
\qquad
V_{ai,bj} = \sum_q W_q\, \operatorname{tr}\boldsymbol{\varepsilon}_q[\phi_a\mathbf{e}_i]\,
                         \operatorname{tr}\boldsymbol{\varepsilon}_q[\phi_b\mathbf{e}_j],
$$

where $\boldsymbol{\varepsilon}_q[\phi_a\mathbf{e}_i]$ is the symmetric
gradient at quadrature point $q$ of the shape function of node $a$ in
direction $i$. $G$ and $V$ ($d\,2^d \times d\,2^d$, i.e. $24 \times 24$ in 3D)
are computed once from the grid spacing and the element (P1 or Q1, as for
`FEMGradientOperator`).

Consequences of this decomposition:

- **Memory:** a heterogeneous material costs two scalars per pixel
  ($\lambda, \mu$) instead of an element matrix ($576$ entries in 3D) or a
  per-quadrature-point stiffness tensor.
- **Traffic:** one apply reads $\mathbf{u}$, $\lambda$, $\mu$ and writes
  $\mathbf{f}$ — no strain and stress fields of $d^2$ components per
  quadrature point are written and read back, as they are when
  $\mathbf{B}^\top W \mathbf{C}\mathbf{B}$ is evaluated as three operators.

The kernel loops over nodes and gathers the contributions of the $2^d$
elements adjacent to each node, so every output value is written by exactly
one thread and no atomics are needed. Displacement, force and material fields
live on the same collection (the material of pixel $\mathbf{n}$ belongs to the
element whose lowest corner is node $\mathbf{n}$); all need one ghost layer on
each side, and the computable regions of the nodal and material fields must
coincide. The material ghosts must be communicated once after the material is
set.

```python
muGrid.IsotropicStiffnessOperator2D(grid_spacing, element=muGrid.FEMElement.q1)
muGrid.IsotropicStiffnessOperator3D(grid_spacing, element=muGrid.FEMElement.q1)
muGrid.IsotropicStiffnessOperator(spatial_dim, grid_spacing, element=...)
```

| Method | Computes |
|---|---|
| `apply(displacement, lambda_field, mu_field, force)` | $\mathbf{f} = \mathbf{K}\mathbf{u}$ |
| `apply_increment(displacement, lambda_field, mu_field, alpha, force)` | $\mathbf{f} \mathrel{+}= \alpha\,\mathbf{K}\mathbf{u}$ |
| `apply_uniform(displacement, lambda, mu, force)` | $\mathbf{K}\mathbf{u}$ for spatially uniform scalar $\lambda, \mu$ (no material fields) |
| `apply_uniform_increment(displacement, lambda, mu, alpha, force)` | $\mathbf{f} \mathrel{+}= \alpha\,\mathbf{K}\mathbf{u}$, uniform $\lambda, \mu$ |
| `apply_macro_rhs(lambda_field, mu_field, E_macro, force)` | $\mathbf{f} = \mathbf{B}^\top W \mathbf{C}\,\bar{\boldsymbol{\varepsilon}}$ for a uniform macroscopic strain $\bar{\boldsymbol{\varepsilon}}$ |
| `average_stress(displacement, lambda_field, mu_field, E_macro)` | local $\int \mathbf{C} : (\bar{\boldsymbol{\varepsilon}} + \boldsymbol{\varepsilon}(\mathbf{u}))\,dV$ over the rank's subdomain |
| `assemble_diagonal(lambda_field, mu_field, diagonal)` | $\operatorname{diag}(\mathbf{K})$ as a nodal field |
| `compute_sensitivity(forward_disp, forward_macro, costate_disp, costate_macro, g_shear, g_vol)` | per-pixel $\mathbf{a}_e^\top G\,\mathbf{b}_e$ and $\mathbf{a}_e^\top V\,\mathbf{b}_e$ |

`E_macro`, `forward_macro` and `costate_macro` are the $d \times d$ tensors
flattened row-major ($d^2$ values). `average_stress` returns the $d^2$
components of the un-reduced volume integral; sum it over ranks and divide by
the domain volume to obtain $\langle\boldsymbol{\sigma}\rangle$. In
`compute_sensitivity`, $\mathbf{a}_e$ and $\mathbf{b}_e$ are the element
displacements of the total forward and costate strains
($\bar{\boldsymbol{\varepsilon}} + \boldsymbol{\varepsilon}(\mathbf{u})$), so
$\partial (\mathbf{a}^\top\mathbf{K}\mathbf{b}) / \partial\rho_e =
 \tfrac{d(2\mu_e)}{d\rho_e}\, g_\text{shear} + \tfrac{d\lambda_e}{d\rho_e}\, g_\text{vol}$
for any material interpolation $\lambda_e(\rho_e), \mu_e(\rho_e)$.
Properties: `G` and `V` (the $d\,2^d \times d\,2^d$ geometry matrices),
`apply_ghost_requirement` and `ghost_requirement`.

### Example: material fields and apply

```python
import numpy as np
import muGrid

def lame_parameters(E, nu):
    """Lamé parameters from Young's modulus E and Poisson's ratio nu."""
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))

nb_grid_pts = (32, 32)
dim = 2
grid_spacing = tuple(1.0 / n for n in nb_grid_pts)
stiffness = muGrid.IsotropicStiffnessOperator2D(grid_spacing)

comm = muGrid.Communicator()
decomposition = muGrid.CartesianDecomposition(comm, nb_grid_pts,
                                              ghosts=stiffness)

u = decomposition.real_field("displacement", (dim,))
f = decomposition.real_field("force", (dim,))
lam = decomposition.real_field("lambda")
mu = decomposition.real_field("mu")

# circular inclusion, ten times stiffer than the matrix
x, y = decomposition.coords
phase = ((x - 0.5) ** 2 + (y - 0.5) ** 2 < 0.25 ** 2).astype(float)
lam_m, mu_m = lame_parameters(1.0, 0.3)
lam_i, mu_i = lame_parameters(10.0, 0.3)
lam.p[...] = lam_m * (1 - phase) + lam_i * phase
mu.p[...] = mu_m * (1 - phase) + mu_i * phase
decomposition.communicate_ghosts(lam)       # once, after setting the material
decomposition.communicate_ghosts(mu)

def hessp(u_in, f_out):
    decomposition.communicate_ghosts(u_in)
    stiffness.apply(u_in, lam, mu, f_out)
```

`hessp` is the operator argument expected by
`muGrid.Solvers.conjugate_gradients` (see [Linear algebra](linalg.md#conjugate-gradients)).
The complete homogenization workflow — right-hand side
$-\mathbf{B}^\top W \mathbf{C}\bar{\boldsymbol{\varepsilon}}$ via
`apply_macro_rhs`, the CG solve and the effective stress via `average_stress`
— is described under [Examples](examples.md#homogenization).
