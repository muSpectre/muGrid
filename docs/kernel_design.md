# Operator Kernel Design

Design notes for the stencil operators in `src/libmugrid/operators/`: how the
kernels handle ghost regions, which region they compute, how data is laid out,
and how each operator is evaluated. The mathematical definitions and the Python
interface are documented in `docs/operators.md`; the notation follows that page
(dimension $d$, grid point $\mathbf{n}$, spacing $h_\alpha$, unit vector
$\mathbf{e}_\alpha$, corner offsets $\mathbf{a} \in \{0,1\}^d$).

## Operator overview

| Operator | Input → output | Stencil (pixels) | Class template |
|---|---|---|---|
| `GenericLinearOperator` | nodal → nodal or quadrature | arbitrary (offset $\mathbf{o}$, shape $\mathbf{S}$) | — |
| `FEMGradientOperator` | nodal → quadrature | $2^d$ corner nodes | `FEMGradientOperator<Element>` |
| `LaplaceOperator` | nodal → nodal | $2d+1$ points | `LaplaceOperator<Dim>` |
| `IsotropicStiffnessOperator` | nodal → nodal | $3^d$ nodes | `IsotropicStiffnessOperator<Dim>` |

All operators except the stiffness operator derive from `LinearOperator`
(`linear.hh`); the stiffness operator derives from `MaterialOperator`, which
carries the `apply(displacement, λ, μ, force)` signature.

## Common design principles

### No boundary conditions in the kernels

Kernels never wrap indices. Periodicity and distributed-memory parallelism are
implemented by the ghost layers, which `CartesianDecomposition::communicate_ghosts()`
fills before an operator runs: for periodic boundaries the ghost layers hold
copies from the opposite side of the domain (possibly owned by another rank);
for other boundary conditions the caller writes the appropriate ghost values
(e.g. zeros for homogeneous Dirichlet data). One kernel therefore serves
periodic and non-periodic problems, serial and MPI runs, and host and device.

### Stencil requirements and the computed region

Every operator has intrinsic **stencil requirements**: the number of ghost
layers $\ell_\alpha$ (left) and $r_\alpha$ (right) that one output point reads
beyond itself along axis $\alpha$.

| Operator | apply: left / right | transpose: left / right |
|---|---|---|
| `GenericLinearOperator` | $\max(0, -o_\alpha)$ / $\max(0, o_\alpha + S_\alpha - 1)$ | mirrored |
| `FEMGradientOperator` | 0 / 1 | 1 / 0 |
| `LaplaceOperator` | 1 / 1 | 1 / 1 |
| `IsotropicStiffnessOperator` | 1 / 1 | — |

`LinearOperator::get_apply_ghost_requirement()` derives the apply requirement
from `get_offset()` and `get_stencil_shape()`; the default
`get_transpose_ghost_requirement()` is its mirror image, because a gather-style
transpose reads at negated offsets. `get_ghost_requirement()` is the elementwise
maximum.

For a buffer of $L_\alpha$ points along axis $\alpha$ (interior plus ghosts),
the operator writes every index $i_\alpha$ with
$\ell_\alpha \le i_\alpha < L_\alpha - r_\alpha$, i.e. every point whose stencil
lies inside the buffer. The iteration bounds follow from the stencil, not from
the ghost width: with three ghost layers on each side and a requirement of one,
the operator also writes the two outer ghost layers that have valid stencil
data. Kernel pointers are offset to the first computable point, so index $0$ is
the first computed point and negative indices address the left ghost region.

### Upfront validation

Field compatibility (collection type, storage order, component counts) and
ghost sufficiency are checked once per call before the kernel runs. An
insufficient ghost region raises an error naming the operator and the missing
side; the kernels contain no bounds checks.

### Stencil offset convention

The offset places the stencil relative to the output point. The FE operators
use offset $\mathbf{0}$ and stencil shape $(2, \ldots, 2)$: the quadrature
points of the element of pixel $\mathbf{n}$ read the nodes $\mathbf{n} +
\mathbf{a}$. `FEMGradientOperator` exposes this as `offset`, `stencil_shape`
and `coefficients`, so `GenericLinearOperator(op.offset, op.coefficients)`
reproduces it.

### Memory layout

Host fields are array-of-structures (AoS), device fields structure-of-arrays
(SoA). For a vector field $u_c$ on an $L_0 \times L_1 \times L_2$ buffer with
$n_c$ components, the element strides are

| | component | $x$ | $y$ | $z$ |
|---|---|---|---|---|
| AoS (CPU) | $1$ | $n_c$ | $n_c L_0$ | $n_c L_0 L_1$ |
| SoA (GPU) | $L_0 L_1 L_2$ | $1$ | $L_0$ | $L_0 L_1$ |

SoA makes consecutive GPU threads (consecutive $x$) read consecutive addresses
for every component, which coalesces the loads; AoS keeps the components of a
pixel in one cache line on the CPU. Kernels are written against these strides;
CPU and GPU kernels are separate implementations (`*_gpu.cc` files, compiled as
CUDA or HIP) dispatched through `kernel_dispatcher.hh`.

### Precision

Operators are monomorphic in their scalar type at the kernel level. Generic,
Laplace and stiffness operators provide `Real` and `Real32` overloads on one
object; `FEMGradientOperator` is instantiated per precision (`_32` suffix in
the bindings). An overload that an instance does not implement throws.

---

## GenericLinearOperator

The operator is stored as a sparse matrix built from the coefficient tensor
$S_{kqi}(\mathbf{s})$ for the current grid and component count; coefficients
whose magnitude is at most a fixed fraction of the largest one are dropped
(a relative threshold, so the result does not depend on the units of the
coefficients). Two cached orderings exist: row-major (grouped by output
quadrature component, so consecutive entries write the same output location in
`apply`) and column-major (grouped by nodal component, the same property for
`transpose`). Device copies are created lazily and rebuilt when the grid
changes. Coefficients are kept in double precision and cast to the field
precision inside the kernels.

The kernel sweeps the output once per non-zero coefficient, resolving the
input offset of each entry at run time. An apply therefore costs
$O(\mathrm{nnz}\,N)$ operations and $\mathrm{nnz}$ passes over the output;
the dedicated operators below make a single pass.

---

## FEMGradientOperator

The element tables (`fem_element.hh`) hold, for each element type, the
reference shape-function gradients $B_{q\alpha\mathbf{a}}$ (layout
`[quad][dim][node]`, nodes indexed by the binary corner offset with $x$
fastest) and the quadrature volume fractions $\bar{w}_q$. The physical gradient
is $B_{q\alpha\mathbf{a}}/h_\alpha$ and the physical weight
$W_q = \bar{w}_q \prod_\alpha h_\alpha$. Kernels are templated on the element,
so a new element only needs a new table.

| Element | nodes | quadrature points | $\bar{w}_q$ |
|---|---|---|---|
| `P1Tri2D` | 4 | 2 (triangles $\{0,1,2\}$, $\{1,2,3\}$) | $1/2, 1/2$ |
| `P1Tet3D` | 8 | 5 (central tetrahedron, four corner tetrahedra) | $1/3, 1/6, 1/6, 1/6, 1/6$ |
| `Q1Quad2D` | 4 | 4 ($2 \times 2$ Gauss) | $1/4$ each |
| `Q1Hex3D` | 8 | 8 ($2 \times 2 \times 2$ Gauss) | $1/8$ each |

The 3D P1 split (central tetrahedron on the odd-parity corners $(1,0,0)$,
$(0,1,0)$, $(0,0,1)$, $(1,1,1)$; corner tetrahedra with apex $(0,0,0)$,
$(1,1,0)$, $(1,0,1)$, $(0,1,1)$) is not the six-tetrahedron Kuhn
triangulation: it has no preferred diagonal, which avoids grid-anisotropy
artefacts. The same split is used in every voxel so that the operator is
translation invariant; the assembled stiffness is a symmetric 19-point stencil.

**Apply** (gradient, nodal → quadrature): for the element of pixel
$\mathbf{n}$,
$(\mathbf{B}\mathbf{u})_{i\alpha q}(\mathbf{n}) = h_\alpha^{-1}\sum_{\mathbf{a}}
B_{q\alpha\mathbf{a}}\,u_i(\mathbf{n}+\mathbf{a})$. It reads node
$\mathbf{n} + \mathbf{a}$ with $a_\alpha \in \{0, 1\}$, hence one right ghost
layer of the nodal field.

**Transpose** (weighted divergence, quadrature → nodal):
$(\mathbf{B}^\top W \boldsymbol{\sigma})_i(\mathbf{n}) =
\sum_{\mathbf{a}} \sum_{q,\alpha} W_q\, h_\alpha^{-1} B_{q\alpha\mathbf{a}}\,
\sigma_{i\alpha q}(\mathbf{n} - \mathbf{a})$. Each owned node gathers from the
$2^d$ elements that contain it, including the element $\mathbf{n} - \mathbf{1}$
in the left ghost layer of the quadrature field; it needs one left ghost layer.
The gather writes only owned nodes, so `reduce_ghosts` must not be called on the
result. With explicit `weights`, $W_q$ is replaced by the given values.

---

## LaplaceOperator

With per-axis weights $w_\alpha = s/h_\alpha^2$ ($w_\alpha = s$ without grid
spacing), the kernel evaluates

$$
(Lu)(\mathbf{n}) = \sum_\alpha w_\alpha\big(u(\mathbf{n}+\mathbf{e}_\alpha)
  - 2u(\mathbf{n}) + u(\mathbf{n}-\mathbf{e}_\alpha)\big)
$$

on every point with $1 \le i_\alpha < L_\alpha - 1$, in one unit-stride pass
over input and output with the $d$ weights held in registers; the fixed access
pattern lets the compiler vectorise the $x$ loop. Cost: $O(N)$ with $2d+1$
reads (mostly cache hits) and one write per point. The operator is symmetric,
so `transpose` calls the same kernel; quadrature weights are rejected because
the operator has no quadrature points.

---

## IsotropicStiffnessOperator

### Element matrices

At construction the operator computes, from the element table and the grid
spacing, the $d\,2^d \times d\,2^d$ matrices (row/column index
$(\mathbf{a}, i)$ = node, displacement component)

$$
G = \sum_q W_q\, \mathbf{B}_q^\top \mathbb{I}^\text{sym} \mathbf{B}_q,
\qquad
V = \sum_q W_q\, (\mathbf{B}_q^\top \mathbf{m})(\mathbf{m}^\top \mathbf{B}_q),
$$

where $\mathbf{B}_q$ maps element displacements to the strain in Voigt
notation at quadrature point $q$, $\mathbb{I}^\text{sym} =
\operatorname{diag}(1, \ldots, 1, \tfrac12, \ldots, \tfrac12)$ ($d$ normal and
$d(d-1)/2$ engineering-shear entries) and $\mathbf{m}$ selects the trace
(ones on the normal entries). With $\mathbf{C} = \lambda\,\mathbf{m}\mathbf{m}^\top
+ 2\mu\,\mathbb{I}^\text{sym}$ this gives the element stiffness
$K_e = 2\mu_e G + \lambda_e V$. A heterogeneous material therefore needs two
scalars per pixel instead of $(d\,2^d)^2$ matrix entries ($576$ in 3D). The
operator also stores the element-averaged gradient
$\bar{D}_{\alpha\mathbf{a}} = h_\alpha^{-1}\sum_q \bar{w}_q B_{q\alpha\mathbf{a}}$,
used by the macroscopic-strain helpers.

### Gather pattern

Material fields are indexed like the nodal fields: $\lambda(\mathbf{n})$,
$\mu(\mathbf{n})$ belong to the element whose lowest corner is node
$\mathbf{n}$. Node $\mathbf{n}$ is corner $\mathbf{a}$ of element
$\mathbf{n} - \mathbf{a}$, so the kernel computes, for every node,

$$
f_i(\mathbf{n}) = \sum_{\mathbf{a} \in \{0,1\}^d} \sum_{\mathbf{b}, j}
  \Big(2\mu(\mathbf{n}-\mathbf{a})\, G_{\mathbf{a}i,\mathbf{b}j}
       + \lambda(\mathbf{n}-\mathbf{a})\, V_{\mathbf{a}i,\mathbf{b}j}\Big)\,
  u_j(\mathbf{n} - \mathbf{a} + \mathbf{b}).
$$

It reads elements at offset $-1$ and nodes at offset $+1$, hence one ghost
layer on each side of both the displacement and the material fields; the
computable regions of the two collections must coincide. Each output value is
written by exactly one thread (one thread per node on the GPU), so no atomics
are needed. On the GPU the element matrices live in `__constant__` memory (all
threads of a warp read the same entry, which the constant cache broadcasts);
the 3D apply stages data through shared memory.

The other kernels reuse this structure: `apply_uniform` replaces the material
fields by two scalars; `apply_macro_rhs` replaces $\mathbf{u}$ by the constant
element vectors of a uniform strain; `assemble_diagonal` accumulates
$2\mu\,G_{\mathbf{a}i,\mathbf{a}i} + \lambda\,V_{\mathbf{a}i,\mathbf{a}i}$;
`average_stress` integrates
$\mathbf{C}:(\bar{\boldsymbol{\varepsilon}}+\boldsymbol{\varepsilon}(\mathbf{u}))$
element-locally (on the GPU with a two-pass tree reduction);
`compute_sensitivity` evaluates $\mathbf{a}_e^\top G \mathbf{b}_e$ and
$\mathbf{a}_e^\top V \mathbf{b}_e$ per pixel.

---

## File structure

```
src/libmugrid/operators/
├── linear.hh                     # LinearOperator base, ghost requirements
├── kernel_dispatcher.hh          # host/device kernel dispatch
├── convolution_kernels_cpu.hh    # sparse-operator kernels (host)
├── convolution_kernels_gpu.hh    # sparse-operator kernels (device)
├── generic.hh/.cc, generic_gpu.cc            # GenericLinearOperator
├── fem_element.hh                # P1/Q1 element tables
├── fem_gradient.hh, fem_gradient_gpu.cc      # FEMGradientOperator<Element>
├── laplace.hh/.cc, laplace_gpu.cc            # LaplaceOperator<Dim>
├── nodal_moments.hh/.cc, nodal_moments_gpu.cc  # NodalMomentOperator
├── transfer.hh/.cc               # GridTransfer (multigrid prolongation/restriction)
└── solids/
    ├── isotropic_stiffness.hh    # MaterialOperator, IsotropicStiffnessOperator<Dim>
    ├── isotropic_stiffness_2d.cc # 2D element matrices and host kernels
    ├── isotropic_stiffness_3d.cc # 3D element matrices and host kernels
    └── isotropic_stiffness_gpu.cc  # device kernels
```

---

## Testing

`tests/python_isotropic_stiffness_operator_tests.py` covers the stiffness
operator:

- `test_compare_with_generic_operator`, `test_macro_rhs_and_average_vs_generic`:
  the fused kernels against an explicit $\mathbf{B}^\top W\mathbf{C}\mathbf{B}$
  built from generic operators;
- `TestUnitImpulse2D/3D`: response to a unit displacement at single nodes, with
  periodic and non-periodic ghost data;
- `test_symmetry`: symmetry of $\mathbf{K}$;
- `TestValidationGuard2D/3D`: errors for missing ghosts and mismatched
  material fields;
- `TestIsotropicStiffnessOperatorGPUCorrectness`, `TestGPUUnitImpulse`:
  GPU against CPU results;
- `TestFEMElements`, `TestAssembleDiagonal`, `TestIsotropicStiffnessApplyUniform`:
  patch test for both elements, diagonal assembly, uniform-coefficient apply.

The other operators are covered by `python_convolution_operator_tests.py`,
`python_fem_element_tests.py`, `python_fem_gradient_transpose_tests.py`,
`python_fem_gradient_gpu_tests.py`, `python_laplace_operator_tests.py` and the
C++ suite `test_convolution_operator.cc`.
