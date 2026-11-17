# Heterogeneous multigrid preconditioner

`muGrid.Multigrid.MultigridPreconditioner` is a geometric multigrid V(ν,ν)-cycle for
**heterogeneous** nodal operators, used as a fixed symmetric positive definite
preconditioner of [conjugate gradients](linalg.md#conjugate-gradients).

- **Iterations:** its CG count does not depend on the grid or the material contrast. With
  contrast 10⁴ it needs 11–12 iterations at 64³ and 128³, where Green needs 266–533 and
  Green-Jacobi 76–203 (see [Performance](#performance)).
- **What it needs:** no FFT and no element knowledge. It builds its hierarchy from the
  operator's action alone.
- **Where it runs:** CPU and GPU, under MPI, in double or single precision.

The cycle is the one of muFFTTO's multigrid preconditioner:
- Chebyshev–Jacobi smoothing;
- multilinear transfer;
- an exact solve on the coarsest grid.

Two things differ from it:

1. **The coarse operators are Galerkin products R A P, measured by probing.**
   - The multigrid knows the fine operator only through `operator(x, y)`.
   - No element, quadrature rule or material model enters the hierarchy.
2. **The Chebyshev interval uses a guaranteed upper bound of λmax(D⁻¹A) on every
   level, never an estimate.**
   - An eigenvalue estimate, such as power iteration times a safety factor, can
     miss a mode that localises after a material update. One example is a crack
     cutting through the cell in a phase-field fracture run.
   - The Chebyshev polynomial amplifies modes above the interval, so such a miss
     breaks the cycle.

## Usage

```python
import muGrid
from muGrid.Multigrid import MultigridPreconditioner, ScaledLocalMatrices
from muGrid.Solvers import conjugate_gradients

comm = muGrid.Communicator()
N = 64
decomposition = muGrid.CartesianDecomposition(
    comm, [N] * 3, nb_subdivisions=[1] * 3,
    nb_ghosts_left=[1] * 3, nb_ghosts_right=[1] * 3)
op = muGrid.IsotropicStiffnessOperator3D([1 / N] * 3, muGrid.FEMElement.q1)

# Material s(x) C(λ0, μ0): per-pixel Lamé fields and the scale s >= 0
lam, mu = decomposition.real_field("lambda"), decomposition.real_field("mu")
scale = decomposition.real_field("scale")
scale.p[...] = s                          # any array of the subdomain's shape
lam.p[...], mu.p[...] = lam0 * s, mu0 * s
for f in (scale, lam, mu):
    decomposition.communicate_ghosts(f)

def apply(u, f):                          # y = A x; exchanges the ghosts of x
    decomposition.communicate_ghosts(u)
    op.apply(u, lam, mu, f)

mg = MultigridPreconditioner(
    apply, decomposition, 3,
    local_matrices=ScaledLocalMatrices.isotropic_stiffness(op, lam0, mu0, scale),
    communicator=comm)
mg.setup()                                # after every change of the material
conjugate_gradients(comm, decomposition.collection, b, x,
                    hessp=apply, prec=mg, rtol=1e-6)
```

`MultigridPreconditioner(operator, decomposition, nb_components, ...)`:

| argument | meaning |
|---|---|
| `operator` | `operator(x, y)`: y = A x on float64 nodal fields of `decomposition`, exchanging the ghosts of `x` itself (the `hessp` contract of `conjugate_gradients`) |
| `decomposition` | `CartesianDecomposition` or `FFTEngine` with at least one ghost layer per side |
| `nb_components` | unknowns per node (1, 2 or 3), one nodal point per pixel |
| `local_matrices` | a `ScaledLocalMatrices` (below); makes the smoother bound a constant per level. Without it the bound is computed from the operator's entries |
| `diagonal` | optional `diagonal(field)` writing diag(A) (e.g. `IsotropicStiffnessOperator.assemble_diagonal`); default: from `local_matrices`, else probed |
| `degree`, `alpha` | Chebyshev steps per smoothing (2) and interval [b/α, b] (30) |
| `n_coarsest` | coarsen while every axis stays even and at least this wide (4) |
| `n_agglomerate` | MPI: grids of this size and coarser are gathered and solved on rank 0 (16) |
| `precision` | `"double"` or `"single"`: the cycle runs in float32 between float64 CG vectors |
| `operator_single` | the operator on float32 fields; with `"single"` the fine level then runs in float32 too |
| `communicator` | needed under MPI with a `CartesianDecomposition` (an `FFTEngine` brings its own) |
| `coarsening` | `"galerkin"` (default) or a coarsening rule such as `IsotropicRediscretisation` (below) |

`ScaledLocalMatrices(matrices, scales)` declares A = Σ_pixels Σ_k s_k(pixel) K_k:
- `matrices` has shape `(n_k, 2^d nc, 2^d nc)`. These are the fixed local matrices of a pixel's corner
  nodes (row `a nc + i`: component `i` of corner `a = Σ_α a_α 2^α`).
- `scales` is a field with `n_k` values per pixel, all ≥ 0.

Helpers build it without hard-coding an element:
- `ScaledLocalMatrices.isotropic_stiffness(op, λ0, μ0, s)`: the fused operator with λ = sλ₀, μ = sμ₀.
- `ScaledLocalMatrices.fem(gradient, C, s_q)`: Bᵀ W (s_q C) B with a `FEMGradientOperator`, one family
  per quadrature point.
- `probe_pixel_matrix(apply, decomposition, nc, pixel)`: any other operator, probed on one pixel of a
  4^d grid. That grid must keep the problem's pixel size, because 3D element matrices scale with h.

For the fused isotropic operator the coarse levels can instead be re-discretised (see
[Re-discretised coarse levels](#re-discretised-coarse-levels-option)):

```python
from muGrid.MultigridCoarsening import IsotropicRediscretisation

rule = IsotropicRediscretisation(muGrid.FEMElement.q1, [1 / N] * 3, (lam, mu),
                                 lambda_over_mu=lam0 / mu0)   # None: λ, μ independent
mg = MultigridPreconditioner(
    apply, decomposition, 3,
    local_matrices=ScaledLocalMatrices.isotropic_stiffness(op, lam0, mu0, scale),
    communicator=comm, coarsening=rule)
```

The [homogenization example](examples.md#homogenization) runs it with
`-P heterogeneous-multigrid`.

## How it works

### Hierarchy and transfer

- **Grids:** N, N/2, …, down to the first grid that cannot be halved with every axis ≥ `n_coarsest`.
  Each coarse level is a `CartesianDecomposition` with one ghost layer, on the fine level's device.
- **Transfer:** P is multilinear interpolation acting component-wise and R = Pᵀ (`muGrid.GridTransfer`,
  host and device). P reproduces linear fields, so the rigid translations and constant strains lie in
  its range.

### Galerkin coarse operators

- **Coarse stencil:** A₁ = R A₀ P couples each coarse node with its 3^d neighbours. The blocks
  (nc × nc) vary from node to node.
- **Probing:**
  - The coarse nodes are coloured so that nodes at periodic distance 1 or 2 differ: period 3 on axes
    divisible by 3, period 4 on axes divisible by 4, otherwise blocks of 3 and 4.
  - For every colour c and component j, y = R A₀ P e_{c,j} is computed. At every coarse node J, y(J)
    is block column j of A₁(J, I), I being the one node of colour c next to J.
  - Cost: nc ∏ k_α fine applies, k_α the colours of axis α. In 3D elasticity on power-of-two grids that
    is 3 · 4³ = 192.
- **Deeper levels:** A_{l+1} = R A_l P is formed algebraically from the explicit stencils, accumulated
  in double precision.
- **Storage:** the operator is symmetric, so a stencil stores the blocks of the 13 (3D) or 4 (2D)
  "positive" offsets plus the upper triangle of the centre block. That is 123 values per node in 3D
  elasticity. `muGrid.VariableStencilOperator` applies it (host and device kernels, float64 and
  float32).

### Re-discretised coarse levels (option)

- **What:** with `coarsening=IsotropicRediscretisation(...)` every coarse level applies the fine
  operator's own fused kernel at its spacing 2^l h, with λ_l, μ_l the arithmetic means of the 2^d child
  voxels of the next finer level (as muFFTTO's multigrid). No probing, no RAP, no stencils.
- **Element:** Q1 only (bilinear 2D, trilinear 3D), where multilinear interpolation is nested: for a
  homogeneous material the re-discretised levels equal the Galerkin ones. P1 is refused.
- **Bound and coarsest level:** a re-discretised level is again Σ_cells s K of one element matrix
  (one family with `lambda_over_mu`, else the two families λ, μ), so the diagonal, the guaranteed
  bound (the mediant inequality of the element matrix: a constant) and the coarsest dense matrix all
  come from its local matrices.
- **Safeguard:** at construction the rule's operator at spacing h with the fine λ, μ is compared with
  the caller's operator on a random vector, which catches a wrong spacing, element or field.
- **MPI:** the material of the last distributed level is gathered on rank 0, which averages it down its
  serial tail.
- **Trade-off** (CH3D, contrast 1e4, 128³, float32, laptop GPU): 14 instead of 12 CG iterations, a
  V-cycle of 16.0 instead of 16.3 ms, a repeated setup of 0.05 instead of 0.80 s, and 80 B/voxel less
  memory. It pays when the material changes often (topology optimisation) or memory is tight.

### Smoother and its guaranteed bound

- **Smoother:** `degree` Chebyshev steps for A x = b, preconditioned by D⁻¹ = diag(A)⁻¹ on [b/α, b]
  (Saad, *Iterative Methods for Sparse Linear Systems*, Alg. 12.1). They run before and after the
  coarse correction, so the cycle is a fixed SPD operator.
- **Interval:** b must bound λmax(D⁻¹A) from above. Overestimating it only weakens the smoother. Below
  it the Chebyshev polynomial grows fast: for degree 2, |p| exceeds 1 beyond 1.033 b.
- **Mediant inequality:** if A = Σ_e A_e with A_e ⪰ 0 and D = Σ_e diag(A_e), then for every x
  ```latex
  \frac{x^\top A x}{x^\top D x} = \frac{\sum_e x^\top A_e x}{\sum_e x^\top \mathrm{diag}(A_e)\, x}
  \le \max_e \lambda_{\max}\!\left(\mathrm{diag}(A_e)^{-1} A_e\right).
  ```
- **With `ScaledLocalMatrices`:**
  - A Galerkin level is the sum of the atoms T_posᵀ (s K_k) T_pos, with T_pos the multilinear
    interpolation from the corners of a level-l cell to the fine pixel at position `pos` inside it.
  - This gives a constant c_l = max over (k, pos) of λmax(diag(M)⁻¹ M), M = T_posᵀ K_k T_pos. It is
    independent of the material and of the contrast, and tight over all s ≥ 0.
  - c_l is computed once, batched, up to 2¹⁵ atoms per level. For 3D fused Q1 with K = 1, μ = 1/2
    (ν ≈ 0.29) the values are 5.06, 6.25, 8.27, 8.82, 8.95 at levels 0–4.
- **Coarse levels:** they use the smaller of c_l and the Gershgorin bound
  max_i Σ_j |a_ij| / √(a_ii a_jj) of their actual stencil. Both are guaranteed upper bounds, and the
  Gershgorin one is usually the tighter there.
- **Without local structure:** the operator is a black box, and the fine level's Gershgorin bound comes
  from probed entries, which costs one more probing pass.
- **Safeguards:**
  - At construction, the operator is compared with Σ s_k K_k on random patches. A mismatch raises.
  - Every `setup()` checks s ≥ 0 globally. A negative scale raises.
  - Eigenvalue estimates are used only in the tests, to check the bounds.

### Coarsest level, precision, MPI, GPU

- **Coarsest level:** a dense pseudo-inverse assembled from its stencil. The periodic operator is
  singular; its constants are cut off.
- **Single precision:**
  - Stencils, work vectors and the pseudo-inverse are stored in float32. The stencils are measured
    straight into them: a float32 fine level probes with its float32 operator (its entries are float32
    values), RAP accumulates in double and rounds each entry once, the Gershgorin bound promotes to
    double, and the coarsest matrix is factorised in double on the host.
  - The pseudo-inverse cuts at 1e-5 instead of 1e-12 for a float32 coarsest stencil (also under a
    double fine level), whose null space rounding leaves at about 1e-7.
  - The Jacobi step and the float32 ↔ float64 casts run over whole contiguous buffers on a device.
- **MPI:**
  - Levels are distributed with the fine level's subdivisions while every rank keeps an even number of
    at least 4 nodes per axis and the grid stays ≥ `n_agglomerate`. The first coarsening is
    distributed even below `n_agglomerate`.
  - The last distributed level is gathered on rank 0, and a serial tail of the hierarchy runs there.
    Gather and scatter go through host memory.
- **GPU:** every kernel of the cycle runs on the device, so the cycle makes no host round trips. The
  setup uses CuPy array operations.

## Cost

Per fine voxel, 3D elasticity:

| | float64 | float32 cycle |
|---|---|---|
| fine level: r, d, t, D⁻¹ (plus own b, x in float32) | 96 B | 72 B |
| level-1 stencil | 123 B | 62 B |
| level-1 vectors and levels ≥ 2 | ~38 B | ~19 B |

A setup needs about 47 B/voxel beyond these fields (the double fine diagonal, the RAP block; 64³), and
the safeguard's two check vectors are freed after construction. With re-discretised coarse levels the
level stencils are replaced by λ_l, μ_l (about 1 B/voxel in total).

Setup per material update:
- 192 fine applies of the probing (3D, power-of-two grids);
- RAP;
- the bounds;
- the coarsest pseudo-inverse;
- once per hierarchy, c_l.

A V-cycle costs about 4 fine applies plus level-1 stencil applies, which move about twice the bytes of a
fused fine apply.

## Validation

`tests/python_heterogeneous_multigrid_tests.py` checks:
- the stencil kernel against an independent dense assembly, in float64 and float32;
- probed R A P equal to the dense Pᵀ A P to 1e-12, for fused elasticity (2D and 3D, Q1 and P1, contrast
  10³) and for scalar FE chains with a coefficient per quadrature point;
- algebraic RAP equal to Pᵀ A P;
- the bound ≥ the exact λmax(D⁻¹A) on every level, for homogeneous, random 10⁴-contrast, cracked and
  phase-field materials;
- the cycle is SPD;
- CG iterations independent of the grid: 16 at 2D 32², 64² and 128², phase field, contrast 10⁴;
- a crack appearing after setup: the bound stays valid and CG stays fast;
- anisotropic materials per quadrature point;
- CPU = GPU;
- float32 = float64 iterations;
- 1, 2 and 4 MPI ranks = serial (V-cycle output to 1e-9, bounds and iterations).

## Performance

GPU: NVIDIA RTX PRO 1000 (laptop). PCG to a relative residual of 1e-6, float32 preconditioners between
float64 CG vectors.

Problem:
- the Cahn–Hilliard geometry of muFFTTO's multigrid study;
- a tanh interface 2h wide;
- material s C(K = 1, μ = 1/2) with s = φ²(1 − k) + k and k = 10⁻⁴, i.e. contrast 10⁴;
- fused Q1 operator, load E_xx.

| preconditioner | N | iterations | one apply (ms) | CG (s) | setup (s) | GPU B/voxel |
|---|---|---|---|---|---|---|
| heterogeneous multigrid | 64³ | 11 | 3.6 | 0.12 | 0.85 | 680 |
| muFFTTO multigrid (re-discretised) | 64³ | 12 | 3.5 | 0.13 | 1.45 | 536 |
| Green (reference C₀) | 64³ | 266 | 0.30 | 1.63 | 0.01 | 456 |
| Green-Jacobi | 64³ | 76 | 0.33 | 0.48 | 0.04 | 504 |
| heterogeneous multigrid | 128³ | 12 | 20.6 | 0.95 | 1.75 | 530 |
| muFFTTO multigrid (re-discretised) | 128³ | 14 | 16.5 | 1.02 | 1.70 | 417 |
| Green (reference C₀) | 128³ | 533 | 2.5 | 24.8 | 0.03 | 349 |
| Green-Jacobi | 128³ | 203 | 3.4 | 9.8 | 0.14 | 387 |

- **Accuracy:** the homogenised stress of all four agrees to the CG tolerance. Against the muFFTTO
  multigrid it agrees to 13 digits.
- **muFFTTO's own non-fused operators:** Galerkin needed 10–25 % fewer iterations than the
  re-discretised muFFTTO hierarchy on the same operator:
  - linear triangles: 20–30 against 25–38 iterations at 64²–256²;
  - Q1 quadrilaterals: 30 against 39;
  - Q1 hexahedra: 32 against 37;
  - one-point hexahedra with hourglass control: 33–39 against 37–44.
- **Reproducing:** the scripts are in `benchmarks/heterogeneous_multigrid/`. The
  [preconditioner benchmark](benchmark_homogenization_preconditioner.md) runs the multigrid on its own
  problem with `--iter-preconditioners`/`--timing-preconditioners heterogeneous-multigrid`, and runs
  this phase field with `--studies phase_field`.

## Relation to `MultigridReferencePreconditioner`

| | `MultigridReferencePreconditioner` | `MultigridPreconditioner` |
|---|---|---|
| operator | the uniform reference Kʳᵉᶠ | the heterogeneous A itself |
| role | replaces the FFT inside Green or Green-Jacobi | the whole preconditioner |
| coarse operators | re-discretised Kʳᵉᶠ | Galerkin R A P, probed |
| smoother | damped Jacobi, ω from a power estimate | Chebyshev–Jacobi on a guaranteed bound |
| device, MPI | host, serial | host and GPU, MPI |

## Limitations

- **Nodes:** one nodal point per pixel, with 1–3 components. The interpolation is multilinear; P1
  elements still get an exact Galerkin hierarchy for that interpolation, just not the P1-nested one.
- **Grid sizes:** every axis must stay even down to the coarsest level.
- **MPI:** every rank needs an even number of at least 4 nodes per axis on the levels it distributes.
- **Setup cost:** the probing costs nc ∏ k_α fine applies at every material update. In float64 on GPUs
  with weak double throughput this dominates the setup. For the fused isotropic operator the
  re-discretised levels avoid it.

## References

- Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM (2003), Alg. 12.1 (Chebyshev
  acceleration).
- M. Ladecký et al., Appl. Math. Comput. 446 (2023) 127835 (the reference preconditioner compared
  above).
