# Examples

The scripts in `examples/` are small, runnable programs that each demonstrate
one part of µGrid. This page states what each one demonstrates, the problem it
solves and how to run it. The operators, transforms and solvers they use are
defined on the [Operators](operators.md), [FFT](fft.md) and
[Linear algebra](linalg.md) pages.

Run an example from the repository root with the Python that has µGrid
installed (or with `PYTHONPATH` pointing at a build tree, see
[Installation](installation.md)):

```bash
python examples/poisson.py -n 64,64
```

`poisson.py`, `homogenization.py` and the scripts that drive them also
import `muTimer` (a dependency of the µGrid package) and `NuMPI` (part of the
`test` extra); MPI runs need an MPI-enabled µGrid; GPU runs (`-d gpu`) need a CUDA or HIP build and CuPy. The
examples `components.py`, `fft_roundtrip.py`, `field_collection.py`,
`fourier_derivative.py`, `gradient.py`, `homogenization.py`, `io.py`,
`poisson.py` and `sub_points.py` are registered as CTest tests (`example_*`),
together with 3D and MPI variants of the Poisson and homogenization examples.

## Fields and collections

| Script | Demonstrates |
|---|---|
| `field_collection.py` | Creating a `GlobalFieldCollection`, get-or-create with `real_field`, writing through the pixel view `.p`. |
| `scalar.py` | Shapes of a scalar field (`[1, nx, ny]`, one sub-point) versus a one-component field (`[1, 1, nx, ny]`). |
| `components.py` | A tensor-valued field (`(2, 2)` components per pixel). |
| `sub_points.py` | Fields on sub-points (`nb_sub_pts={"element": 5}`) and the two views `.s` (sub-point axis separate) and `.p` (sub-points folded into the components). |
| `io.py` | Writing a field collection to NetCDF with `FileIONetCDF` (skipped when µGrid is built without NetCDF). |

## FFT round trip

`fft_roundtrip.py` transforms a random real field on a $16 \times 20$ grid
forward and back and checks that multiplying by `engine.normalisation`
$= 1/(16 \cdot 20)$ recovers the input to $10^{-14}$, i.e. that
`ifft(fft(u))` $= \prod_\alpha N_\alpha\,u$.

## Fourier derivative

`fourier_derivative.py` differentiates $f(x, y) = \sin x \cos y$ on the periodic
domain $[0, 2\pi)^2$ with $64 \times 64$ points spectrally: it transforms $f$,
multiplies by $i k_x$ and $i k_y$ with $k_\alpha = 2\pi q_\alpha N_\alpha /
L_\alpha$ (from `engine.fftfreq`), transforms back, and checks
$\partial_x f = \cos x \cos y$ to $10^{-10}$. Spectral differentiation is exact
for band-limited data up to rounding.

## Stencil gradient

`gradient.py` builds the P1 (two triangles per pixel) gradient as a
`GenericLinearOperator` with two output components, two quadrature points and
a $2 \times 2$ stencil, sizes the ghost layers from
`op.apply_ghost_requirement`, fills them periodically with `np.pad(...,
mode="wrap")`, and applies it to $u = \sin(2\pi x/N_x)$ on a $1024 \times 2$
grid. The result is checked against the analytic derivative at the edge
midpoints, $\frac{2\pi}{N_x}\cos\big(2\pi (x + \tfrac12)/N_x\big)$, to
$10^{-8}$ (the forward difference is second-order accurate at the midpoint).
It runs serially or under MPI.

```bash
python examples/gradient.py
mpiexec -n 2 python examples/gradient.py
```

## Poisson equation

`poisson.py` solves

$$
-\nabla^2 u = f \quad \text{on } [0, 1)^d, \text{ periodic},
\qquad
f(\mathbf{x}) = g(\mathbf{x}) - \langle g\rangle,\quad
g = \Big(1 + \prod_{\alpha} \cos 2\pi x_\alpha\Big)^{10},
$$

in 2D or 3D with the second-order finite-difference Laplacian
($s = -1/\bar{h}^2$, $\bar{h}$ the mean grid spacing; see
[LaplaceOperator](operators.md#laplaceoperator)) and conjugate gradients to a
relative residual of $10^{-6}$. The periodic Laplacian is singular with the
constants as null space; subtracting the mean makes $f$ orthogonal to it, so
CG converges to the zero-mean solution.

It demonstrates:

- a `CartesianDecomposition` with ghost layers sized from the operator
  (`ghosts=laplace`) and `communicate_ghosts` inside the operator callback;
- the dedicated `LaplaceOperator` versus the same stencil as a
  `GenericLinearOperator` (`-k hardcoded` / `-k generic`);
- spectral preconditioning (`-P`): `fourier` uses the inverse continuum
  symbol $1/|\mathbf{k}|^2$, which is spectrally equivalent to the stencil and
  gives a grid-independent iteration count; `fourier-exact` uses the exact
  inverse of the stencil symbol, so CG converges in one iteration. With a
  preconditioner the `FFTEngine` serves as the domain decomposition (see
  [Preconditioners](linalg.md#preconditioners));
- device execution (`-d gpu`) with the same code;
- a timing summary with lattice updates per second and estimated memory
  throughput (`--json`/`--json-out` for machine-readable output).

Unpreconditioned, the iteration count grows as $O(N_\alpha)$ because
$\kappa(-L) = O(h^{-2})$.

```bash
python examples/poisson.py -n 256,256
python examples/poisson.py -n 256,256 -P fourier
python examples/poisson.py -n 128,128,128 -d gpu
mpiexec -n 4 python examples/poisson.py -n 128,128,128 -P fourier
```

Options: `-n` grid (`nx,ny[,nz]`), `-d cpu|gpu`, `-i` maximum iterations
(default 1000), `-k hardcoded|generic`, `-P none|fourier|fourier-exact`, `-p`
plot (2D), `-q` quiet, `--json`, `--json-out FILE`.

The core of the solver, without preconditioner:

```python
import numpy as np
import muGrid
from muGrid.Solvers import conjugate_gradients

comm = muGrid.Communicator()
nb_grid_pts = (64, 64)
h = 1.0 / nb_grid_pts[0]

laplace = muGrid.LaplaceOperator(2, -1.0 / h**2)
decomposition = muGrid.CartesianDecomposition(comm, nb_grid_pts,
                                              ghosts=laplace)
rhs = decomposition.real_field("rhs")
solution = decomposition.real_field("solution")

x, y = decomposition.coords
rhs.p[...] = (1 + np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)) ** 10
rhs.p[...] -= comm.sum(rhs.p.sum()) / np.prod(nb_grid_pts)

def hessp(u, Au):
    decomposition.communicate_ghosts(u)
    laplace.apply(u, Au)

conjugate_gradients(comm, decomposition, rhs, solution, hessp=hessp,
                    rtol=1e-6, maxiter=1000)
```

## Homogenization

`homogenization.py` computes the effective elastic stiffness of a periodic
two-phase microstructure on the unit cell $\Omega = [0, 1)^d$ (2D or 3D) with
the finite-element method.

**Problem.** For a prescribed macroscopic strain $\bar{\boldsymbol{\varepsilon}}$
the displacement is $\mathbf{u}(\mathbf{x}) = \bar{\boldsymbol{\varepsilon}}\mathbf{x}
+ \tilde{\mathbf{u}}(\mathbf{x})$ with a periodic fluctuation
$\tilde{\mathbf{u}}$ that satisfies equilibrium,
$\nabla\cdot\big(\mathbf{C}(\mathbf{x}) : (\bar{\boldsymbol{\varepsilon}} +
\boldsymbol{\varepsilon}(\tilde{\mathbf{u}}))\big) = 0$. Discretised with
`FEMGradientOperator` elements (Q1 by default, P1 with `--element p1`) this is

$$
\mathbf{K}\tilde{\mathbf{u}} = -\mathbf{B}^\top W \mathbf{C}\,\bar{\boldsymbol{\varepsilon}},
\qquad \mathbf{K} = \mathbf{B}^\top W \mathbf{C}\mathbf{B},
$$

solved with conjugate gradients to a relative residual `--tol` (default
$10^{-6}$). The homogenised stress is
$\langle\boldsymbol{\sigma}\rangle = |\Omega|^{-1}\int_\Omega \mathbf{C} :
(\bar{\boldsymbol{\varepsilon}} + \boldsymbol{\varepsilon}(\tilde{\mathbf{u}}))\,dV$.
Applying the $d(d+1)/2$ unit strains ($xx, yy, xy$ in 2D; $xx, yy, zz, yz, xz,
xy$ in 3D, off-diagonal cases with $\bar\varepsilon_{ij} = \bar\varepsilon_{ji}
= 1$) yields the columns of the effective stiffness $\mathbf{C}^\text{eff}$ in
Voigt notation.

**Material.** Both phases are isotropic with Poisson's ratio `--nu` (default
0.3) and Young's moduli `--E-matrix` (1) and `--E-inclusion` (10), converted
to $\lambda = E\nu/((1+\nu)(1-2\nu))$, $\mu = E/(2(1+\nu))$. The
microstructure (`--inclusion-type`) is a centred spherical (circular) inclusion
of radius `--inclusion-radius` (0.25), a $2^d$ checkerboard, or a random
30 % inclusion fraction.

It demonstrates:

- the fused `IsotropicStiffnessOperator` (`-k fused`, default) with
  `apply_macro_rhs` for the right-hand side and `average_stress` for
  $\langle\boldsymbol{\sigma}\rangle$, so no strain or stress field is stored,
  versus an explicit gradient–stress–divergence sequence with the full
  per-quadrature-point $\mathbf{C}$ (`-k generic`);
- preconditioners (`-P`): `reference` (the reference-material Green's function
  in Fourier space), `multigrid` (its V-cycle approximation with an FFT solve
  on the coarsest grid) and `hybrid` (FFT in the rank-local axes, tridiagonal
  solve along the distributed axis); see
  [Preconditioners](linalg.md#preconditioners);
- the choice of domain decomposition (`--decomposition auto|cartesian|fft`):
  `auto` uses the `FFTEngine` as decomposition for `-P reference` (whose
  transforms need it) and a `CartesianDecomposition` otherwise (split only
  along the last axis for `-P hybrid`);
- single precision (`--precision single`), MPI and GPU execution with the same
  code.

```bash
python examples/homogenization.py -n 64,64
python examples/homogenization.py -n 32,32,32 -P reference
mpiexec -n 4 python examples/homogenization.py -n 128,128
python examples/homogenization.py -n 256,256 -d gpu
```

Further options: `-i` maximum CG iterations per load case (default 500),
`-p` plot (2D), `-q` quiet, `--json`/`--json-out FILE`, `--mg-nu`,
`--mg-cycles` (multigrid smoothing steps and cycles), `--sync-timers`,
`--profile-memory`.

## Single versus double precision

`precision_homogenization.py` solves the 2D homogenization problem above
(Q1 elements, circular inclusion) twice, with `float64` and with `float32`
fields, and prints both effective stiffness tensors, their relative difference
and the per-field memory footprint. Precision is selected only through the
`dtype` of the fields; operator, `linalg` kernels and CG follow it.

```bash
python examples/precision_homogenization.py -n 128,128
python examples/precision_homogenization.py -n 512,512 -d gpu
```

Options: `-n`, `-d`, `-i` (default 2000), `-t` relative tolerance (default
$10^{-5}$), `--E-matrix`, `--E-inclusion`, `--nu`, `--inclusion-radius`.

## Preconditioner cost studies

Two drivers run `homogenization.py` as subprocesses to separate cost
contributions of the reference preconditioner:

- `vcycle_vs_fft.py` measures, on one device, the ratio $R$ of the cost of one
  V-cycle apply (`-P multigrid`) to one FFT apply (`-P reference`), either from
  a model in units of fine-grid matvecs (a V-cycle with $\nu$ pre- and
  post-smoothing steps costs $W = (2\nu+1)\sum_\ell 2^{-d\ell}$ matvecs),
  from timed parts (`--measure-parts`), or end to end (`--measure`).
- `decomposition_baseline.py` fixes the preconditioner at `-P none` and varies
  only the domain decomposition (`--decomposition fft` versus `cartesian`), to
  measure what the FFT engine's slab decomposition costs in halo exchange
  independently of any transform.

```bash
python examples/vcycle_vs_fft.py -n 64 128 -d cpu --measure
python examples/decomposition_baseline.py -n 128 -p 1 2 4 8
```

## Benchmark drivers

`benchmark.py`, `benchmark_homogenization.py` and
`benchmark_homogenization_preconditioner.py` run `poisson.py` and
`homogenization.py` over ranges of grid sizes, devices and MPI configurations
and record the results in `benchmarks/results.csv` (schema in
`examples/benchmark_db.py`). What they measure and the recorded results are on
the benchmark pages: [Poisson](benchmark.md),
[homogenization](benchmark_homogenization.md) and
[preconditioner](benchmark_homogenization_preconditioner.md).
