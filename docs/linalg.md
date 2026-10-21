# Linear algebra

The `muGrid.linalg` module provides field-level BLAS-1 operations — vector
updates and inner products that act directly on µGrid fields on host and device
— and two fused per-pixel vector kernels. The Krylov solvers in
`muGrid.Solvers` and the preconditioners in `muGrid.Preconditioners` are built
on them; both are described at the end of this page.

## Notation

A field $x$ has components $c = 1, \ldots, n_c$ (all sub-points included) at
the pixels $p$ of its collection. $\Omega_\text{int}$ denotes the interior
(owned) pixels of the rank and $\Omega_\text{buf} \supseteq \Omega_\text{int}$
the whole buffer including ghost layers. A global quantity is the sum of the
rank-local values over all $P$ ranks.

## Operations

| Operation | Definition | Region | BLAS analogue |
|---|---|---|---|
| `vecdot(a, b)` | returns $\sum_{p,c} \overline{a_{pc}}\, b_{pc}$ | $\Omega_\text{int}$ | `xDOT` / `xDOTC` |
| `norm_sq(x)` | returns $\sum_{p,c} \lvert x_{pc}\rvert^2$ | $\Omega_\text{int}$ | $\lVert x\rVert_2^2$ (`xNRM2`$^2$) |
| `axpy(alpha, x, y)` | $y \leftarrow \alpha x + y$ | $\Omega_\text{buf}$ | `xAXPY` |
| `scal(alpha, x)` | $x \leftarrow \alpha x$ | $\Omega_\text{buf}$ | `xSCAL` |
| `scal(alpha_field, x)` | $x_{pc} \leftarrow \alpha_{pc}\, x_{pc}$ (a single-component $\alpha$ broadcasts over the components of $x$) | $\Omega_\text{buf}$ | — |
| `axpby(alpha, x, beta, y)` | $y \leftarrow \alpha x + \beta y$ | $\Omega_\text{buf}$ | `xAXPBY` |
| `copy(src, dst)` | $\text{dst} \leftarrow \text{src}$ | $\Omega_\text{buf}$ | `xCOPY` |
| `axpy_norm_sq(alpha, x, y)` | $y \leftarrow \alpha x + y$, returns $\sum_{p \in \Omega_\text{int},c} \lvert y_{pc}\rvert^2$ | update $\Omega_\text{buf}$, sum $\Omega_\text{int}$ | — |
| `pipelined_cg_dots(r, u, w)` | returns $[\langle r,u\rangle, \langle w,u\rangle, \langle r,r\rangle]$ in one pass | $\Omega_\text{int}$ | — |
| `cross(a, b, out)` | $\text{out}_p = a_p \times b_p$ (three-vectors) | $\Omega_\text{buf}$ | — |
| `leray_project(k, invk, N, out)` | $\text{out}_{pc} \mathrel{-}= k_{pc} \sum_{d} \text{invk}_{pd}\, N_{pd}$ | $\Omega_\text{buf}$ | — |

All operations require their fields to share a collection and component
shape. Real (`float64`, `float32`) and complex (`complex128`, `complex64`)
fields are supported, except that `scal(alpha_field, x)` takes a real
`alpha_field`, `cross` takes three real or three complex fields, and
`leray_project` takes real `k`, `invk` and complex `N`, `out`. Every operation
is $O(n_c\,|\Omega|)$ and streams each operand once.

### Interior versus full buffer

- **Reductions** (`vecdot`, `norm_sq`, the sum of `axpy_norm_sq`,
  `pipelined_cg_dots`) sum over $\Omega_\text{int}$ only. Ghost values are
  copies of values owned by a neighbouring rank (or, for periodic boundaries,
  of the opposite side of the domain), so including them would count those
  values twice. The result is rank-local; reduce it over ranks (e.g.
  `comm.sum(value)` of the `muGrid.Communicator`, or `MPI.COMM_WORLD.allreduce`)
  to obtain the global value.
- **Updates** run over $\Omega_\text{buf}$. The buffer is contiguous, so a
  single loop without per-row index arithmetic covers it. The updates are
  pointwise, so ghost values that were consistent before stay consistent;
  either way, stencil operators read ghosts only after `communicate_ghosts()`
  has refreshed them.

### Precision of reductions

Reductions accumulate in double precision and return double precision: a
reduction over a `float32` (`Real32`) field returns `Real`, over a `complex64`
(`Complex32`) field `Complex` (in C++ `linalg::reduction_result_t<T>`; in
Python a `float`/`complex` either way). Each operand is widened before
multiplication. The reasons:

- A running `float32` sum of $N$ terms has a rounding error that grows as
  $O(N\,\epsilon_{32})$ in the worst case, which can corrupt CG's
  termination test.
- A `float32` result overflows at $\approx 3.4 \times 10^{38}$. In CG,
  $p^\top A p \le N\,\max_p |p|^2\,\lVert A\rVert$ with
  $\lVert A\rVert = O(h^{-2})$ for second-order operators, so the admissible
  magnitude of the entries shrinks like $h/\sqrt{N}$ as the grid is refined.
  An infinite $p^\top A p$ makes the step length zero and stalls the solve
  without an error.

### Complex fields

`vecdot` conjugates its first argument, so `vecdot(a, a)` equals
`norm_sq(a)`. On the device, complex buffers are processed through their
real and imaginary parts, so the kernels need no device complex type; the
sesquilinear `vecdot` accumulates the real and imaginary parts of
$\overline{a}\,b$ separately.

### Fused update and norm

`axpy_norm_sq` updates $y$ and returns its squared interior norm in one pass.
Separately, `axpy` followed by `norm_sq` reads $x$ once and $y$ twice and
writes $y$ once; the fused kernel reads $x$ and $y$ once each and writes $y$
once. BLAS-1 operations are memory-bandwidth bound, so the fused kernel saves
one of four memory sweeps.

### Per-pixel vector kernels

`cross` and `leray_project` act on three-component fields one pixel at a time.
Written as arithmetic on array views they would allocate temporaries and
traverse memory several times; as fused kernels they make one pass and run
unchanged on host and device.

**`cross(a, b, out)`** computes

$$
\text{out}_0 = a_1 b_2 - a_2 b_1, \quad
\text{out}_1 = a_2 b_0 - a_0 b_2, \quad
\text{out}_2 = a_0 b_1 - a_1 b_0 .
$$

`out` must be distinct from `a` and `b`, because the first component written
would otherwise be read back while forming the others. Typical uses are the
Fourier-space vorticity $i\mathbf{k}\times\hat{\mathbf{u}}$ (complex fields)
and the real-space Lamb vector $\mathbf{u}\times\boldsymbol{\omega}$.

**`leray_project(k, invk, N, out)`** computes
$\text{out}_c \mathrel{-}= k_c \sum_d \text{invk}_d\, N_d$. With the
wavevector $\mathbf{k}$ in `k` and $\mathbf{k}/|\mathbf{k}|^2$ in `invk` (the
$\mathbf{k} = 0$ entry set to zero by the caller), this subtracts
$\mathbf{k}\,(\mathbf{k}\cdot\mathbf{N})/|\mathbf{k}|^2$, i.e. the
longitudinal part of $\mathbf{N}$, from `out` (the Helmholtz/Leray projection
onto divergence-free modes). The coefficients are real, so real and imaginary
parts are updated independently; `out` may alias `N`.

## Usage

### Python

```python
import muGrid
import muGrid.linalg as la

fc = muGrid.GlobalFieldCollection([64, 64])
x = fc.real_field("x")
y = fc.real_field("y")
x.p[...] = 1.0
y.p[...] = 2.0

dot = la.vecdot(x, y)            # rank-local
la.axpy(0.5, x, y)               # y = 0.5 x + y
la.scal(2.0, x)                  # x = 2 x
la.axpby(1.0, x, -1.0, y)        # y = x - y
la.copy(x, y)                    # y = x
norm2 = la.norm_sq(x)
norm2 = la.axpy_norm_sq(0.5, x, y)

alpha = fc.real_field("alpha")   # per-pixel multiplier
alpha.p[...] = 0.5
la.scal(alpha, x)                # x_pc = alpha_p x_pc
```

Under MPI, reduce scalar results over ranks:

```python
import muGrid
import muGrid.linalg as la

comm = muGrid.Communicator.world()
decomp = muGrid.CartesianDecomposition(comm, [128, 128],
                                       nb_ghosts_left=[1, 1],
                                       nb_ghosts_right=[1, 1])
x = decomp.real_field("x")
y = decomp.real_field("y")
global_dot = comm.sum(la.vecdot(x, y))
```

Vector kernels:

```python
fc = muGrid.GlobalFieldCollection([64, 64, 64])
a = fc.real_field("a", (3,))
b = fc.real_field("b", (3,))
omega = fc.real_field("omega", (3,))
la.cross(a, b, omega)                 # omega = a x b

k = fc.real_field("k", (3,))          # wavevector
invk = fc.real_field("invk", (3,))    # k / |k|^2, k = 0 entry zero
N = fc.complex_field("N", (3,))
out = fc.complex_field("out", (3,))
la.copy(N, out)
la.leray_project(k, invk, N, out)     # out = N - k (k.N) / |k|^2
```

All operations run on device fields as well:

```python
if muGrid.has_gpu:
    fc = muGrid.GlobalFieldCollection([64, 64], device="gpu")
    x = fc.real_field("x")
    y = fc.real_field("y")
    la.axpy(0.5, x, y)          # device kernel
    dot = la.vecdot(x, y)       # device reduction, result returned to host
```

### C++

```cpp
#include "linalg/linalg.hh"

using namespace muGrid;

// x, y, src, dst: TypedField<Real, HostSpace> on one collection
Real dot = linalg::vecdot(x, y);              // interior only
linalg::axpy(0.5, x, y);                      // y = 0.5 x + y
linalg::scal(2.0, x);                         // x = 2 x
linalg::axpby(1.0, x, -1.0, y);               // y = x - y
linalg::copy(src, dst);                       // dst = src
Real norm2 = linalg::norm_sq(x);              // interior only
Real new_norm2 = linalg::axpy_norm_sq(0.5, x, y);
```

## Conjugate gradients

`muGrid.Solvers.conjugate_gradients` solves $\mathbf{A}\mathbf{x} = \mathbf{b}$
for symmetric positive (semi)definite $\mathbf{A}$ given only through its
action, optionally with a symmetric positive definite preconditioner
$\mathbf{M} \approx \mathbf{A}^{-1}$:

```python
from muGrid.Solvers import conjugate_gradients

conjugate_gradients(comm, fc, b, x, hessp, prec=None, maxiter=1000,
                    callback=None, timer=None, rtol=1e-6, atol=0.0,
                    residual=None)
```

- `hessp(p, Ap)` writes $\mathbf{A}\mathbf{p}$ into `Ap` (including any ghost
  communication it needs); `prec(r, z)` writes $\mathbf{M}\mathbf{r}$ into `z`.
- `fc` is the collection (`GlobalFieldCollection`, `CartesianDecomposition`, or
  an `FFTEngine`'s `real_space_collection`) on which the work fields are
  created; `x` holds the initial guess and is overwritten with the solution.
- The solve has converged when $\lVert \mathbf{b} - \mathbf{A}\mathbf{x}\rVert
  \le \max(\text{rtol}\,\lVert\mathbf{b}\rVert, \text{atol})$. The deprecated
  `tol` is an alias for `atol` (with `rtol = 0`).
- `callback(iteration, state)` is called before the first and after every
  iteration; `state` contains the fields `x`, `r`, `p`, `z` and the global
  scalars `rr` $= \mathbf{r}^\top\mathbf{r}$ and `rz` $= \mathbf{r}^\top\mathbf{z}$.
- `residual`, if given, receives the final residual.
- `ConvergenceError` is raised when `maxiter` is exceeded, when the true
  residual stagnates above the tolerance, when $\mathbf{p}^\top\mathbf{A}\mathbf{p}$
  or $\mathbf{r}^\top\mathbf{M}\mathbf{r}$ is not positive, or when the residual
  is not finite.

Starting from $\mathbf{r}_0 = \mathbf{b} - \mathbf{A}\mathbf{x}_0$,
$\mathbf{z}_0 = \mathbf{M}\mathbf{r}_0$, $\mathbf{p}_0 = \mathbf{z}_0$, each
iteration computes

$$
\alpha_k = \frac{\mathbf{r}_k^\top \mathbf{z}_k}{\mathbf{p}_k^\top \mathbf{A}\mathbf{p}_k},\quad
\mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k \mathbf{p}_k,\quad
\mathbf{r}_{k+1} = \mathbf{r}_k - \alpha_k \mathbf{A}\mathbf{p}_k,
$$

$$
\mathbf{z}_{k+1} = \mathbf{M}\mathbf{r}_{k+1},\quad
\beta_k = \frac{\mathbf{r}_{k+1}^\top \mathbf{z}_{k+1}}{\mathbf{r}_k^\top \mathbf{z}_k},\quad
\mathbf{p}_{k+1} = \mathbf{z}_{k+1} + \beta_k \mathbf{p}_k .
$$

In exact arithmetic the error contracts at least as

$$
\lVert \mathbf{x}_k - \mathbf{x}^*\rVert_{\mathbf{A}} \le
2\left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^k
\lVert \mathbf{x}_0 - \mathbf{x}^*\rVert_{\mathbf{A}},
\qquad \kappa = \kappa(\mathbf{M}\mathbf{A}),
$$

so the iteration count grows as $O(\sqrt{\kappa})$. For a second-order
operator discretised on a grid with spacing $h$, $\kappa(\mathbf{A}) =
O(h^{-2})$, and unpreconditioned CG needs $O(N_\alpha)$ iterations. Each
iteration costs one `hessp`, one `prec`, three global reductions and four
vector updates. When the recursive residual meets the tolerance, the true
residual $\mathbf{b} - \mathbf{A}\mathbf{x}$ is recomputed (one extra
`hessp`); the solve returns only if it also meets the tolerance, and otherwise
restarts from it. The work fields take the dtype of `b`, so a `float32`
right-hand side gives a single-precision solve.

`conjugate_gradients_pipelined` has the same signature (plus
`replacement_interval`) and implements the pipelined variant of Ghysels and
Vanroose (Parallel Computing 40 (2014) 224). It carries
$\mathbf{A}\mathbf{p}$, $\mathbf{M}\mathbf{r}$ and related images as
auxiliary recurrences, so the three inner products of an iteration form a
single non-blocking reduction that overlaps with the operator and
preconditioner applies. Per iteration it performs one combined reduction
instead of three, at the cost of about eight vector updates instead of four
and nine work fields instead of four; residual replacement every
`replacement_interval` iterations bounds the drift between recursive and true
residual. It pays off when global reductions are latency-bound (MPI
allreduce at high rank counts, device-to-host synchronisation on GPUs).

## Preconditioners

`muGrid.Preconditioners` provides preconditioners with the call signature
`prec(r, z)` ($\mathbf{z} = \mathbf{M}\mathbf{r}$) expected by the solvers. All
of them store their data in fields on the solver's collection and apply it
with the `linalg` kernels and the FFT engine, so they run on host and device
and under MPI.

| Preconditioner | $\mathbf{M}$ | Cost per apply |
|---|---|---|
| `IdentityPreconditioner()` | $\mathbf{I}$ (same as `prec=None`) | $O(N)$ copy |
| `JacobiPreconditioner(diagonal)` | $\operatorname{diag}(\mathbf{A})^{-1}$ | $O(N)$ |
| `FourierPreconditioner(engine, kernel)` | $\mathcal{F}^{-1}\,k(\mathbf{q})\,\mathcal{F}$, scalar kernel per mode | $O(N \log N)$ |
| `BlockFourierPreconditioner(engine, blocks)` | $\mathcal{F}^{-1}\,\hat{K}(\mathbf{q})^{-1}\,\mathcal{F}$, $n \times n$ block per mode | $O(n N \log N + n^2 N)$ |
| `make_reference_stiffness_preconditioner(engine, ...)` | Green's function of a homogeneous reference stiffness (returns a `BlockFourierPreconditioner` or `AnalyticReferencePreconditioner`) | $O(n N \log N + n^2 N)$ |
| `GreenJacobiPreconditioner(green, diagonal)` | $\mathbf{J}^{1/2}\,\mathbf{G}\,\mathbf{J}^{1/2}$, $\mathbf{J} = \operatorname{diag}(\mathbf{K})^{-1}$ | one apply of $\mathbf{G}$ plus $O(N)$ |
| `MultigridReferencePreconditioner(...)` | V-cycle approximation of the reference Green's function | $O(N)$ plus a coarse-grid FFT solve |
| `HybridFourierTridiagonalPreconditioner(...)` | reference Green's function via FFT in the rank-local axes and a block-tridiagonal solve in the distributed axis | no all-to-all |

**Jacobi.** $\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$ has unit diagonal,
and among all diagonal scalings Jacobi is within a factor of the number of
non-zeros per row of the optimal condition number (van der Sluis, 1969). It
helps when the diagonal varies strongly in space (heterogeneous coefficients);
for a constant diagonal it only rescales the system. The diagonal must be
positive.

**Fourier.** On a periodic grid the FFT diagonalises every
translation-invariant operator. With the exact inverse symbol $k = 1/\hat{A}$,
$\mathbf{M} = \mathbf{A}^{+}$ and CG converges in one iteration. For a
variable-coefficient operator $\mathbf{A} = \mathbf{B}^\top W\mathbf{C}(x)\mathbf{B}$
preconditioned with the inverse of a reference operator
$\mathbf{B}^\top W \mathbf{C}^\text{ref}\mathbf{B}$ and
$c_\text{min}\mathbf{C}^\text{ref} \le \mathbf{C}(x) \le c_\text{max}\mathbf{C}^\text{ref}$,
$\kappa(\mathbf{M}\mathbf{A}) \le c_\text{max}/c_\text{min}$, independent of
the grid. `kernel` is an array on the local Fourier subdomain or a callable
`kernel(engine)` evaluated once; modes where the kernel is zero are projected
out (for singular operators such as the periodic Laplacian, set $k(0) = 0$
and keep the right-hand side free of that mode). The kernel broadcasts over
field components, and the FFT `normalisation` is folded into the stored
kernel. Fields must live on `engine.real_space_collection`.

**Reference stiffness (Green's function).** For an FE problem with uniform
reference data, $\mathbf{K}^\text{ref} = \mathbf{B}^\top W\mathbf{C}^\text{ref}\mathbf{B}$
is block-circulant and therefore block-diagonal in Fourier space, with one
$n \times n$ block $\hat{K}(\mathbf{q})$ per mode ($n = d$ displacement
components). The preconditioner applies
$\mathbf{G}\mathbf{r} = \mathcal{F}^{-1}[\hat{K}(\mathbf{q})^{+}\,\mathcal{F}\mathbf{r}]$
with $\hat{K}(\mathbf{0})^{+} = 0$ (the rigid translations are projected out).
$\hat{K}(\mathbf{q})$ comes either from the impulse response of
`apply_reference_stiffness` ($n$ operator applies and FFTs at setup) or
analytically from the element, grid spacing and reference Lamé parameters
($\hat{K}(\mathbf{q}) = \sum_\mathbf{s} S(\mathbf{s})\,e^{-2\pi i\mathbf{q}\cdot\mathbf{s}}$
for the $3^d$ stencil $S$). The symbol is assembled and inverted in double
precision because the stencil rows sum to zero, so the low-$\mathbf{q}$ blocks
are differences of $O(1)$ terms of size $O(|\mathbf{q}|^2)$. This is the
preconditioner of Ladecký et al., Appl. Math. Comput. 446 (2023) 127835.

**Green-Jacobi.** For smoothly varying, high-contrast material data the
spectrum of $\mathbf{G}\mathbf{K}$ spreads out. Scaling symmetrically with the
local Jacobi diagonal, $\mathbf{M} = \mathbf{J}^{1/2}\mathbf{G}\mathbf{J}^{1/2}$,
re-clusters it while keeping $\mathbf{M}$ symmetric positive definite (the
J-FFT scheme of Ladecký et al.). `make_green_jacobi_preconditioner` builds
$\mathbf{G}$ and $\operatorname{diag}(\mathbf{K})$ (via
`IsotropicStiffnessOperator.assemble_diagonal`) together.

### Example: spectral preconditioning of the Poisson problem

The finite-difference Laplacian $-L$ with $s = -1/h^2$ has the symbol
$\sum_\alpha 4\sin^2(\pi q_\alpha)/h^2$. Its inverse as kernel makes CG a
direct solver. The solver fields must live on the engine's real-space
collection, so the engine also serves as the ghosted domain decomposition:

```python
import numpy as np
import muGrid
from muGrid.Preconditioners import FourierPreconditioner
from muGrid.Solvers import conjugate_gradients

comm = muGrid.Communicator()
nb_grid_pts = (64, 64)
h = 1.0 / nb_grid_pts[0]
laplace = muGrid.LaplaceOperator(2, -1.0 / h**2)
engine = muGrid.FFTEngine(nb_grid_pts, comm, ghosts=laplace)

rhs = engine.real_space_field("rhs")
solution = engine.real_space_field("solution")
x, y = engine.coords
rhs.p[...] = np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

def hessp(u, Au):
    engine.communicate_ghosts(u)
    laplace.apply(u, Au)

def inverse_fd_laplacian(engine):
    q = engine.fftfreq
    denom = (4 * np.sin(np.pi * q) ** 2 / h**2).sum(axis=0)
    return np.where(denom > 0, 1 / np.where(denom > 0, denom, 1), 0.0)

prec = FourierPreconditioner(engine, inverse_fd_laplacian)
conjugate_gradients(comm, engine.real_space_collection, rhs, solution,
                    hessp=hessp, prec=prec, rtol=1e-8, maxiter=10)
```

The kernel is evaluated on the rank-local Fourier subdomain, so the code is
the same in serial and MPI runs.

### Example: Jacobi preconditioning of a heterogeneous problem

For the screened Poisson operator $-L + c(\mathbf{x})$ with a coefficient
$c$ that varies over orders of magnitude, the diagonal is
$2\sum_\alpha h_\alpha^{-2} + c(\mathbf{x})$:

```python
from muGrid.Preconditioners import JacobiPreconditioner

diag = engine.real_space_field("diagonal")
diag.p[...] = 4 / h**2 + c          # 2D stencil centre plus screening
conjugate_gradients(comm, engine.real_space_collection, rhs, solution,
                    hessp=hessp_screened,
                    prec=JacobiPreconditioner(diag, communicator=comm),
                    rtol=1e-8, maxiter=1000)
```
