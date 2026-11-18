# Bound-constrained L-BFGS in muGrid

`muGrid.Optimize.l_bfgs_bounded` minimises a function of muGrid fields under box constraints. This
note describes the method, the API, the kernels it adds to `muGrid.linalg`, and how it was validated
against NuMPI, the reference implementation of the same algorithm.

## 1. What it is

Minimise f(x) subject to lower ≤ x ≤ upper, with f and its gradient given by one callable.

- **Method:** L-BFGS on the free variables, with a projected Armijo backtracking line search. The
  specification is NuMPI's `l_bfgs_bounded` with `precond` (PR 112); the iterates must be the same.
- **Preconditioned initial inverse Hessian:** H₀ = γ P M P.
  - M is the user's symmetric positive definite preconditioner, for topology optimisation a Helmholtz
    smoother applied with muGrid's FFT.
  - P is the 0/1 mask of the free variables: those not sitting on a bound with the gradient pushing
    into it.
  - γ = ⟨s, y⟩ / ⟨y, P M P y⟩ for the newest curvature pair (s, y).
- **Why in muGrid:** all vectors are muGrid fields, so they live on the GPU and are distributed under
  MPI like the solver's. Only scalars reach the host. The memory is fixed: every vector is allocated
  once.

## 2. API

A new module `muGrid/Optimize.py` (capitalised like `Solvers` and `Preconditioners`):

```python
from muGrid.Optimize import l_bfgs_bounded

result = l_bfgs_bounded(
    comm, fc,                 # communicator and field collection, as in conjugate_gradients
    fun,                      # fun(x, gradient) -> float: writes the gradient field, returns the
                              # objective summed over all ranks
    x,                        # muGrid real field: start, overwritten with the solution
    lower=0.0, upper=1.0,     # scalars, fields of the shape of x, or None for no bound
    precond=None,             # precond(v, Mv): writes M v, like `prec` of conjugate_gradients
    maxcor=20,                # number of curvature pairs kept
    gtol=1e-5, xtol=0.0, ftol=0.0, maxiter=500,
    c1=1e-4, max_halvings=40, # Armijo constant and maximal number of step halvings
    callback=None,            # callback(iteration, state) -> True stops the run
    timer=None,
    pair_store="device")      # "host": the curvature pairs in page-locked host memory (§4)
result.success, result.message, result.fun, result.nit, result.nfev, result.max_grad
```

- The conventions are those of `muGrid.Solvers`: fields in and out, callables that write into their
  second argument, work fields registered in `fc` under names starting with `lbfgs-`, and the dtype of
  `x`.
- `result` is a small dataclass. The solution is in `x`; the gradient at the solution is the field
  `result.gradient` (one of the work fields, valid until the next call).
- `callback(iteration, state)` gets `x`, `gradient`, `fun`, `max_grad` and `step` (the accepted α).
- Differences from NuMPI's signature, on purpose:
  - no `zero_mask` (pinned indices). A bound pair lower = upper = 0 on those nodes does the same.
  - `fun` writes the gradient into a field and returns no array, so no gradient is copied.
  - reaching `maxiter` returns `success=False` with a message, as in NuMPI. It does not raise: an
    optimisation stopped at `maxiter` is a usable result, unlike a linear solve.

## 3. The algorithm, step by step

Per iteration, with x the iterate, g its gradient and r the masked gradient (g on the free variables,
0 on the others):

1. **Direction** d = −H r by the two-loop recursion over the stored pairs, newest first, with the
   initial matrix γ I, or γ P M P with a preconditioner. Without pairs: d = −r, or −P M P r.
2. If ⟨r, d⟩ ≥ 0 the direction is not a descent direction: drop all pairs and take the
   steepest-descent direction of step 1 without pairs.
3. **Line search:** x_try = clip(x + α d), starting with α = 1 and halving until
   f(x_try) ≤ f(x) + c₁ α ⟨r, d⟩.
4. If the line search fails: drop all pairs, take the steepest-descent direction with
   α = 1 / ‖d‖ and search again. If that fails too, stop with "line-search did not converge".
5. **Curvature pair:** s = x_try − x, y = r_try − r; store it if ⟨s, y⟩ > 1e-12 ‖s‖ ‖y‖.
6. **Stopping tests**, in this order: ‖s‖_∞ < xtol (not on a step made without pairs), relative
   decrease of f below ftol, the callback, ‖r‖_∞ < gtol.

Every one of these steps follows NuMPI line by line, including the order of the tests.

## 4. Vector operations and memory

| operation | how |
|---|---|
| inner products, axpy, axpby, copy, scale | existing `muGrid.linalg` kernels: no temporaries, host and device |
| projection clip(x + α d), masked gradient, P v, ‖·‖_∞ | new small kernels in `linalg` (below) |
| global sums and maxima | `comm.sum`, `comm.max` of host scalars: one allreduce each, never a device buffer |

New `linalg` kernels, each one pass over the fields, host and device, float32 and float64:

- `norm_inf(x)`: the largest absolute value;
- `axpy_clip(alpha, d, x, lower, upper, out)`: out = clip(x + α d);
- `mask_bound_active(g, x, lower, upper, v, out)`: out = v on the free variables and 0 on the
  variables that sit on a bound (within 1e-12, as in NuMPI) with g pointing into it. With v = g this is
  the masked gradient r; with any other v it is P v.

With these, the mask is never stored: it is recomputed from g and x where it is needed. This removes
the four boolean arrays of the reference `lbfgs_xp.py`.

**Memory:** 2 · maxcor pair fields + 11 work fields, plus the caller's `x`:
- x_try, g, g_try, r, r_try (the two iterates swap roles when a step is accepted, without a copy);
- d;
- two scratch fields for P M P;
- the candidate pair s, y;
- one spare.

Pair fields are allocated when first needed and recycled when a pair drops out.

**Pairs in host memory** (`pair_store="host"`), for grids where 2 · maxcor fields do not fit on the
GPU: the candidate pair is the only pair of fields on the device and stages the stored pairs, one at a
time, for the two-loop recursion; a pair already there is not copied again (about 4 · maxcor field
copies per iteration). Copies move the whole buffer of a field (ghosts included, contiguous), so no
device temporary is needed. γ of the newest pair is computed before the first loop (it does not
depend on the direction), which keeps that pair on the device. The iterates are those of
`pair_store="device"` bit for bit. On the CPU "host" is the same as "device". Device memory: 2 pair
fields instead of 2 · (maxcor + 1).

## 5. Reproducing NuMPI exactly

The fast kernels round differently from NuMPI's numpy expressions (fused multiply-add, the order of
the sums). Near a tie in the Armijo test that changes a decision, and the runs then drift apart. For
the validation there is therefore a private check mode, `_arithmetic="numpy"`, that performs every
operation with the numpy expression NuMPI uses. In this mode the iterates must be bitwise NuMPI's on
one process. The default mode must agree to round-off, with the same `nit`, `nfev` and message on
NuMPI's test problems.

## 6. Known limitations

- **The first step after a (re)start.** With an empty memory the first step is α = 1 along −P M P r.
  When M is not scaled to the problem (a smoother of norm 1 against a gradient that scales with the
  cell volume), that step can be orders of magnitude too small; its curvature pair is then noise and
  the following directions overshoot. A caller that knows a sensible size of the first step can scale
  M once, at its first application: later quasi-Newton directions do not depend on that scale, since γ
  divides it out.
- The host `mask_bound_active` runs at about 7 ns per value, three times the memory-bandwidth limit
  (`axpy_clip`: 1.8 ns).

## 7. Validation and speed

### 7.1 Against NuMPI

`benchmarks/lbfgs/cross_check_numpi.py`, 12 problems (quadratic; solution on the box; ill-conditioned
quadratic with scalar and with array bounds; Laplacian without, with the exact and with a smoothing FFT
Helmholtz preconditioner, also in a box; many active bounds without and with a smoother, and stopped
by xtol and by ftol):

- **Check arithmetic** (`_arithmetic="numpy"`): the solution is bitwise NuMPI's on all 12, with the
  same `nit`, `nfev`, objective and message.
- **Kernel arithmetic** (the default): the same `nit`, `nfev` and message on 7 of the 12, with
  solutions equal to 1e-12 or better. On the other 5 (both ill-conditioned box problems, the Laplacian
  without preconditioner and in a box, many active bounds with gtol) the counts differ, e.g. 95 against
  89 iterations: a rounding difference flips one line-search decision and the two runs take different
  paths to the same minimum (objectives equal to 1e-9 or better where both converge). `lbfgs_xp.py`
  documents the same for its fast mode.

### 7.2 Tests

`tests/python_optimize_tests.py`, 77 tests, on 1, 2 and 4 ranks:
- the three kernels against numpy expressions, with ghosts filled with large values, scalar and field
  bounds, in place, both precisions, CPU and GPU;
- check arithmetic bitwise equal to a transcription of NuMPI's algorithm on 13 problems;
- kernel arithmetic: iteration by iteration on the well-conditioned problems, the same minimum on the
  sensitive ones, CPU and GPU;
- stopping tests, callback, failed line search, `maxiter`;
- the solution is in the caller's field whichever of the two iterate fields holds it;
- the number of work fields: 2 per stored pair + 8 (10 with a preconditioner), none for scalar bounds;
- a box-constrained problem on a decomposed grid with ghosts, with and without a preconditioner,
  float64 and float32, CPU and GPU, against a serial reference of the whole grid.

Two things learned on the way, both documented in the docstring:
- `gtol` cannot be reached when its square is below the round-off of the objective; the line search
  then stalls until `maxiter`. NuMPI's own test notes the same.
- With float32 fields the objective must be evaluated in double, operator product included.

### 7.3 Speed: vector work per iteration

`benchmarks/lbfgs/vector_work.py`, separable quadratic in a box, 4 M values, maxcor 20, mean over the
iterations with a full memory, time outside the objective. Last column: scaled to 33.5 M values
(512³ on 4 ranks). The speed at 33.5 M values is extrapolated from 4 M values.

| implementation | device | preconditioner | ns per value | s at 33.5 M |
|---|---|---|---|---|
| muGrid | GPU | no | 7.1 | 0.24 |
| `lbfgs_xp.py` | GPU | no | 11.2 | 0.38 |
| muGrid | GPU | diagonal | 7.7 | 0.26 |
| `lbfgs_xp.py` | GPU | diagonal | 9.7 | 0.33 |
| muGrid | CPU | no | 115 | 3.9 |
| `lbfgs_xp.py` | CPU | no | 143 | 4.8 |
| NuMPI | CPU | no | 215 | 7.2 |
| muGrid | CPU | diagonal | 155 | 5.2 |
| `lbfgs_xp.py` | CPU | diagonal | 191 | 6.4 |
| NuMPI | CPU | diagonal | 225 | 7.5 |

The target of 0.3 s on the laptop GPU is met (0.24–0.26 s). The numbers include the line search's
projected steps, whose count differs a little between the implementations on the CPU.

### 7.4 Topology optimisation

muFFTTO's 3D topology-optimisation driver at 64³ (fused operator, multigrid in single precision,
energy objective, Helmholtz preconditioner by FFT, GPU, 30 iterations), once with its own optimiser
`lbfgs_xp.py` and once with `muGrid.Optimize.l_bfgs_bounded` in its place.

- The objective is identical at all 33 evaluations, to the 13 digits the driver prints; the 31 step
  lengths are identical; CG iterations 3533 mechanical and 3667 adjoint in both.
- Wall time 80.6 s against 80.8 s; the optimiser's vector work is 0.2 s of that in both.
