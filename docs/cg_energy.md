# Energy-error stopping test for CG

`conjugate_gradients_energy` stops the preconditioned conjugate gradients on an estimate of the
energy norm of the error, ‖x − x_k‖_A, relative to the energy of the solution, instead of on the
residual. In homogenization this is the error that enters the effective stiffness, so the solve stops
when the quantity of interest is accurate, typically much earlier than a residual test of the same
tolerance. The estimator follows muFFTTO's implementation, with which iterates and iteration counts
agree exactly.

## What it computes

CG minimises the energy norm of the error, and its steps are A-orthogonal, so

    ‖x − x_k‖²_A = Σ_{j ≥ k} Δ_j,      Δ_j = α_j (r_j, z_j) = ‖x_{j+1} − x_j‖²_A.

- Truncating the sum after `d` terms gives a **lower** estimate of the error at iteration k, available
  at iteration k + d (Strakoš & Tichý 2002). It costs nothing: α_j and (r_j, z_j) are CG's own scalars.
- The delay `d` is chosen adaptively (Meurant, Papež & Tichý 2021): the estimate for iteration ℓ is
  accepted at iteration k when `S · Δ_k / Σ_{j=ℓ}^{k−1} Δ_j ≤ τ`, with the safety factor S of the paper
  (`findS` in muFFTTO) and τ = 0.25. The rule aims at an estimate of at least 1 − τ times the true
  squared error. That is a heuristic; only "never above the true error" is certain.
  `d_min` is a lower limit on the delay.
- Stop when `estimate ≤ rtol² · E`, where E is the energy of the solution:
  - warm start (x₀ ≠ 0): E = (x₀, A x₀). The product A x₀ is needed for r₀ anyway, so this costs one
    dot product.
  - cold start (x₀ = 0): E = the current estimate of ‖x − x₀‖²_A = Σ_{j ≥ 0} Δ_j.

  These are exactly the two cases of muFFTTO's `energy_stop_kwargs`.

## API

Two pieces in `muGrid/Solvers.py`:

```python
class EnergyErrorEstimate:
    """Adaptive-delay lower estimate of the squared energy error of CG. Scalars only."""
    def __init__(self, tau=0.25, d_min=0): ...
    def update(self, alpha, rz): ...   # call once per iteration with α_k and (r_k, z_k)
    error_sq                            # latest accepted estimate of ‖x − x_ℓ‖²_A, or None
    iteration                           # the ℓ it belongs to
    initial_error_sq                    # Σ_j Δ_j, the estimate of ‖x − x₀‖²_A

def conjugate_gradients_energy(comm, fc, b, x, hessp, prec=None, rtol=1e-6, d_min=0,
                               tau=0.25, maxiter=1000, callback=None, timer=None):
    """PCG, stopped when the estimated energy error is below rtol · ‖x‖_A. Returns x."""
```

- Same calling convention as `conjugate_gradients`: fields, `hessp(x, y)`, `prec(r, z)`, work fields
  from `_WorkFields`, dtype taken from `b`, `ConvergenceError` at `maxiter` or for a non-positive
  `pᵀAp` or `rᵀz`.
- `callback(iteration, state)` gets `x, r, p, z, rz` and the new entries `energy_error_sq` (the
  estimate or None), `energy_error_iteration` and `energy_sq` (the squared energy it is compared with).

**Why a separate function and not an argument of `conjugate_gradients`.** The residual test is woven
into that loop: `|r|²` is fused into the residual update, and convergence is confirmed on the true
residual `b − Ax` with a restart if it fails. The energy test needs none of that, and it needs the
start energy and different callback entries. An argument `criterion="energy"` would branch in four
places of one loop. Two short loops that share the helpers (`_WorkFields`, the guards) are easier to
read. The estimator is its own class, so it is tested alone and can later serve
`conjugate_gradients_pipelined` or a residual-stopped solve that only wants to report the error.

Alternatives considered:
- `conjugate_gradients(..., criterion="energy")`: one entry point, but a cluttered loop (above).
- a stopping-criterion object passed to `conjugate_gradients`: the most general, but the object would
  have to dictate which reductions the loop performs. Too much machinery for two criteria.

## Cost per solve, compared with `conjugate_gradients`

- One operator product less: no true-residual check at the end.
- One reduction less per iteration: `|r|²` is not computed. Each iteration has two reductions,
  `pᵀAp` and `rᵀz`.
- No `bᵀb`; one extra dot product `(x₀, A x₀)` at the start.
- No temporaries: `linalg.copy / axpy / axpby / vecdot` on the work fields (3 without, 4 with a
  preconditioner).

This is the operation count of muFFTTO's reference implementation `cg_energy.py`.

## Details that follow the reference

- The scalar arithmetic of the estimator (order of the sums, `findS`) is kept as in muFFTTO, so the
  iteration counts are identical and not only close.
- The estimator works on Python floats, identical on every rank, so MPI needs no further reductions.
- Differences in behaviour, all on the side of muGrid's conventions: `maxiter` raises
  `ConvergenceError` (the reference returns silently), and non-finite scalars raise.
- float32: the scalars are accumulated in double. Past the attainable accuracy of float32 the
  recursive scalars no longer describe the true error; the docstring says so, and `rtol` below
  `FLOAT32_RTOL_FLOOR` warns as in `conjugate_gradients`.

## Details of the implementation

- The reference keeps a counter `delay` next to the index of the iterate being estimated. It is
  equivalent to requiring at least two steps in the denominator of the rule, which is how the muGrid
  code states it. The test `test_estimator_accepts_as_the_reference` checks the equivalence bit for
  bit on four kinds of step sequences.
- The start test is `rᵀz = 0` (the reference reduces `rᵀr` for it), which saves one reduction.
- If `rᵀz` becomes exactly zero during the iteration, the solve returns: the residual vanished.

## Validation and speed

**Against muFFTTO** (heterogeneous scalar problem, 24³, contrast 1e4, energy rtol 1e-6). Iterations of
muGrid / `cg_energy.py` / `conjugate_gradients_mugrid_experimental`:

| preconditioner | start | `d_min` 0 | `d_min` 10 |
|---|---|---|---|
| none | cold | 424 / 424 / 424 | 424 / 424 / 424 |
| none | warm | 375 / 375 / 375 | 375 / 375 / 375 |
| Jacobi | cold | 187 / 187 / 187 | 187 / 187 / 187 |
| Jacobi | warm | 163 / 163 / 163 | 164 / 164 / 164 |

The solution is bitwise equal to that of `cg_energy.py` in all eight cases. It differs from the
experimental solver's by 1e-15 with Jacobi and up to 1.5e-7 without (that solver uses numpy
arithmetic for its updates; the unpreconditioned runs take about 400 iterations at contrast 1e4).

**Tests** (`tests/python_cg_energy_tests.py`, 46):
- the estimator equals the reference bookkeeping bit for bit;
- the estimate never exceeds the true energy error; on a well-conditioned system it is at least half
  of it;
- iterates, iteration counts and estimates equal a dense reference CG, cold and warm, with and
  without Jacobi, `d_min` 0 and 10; one operator and one preconditioner product per iteration plus
  one each at the start;
- the true energy error of the result is below 1.5 · rtol · ‖x‖_A;
- 1, 2 and 4 MPI ranks, CPU and GPU: the same iteration count as a serial numpy solve of the whole
  grid in float64; within 2 iterations in float32 (rtol 1e-4);
- error paths: `maxiter`, indefinite operator, indefinite preconditioner, bad parameters.

**Speed** (GPU, 128³ scalar screened Poisson, Jacobi, best of 3):

| | products | time s | ms per product |
|---|---|---|---|
| muGrid, float64 | 234 | 0.531 | 2.27 |
| `cg_energy.py`, float64 | 234 | 0.616 | 2.63 |
| muGrid, float32 | 153 | 0.211 | 1.380 |
| `cg_energy.py`, float32 | 153 | 0.212 | 1.383 |

Same products and reductions per iteration as `cg_energy.py`, and one reduction less at the start.

## Limitations

- `conjugate_gradients_pipelined` has no energy test. `EnergyErrorEstimate` could be fed there too.
- No floor on the attainable accuracy: in float32, an `rtol` below about 1e-5 relies on recursive
  scalars that no longer describe the true error. A warning is issued below `FLOAT32_RTOL_FLOOR`.
