# Numerics audit of µGrid (October 2026) and GPU follow-up

This note records a numerical audit of µGrid, the two critical fixes made in
the PR that adds this file (S1, S2), their validation on GPUs, and what
remains to be done. It is written so that the work can be picked up without
the original session.

- Audited state: `origin/main` at `f83ce587` (Python and C++ line numbers below
  refer to that commit unless stated otherwise).
- Machines used: macOS, CPU only, OpenMPI via `benchenv.sh` for the audit and
  the fixes; MI300A, H200 and L40S nodes for the GPU validation (section 2).
  GPU statements in section 4 are still from reading the code.
- Method: five areas audited in parallel (solvers/preconditioners, linalg
  kernels, FFT, operators, MPI/fields/IO). Most findings were checked with
  small scripts against the build; those are marked CONFIRMED. SUSPECTED means
  traced in code but not demonstrated.

---

## 1. What this PR changes

### S1 — Pipelined CG declared convergence on a drifted residual (critical)

`conjugate_gradients_pipelined` (`language_bindings/python/muGrid/Solvers.py`)
tested only the recursive residual. In the Ghysels–Vanroose formulation the
rounding errors of the recurrences are amplified (Cools et al., SIAM J. Matrix
Anal. Appl. 39 (2018) 426), so the recursive residual kept shrinking while the
true one stalled.

Before (64×64 diagonal SPD operator, log-spaced spectrum):

| dtype | cond | rtol | true ‖b−Ax‖/‖b‖ on "convergence" |
|---|---|---|---|
| float64 | 1e5 | 1e-8 | 6.0e-7 |
| float64 | 1e6 | 1e-8 | 5.5e-5 |
| float64 | 1e5 | 1e-10 | 6.9e-7 |
| float32 | 1e2 | 1e-5 | 2.8e-4 |
| float32 | 1e3 | 1e-5 | ~20 (diverged, still returned) |

Fix:

- **True-residual check on exit.** When the recursive residual meets the
  tolerance, `b − Ax` is computed (one extra operator apply). Only it may
  declare convergence. If it fails, CG restarts from the true residual; if the
  true residual has not decreased since the previous failed check, a
  `ConvergenceError` reports the stagnation.
- **Periodic residual replacement**, new keyword `replacement_interval`
  (default 50 in float64, 20 in float32; `0` disables): r, u, w, s, q, z are
  recomputed from x and p (4 operator + 2 preconditioner applies). If the true
  and recursive residual norms differ by more than 2×, CG restarts instead of
  keeping the drifted search direction — keeping it was seen to stall float32
  solves.
- The `maxiter` error message now reports the residual reached.

After: every float64 case above reaches the requested true residual with
~9–11% more operator applies than `conjugate_gradients`; float32 cases either
converge or raise. Interval sweep that chose the defaults:

| case | std CG | interval 100 | 50 | 20 |
|---|---|---|---|---|
| f64 cond 1e6 rtol 1e-8 | 9745 | 10343 | 10701 | 11855 |
| f64 cond 1e7 rtol 1e-9 | 33862 | 36532 | 37720 | 41820 |
| f32 cond 1e3 rtol 1e-5 | 199 | 210 | 215 | 239 |
| f32 cond 1e4 rtol 1e-5 | 639 (true 1.01e-5, false conv.) | fails | fails | 775 |

(numbers are operator applies; "fails" = `ConvergenceError`, never a false
return.)

Known limitation: an unattainable tolerance can still run to `maxiter` before
raising — slow, but no longer silent.

Tests: `tests/python_solvers_tests.py::test_pipelined_cg_converges_in_the_true_residual`,
`::test_pipelined_cg_without_replacement_never_returns_a_drifted_iterate`.

### S2 — float32 evaluated Green symbol lost the long-wavelength modes (critical)

`green_symbol.cc` / `green_symbol_gpu.cc` (used by
`AnalyticReferencePreconditioner`, which `make_reference_stiffness_preconditioner`
selects **by default on a GPU**) built the symbol
`K(q) = Σ_d S_d exp(−2πi q·d)` and its cofactor inverse in the field precision.

- The stencil rows sum to zero (rigid-body null space), so at low q the sum
  cancels O(S) terms to O(S q²): relative error ≈ eps·N², order one in float32
  at N = 4096.
- `|det|²` (the Dim-th power of O(S q²) entries) underflowed; the kernel then
  took the mode for the q = 0 null space and **zeroed it**.

Before, float32 kernel vs float64 numpy reference on the lowest modes (axis 0
in full, first 8 modes of the other axes):

| dim | N | h | λ = μ | max rel. err | modes zeroed |
|---|---|---|---|---|---|
| 3 | 256 | 1/N | 1 | 1.4e-3 | 0 |
| 3 | 2048 | 1/N | 1 | 1.0 | 10 |
| 3 | 4096 | 1/N | 1 | 1.0 | 157 |
| 3 | 256 | 0.01 | 1e-3 | 1.0 | 7 |
| 2 | 65536 | 1/N | 1e-3 | 3.2 | 4 |

Fix: the symbol is built and inverted in **double for every field precision**;
only the load/store of the mode's components is in the field precision. The
block is scaled by its largest entry before the cofactor inverse so `|det|²`
cannot underflow. The `apply_green_symbol_*` (host) and
`apply_green_symbol_gpu_*` (device) bindings now take stencil, frequency
tables and normalisation as **float64**; `AnalyticReferencePreconditioner`
passes them so (the device kernel reads the tables through raw pointers, so
the dtype is load-bearing).

After: float32 max rel. error ≈ 1e-7 and nothing zeroed in all cases above;
float64 results bit-for-bit unchanged.

Tests: `tests/python_preconditioner_tests.py::test_evaluated_symbol_is_accurate_in_single_precision`
(fails on the old build, passes now) and
`::test_evaluated_symbol_on_device_is_accurate_in_single_precision` (GPU;
skipped without one).

### Verification done (CPU)

- Full Python suite: 617 passed, 87 skipped (GPU).
- `mpirun -np 2` and `-np 3` on solver and preconditioner tests: pass.
- C++ `mugrid_main_test_suite`: pass.

---

## 2. GPU validation (done 2026-10-17)

Built this branch and `main` (`f83ce587`, via a worktree) for HIP (MI300A,
gfx942, ROCm 6.4.1) and CUDA (sm_89 + sm_90, CUDA 13.3), Release, no MPI.
Run on one MI300A, one L40S and one H200.

- **Builds**: `green_symbol_gpu.cc` compiles cleanly for both back ends.
- **C++ suite**: passes on all three GPUs.
- **S2 device tests**: `test_evaluated_symbol_on_device_is_accurate_in_single_precision`
  passes on all three, and `main` fails 3 of its cases (the 2048³/4096³ and
  h = 0.01 rows of the table above).
- **`test_evaluated_symbol_on_device_matches_stored_on_host[float32-2-shape2]`
  failed on all three GPUs** — the reference was wrong, not the kernel.
  Against a float64 stored reference the errors on the (2, 64) grid are:
  host stored float32 2.4e-5 (finding S6), device evaluated float32 1.2e-7 on
  this branch (1.7e-5 on `main`). The test now takes the host reference in
  float64 for both precisions and its float32 tolerance is tightened from 1e-5
  to 1e-6. It passes on this branch and fails on `main` for 3 of 4 float32
  grids, so it guards S2 as well.
- **Full Python suite on the branch**: CUDA (L40S, H200) 743 passed, 0 other
  failures. HIP 748 passed, 31 failed — the same 31 fail on `main`. They are
  the tests in `python_convolution_operator_tests.py`,
  `python_convolution_performance_tests.py` and `python_field_gpu_tests.py`
  that hard-code `muGrid.Device.cuda()`, which a ROCm build rejects. They are
  unrelated to this PR (new finding T1 in section 4).

Benchmark script below; at 512³ the stored-symbol comparison is assembled on
the host and needs a large `--mem` in Slurm, or the job is OOM-killed.

### Benchmark script

Times the full apply and its parts with CUDA events. The symbol kernel is
timed on a copy of the Fourier data so repeated in-place application cannot
drift into denormals.

```python
import numpy as np
import cupy as cp
import muGrid
from muGrid.Preconditioners import make_reference_stiffness_preconditioner

n, dim, reps = 512, 3, 20


def timed(f):
    f()
    cp.cuda.Device().synchronize()
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    for _ in range(reps):
        f()
    stop.record()
    stop.synchronize()
    return cp.cuda.get_elapsed_time(start, stop) / reps  # ms


for dtype in (np.float32, np.float64):
    name = np.dtype(dtype).name
    eng = muGrid.FFTEngine((n,) * dim, muGrid.Communicator(),
                           device=muGrid.Device.gpu())
    prec = make_reference_stiffness_preconditioner(
        eng, nb_components=dim, dtype=dtype, element=muGrid.FEMElement.q1,
        grid_spacing=(1.0 / n,) * dim, lambda_ref=1.3, mu_ref=0.7,
        name=f"bench-{name}")
    r = eng.real_space_field(f"r-{name}", components=(dim,), dtype=dtype)
    z = eng.real_space_field(f"z-{name}", components=(dim,), dtype=dtype)
    r.p[...] = cp.random.standard_normal(r.p.shape).astype(dtype)

    eng.fft(r, prec._work)
    work = prec._work.s
    pristine = work.copy()

    def kernel():
        prec._kernel(work.data.ptr, prec._stride_component,
                     prec._stride_mode, prec._stencil, prec._q_ptrs,
                     list(prec._fourier_shape), prec._normalisation)

    t_copy = timed(lambda: cp.copyto(work, pristine))
    t_kernel = timed(lambda: (cp.copyto(work, pristine), kernel())) - t_copy
    t_fft = timed(lambda: eng.fft(r, prec._work))
    t_ifft = timed(lambda: eng.ifft(prec._work, z))
    t_apply = timed(lambda: prec(r, z))
    print(f"{name}: apply {t_apply:.2f} ms = fft {t_fft:.2f} + symbol "
          f"{t_kernel:.2f} + ifft {t_ifft:.2f}")
```

(On `main` the kernel's `_stencil`/`_q` are in the field precision; the script
works unchanged on both.)

---

## 3. Performance cost of S2 on GPUs (measured)

Times in ms from the benchmark script, 3 components, Q1, CUDA/HIP events,
mean of 20. "apply" is FFT + symbol kernel + iFFT; "stored" is the
`BlockFourierPreconditioner` that the factory picks when `evaluate_symbol=False`
(unchanged by this PR).

float32:

| GPU | N | kernel `main` → branch | apply `main` → branch | stored apply |
|---|---|---|---|---|
| MI300A | 256³ | 0.29 → 0.35 | 2.11 → 2.22 (+5%) | 3.65 |
| MI300A | 512³ | 2.87 → 3.79 | 21.98 → 22.70 (+3%) | 58.3 |
| H200 | 256³ | 0.32 → 0.49 | 1.18 → 1.35 (+14%) | 1.79 |
| H200 | 512³ | 2.34 → 3.79 | 9.28 → 10.73 (+16%) | 25.6 |
| L40S | 256³ | 0.46 → 5.92 | 5.42 → 10.92 (**2.0×**) | **7.13** |
| L40S | 512³ | 4.85 → 46.97 | 44.93 → 87.19 (**1.9×**) | 129.8 |

float64 (should be unchanged; the pre-inversion scaling costs a little):

| GPU | N | kernel `main` → branch | apply `main` → branch |
|---|---|---|---|
| MI300A | 512³ | 3.59 → 3.70 | 29.87 → 29.81 |
| H200 | 512³ | 3.67 → 3.86 | 16.16 → 16.34 |
| L40S | 256³ | 5.37 → 5.83 | 16.78 → 17.24 |
| L40S | 512³ | 41.82 → 45.62 | 134.95 → 138.91 |

Register use of the 3D kernel (`-Xptxas -v`, `-Rpass-analysis=kernel-resource-usage`):

| | float32 `main` | float32 branch | float64 `main` | float64 branch |
|---|---|---|---|---|
| sm_89 | 80 | 104 | 86 | 104 |
| sm_90 | 80 | 98 | 84 | 98 |
| gfx942 VGPRs (waves/SIMD) | 87 (5), 256 B scratch | 152 (3) | 152 (3) | 152 (3) |

No spills anywhere. The 432 B stack frame is the per-thread `P[NP]` of the
short-axis-0 fallback path; `main` already had it in float64.

Reading:

- **Datacenter GPUs (MI300A, H200)**: the estimate held. The float32 kernel
  now costs the same as the float64 one, and the full apply is 3% (MI300A)
  to 16% (H200) slower. It is still 1.3–2.6× faster than the stored symbol.
- **L40S (FP64 : FP32 = 1 : 64)**: the predicted severe case. The kernel is
  10–13× slower, the apply about 2×, and at 256³ the evaluated symbol is now
  **slower than the stored one** (10.9 vs 7.1 ms). That defeats the reason the
  factory picks it on a device. At 512³ it still wins, because the stored
  symbol is memory-bound.
- On `main` the float32 kernel on MI300A used 256 B of scratch per lane;
  moving to double removed it.

**Decision**: the cost is acceptable on FP64-capable GPUs but not on
consumer/workstation cards, so float32 now takes single precision with a
double retry for the modes that need it (next subsection).

### Implemented: single-precision attempt with a double retry

`green_symbol_gpu.cc`, float32 fields only (float64 fields and the host stay
in double throughout):

- Every mode is first finished in float32: hoisted sums, symbol, scaled
  cofactor inverse.
- **Acceptance test, a posteriori on that block.** Rounding the phase sum
  leaves an absolute error of about `E = eps32 · max_ab Σ_d |S_d,ab|` in K. In
  a numpy emulation over isotropic, λ/μ = 100 and anisotropic (h ratio 8–10)
  stencils in 2D and 3D, the mode's relative error stayed below 0.4·E/b with
  `b = det / e2` (`e2` = sum of the principal 2×2 minors, i.e. the diagonal
  cofactors; the trace in 2D), which lower-bounds λ_min of a Hermitian
  positive definite block within a factor of Dim. The float result is kept
  iff the block is positive definite (Sylvester on the leading minors) and
  `b ≥ 2·E / 1e-4`. The definiteness check is load-bearing. Without it, a
  float K with two negative eigenvalues still has det > 0, and passed with
  errors of 7e2 (λ/μ = 100) and 1.2 (anisotropic) on the lowest modes.
- Rejected modes are redone in double. A block computes its double hoisted
  sums only for the lines holding a rejected mode, behind `__syncthreads_or`,
  so blocks without one pay nothing. Routed to double (emulation, 256³ modes):
  0.05% for an isotropic medium, 3% at λ/μ = 100, 20% for strong anisotropy.
- Tolerance 1e-4 rather than float32 resolution. A preconditioner this close
  to exact changes the PCG iteration count by O(1e-4), and the attainable
  accuracy is set by the operator. A tighter target is not cheap: the float
  inversion alone is only good to about eps·κ ≈ 1e-5 at κ ≈ 100, and
  δ = 1e-5 would send 50–100% of the modes of stiff or anisotropic media to
  double.
- The shared-memory and per-thread (short axis 0) paths are now separate
  kernel instantiations, chosen at launch from n0. Sharing one register
  allocation made the 3D float32 kernel take 217 VGPRs on gfx942 (occupancy 2).

Measured (512³, ms, float32 symbol kernel / full apply):

| GPU | `main` | double throughout | this |
|---|---|---|---|
| MI300A | 2.87 / 21.98 | 3.79 / 22.70 | 3.15 / 22.03 |
| H200 | 2.34 / 9.28 | 3.79 / 10.73 | 2.33 / 9.28 |
| L40S | 4.85 / 44.93 | 46.97 / 87.19 | 4.82 / 45.05 |

At 256³ the float32 apply is 2.06 / 1.18 / 5.45 ms (MI300A / H200 / L40S),
the same as `main`. The evaluated symbol again beats the stored one on all
three GPUs (L40S 256³: 5.45 vs 7.13 ms). The float64 kernel moved by −15%
(H200) to +9% (L40S, the pre-inversion scaling). Registers of the 3D shared
kernel: sm_89 72 (float32) / 68 (float64); gfx942 84 VGPRs, occupancy 5, no
scratch (float32), 78 VGPRs, occupancy 6 (float64).

Accuracy, float32 device kernel against the float64 host kernel, max relative
error per mode on the lowest modes of N = 4096 (3D) / 65536 (2D) and on full
small grids: ≤ 1.5e-5 in every case, including λ/μ = 100 and h ratio 8. No
mode is zeroed. `test_evaluated_symbol_on_device_is_accurate_in_single_precision`
now covers the stiff and anisotropic cases at a 1e-4 tolerance, and the
float32 tolerance of `..._matches_stored_on_host` is 1e-5 (measured 2e-6).

Not done: the cancellation-free form `K(q) = Σ_d S_d (e^{−iθ·d} − 1)`
(error ≈ eps/q instead of eps/q²) would shrink the double set further. It
still hoists: `K = Σ_{d0} [φ0 P'[d0] + (φ0 − 1) R[d0]]` with
`P'[d0] = Σ_{d12} S (φ12 − 1)` and the constant `R[d0] = Σ_{d12} S`. With
0.05% of modes in double it is not worth the complexity for isotropic media,
but it would help nearly incompressible or strongly anisotropic ones.

---

## 4. Full audit findings

Abbreviations: S = `language_bindings/python/muGrid/Solvers.py`,
P = `language_bindings/python/muGrid/Preconditioners.py`.

### Clean areas (verified numerically)

- **FFT on CPU**: forward/inverse vs numpy for odd/even sizes, 1D/2D/3D,
  components and sub-points, float32/float64; serial, `MUGRID_FFT_NO_ND=1`,
  MPI np = 2/3/4 (1×P, P×1, 2×2), staged transpose, empty ranks.
  Unnormalised both ways, consistent across backends.
- **Ghost exchange/accumulation**: exact adjoints, incl. ghosts wider than the
  domain and a rank that is its own neighbour.
- **FE tables, stiffness**: B tables, quadrature, fused stiffness = assembled
  BᵀCB, diagonal, sensitivities, macro RHS; grid transfer R = Pᵀ exactly.
- **Reductions**: conj(a)·b on all paths; widening to double before multiply;
  bit-reproducible per device and rank count.
- **NetCDF** layout and parallel offsets; units module.
- **Textbook PCG** recurrences; MPI reductions in solvers are global wherever
  a branch depends on them.

### Parameter inventory

| Location | Parameter | Value | Settable? | Abs/rel, assessment |
|---|---|---|---|---|
| S:168, S:407 | `rtol` | 1e-6 | kwarg | Relative to global ‖b‖₂, confirmed on the true residual (S1, S3 fixed) |
| S:83, S:357 | `atol` | 0.0 | kwarg | Absolute; b=0, x₀≠0 → tolerance 0, runs to maxiter |
| S:79, S:353 | `maxiter` | 1000 | kwarg | Raises `ConvergenceError` |
| S:25 | `FLOAT32_RTOL_FLOOR` | 1e-6 | module const | Warning only; text corrected (S3) |
| S:264, S:278, S:523, S:531 | breakdown tests | `<= 0.0` | hard-coded | Negative curvature and indefinite preconditioners raise (S4 fixed) |
| P:130 | Jacobi diagonal check | `> 0` and finite | `communicator=` | Collective with a communicator (S4 fixed) |
| P:446 | Hermitian-symbol test | 1e-10·max\|B\| | no | Not dtype-aware → float32 always dense |
| P:652, P:681 | `void_tol` | 0.0 | kwarg | Absolute (unit-dependent) |
| P:1154 | per-block `np.linalg.inv` | – | – | No singularity/residual check |
| P:1446 | MG off-diagonal test | 1e-10·max(scale, 1) | no | Mixed abs/rel |
| P:1464 | power iteration for λ_max | 100 its, seed 0 | no | No convergence check; ω = 1.7/λ_est may be too large |
| P:1578–1582 | MG `SAFETY`, `nu`, `nb_cycles`, `min_coarse` | 1.7/2/1/32 | kwargs except SAFETY | – |
| P:1779 | `_INVERSE_RESIDUAL_EPS` | 1e3·eps(dtype) | const | Dtype-aware, good |
| P:2393 | factor compression | 1e-12·scale | no | Not dtype-aware (memory only) |
| `green_symbol*.cc` | singularity test | `|det|² > 0` | no | Underflowed in float32 — **fixed (S2)** |
| new, S | `replacement_interval` | 50 (f64) / 20 (f32) | kwarg | **added (S1)** |
| `generic.hh:382` | `zero_tolerance` | 1e-14 | no | **Absolute**, drops coefficients (O3) |
| `fem_gradient.hh:366` | grid spacing | 1.0 | yes | – |
| Wrappers.py:1482, 1630, 1685 | default element | Q1 | yes | Docstrings say P1 (O9) |
| `laplace.hh:157` | `scale` | 1.0 | yes | Single scalar, no per-axis h (O8) |
| `transfer.cc` | prolongation ½, ¼, ⅛; factor 2 | – | no | Galerkin-exact for Q1 only (O6) |
| isotropic stiffness | 2D model | plane strain | no | Undocumented, no plane-stress option |
| `fft_engine_base.cc` ~L130 | x-ghost padding to even | – | no | Silently enlarges ghosts (F5) |
| `fft_utils.cc` | process-grid selection | slab if P ≤ Nz else pencil | `decomposition=` | Can idle ranks (F4) |
| `python_helpers.hh:51` | coordinates | `(i mod N)/N` | no | Fractional, ghosts wrap (M9) |
| `cartesian_communicator.cc:295` | topology | periodic in all dims | no | – |
| env `MUGRID_GPU_AWARE_MPI`, `MUGRID_UNIFIED_MEMORY` | – | first char 1/y/t | env | Cached on first read; `"on"` = false |
| env `MUGRID_FFT_NO_ND`, `MUGRID_STAGED_TRANSPOSE` | – | only `'1'` | env | Read at construction |
| `tensor_algebra.hh:351` | `Iasymm` | 1/3 | no | Hard-codes dim = 3 (L9) |

### Solvers and preconditioners

- **S1 — FIXED in this PR** (section 1).
- **S2 — FIXED in this PR** (section 1).
- **S3 — FIXED** (follow-up PR): true-residual check on exit, restart,
  stagnation error; warning text corrected.
  ~~**S3. Standard CG in float32 returns above the requested tolerance**
  (significant, CONFIRMED). S:28–41, S:324: cond 1e3, rtol 1e-8 returned with
  true residual 6.2e-7; cond 1e4, rtol 1e-5 returned at 1.01e-5. The warning
  claims the solve "will run to maxiter". Fix: true-residual check before
  return (one `hessp`), as in S1, or clamp rtol to the floor.~~
- **S4 — FIXED** (follow-up PR): `p^T A p <= 0` / `r^T M r <= 0` raise
  (pipelined: after a restart on fresh vectors); Jacobi rejects
  non-positive or non-finite entries, collectively with `communicator=`.
  ~~**S4. Indefiniteness not detected** (significant, CONFIRMED). Only `pAp == 0`
  / `alpha == 0` checked; an operator with 1/7 negative eigenvalues ran 17,529
  iterations without error. Jacobi accepts negative diagonals and its check is
  rank-local (one rank raising deadlocks the others). Fix: raise on
  `pAp <= 0`, `rz <= 0`; collective `diag <= 0` check.~~
- **S5 — FIXED** (follow-up PR). Unique per-call work-field names, popped on
  exit; float32 get-or-create checks the existing field.
  ~~**S5. Solver work fields are fixed-name get-or-create fields in the
  caller's collection** (significant, CONFIRMED). S:197–211, S:424–450;
  `Wrappers.py:93–95`. A nested CG on the same collection corrupts the outer
  solve (reported convergence at true residual 1.0); float32 after float64 on
  the same collection raises TypeError; `real_field` ignores dtype/shape on
  re-get. Fix: unique per-call names + `pop_field` in `finally`, or a
  workspace object; validate dtype/shape on re-get.~~
- **S6 — FIXED** (follow-up PR): built and inverted in double per slab, symmetrised,
  stored as the upper triangle; float32 error on the lowest modes 4e-2 → 2e-7 at 4096².
  ~~**S6. Stored Green preconditioner in float32 assembled in single
  precision** (significant, CONFIRMED). P:983–986 claims double; `K_hat` is
  complex64 (P:1081). Lowest-mode inverse error 3e-4/7e-3/0.13 at N =
  256/1024/4096; Hermitian test fails in float32 so storage is always dense.
  Fix: assemble in complex128 slab-wise, test Hermiticity before casting,
  symmetrise and store the upper triangle.~~
- **S7 — FIXED** (follow-up PR): outputs `noconvert` and writeable; shapes, slots and
  strides validated.
  ~~**S7. In-place kernels write into a temporary on dtype/layout mismatch**
  (significant, latent, CONFIRMED). `bind_py_linalg.cc` uses
  `py::array_t<…, c_style | forcecast>` for the *outputs* of
  `apply_green_symbol_*` and `block_thomas_*`; a strided or wrong-dtype output
  comes back unchanged without error. Fix: `.noconvert()`, require writeable,
  check sizes (`nb_head == 0` indexes −1).~~
- **S8. Minor**: Jacobi `_inverse_diagonal_field` (P:154) always float64, so a
  float32 solve raises; `_build_pcr` docstring (P:2606) claims a check that
  does not exist; `MultigridReferencePreconditioner` (P:1612) guards MPI only
  through the `communicator` argument (SUSPECTED); power iteration has no
  convergence check; pipelined solver has no `residual=` argument.

### Linalg kernels and reductions

- **L1 — FIXED** (follow-up PR): sub-points checked in `check_three_vector`.
  ~~**L1. `cross` / `leray_project` read out of bounds on sub-point mismatch**
  (significant, CONFIRMED). `linalg_host.cc:709–795` + GPU: check equal
  `get_nb_sub_pts()`.~~
- **L2 — FIXED** (follow-up PR): shared `check_pipelined_cg_operands`.
  ~~**L2. `pipelined_cg_dots<Real32, HostSpace>` has no operand validation**
  (significant, CONFIRMED). `linalg_host.cc:604–625`: returns garbage or reads
  OOB where Real and GPU paths raise. Shared validation helper.~~
- **L3 — FIXED** (follow-up PR): shared `check_contiguous` in the flat-buffer
  kernels (host tests only; the device check is compiled but not exercised
  with padded device storage).
  ~~**L3. Kernels assume dense pixel buffers** (significant; CONFIRMED host,
  SUSPECTED GPU). `scal(field, field)`, `cross`/`leray`, all GPU full-buffer
  ops (`linalg_gpu.cc:913, 949`): padded `pixels_strides` silently give wrong
  results on GPU while host Eigen paths throw. Shared `check_contiguous`.~~
- **L4. GPU reduction scratch is process-global and kernels launch on the
  current device** (significant, SUSPECTED, multi-GPU per process).
  `linalg_gpu.cc:739–756`; same in `ghost_accumulate_gpu.cc:62–75`. RAII
  device guard from `field.get_device_id()`, scratch keyed by device.
- **L5.** Host `leray_project` accumulates in complex64 for float32, GPU in
  double (`linalg_host.cc:688/697`) — minor.
- **L6.** Complex `norm_sq` can return a non-zero imaginary part (FMA
  contraction) — return `Real`. Minor.
- **L7.** Host/GPU asymmetry on SoA buffers; `eigen_pixel`/`eigen_sub_pt`
  misread host SoA silently. Minor.
- **L8.** Full-buffer GPU reductions launch `ceil(n/256)` blocks instead of
  `reduction_blocks()` (`linalg_gpu.cc:832, 1067`). Minor/perf.
- **L9.** `Iasymm<dim>` is the deviatoric projector, correct only for dim = 3
  (unused); `ct_sqrt` accepts non-squares. Minor.
- **L10.** `vecdot` docs omit conj(a)·b; float32 `axpy` coefficients rounded
  to float32. Minor.

### FFT

- **F1. Serial float32 FFT throws under `MUGRID_FFT_NO_ND=1`** (significant,
  CONFIRMED). `fft_engine.hh:196–203, 247–254`: fall through to the existing
  `fft_{1,2,3}d<Real32>` paths when `!supports_nd()`.
- **F2.** Serial float64 2D never uses the N-D transform, contrary to
  `docs/gpu.md:312`; on GPU ~2·nb_dof syncs + a copy. Minor.
- **F3. cuFFT plan parameters cast to `int` without checks** (significant for
  > ~1291³ single-GPU grids, SUSPECTED). `cufft_backend.cc`: use
  `cufftMakePlanMany64` or range-check.
- **F4.** Pencil heuristic can idle ranks (np 16 on (16,16,3) → 4 empty).
  Minor.
- **F5.** x-ghosts silently padded to even counts; `.sg` shape differs from
  docs. Minor.
- **F6.** Nyquist sign on the half-complex axis follows `fftfreq` (−0.5), not
  `rfftfreq`; inconsistent with C++ `rfft_freq` and P:1930. Minor.
- **F7.** GPU multi-axis C2R on non-Hermitian input undefined in cuFFT
  (SUSPECTED). Minor.
- **F8.** Slab fast-path probe catches every `RuntimeError`; a transient OOM
  permanently downgrades the engine. Minor.
- **F9.** Caches never evicted; `docs/architecture.md` AoS-only claim stale.

### Operators

- **O1 — FIXED** (follow-up PR). The transpose writes owned nodes only and
  requires one left ghost layer; protocol "communicate the input, never
  reduce". Host and device computed the same thing before -- every element in
  the buffer into every node -- so the defect was the protocol, not a
  host/device split.
  ~~**O1. FEM `transpose` correctness depends on ghost layout and call order**
  (significant, CONFIRMED). `fem_gradient.hh:498–507`; host divergence
  scatters from ghost elements too. With the operator's reported ghosts (0,1),
  "communicate then transpose" gives rel. error 5–30; with (1,1), "transpose
  then reduce" double-counts. Docs and C++ comment disagree. Fix: make the
  host divergence a gather onto owned nodes (as on the GPU), report {1,0},
  document "communicate input, never reduce".~~
- **O2 — FIXED** (follow-up PR): `LinearOperator::check_transpose_weights`.
  ~~**O2. Transpose `weights` length not validated** (significant, CONFIRMED):
  OOB read. `fem_gradient.hh:632/720`, `generic.cc:781`, `generic_gpu.cc:165`.~~
- **O3 — FIXED** (follow-up PR): threshold relative to the largest coefficient.
  ~~**O3. `GenericLinearOperator` drops |c| < 1e-14 (absolute)** (significant,
  CONFIRMED). `generic.hh:382`: identity × 1e-15 → zero; SI-unit micro-scale
  operators vanish. Drop exact zeros or use a relative tolerance.~~
- **O4.** Generic and FEM transposes default to different weights (ratio
  Wfrac·∏h). Significant (leaky API).
- **O5.** 3D "P1 Kuhn" is a non-conforming, same-orientation 5-tet split
  (`fem_element.hh:97–126`). Minor; use 6-tet Kuhn or alternate.
- **O6.** Prolongation Galerkin-exact only for Q1 (P1: 0.26/0.50 rel. error
  in RKP vs K_2h). Minor.
- **O7.** `fourier()` sums over outputs/quad points (`generic.cc:622–660`).
- **O8.** `LaplaceOperator`: no anisotropic h; sign div grad vs FE −h^d∇².
- **O9.** Docs say P1, default Q1; stale `nb_quad`; `NodalMomentOperator` not
  exported; float32 FEM gradient not reachable via wrapper.
- **O10.** Duplicated validation; GPU stiffness matrices in process-global
  `__constant__` memory (SUSPECTED race with concurrent streams/devices).

### MPI, fields and I/O

- **M1. `.p`/`.pg` order the merged component×sub-point axis differently on
  host and device** (significant, CONFIRMED). `field.cc:283–310`,
  `Field.py:134–177`. Build `.p` from `.s` in one canonical order.
- **M2 — FIXED** (follow-up PR): collective check, all ranks raise.
  ~~**M2. `Communicator.gather` drops data on uneven rows** (significant,
  CONFIRMED). `communicator.hh:282–383`: row check is an `assert`.~~
- **M3 — FIXED** (follow-up PR): integer identities from `np.iinfo`.
  ~~**M3. Integer min/max on an empty rank deadlocks** (significant, CONFIRMED).
  `Parallel.py:453–467`: ±inf identity overflows on one rank only.~~
- **M4 — FIXED** (follow-up PR): scalar reductions typed by dtype
  (`_sum_int64`, `_sum_complex128`, ...).
  ~~**M4. `reduce_sum` / `Communicator.sum` lose data** (significant,
  CONFIRMED): imaginary parts dropped, int64 > 2⁵³ rounded.~~
- **M5 — FIXED** (follow-up PR): narrowing reads raise; widening ones still convert.
  ~~**M5. NetCDF reads narrow dtype silently** (significant, CONFIRMED).
  `file_io_netcdf.cc:244, 971–994, 2307`: compare `nc_inq_vartype`.~~
- **M6. Python GPU reductions ignore GPU-aware-MPI detection** (significant,
  SUSPECTED). `Parallel.py` `_sum_gpu`, `_max_gpu`, `_allreduce_scalar`.
- **M7.** `compute_axes_order` mis-breaks stride ties for size-1 axes. Minor.
- **M8.** "SoA" means row-major pixels on host, column-major on device;
  `communicate_ghosts` assumes column-major (FIXME at
  `cartesian_decomposition.cc:245`). Minor.
- **M9. Minor**: Python `CartesianDecomposition` default subdivisions throw;
  `coords` fractional not physical and ghosts wrap; `nb_grid_pts` local vs
  global naming; NetCDF local write narrows `num_requests`; `import muGrid`
  always initialises MPI in MPI builds.

### Tests

- **T1 — FIXED** (follow-up PR): `Device.gpu()`; the three files pass on
  MI300A. ~~**T1. Device tests hard-code `muGrid.Device.cuda()`** (minor,
  CONFIRMED on MI300A). 31 tests in `python_convolution_operator_tests.py`,
  `python_convolution_performance_tests.py` and `python_field_gpu_tests.py`
  fail on every ROCm build; use `Device.gpu()` / `create_device("gpu")`.~~

---

## 5. Suggested order for the remaining work

1. ~~Low-frequency double path for the evaluated symbol~~ — done
   (section 3).
2. ~~O1 (FEM transpose protocol) and S5 (named solver work fields)~~ — done,
   with T1.
3. ~~S3 (same true-residual check for standard CG), S4 (indefiniteness)~~
   — done.
4. ~~S6, S7, O3, M5 (silent precision/data loss at interfaces)~~ — done.
5. ~~L1–L3, O2, M2–M4 (input validation / out-of-bounds)~~ — done.
