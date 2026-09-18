# Multigrid-accelerated reference preconditioner — implementation plan

Target: replace the fine-grid FFT in the reference-material (Green) preconditioner
with a multigrid V-cycle whose **coarsest level** is solved by the existing
FFT-based block symbol. Motivation is multi-GPU MPI scaling; see
[Why](#1-why-the-current-preconditioner-does-not-scale).

**Status:** Stages 1 and 2 complete (PR #206). Stage 3 (GPU) and Stage 4 (MPI)
remain, and one gating measurement is outstanding.

**If you are picking this up cold, read [§0 Start here](#0-start-here) first** —
current state, where the code lives, how to build and test in this repo, what to
do next, and the traps that have already cost time. Sections 1–6 are the design
and the measurements behind it; they explain *why*, and can be read on demand.

---

## 0. Start here

Self-contained orientation for someone (or something) picking this up cold.
Sections 1–6 are the design and the measurements behind it; this section is
what you need to *operate*.

### 0.1 Current state

Stages 1 and 2 are **done**. Stages 3 (GPU) and 4 (MPI) remain, and one
gating measurement is outstanding (§0.4).

- **PR #206** — `feat/multigrid-reference-preconditioner`, one commit, open
  against `main`, mergeable. Contains `GridTransfer{2,3}D` (host kernels +
  bindings + wrapper), `MultigridReferencePreconditioner` (serial),
  19 tests, CHANGELOG entries.
- **Not in the PR, local and untracked**: this document, and
  `mg_prototype.py` at the repo root. Both are deliberate — the repo's
  convention is that planning docs stay untracked (cf.
  `docs/homogenization_memory_migration.md`), and an unlisted page under
  `docs/` risks the `mkdocs build --strict` job. Add them if you disagree; it
  is one commit.
- The branch was rebased onto `main` to drop two unrelated commits from
  `doc/history-independent-comments`. Do not branch from that branch.

### 0.2 Where the code lives

| path | what |
|---|---|
| `src/libmugrid/operators/transfer.{hh,cc}` | `GridTransfer<Dim>` + host kernels, `Real`/`Real32` |
| `language_bindings/python/bind_py_operators.cc` | `add_grid_transfer<Dim>`, near the end |
| `language_bindings/python/muGrid/Wrappers.py` | `GridTransfer` wrapper, at EOF |
| `language_bindings/python/muGrid/Preconditioners.py` | `_MultigridLevel`, `MultigridReferencePreconditioner`, at EOF |
| `tests/python_multigrid_tests.py` | 19 tests (13 transfer, 6 preconditioner) |
| `mg_prototype.py` (repo root, untracked) | **the NumPy oracle** — `--probe`, `--check`, `--rho`, `--cg` |

`mg_prototype.py` is the reference implementation that validated everything in
§6. Keep it working: it is how Stage 3 and Stage 4 will be checked, because it
shares nothing with the production path but the mathematics.

### 0.3 Build and test — read before running anything

```bash
source benchenv.sh          # from the workspace root: venv + homebrew libmpi
cd muGrid
```

**The workspace venv will not see your build tree.** It carries a
scikit-build-core *editable* install of muGrid whose
`ScikitBuildRedirectingFinder` sits on `sys.meta_path` and therefore beats
`PYTHONPATH`; its `rebuild()` hook also fails ("no persistent build
directory"). Two ways out — either

```bash
pip install -e . -Cbuild-dir=build     # gives the editable install a rebuild hook
```

or keep site-packages untouched and drop the finder in the test process:

```python
# runtests.py
import sys
sys.meta_path = [f for f in sys.meta_path
                 if type(f).__name__ != "ScikitBuildRedirectingFinder"]
import pytest
sys.exit(pytest.main(sys.argv[1:]))
```

Configure a build that reuses the already-fetched dependencies:

```bash
cmake -S . -B cmake-build-mg -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE=$(which python3) -DMUGRID_ENABLE_TESTS=ON \
  -DFETCHCONTENT_SOURCE_DIR_EIGEN=$PWD/cmake-build-debug/_deps/eigen-src \
  -DFETCHCONTENT_SOURCE_DIR_PYBIND11=$PWD/cmake-build-debug/_deps/pybind11-src \
  -DFETCHCONTENT_SOURCE_DIR_DLPACK=$PWD/cmake-build-debug/_deps/dlpack-src
cmake --build cmake-build-mg -j8
```

Run the tests:

```bash
export PYTHONPATH=$PWD/cmake-build-mg/language_bindings/python:$PWD/language_bindings/python
python runtests.py tests/ -q                                    # 546 passed, 68 skipped
mpiexec -n 4 --oversubscribe python runtests.py tests/python_multigrid_tests.py -q
cd cmake-build-mg && ctest --output-on-failure                  # C++
```

### 0.4 What to do next, in order

**1. The gating measurement — needs one GPU, not four.** Nothing else should be
built until this number exists.

A V-cycle replaces exactly one thing, the FFT pair, and costs about 5.7 uniform
matvecs to do it. The decisive quantity is therefore

```
R  =  5.7 × t(fused_kernel) / [ t(fft) + t(scale) + t(ifft) ]
```

on a single device, because a V-cycle's cost is halo-only and scales well with
ranks while the FFT's all-to-all scales badly — so `R` at P=1 plus that
asymmetry predicts the crossover without observing it. Every term is already
instrumented; one `-P reference` GPU run gives all four:

- `total_solve/iteration/hessp/apply_stiffness/fused_kernel`
- `total_solve/iteration/prec/fft`, `/prec/scale`, `/prec/ifft`

```bash
for N in 128 192 256; do
  python examples/homogenization.py -n $N,$N,$N -d gpu -k fused -P reference
done
```

No GPU multigrid code is needed: the V-cycle's cost is dominated by
`apply_uniform`, which is already GPU-enabled.

**`R > 1` on one device is expected and is not a failure signal** — there the
FFT has no communication at all and rocFFT's native 3-D transform is excellent.
Read it as the deficit the all-to-all must make up:

| `R` | reading |
|---|---|
| 1–2 | wins at modest rank counts — clear go |
| 3–5 | needs the FFT to degrade 3–5×, i.e. roughly 8–32 ranks — go, but 4 GPUs may only just show it |
| > 8 | flop cost too steep — revisit `ν`, or push the coarse level cheaper |

Two more single-GPU results worth taking in the same session: whether multigrid
reaches `384³` on one device where `-P reference` OOMs (see the table in
`docs/benchmark_homogenization_preconditioner.md`), and **Stage 0**, which needs
no GPU at all — `-P none` on a 3D `CartesianDecomposition` versus the FFT
engine's pencil, so the eventual multi-GPU numbers can separate "removed the
FFT" from "removed the pencil".

Running several MPI ranks on a single device will produce numbers, but they are
meaningless for this question: the all-to-all never leaves one device's memory.
That configuration is useful for Stage 4 *correctness* only.

**2. Stage 3 — GPU transfers and Chebyshev.** `transfer_gpu.cc` mirroring
`transfer.cc`. The kernels already take explicit strides, so only the caller's
stride computation changes: host is array-of-structures (`stride_dof = 1`,
`stride_x = nb_dof`), device is structure-of-arrays (`stride_dof = nb_pixels`,
`stride_x = 1`). This is the likeliest source of a silent bug and will not show
up on the host. Then Chebyshev, reusing the `λ_max` the damping rule already
computes.

**3. Stage 4 — MPI.** Power-of-two subdivisions pinned across levels (see the
nesting precondition, §2.3, and `_power_of_two_subdivisions` in the tests), the
redundant coarsest level via `comm.sum` (§2.6), and lifting the
`NotImplementedError` in `MultigridReferencePreconditioner.__init__`.

**Acceptance, at every stage: the production path must reproduce
`mg_prototype.py`'s CG iteration counts to the digit** — 36/36/36 and 26/35/53
on the sharp inclusion at contrast 10, 35/36/37 and 15/15/15 on smooth, for
n = 16/32/64 in 3D. That is a far sharper criterion than "it converges", and it
is why the prototype exists.

### 0.5 Traps that have already cost time

- `CartesianDecomposition(nb_subdivisions=None)` documents "Default is
  automatic" but passes `[0]*dim` and raises *"The total number of subdivisions
  (0) does not match the size of the communicator"*. Always pass it explicitly.
- `Communicator.sum` is bound through Eigen and takes **only** scalars and 2D
  Fortran-contiguous `float64`. A 3D or C-contiguous array raises `TypeError`.
  Reshape to `(nb_dof, -1)` and `np.asfortranarray` — see `_gather_interior` in
  the tests. This matters for the redundant coarsest level.
- `collection.register_real32_field(...)` returns the **bare C++ field**;
  `real_field(...)` returns the Python wrapper. Wrap it with `muGrid.wrap_field`
  or `.s` / `.p` will not exist.
- Test files must match `python_*_tests.py` or pytest silently skips them —
  `tests/python_solvers_test.py` (singular) is not collected today.
- A new `.cc` must be added to `src/libmugrid/CMakeLists.txt` in
  `MUGRID_SOURCES`, in **both** the CUDA and HIP branches, **and** in the
  matching `set_source_files_properties(... LANGUAGE ...)` lists. Missing the
  last one compiles it as plain C++ and fails obscurely.
- pre-commit runs flake8 *before* isort, so isort's rewrite is unchecked. Run
  flake8 again after a commit that isort modified.
- `Solvers.py` has pre-existing E501 warnings. They are not yours.

---

## 1. Why the current preconditioner does not scale

With `-P reference` the solver fields live on the FFT engine's collection
(`examples/homogenization.py:408-422`), and the engine pins

```cpp
nb_subdivisions[0] = 1;  // X is not distributed in real space
```

(`src/libmugrid/fft/fft_engine_base.cc:110`). Three consequences:

1. **Two all-to-alls per transform, four per preconditioner apply.** All-to-all is
   latency- and bisection-bandwidth-bound and is a full barrier across all ranks.
2. **X is never distributed**, so every rank holds the full X extent — a memory
   floor per device, and the reason `384³` OOMs on a single GPU.
3. The real-space decomposition is pencil at best, never 3D, because the
   preconditioner owns it.

Measured (`docs/benchmark_homogenization_preconditioner.md`, MI300A):

| grid | 1 GPU | 4 GPUs | speedup |
|---|---|---|---|
| 128³ | 1.99 s | 1.73 s | 1.15× |
| 192³ | 5.89 s | 5.12 s | 1.15× |
| 256³ | 13.2 s | 10.2 s | 1.29× |

---

## 2. The design

### 2.1 Key decision: the cycle runs on the *reference* operator only

The preconditioner being approximated is `Kʳᵉᶠ` — **spatially uniform** Lamé
parameters (the volume means). The multigrid cycle therefore never sees the
heterogeneous material. Heterogeneity is handled exactly where it is handled
today: by the symmetric Jacobi scaling of `GreenJacobiPreconditioner`
(Ladecký et al., J-FFT).

```
M⁻¹  =  J^{1/2} · V-cycle(Kʳᵉᶠ) · J^{1/2}
                  ^^^^^^^^^^^^^
                  replaces the fine-grid FFT solve
```

This is the decision that collapses most of the work:

| Would have been needed | Status under this design |
|---|---|
| Material restriction (coarse λ, μ) | **Not needed** — every level is uniform |
| Galerkin/RAP coarse operator | **Not needed** — rediscretize `Kʳᵉᶠ` with `2h` |
| Contrast-robust interpolation | **Not needed** — cycle sees no contrast |
| `assemble_block_diagonal` field + per-node inverse kernel | **Collapses to one constant `dim×dim` matrix per level** (§2.4) |
| Operator-dependent smoothers | **Not needed** — textbook constant-coefficient MG |

The coarse operator at level `l` is just

```python
muGrid.IsotropicStiffnessOperator(dim, 2**l * grid_spacing, element)
```

applied through `apply_uniform` / `apply_uniform_increment`
(`src/libmugrid/operators/solids/isotropic_stiffness.hh:392,399`), which already
exist on host and device.

### 2.2 Level structure

| Level | Grid | Decomposition | Operator | Role |
|---|---|---|---|---|
| `0` | `N³` | 3D `CartesianDecomposition`, `nb_subdivisions = s` | `Kʳᵉᶠ(h)` | smoothed |
| `1 … L-1` | `N/2^l` | same `s`, nested | `Kʳᵉᶠ(2^l h)` | smoothed |
| `L` | `N/2^L` | **serial, redundant on every rank** | FFT block symbol | solved exactly |

**Choosing `L` is not a numerical decision** (§6.6): ρ and the CG count are flat
in the coarsest grid size across an 8–32× range, so nothing in the numerics
prefers one `L` over another. Pick it from the *parallel* side instead:

> Go redundant at the first level where the local subdomain would fall below
> ~8 points per direction.

That keeps every distributed level out of the latency-dominated regime — where a
1-layer ghost halo costs more than the interior — and makes the choice
rank-count-aware rather than a hardcoded constant. The Allreduce is negligible at
any resulting size (`512³`: `32³` coarsest is 0.79 MB/rank, `64³` is 6.3 MB).

**The coarsest level is the reason this is worth doing.** Today the block symbol
is stored on the fine grid: `n² = 9` reals per Fourier mode over `N³/2` modes
≈ `4.5 N³` reals, i.e. 1.5 fine solution vectors. Moving it to level `L` shrinks
it by `8^L` — 512× at `L = 3` — and the transform it drives becomes rank-local.

### 2.3 Nesting precondition (must be asserted at setup)

Grid transfers are rank-local iff coarse subdomains are exactly half the fine
ones. The default constructor
(`src/libmugrid/mpi/cartesian_decomposition.cc:182-193`) computes
`N / P` and distributes the remainder, so nesting holds **iff `P_d` divides
`N_d / 2^l` exactly at every level**.

With power-of-two grids this reduces to: **every subdivision count must be a power
of two.** Note `suggest_subdivisions(dim, comm.size)` does *not* guarantee this —
`comm.size = 12` gives `[2, 2, 3]`. So:

- factor the rank count into powers of two per direction explicitly, and
- assert `N_d % (P_d << L) == 0` at construction, with a clear error.

If non-power-of-two rank counts must be supported later, the C++
`CartesianDecomposition::initialise` overload taking explicit
`nb_subdomain_grid_pts` / `subdomain_locations`
(`src/libmugrid/mpi/cartesian_decomposition.hh:51`) already exists but is **not
bound to Python** — only the subdivisions-based constructor is
(`language_bindings/python/bind_py_decomposition.cc:184`). Binding it is the
escape hatch.

### 2.4 Smoother

For a **uniform** operator on a regular periodic grid every node has an identical
element neighbourhood, so the nodal `dim×dim` block `K_nn` is **one constant
matrix for the whole level**, not a field.

**Measured** (§6.1): for the production configuration — 3D, Q1, isotropic `h` — it
is `c·I`, so damped Jacobi is `z += (ω/c) · t`, a scalar `linalg.axpy`, and the
smoother needs **no kernel of its own**. Only 2D P1 needs the general
`dim×dim` path. Recover the block at setup with `dim` applies of `apply_uniform`
to unit impulses, reading the response at the impulse node — the impulse-response
trick already used in `make_reference_stiffness_preconditioner`
(`language_bindings/python/muGrid/Preconditioners.py:836-856`).

**The damping `ω` must be derived, not tuned** (§6.3). The stability limit is
`ω < 2/λ_max(D⁻¹K)`, and `λ_max` moves with dimension and element kind, so a
hardcoded value that is optimal in 2D *diverges* in 3D. Use

```
ω = 1.7 / λ_max(D⁻¹K)
```

with `λ_max` from ~100 power iterations at setup. `D⁻¹K` is invariant under
uniform refinement — `D` and `K` carry the same power of `h` — so one estimate on
the *coarsest* level serves the whole hierarchy.

**Chebyshev** (Stage 3) remains preferable to plain damped Jacobi: no ordering, no
halo serialization, only matvecs and axpys — and it needs the same `λ_max` that
the damping rule already computes, so the machinery is shared.

### 2.5 Grid transfer

Standard multilinear (2:1) prolongation on nodal fields; `R = Pᵀ` full weighting.
Transfers act component-wise — no coupling between displacement components.

**Write both in gather form.** Then each needs only `communicate_ghosts` on its
*input* and no `reduce_ghosts`:

- **Prolongation** — loop over fine nodes, gather from 1–2^d coarse nodes.
  Needs 1 coarse ghost layer.
- **Restriction** — loop over coarse nodes, gather the surrounding 3^d fine nodes
  with full-weighting coefficients. Needs 1 fine ghost layer, which the stiffness
  stencil already requires.

(`reduce_ghosts` — `cartesian_decomposition.hh:80` — remains the tool if a scatter
formulation is ever preferred.)

Because multilinear interpolation reproduces linear displacement fields exactly,
its range contains all rigid-body modes and all constant strains. This is the
near-nullspace property AMG must be told explicitly; geometric MG gets it free.

### 2.6 The redundant coarsest level

Level `L` uses a **serial** `muGrid.Communicator()` on every rank. This is
explicitly supported — `Parallel.py:588-598` warns that "every rank will
redundantly work on the full problem by itself", which is precisely the intent
(the warning is already filtered in `pyproject.toml`).

- **Restrict into it:** each rank restricts its own piece into a zero-padded full
  coarse array, then one `comm.sum(array)` — GPU-aware, already implemented
  (`Parallel.py:159-183`). No new redistribution code. Note the binding is via
  Eigen and accepts only scalars and **2D Fortran-contiguous** `float64`
  arrays, so the coarse array has to be reshaped to `(nb_dof, -1)` and passed
  through `np.asfortranarray` — see `_gather_interior` in
  `tests/python_multigrid_tests.py`. A 3D or C-contiguous array raises
  `TypeError`.
- **Solve:** `make_reference_stiffness_preconditioner` **verbatim**, on a small
  serial `FFTEngine` at level `L`. The `q = 0` block is already replaced by its
  pseudo-inverse there, which is exactly the nullspace handling needed.
- **Prolong out of it:** purely local — every rank already holds the whole coarse
  solution and reads its own part.

This replaces 4 fine-grid all-to-alls per apply with **one Allreduce of a few MB**.

### 2.7 Symmetry — a hard requirement

Standard CG requires `M⁻¹` to be a fixed, symmetric, positive-definite linear
operator. Therefore:

- **fixed** cycle count and **fixed** smoothing-step count (no inner
  convergence test — that would make `M⁻¹` nonlinear and require flexible CG);
- `ν₁ = ν₂` with a symmetric smoother, and `R = Pᵀ` exactly.

The nullspace (the `dim` constant translations under periodic BCs; rigid rotations
are not periodic, so they are *not* in it) is projected out on entry and exit — one
global reduction per apply, against today's four all-to-alls.

### 2.8 Cost

**Measured** (§6.4, §6.5): one V-cycle with `ν = (2,2)` gives `ρ ≈ 0.355`,
independent of grid size *and* of dimension. In the J-scaled configuration that
production actually uses, replacing the exact fine-grid FFT by that V-cycle
changes the CG count by **0–4%**.

Per apply, a V-cycle costs `(ν₁ + ν₂ + 1) × 1.14 ≈ 5.7` fine matvecs against the
FFT's one forward/inverse pair, so **this trades flops for communication**. On a
single device the FFT wins; the crossover is a rank-count question that only
Stage 4 answers. That is what makes the Stage 0 baseline worth measuring first.

V-cycle communication is a geometric series: `1 + 1/8 + 1/64 + … ≈ 1.14×` the
fine-level halo exchange, plus the coarse Allreduce.

---

## 3. What has to be written

| # | Item | Where | Effort |
|---|---|---|---|
| 1 | ~~`prolong` / `restrict` kernels, host~~ | `src/libmugrid/operators/transfer.{hh,cc}` | **done** |
| 2 | Same, device | `transfer_gpu.cc`, via `KernelDispatcher` | medium |
| 3 | ~~Python bindings + wrappers~~ | `bind_py_operators.cc`, `Wrappers.py` | **done** |
| 4 | ~~`MultigridReferencePreconditioner(Preconditioner)`~~ | `muGrid/Preconditioners.py` | **done (serial)** |
| 5 | `linalg.sum` (interior-only) + `linalg.add_constant` | `linalg/linalg.hh` + `_gpu.cc` | small — **still open**: `_project_constants_out` is the one remaining array-library call in the hot path (twice per apply, not per level) |
| 6 | Power-of-two subdivision helper + nesting assertions | `Preconditioners.py` | small |
| 7 | `-P multigrid` flag and timers | `examples/homogenization.py` | small |

Item 5 is optional for a prototype (do the mean projection with `comm.sum` and
numpy/cupy on `.s`), but the hot loop should not depend on an array library —
that discipline is kept everywhere else in `Preconditioners.py`.

**Not needed:** material restriction, block-diagonal *fields*, per-node block
inverse kernels, Galerkin coarsening, agglomeration/sub-communicators.

Nothing in `Solvers.py` changes: the result conforms to the existing
`apply(r, z)` contract and drops straight into `conjugate_gradients`, and into
`GreenJacobiPreconditioner` as its `green` argument.

---

## 4. Staging

**Stage 0 — measurement baseline.**
Add `-P multigrid` plumbing and timers. Separately measure `-P none` on a *3D*
`CartesianDecomposition` versus the FFT engine's pencil, to separate "removed the
FFT" from "removed the pencil" in every later number.

**Stage 1 — serial host prototype, 2D, pure Python.**
Whole V-cycle on `.s` views with numpy, `apply_uniform` for matvecs, existing
`make_reference_stiffness_preconditioner` at the coarsest level. Confirm before
writing any C++:
- two-grid convergence factor `ρ` on uniform material;
- iteration count independent of `N`;
- `K_nn` really is `c·I` for Q1 (and what it is for P1).

**Stage 2 — transfer kernels in C++ (host, then GPU).**
`tests/python_multigrid_tests.py`. Three tests catch nearly everything:
- **adjointness** `⟨P c, f⟩ = ⟨c, Pᵀ f⟩` to round-off;
- **polynomial reproduction** — a linear displacement field prolongs exactly;
- **preconditioner symmetry** `⟨M⁻¹a, b⟩ = ⟨a, M⁻¹b⟩` (§2.7), the one that
  protects CG.

**Stage 3 — Chebyshev smoother**, spectral bound from the symbol or power
iteration.

**Stage 4 — MPI.** Power-of-two subdivisions pinned across levels, nesting
assertions, redundant coarse level via `comm.sum`. Re-run the 4-GPU benchmark;
this is where the design either pays off or does not.

**Stage 5 — compose with `GreenJacobiPreconditioner`** and measure against
contrast on the real microstructures (inclusions, SIMP near-void).

---

## 5. Open questions

- ~~**Q1 vs P1.**~~ **Resolved in Stage 1** (§6.1): Q1 gives `c·I` in both
  dimensions and 3D P1 does too; only 2D P1 needs the general constant-matrix
  multiply, and it is not a production target.
- **Where to stop coarsening.** Trade the Allreduce volume at level `L` against
  the extra smoothing work of more levels. `64³` and `32³` are both cheap;
  measure.
- **Does the coarse level want the heterogeneous operator?** Solving the *actual*
  coarse system (rediscretized λ, μ) by CG-preconditioned-by-FFT would capture
  long-wavelength heterogeneity that `Kʳᵉᶠ` misses, at the price of an inner
  iteration — which breaks linearity unless the count is fixed. Stage 1 (§6.5)
  makes this concrete: on a *sharp* interface the J-scaled preconditioner loses
  grid-independence (25 → 35 → 52) whether the inner solve is FFT or MG, so the
  ceiling there belongs to the J-FFT scheme, not to the V-cycle. If sharp
  high-contrast microstructures matter, this is the lever — and it would improve
  the existing FFT preconditioner just as much.
- **Is the iteration growth on sharp interfaces worth attacking separately?**
  It is a property of the preconditioner already in production and is orthogonal
  to the parallel-scaling work this plan is about. Flagged here because Stage 1
  measured it, not because this plan should fix it.

---

## 6. Stage 1 findings

Measured with `mg_prototype.py` (serial, host, NumPy; muGrid's own
`apply_uniform` for every matvec). Reproduce with `--probe`, `--check`, `--rho`,
`--cg`.

### 6.1 The nodal block `K_nn` — question 1

λ = 1.3, μ = 0.7, probed by impulse response:

| config | isotropic `h` | anisotropic `h` |
|---|---|---|
| 2D Q1 | `c·I` | diagonal, unequal entries |
| **2D P1** | **full block**, `K₀₁ = λ + μ` exactly | **full block** |
| 3D Q1 | `c·I` | diagonal, unequal entries |
| 3D P1 | `c·I` | diagonal, unequal entries |

- **The production configuration (3D, Q1, isotropic `h`) gives `c·I`**, so the
  smoother is a scalar `axpy` and needs no new kernel.
- The exception is **2D P1**, not 3D P1 as this plan originally guessed. The
  off-diagonal is exactly `λ + μ` for every `(λ, μ)` tested, which confirms the
  mechanism: the two-triangle Kuhn split breaks the `x ↔ y` symmetry that
  cancels `∫ N_n,x N_n,y`. The 3D five-tetrahedron decomposition keeps enough
  symmetry for the cancellation to survive. 2D P1 is a test configuration, so
  this is not on the critical path — but keep the general constant-matrix path.
- Anisotropic spacing keeps the block diagonal but with **unequal** entries, so
  store a `dim`-vector rather than a scalar. Still no kernel.

### 6.2 Transfer and symmetry checks — question 2

2D, fine `n = 32`. All three of the tests earmarked for Stage 2 already pass:

| check | relative error |
|---|---|
| adjointness `⟨Pc, f⟩ = ⟨c, Pᵀf⟩` | `2.9e-16` |
| linear-field reproduction `P(lin_H) = lin_h` | `1.4e-14` |
| preconditioner symmetry `⟨M⁻¹a, b⟩ = ⟨a, M⁻¹b⟩` | `7.2e-16` |

The prototype's `_prolong_axis` / `_restrict_axis` are a direct specification for
the C++ kernels: a tensor product of one 1D pass per spatial axis, with the
component axis untouched.

### 6.3 Damping — question 3, and an unanticipated result

`λ_max(D⁻¹K)`: 2D Q1 **2.38**, 2D P1 **2.45**, 3D Q1 **2.96**, 3D P1 **1.78**.

The measured divergence threshold matches `2/λ_max` in every case. Consequently
**a hardcoded `ω` is unsafe**: `ω = 0.7` is the 2D Q1 optimum and makes 3D Q1
diverge outright (`ρ = 1.33`). Normalising by `λ_max` collapses all
configurations onto one curve:

| `ω · λ_max` | 1.5 | 1.6 | **1.7** | 1.8 | 1.9 |
|---|---|---|---|---|---|
| ρ, 2D Q1 | 0.411 | 0.384 | **0.360** | 0.395 | 0.657 |
| ρ, 3D Q1 | 0.406 | 0.380 | **0.355** | 0.388 | 0.664 |
| ρ, 3D P1 | 0.456 | 0.430 | **0.406** | 0.404 | 0.663 |

Hence `ω = 1.7 / λ_max`, with ~15% margin to the cliff at 1.9.

### 6.4 V-cycle convergence factor — question 3

With the derived `ω` and `ν = (2,2)`:

| `n` | 2D | | `n` | 3D |
|---|---|---|---|---|
| 32 | 0.356 | | 16 | 0.354 |
| 64 | 0.363 | | 32 | 0.352 |
| 128 | 0.365 | | 64 | 0.355 |
| 256 | 0.367 | | | |

**ρ ≈ 0.355, flat in `n` and identical in 2D and 3D.** This is worse than the
`ρ ≈ 0.1` this plan assumed, but grid-independence — the property the design
needs — holds exactly. More smoothing lowers ρ (ν=4 reaches 0.165) but
work-normalised efficiency is flat at ≈0.82 per matvec across ν = 1…4, so
ν = 2 is the right default.

### 6.5 CG iteration count — question 4

The decisive measurement: PCG on the **heterogeneous** system (spherical
inclusion or a smooth sinusoidal field), tolerance `1e-8`. Both `FFT` and `MG`
approximate the same `Kʳᵉᶠ` built from the volume-mean Lamé parameters; `J·…·J`
is the Green-Jacobi composition that production uses.

3D Q1, iterations (`-1` = no convergence in 200):

| material | contrast | `n` | none | FFT | MG | J·FFT·J | J·MG·J |
|---|---|---|---|---|---|---|---|
| inclusion | 10 | 16/32/64 | – | 23/23/22 | **36/36/36** | 25/35/52 | 26/35/53 |
| inclusion | 1000 | 16/32/64 | – | 45/45/46 | **144/145/147** | 56/81/136 | 56/85/137 |
| smooth | 10 | 16/32/64 | 180/–/– | 29/29/29 | **35/36/37** | 14/14/13 | 15/15/15 |
| smooth | 1000 | 16/32/64 | – | – | – | 33/42/49 | 33/41/49 |

Five readings:

1. **MG alone is exactly grid-independent** — 36/36/36, 144/145/147, 35/36/37.
   The core claim of the design holds.
2. Against the *unscaled* Green preconditioner, MG costs 1.6× the iterations at
   contrast 10 and 3.2× at contrast 1000.
3. **`J·MG·J` tracks `J·FFT·J` to within 0–4% in every regime.** Since production
   is the J-scaled form, the V-cycle is a faithful drop-in there — the penalty in
   the configuration that matters is a few percent, not the 60% the unscaled
   column suggests.
4. In the hardest regime — smooth data at contrast 1000 — **plain Green fails
   entirely (no convergence in 200 iterations) whether it is applied by FFT or by
   MG, and only the J-scaled form converges at all**, where MG again matches the
   FFT exactly (33/41/49 vs 33/42/49). This is the clearest evidence that the
   V-cycle is interchangeable with the exact transform: the two agree even where
   the thing they approximate is, on its own, useless.
5. Independent of this project: **J-scaling breaks grid-independence on sharp
   interfaces** (25 → 35 → 52) while being indispensable on smooth high-contrast
   data. That matches the stated scope of Ladecký et al. — *smooth* high-contrast
   data — and is worth knowing about the preconditioner already in production.

### 6.6 Coarsest-level size — how to pick `L`

ρ is essentially independent of where the hierarchy stops. 2D `n=256` and 3D
`n=64`, `ν=(2,2)`, derived `ω`:

| coarsest | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| ρ, 2D `n=256` | 0.3710 | 0.3704 | 0.3694 | 0.3664 | 0.3622 | 0.3592 |
| ρ, 3D `n=64` | 0.3639 | 0.3610 | 0.3580 | 0.3528 | — | — |

A 32× change in the coarsest grid moves ρ by 3%. That is textbook behaviour —
the two-grid factor propagates through the hierarchy — and it holds here even
though the bottom solve is *exact* rather than approximate.

The production configuration does not move **at all**. 3D `n=64`, sharp
inclusion, contrast 10:

| coarsest | levels | MG | J·MG·J | (FFT) | (J·FFT·J) |
|---|---|---|---|---|---|
| 8 | 4 | 36 | 53 | 22 | 52 |
| 16 | 3 | 35 | 53 | 22 | 52 |
| 32 | 2 | 35 | 53 | 22 | 52 |

**Consequence:** choose `L` on cost, not convergence. Redundant-level Allreduce
for a `512³` run — `16³` 0.10 MB, `32³` 0.79 MB, `64³` 6.3 MB, `128³` 50 MB — is
negligible below `64³`, so the binding constraint is the *other* end: how few
points per rank the deepest **distributed** level may have before its ghost halo
costs more than its interior. Hence the rule in §2.2. For `512³` on 64 ranks
(4×4×4) that lands on a `32³` coarsest level — five levels, 0.79 MB/rank.

One caveat this serial sweep cannot see: work-normalised efficiency `ρ^(1/work)`
mildly *favours* shallower hierarchies (0.812 at coarsest 32 vs 0.838 at
coarsest 4, in 3D), but that metric charges nothing for the coarse solve or its
Allreduce. Do not read it as an argument for stopping early.

### 6.7 Consequences for the plan

- **Gap list unchanged** except that the smoother needs a setup-time `λ_max`
  power iteration (free: `vecdot` / `norm_sq` already exist) and the constant
  nodal block is a `dim`-vector, not a `dim×dim` matrix, in every production
  configuration.
- **Open question Q1 is resolved**: Q1 needs nothing; only 2D P1 would need the
  full block, and it is not a production target.
- The transfers and the three Stage 2 tests are specified and already pass in
  NumPy, so Stage 2 is a port rather than a design.
- `L` is a parallel-tuning parameter, not a numerical one (§6.6), so it can be
  chosen in Stage 4 from measured per-rank sizes rather than fixed now.
- Nothing here contradicts the design, but §2.8 now says plainly that MG trades
  flops for communication. Stage 0's baseline (3D decomposition vs pencil,
  unpreconditioned) is what will make the Stage 4 numbers interpretable.

### 6.8 Unrelated papercut found on the way

`muGrid.CartesianDecomposition(nb_subdivisions=None)` documents "Default is
automatic" but passes `[0] * nb_dims` to C++, which rejects it with
*"The total number of subdivisions (0) does not match the size of the
communicator (1)"*. Every caller must pass `nb_subdivisions` explicitly. Worth a
small separate fix in `Wrappers.py`.
