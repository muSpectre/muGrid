# Analytic reference symbol — implementation plan

Target: stop storing the reference-material (Green) preconditioner's inverse
symbol. A uniform operator is a `3^dim` stencil — **243 numbers in 3D** — so the
symbol `K(q) = Σ_d S[d] exp(-2πi q·d)` can be rebuilt per mode, inverted and
applied in registers. The stored symbol is `n²` complex values per Fourier mode:
**2.26 GB at 512³** for three components in single precision, growing with the
grid, and it is the largest single allocation in a large single-precision
topology-optimization run.

**Status:** host and device paths complete and measured (H200, CUDA 13.3).
Evaluated is the **default on a device** and stored stays the default on the
host; see [§0.6](#06-result-of-the-05-measurement). Not yet done: a HIP compile
and run (the source is written for both, but only nvcc has seen it), and an MPI
run of the device path.

**If you are picking this up cold, read [§0 Start here](#0-start-here) first.**
Sections 1–4 are the design, the measurements and the traps behind it.

---

## 0. Start here

### 0.1 Current state

Branch `feat/analytic-reference-symbol`, branched from `release/1.4.0`
(PR #208). Two commits, all 35 ctest green including MPI np = 1, 2, 4, 8:

- `65b23856` — assemble the symbol from its stencil instead of by impulse
  response. `reference_stencil` is public and is the single definition of the
  uniform reference operator; `stencil_symbol` generalises `_z_coupling_blocks`
  (the hybrid preconditioner already did this analytically) to either phase
  every axis or leave one explicit.
- `384f5e61` — `AnalyticReferencePreconditioner` plus the host kernel
  `linalg/green_symbol.{hh,cc}`, bound as `linalg.apply_green_symbol_{2,3}d[_f32]`.
  Stores the stencil, nothing per mode.

Measured, 128³, three components, float32:

| | build peak | total peak | apply |
|---|---|---|---|
| stored symbol | 603.8 MB | 802.6 MB | 59.0 ms |
| evaluated per mode | **51.3 MB** | **258.1 MB** | 75.3 ms |

**This is a memory optimisation, not a speed one.** Rebuilding costs ~630
flops/mode against ~36 bytes read, so it needs ~17.6 flops/byte to break even:
an A100 balances at 12.6, an H100 at 17.9, a CPU socket nearer 5. On CPU the
apply is 28% slower, and that is only this good because the apply is
FFT-dominated. Do not expect the GPU kernel to be faster than reading the
stored symbol — expect it to be comparable and to use no memory.

### 0.2 Where the code lives

| what | where |
|---|---|
| host kernel | `src/libmugrid/linalg/green_symbol.{hh,cc}` |
| **device kernel (to write)** | `src/libmugrid/linalg/green_symbol_gpu.cc` |
| bindings | `language_bindings/python/bind_py_linalg.cc`, `bind_green_symbol` |
| Python re-export | `language_bindings/python/muGrid/linalg.py` |
| preconditioner + stencil + factories | `language_bindings/python/muGrid/Preconditioners.py` |
| tests | `tests/python_preconditioner_tests.py` |

Model everything on `linalg/block_thomas.{hh,cc}` + `block_thomas_gpu.cc` and
its bindings — same namespace shape, same host/device split, same `Dim` as a
compile-time parameter, same "buffers not fields" convention (a Fourier array
has a mode axis, which no field collection describes).

### 0.3 Build and test — read before running anything

```bash
source ../benchenv.sh            # venv + local libmpi; run from the workspace root
cd muGrid
cmake -S . -B build-gpu -DMUGRID_ENABLE_CUDA=ON   # or -DMUGRID_ENABLE_HIP=ON
cmake --build build-gpu -j8
cd build-gpu && ctest --output-on-failure
```

Traps, all of which have already cost time in this repo:

- **The editable install copies the Python layer.** Editing
  `language_bindings/python/muGrid/*.py` has *no effect* until `pip install -e .`
  is re-run, and the `.pth` finder beats `PYTHONPATH`, so pointing that at the
  build tree does not help either. C++ edits need the same reinstall.
- **Do not use `pip install -e ".[test]"`.** The `test` extra pulls
  `pytest-flake8`, which pins `flake8<5` and breaks `pytest` for every project in
  the shared venv. Use `pip install -e .`.
- **Lint through `pre-commit run --files ...`**, which has its own isolated
  environment.
- A new `*_gpu.cc` must go in **both** the `MUGRID_ENABLE_CUDA` and
  `MUGRID_ENABLE_HIP` branches of `src/libmugrid/CMakeLists.txt` *and* in the
  matching `set_source_files_properties(... LANGUAGE CUDA/HIP)` list. Forgetting
  the second compiles it as plain C++ and fails obscurely.
- Guard `__device__` code on `__CUDACC__` / `__HIPCC__`, **not** on
  `MUGRID_ENABLE_CUDA`. The feature macro says the build has GPU support; it says
  nothing about which compiler is reading the header, and most translation units
  that include a GPU header are compiled by the host compiler. This exact mistake
  broke `cuda-compile` and `hip-compile` earlier; see `memory/gpu_runtime.hh`,
  where `global_thread_{x,y,z}()` and `grid_stride_x()` live behind that guard.
  Use those helpers rather than open-coding `blockIdx.x * blockDim.x`, which
  evaluates in 32-bit and wraps.

### 0.4 What to do next, in order

**1. Write `green_symbol_gpu.cc`.** Mirror `apply_inverse` from
`green_symbol.cc`. The signature is already declared in `green_symbol.hh` as
`apply_inverse_gpu`, behind the `MUGRID_ENABLE_CUDA || MUGRID_ENABLE_HIP` guard.

The parallelisation is **not** a transcription of the host loop, and this is the
one real design decision left — see [§2](#2-the-gpu-parallelisation-decision).

**2. Bind it.** Follow `bind_block_thomas_gpu` in `bind_py_linalg.cc`: the device
binding takes raw device addresses as integers (`array.data.ptr`), because CuPy
arrays are not muGrid fields and nothing in the buffer protocol describes device
memory. Re-export in `linalg.py` next to the `block_thomas_gpu_*` loop, which is
already `hasattr`-guarded for non-GPU builds.

**3. Make `AnalyticReferencePreconditioner` device-aware.** It currently assumes
a host buffer:

```python
flat = np.asarray(self._work.s).ravel(order="F")
```

On device, `self._work.s` is a CuPy view. Take the device pointer and call the
GPU binding. The stencil and the per-axis frequency tables are tiny — copy them
to the device once in `__init__`, not per apply. The stencil is 243 floats and
belongs in `__constant__` memory.

**4. Select by device in the factories.** In
`make_reference_stiffness_preconditioner`, the analytic route currently always
assembles and hands the symbol to `BlockFourierPreconditioner`. Make it return
`AnalyticReferencePreconditioner` when the fields live on a device, and keep the
stored symbol on host until §0.5's measurement says otherwise.
`make_green_jacobi_preconditioner` already forwards `element` / `grid_spacing`,
so muTopOpt's default `green-jacobi` path inherits it.

**5. Measure, then choose the host default.** See §0.5.

### 0.5 The measurement that decides the default (done — see §0.6)

Per-apply time and peak device allocation, evaluated vs stored, at 256³ and
512³, three components, float32 and float64. The host numbers are in §0.1; what
is unknown is where the GPU lands relative to its 12.6–17.9 flops/byte balance.

If the GPU apply is within ~30% of the stored symbol, make evaluated the default
on device — 2.26 GB at 512³ is worth that. If it is much worse, the hoisting in
§2 is probably not working; check that before concluding anything.

### 0.6 Result of the §0.5 measurement

H200, three components, one process per row, from
`benchmarks/bench_green_symbol.py N dtype {stored,evaluated}`. *Symbol* = apply − bare
fft+ifft on the same work field; *memory* = device memory held after setup and
20 applies (muGrid fields plus the CuPy pool's high-water mark).

| grid | dtype | mode | setup | memory | apply | symbol |
|---|---|---|---|---|---|---|
| 256³ | f32 | stored | 21.0 s | 1028 MB | 1.79 ms | 0.94 ms |
| 256³ | f32 | evaluated | 0.17 s | 417 MB | **1.18 ms** | 0.33 ms |
| 256³ | f64 | stored | 20.7 s | 1432 MB | 2.62 ms | 1.07 ms |
| 256³ | f64 | evaluated | 0.17 s | 822 MB | **2.05 ms** | 0.50 ms |
| 512³ | f32 | stored | 168.5 s | 8110 MB | 25.5 ms | 18.6 ms |
| 512³ | f32 | evaluated | 1.3 s | 3244 MB | **9.3 ms** | 2.3 ms |
| 512³ | f64 | stored | 167.5 s | 11329 MB | 31.2 ms | 18.7 ms |
| 512³ | f64 | evaluated | 1.3 s | 6476 MB | **16.1 ms** | 3.6 ms |

**Evaluated is faster, not merely comparable**: 2.7× (f32) and 1.9× (f64) per
apply at 512³, 4.9 GB less device memory, and no 3-minute host assembly. The
stored path's multiply is far from its own bandwidth floor — it runs
component by component through CuPy — so §0.1's flops/byte argument
understated the win on a device. The default is therefore evaluated on device,
through `evaluate_symbol=None` in `make_reference_stiffness_preconditioner`
(forwarded by `make_green_jacobi_preconditioner`); pass `False` to get the
stored symbol back.

**The hoist is working** (no profiler counters on the node, so by ablation, 512³
kernel alone): with the shared-memory hoist forced off, the f64 kernel goes
from 3.67 to 8.94 ms; f32 from 2.34 to 2.82 ms. A pure read+write pass over the
same buffer takes 0.96 / 1.51 ms, so the kernel sits at ~2.4× the bandwidth
floor. The f32 kernel is not arithmetic-bound, so there is headroom there,
worth at most ~1.4 ms of a 9.3 ms apply. A Hermitian-only build (6 of 9
entries) would help f64 more than f32.

Implementation notes that differ from §0.4's sketch:

- The stencil is a **kernel argument by value**, not a `__constant__` array.
  Kernel parameters live in the constant bank, so the access is the same, but
  a module-global `__constant__` symbol would be shared by every preconditioner
  in the process and need a memcpy per apply.
- A block covers 256 consecutive *modes*, not one line: at 512³ lines are 257
  modes long, which a line-per-block layout fills poorly. The block builds `P`
  for the ≤ 8 lines it touches; when axis 0 is so short that it touches more,
  each thread builds its own `P` (tested by the `(2, 64)` and `(3, 16, 16)`
  cases).
- Strides are read off the work field's view, not assumed: host AoS gives
  (1, Dim), device SoA gives (nb_modes, 1), both verified on the H200 build.

Tests: `test_evaluated_symbol_on_device_matches_stored_on_host`,
`test_reference_factory_chooses_storage_by_device`. ctest on the H200 no-MPI
build: 16/16.

---

## 1. What is already proven

Do not re-derive these; they are measured and have tests.

- **The analytic symbol equals the assembled one.** 4.9e-16 in 3D, 1.7e-16 in
  2D, comparing the raw symbol; 1.0e-15 / 4.0e-15 comparing the applied
  preconditioner at float64, ~1e-06 at float32. Test:
  `test_analytic_reference_symbol_matches_impulse_assembly`.
- **The stencil matches the C++ operator by construction.** `reference_stencil`
  *probes* the real operator on an 8^dim serial grid rather than re-deriving
  `B(q)` from the element tables, so there is no second expression of the same
  maths to drift out of sync. This is why the analytic route was much less risky
  than it first looked.
- **The symbol is Hermitian by construction.** The stencil of a self-adjoint
  operator satisfies `S[d] = S[-d]ᵀ` — measured *exactly* 0.0 — so the symbol is
  Hermitian to 8.7e-17. Only the upper triangle is needed.
- **The host kernel is correct**, including under MPI. Test:
  `test_evaluated_symbol_matches_stored_symbol`.

## 2. The GPU parallelisation decision

The host kernel hoists the sums over the slower axes out of the fastest-axis
loop: every mode along axis 0 shares `q1` and `q2`, so
`P[d0] = Σ_{d1,d2} phase · S[d0,d1,d2]` is loop-invariant there.

**This is not an optimisation, it is load-bearing.** It takes the symbol build
from `3^dim · Dim²` complex FMAs per mode to `3 · Dim²` — 1944 flops/mode to
216. Without it the kernel is several times slower than reading the stored
symbol on any hardware, and the whole trade collapses.

The tension: one thread per mode is what coalescing wants, but each thread would
then recompute `P[d0]` for itself, losing the hoist. The suggested structure is a
**block per line along axis 0**:

- a thread block covers a chunk of consecutive `i0` at fixed `(i1, i2)`;
- the block cooperatively computes `P[d0]` — three `Dim × Dim` complex blocks —
  once into shared memory;
- each thread then does `3 · Dim²` complex FMAs, the `Dim × Dim` cofactor
  inverse and the matvec, all in registers.

Consecutive threads touch consecutive modes, so the field access stays
coalesced: the Fourier buffer is Fortran-ordered with **axis 0 fastest**, and
components are the fastest index of all (see §3).

This is a hypothesis, not a measurement. If it underperforms, the fallback is one
thread per mode with the naive build, which is simpler and still uses no memory —
just check it against §0.5 before settling.

## 3. Traps specific to this kernel

Both of these are silent when wrong — they produce a plausible-looking result.

- **Mode ordering is axis 0 fastest.** The kernel decomposes the flat mode index
  against `nb_fourier_grid_pts` with axis 0 varying fastest, matching the Fourier
  field's Fortran-ordered buffer. A C-order reading of the same buffer gives a
  wrong answer that looks entirely reasonable. On the host side the buffer is
  handed over as `np.asarray(field.s).ravel(order="F")`, which is a *view*, so
  nothing is copied; `stride_component = 1`, `stride_mode = Dim`. On device the
  storage order is structure-of-arrays, so those strides become `nb_modes` and
  `1` — the kernel already takes both as parameters, but **check the actual
  layout rather than assuming**.
- **muGrid halves axis 0, numpy halves the last.** `engine.fftfreq` follows
  muGrid's convention (`nb_fourier_subdomain_grid_pts` is `(n/2+1, n, n)`), while
  the hybrid preconditioner transforms its rank-local axes with
  `numpy.fft.rfftn` semantics. `stencil_symbol` therefore takes the frequencies
  from the caller rather than constructing them — do not "helpfully" centralise
  that.
- **A rank can own no Fourier modes at all**, when there are more ranks than
  planes along the split axis. It still takes part in the transforms, which are
  collective. This was found by the np = 8 run, not by inspection; the guard is
  in `AnalyticReferencePreconditioner.__init__` and `apply`.
- **Frequencies are per axis, not per mode.** A `(Dim, nb_modes)` frequency array
  is 809 MB at 512³ in single precision — most of what not storing the symbol
  saves. Keep the per-axis tables.

## 4. Sequencing and dependencies

- This branch sits on `release/1.4.0`; **PR #208 should land first.**
- **PR #209** (pinning the GPU compile jobs back to `ubuntu-24.04`) must land
  before any GPU code here can be compile-checked in CI. Both `cuda-compile` and
  `hip-compile` were broken on `main` by the Ubuntu 26.04 runner bump — NVIDIA
  publishes no CUDA 12.6 for 26.04, and the ROCm apt line serves `jammy`
  packages.
- muTopOpt needs no change: both its preconditioner paths go through
  `make_green_jacobi_preconditioner` / `make_reference_stiffness_preconditioner`.
