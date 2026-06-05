# µGrid — Critical Code-Coverage Analysis

This document reports the results of the first systematic code-coverage analysis
of µGrid, and describes the tooling that was added so the analysis can be
repeated and tracked over time.

> **How to reproduce.** See `doc/source/Coverage.rst`. In short:
> ```
> cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DMUGRID_ENABLE_COVERAGE=ON
> cmake --build build
> ctest --test-dir build
> cmake --build build --target coverage
> ```

## 1. Measurement configuration & caveats

The numbers below were produced with:

- Compiler: GCC 13.3, `CMAKE_BUILD_TYPE=Debug`, `-O0 --coverage`
- `MUGRID_ENABLE_MPI=OFF`, GPU backends **off**, NetCDF **on**, Python **on**
- Test runners: the C++ Boost.Test suites (`ctest`) **and** the full `pytest`
  Python binding suite (211 passed, 53 skipped)

**Two caveats are essential when reading the per-file numbers:**

1. **Serial build.** MPI was disabled, so all MPI-only code reports as
   unexecuted. This is a property of the *configuration*, not of the test
   suite — those paths are covered only by the `mpi_*` runners under `mpiexec`.
   The clearest example is `fft/transpose.cc` (0%), which is the MPI pencil
   transpose and contains no serial code path.
2. **CPU build.** GPU backends were off, so `memory/device.cc` and the `*_gpu`
   translation units are absent or unexercised.

A complete picture therefore also requires an MPI + GPU coverage run; the
`Coverage` CI workflow currently runs the serial+CPU configuration, which is the
floor, not the ceiling, of achievable coverage.

## 2. Headline numbers

C++ library (`src/libmugrid`, serial/CPU configuration):

| Metric    | Covered / Total   | Coverage |
|-----------|-------------------|----------|
| Lines     | 11 739 / 20 510   | **57.2 %** |
| Functions | 2 883 / 3 893     | **74.1 %** |
| Branches  | 6 261 / 15 944    | **39.3 %** |

Python wrapper layer (`language_bindings/python/muGrid`):

| Module        | Coverage | Note |
|---------------|----------|------|
| Solvers.py    | 96 % | |
| linalg.py     | 90 % | |
| Field.py      | 84 % | |
| __init__.py   | 78 % | |
| Wrappers.py   | 76 % | |
| Parallel.py   | 38 % | MPI-dependent; low only in serial run |
| **Total**     | **71 %** | |

The low branch coverage (39 %) is the most actionable headline: more than half
of all conditional branches in the library are never exercised in both
directions. Much of this is untested error handling (`throw` branches were
*excluded* from the count, so this is genuine logic, not exception plumbing).

## 3. Dead / dangling code (high confidence)

These symbols are compiled into the library but have **no callers anywhere** in
`src/`, `language_bindings/`, `tests/` or `examples/`. They should be removed or,
if they are intended public API for downstream consumers (e.g. muSpectre),
documented and given at least one test.

### 3.1 `fft/fft_backend_factory.cc` — superseded factory functions (0 %)

```cpp
std::unique_ptr<FFT1DBackend> get_host_fft_backend();    // never called
std::unique_ptr<FFT1DBackend> get_device_fft_backend();  // never called
```

The FFT engine constructs its backend exclusively through the templated
`create_fft_backend<MemorySpace>()` /
`FFTBackendSelector<MemorySpace>::create()` trait in `fft_backend_traits.hh`
(`fft_engine.hh:92`). The free factory functions in this `.cc` are a leftover
from the pre-trait design and are entirely unreachable. The whole translation
unit can almost certainly be deleted (it and its declarations in
`fft_1d_backend.hh`).

### 3.2 `grid/index_ops.cc` — unused dynamic overloads (0 %)

```cpp
Real CcoordOps::compute_pixel_volume(const DynGridIndex &, const DynCoord<...> &);
Dim_t CcoordOps::get_index(const DynGridIndex &, const DynGridIndex &,
                           const DynGridIndex &);
```

`compute_pixel_volume` has **no callers at all**. The dynamic `get_index`
overload is never executed: every caller (tests, `bind_py_common_mugrid.cc`)
resolves to the compile-time `get_index<Dim>` template in `index_ops.hh`. Verify
against downstream usage, then remove or test.

## 4. Untested paths (covered code that no test reaches)

Unlike §3, the code below *is* reachable and likely used in production, but the
test suite never drives it.

### 4.1 Laplace operators — bound to Python but never tested (~3 %)

`operators/laplace_2d.cc` (3 %) and `operators/laplace_3d.cc` (3 %) implement
`LaplaceOperator2D` / `LaplaceOperator3D`, which **are exported to Python**
(`bind_py_operators.cc`, `Wrappers.py::LaplaceOperator`) and advertised in the
README as a headline feature. Yet there is no `python_laplace_*_tests.py` and no
C++ test instantiates them — only the constructor registration is touched.
Sibling operators (convolution, FEM gradient, isotropic stiffness) all have
dedicated test files; Laplace is the gap. **This is the single highest-value
test to add** because it is a shipped, documented, public operator with
essentially zero coverage.

### 4.2 NetCDF type descriptors / I/O details

- `io/type_descriptor_netcdf.cc` — 22 %
- `io/file_io_netcdf.hh` — 31 % (large header, many type-dispatch branches)
- `io/file_io_netcdf.cc` — 77 % (better, but the error/edge branches are open)

The serial NetCDF reader/writer is exercised, but type-dispatch and error paths
for less common dtypes are not.

### 4.3 Core utilities with thin coverage

- `core/type_descriptor.cc` — 19 % (dtype name/size lookup table — cheap to test
  exhaustively)
- `core/enums.cc` — 0 %: the `operator<<` for `IterUnit` and `StorageOrder`.
  These **are** used (e.g. `field_typed.cc:158`, `field_map.cc:57`) but only on
  error/diagnostic paths the tests never hit. A trivial round-trip test would
  cover them.
- `core/exception.cc` — 72 %: the stack-trace formatting branches are untested.
- `grid/strides.cc` — 32 %, `grid/pixels.cc` — 56 %.

### 4.4 Field internals (template-heavy — read with care)

`field/field_typed.cc` (13 %), `field/field_map.cc` (21 %),
`field/state_field_map.cc` (15 %) and `field/field_map_static.hh` (64 %) report
low line coverage, but a large fraction of the "missing" lines are
*uninstantiated template specializations* rather than untested logic — gcov
attributes every potential instantiation to a line. The real signal here is at
the **branch** level and in which concrete `T`/`Dim`/`IterUnit` combinations the
tests instantiate. Treat these percentages as a prompt to audit *which template
parameter combinations* are tested, not as 80 % dead code.

### 4.5 Branch coverage hotspots

With overall branch coverage at 39 %, the recommended next step is a
branch-focused report (`gcovr --html-details` already emits per-line branch
data) over `field_collection.cc` (62 % line), `units.cc` (63 % line),
`generic.cc` (55 % line) and `fft_engine.hh` (61 % line) — these combine high
statement count with many untaken branches.

## 5. What was added to the repository

| File | Purpose |
|------|---------|
| `cmake/Coverage.cmake` | `MUGRID_ENABLE_COVERAGE` option, the `mugrid::coverage` interface target, and the `coverage[-xml/-html/-summary]` report targets (gcovr). |
| `CMakeLists.txt` | Includes the module and prints coverage status in the config summary. |
| `src/libmugrid/CMakeLists.txt` | Links the library (and, transitively, tests + bindings) against `mugrid::coverage`; exports the interface target. |
| `.github/workflows/coverage.yml` | CI job: instrumented build → tests → C++ & Python reports → artifacts + best-effort Codecov upload. |
| `doc/source/Coverage.rst` | User documentation. |
| `.gitignore` | Ignores `*.gcda/*.gcno/*.gcov`, `coverage_html/`, `coverage*.xml`, `.coverage`. |

## 6. Prioritised recommendations

1. **Remove the dead code in §3** (or cover it if it is intended public API) —
   `fft_backend_factory.cc` and the unused `index_ops.cc` overloads.
2. **Add a Laplace operator test** (§4.1) — highest-value gap: a shipped,
   documented operator at ~3 % coverage.
3. **Add an MPI + GPU coverage CI run** so MPI-only files (`transpose.cc`,
   `cartesian_*`) and GPU files stop reading as false 0 %.
4. **Cheap wins**: exhaustively test `type_descriptor.cc` and the `enums.cc`
   stream operators (§4.3) — small files, large percentage gains.
5. **Drive branch coverage up** in the four hotspots in §4.5, focusing on the
   untested error/edge branches that dominate the 39 % branch figure.
6. **Track coverage in PRs** via the new workflow; consider a Codecov status
   check once a baseline is established.
