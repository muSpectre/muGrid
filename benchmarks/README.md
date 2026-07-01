# Benchmark database

`results.csv` is a small, git-friendly, **append-only** database of benchmark
measurements. Every benchmark run adds a fresh batch of rows stamped with the
date, code version, and machine, so the file is a growing, diffable history that
is meant to be committed and updated regularly.

Page generation is fully separated from data collection: the benchmark scripts
read rows back from this file and render the documentation tables and plots, so a
page can be regenerated at any time — and any historical run reproduced — without
re-measuring.

Underneath the CSV sits a per-point **result cache** (`benchmarks/cache/`,
git-ignored) that makes long collection jobs resumable — see
[Result cache & resumability](#result-cache--resumability). The CSV stays the
committed, human-facing aggregate; the cache is a regenerable work ledger.

## Format

One row per measured data point ("long" format), so new studies and series just
add rows (and, if needed, leave unrelated columns blank) rather than reshaping
the file. Columns:

| group | columns | meaning |
|---|---|---|
| provenance | `timestamp`, `version`, `commit`, `dirty`, `cpu`, `gpu` | when/what/where — identical across all rows of one run |
| identity | `benchmark`, `study`, `label` | which plot/series the point belongs to |
| parameters | `device`, `nranks`, `dim`, `n`, `npts`, `precond`, `precision`, `maxiter`, `tol` | enough to reproduce the run |
| results | `iters`, `secs`, `gbps`, `E` | the measurement |

- `version` is `git describe --tags --always --dirty`; `commit` is the short
  hash; `dirty=1` flags an uncommitted working tree (avoid for runs you intend
  to keep).
- `precision` is `double` (fp64) or `single` (fp32). Legacy rows predate this
  column and are blank; they are read as `double`. Each benchmark is run once
  per precision (separate jobs, hence separate `timestamp`s); the pages overlay
  whatever precisions are present — latest run *per precision* — with fp64 drawn
  solid and fp32 dashed.
- A **run** is one invocation of a benchmark script: all its rows share one
  `timestamp`. Rendering selects the most recent run for a benchmark by default
  (`--timestamp <date-or-prefix>` picks an older one).

Current `benchmark`/`study` values:

| benchmark | study | series (`label`) |
|---|---|---|
| `poisson` | `time_vs_size` | config key: `cpu1`, `gpu1` |
| `homogenization` | `time_vs_size` | config key: `cpu1`, `cpuN`, `gpu1`, `gpuN` |
| `homogenization_preconditioner` | `iterations` | `none`, `reference` |
| `homogenization_preconditioner` | `reference_timing` | config key |

## Workflow

Measuring (which appends a new dated run to the DB) and rendering the pages are
**separate steps**. The measurements are split into independent cases, one SLURM
job script each. The three homogenization cases each have a **double- (`_double`,
fp64) and a single-precision (`_single`, fp32) variant** that differ only in the
`--precision` they pass; both append to the same DB and the pages overlay the two
precisions. The Poisson case is a single (double-precision) script:

| case | script(s) | study appended |
|---|---|---|
| Poisson CG solve time, CPU core vs GPU | `job_poisson.sh` | `poisson` / `time_vs_size` |
| unpreconditioned vs reference, CG iteration count (1 CPU core) | `job_compare_preconditioner_{double,single}.sh` | `homogenization_preconditioner` / `iterations` |
| unpreconditioned solve time, device/MPI scaling | `job_scaling_unpreconditioned_{double,single}.sh` | `homogenization` / `time_vs_size` |
| reference-preconditioned solve time, device/MPI scaling | `job_scaling_preconditioned_{double,single}.sh` | `homogenization_preconditioner` / `reference_timing` |

```bash
# 1. Measure — submit any subset (each appends to the DB). Run both precisions
#    of a homogenization case to get both curves on the page; either alone
#    renders fine too:
sbatch benchmarks/job_poisson.sh
sbatch benchmarks/job_compare_preconditioner_double.sh
sbatch benchmarks/job_compare_preconditioner_single.sh
sbatch benchmarks/job_scaling_unpreconditioned_double.sh
sbatch benchmarks/job_scaling_unpreconditioned_single.sh
sbatch benchmarks/job_scaling_preconditioned_double.sh
sbatch benchmarks/job_scaling_preconditioned_single.sh

# 2. Render ALL pages and plots from the DB (no measuring; runs on the head
#    node). Each page combines the latest run of each of its studies AND each
#    precision, so the double/single jobs of a case can run separately:
benchmarks/make_docs.sh

# 3. Commit the updated database and pages:
git add benchmarks/results.csv docs/benchmark*.md docs/benchmark*.png
```

The job scripts only collect data (they pass `--collect-only`, `--precision`, and
`--jobs`; the drivers can also restrict to a single `--studies` value). To
re-render a single page directly without `make_docs.sh`:

```bash
python examples/benchmark_homogenization.py --render-only \
    --doc-out docs/benchmark_homogenization.md
```

Shared SLURM/toolchain setup for all scripts lives in `benchmarks/_env.sh`
(sourced, not submitted). The shared Python helpers (schema, provenance capture,
CSV I/O, the per-point cache, device/MPI config vocabulary, per-study/precision
selection) live in `examples/benchmark_db.py`; new benchmarks should import it and
append rows with the same schema.

## Result cache & resumability

Every data point (one benchmark/study/config/size/precision/… combination) is an
independent measurement of one code version, reproducible up to timing noise. As
each point finishes, its result is written to its own JSON file under
`benchmarks/cache/`, keyed by the **commit** plus the point's identity. This buys
three things:

- **Resumability.** If a collection job hits its wall-clock limit or is killed,
  just **resubmit it**: the finished points replay from the cache in seconds and
  the sweep continues where it stopped. (The examples also write their JSON to a
  private `--json-out` file rather than stdout, so a point either lands in the
  cache intact or not at all.)
- **Cheap re-runs / partial sweeps.** Re-running a script recomputes nothing that
  is already cached at the current commit. `--force` recomputes and refreshes the
  cache anyway.
- **Safe fan-out.** Independent points can be measured concurrently — each writes
  its own file, so there are no shared-file races. `--jobs N` uses this to run the
  single-CPU-core sweep N points at a time (every point is one core); the MPI and
  GPU configs, which already saturate the node or a whole device per point, stay
  serial.

The cache is keyed by commit, so a measurement is only ever reused for the code
that produced it; bump the commit and the points recompute. **Dirty working trees
are never cached** (the commit alone would not identify the code) — so only
benchmark committed code, which you want for a keepable run anyway.

If a sweep is so large it cannot finish even one full pass within a single job's
wall clock, run the collection job(s) as many times as needed (each resubmit
extends the cache), then snapshot the cache into the DB without running anything:

```bash
python examples/benchmark_homogenization.py --aggregate-only   # append one run
benchmarks/make_docs.sh                                         # then render
```

The cache is regenerable and **git-ignored**; only `results.csv` (the aggregate)
and the rendered pages are committed.
