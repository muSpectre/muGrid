# Benchmark database

`results.csv` is a small, git-friendly, **append-only** database of benchmark
measurements. Every benchmark run adds a fresh batch of rows stamped with the
date, code version, and machine, so the file is a growing, diffable history that
is meant to be committed and updated regularly.

Page generation is fully separated from data collection: the benchmark scripts
read rows back from this file and render the documentation tables and plots, so a
page can be regenerated at any time — and any historical run reproduced — without
re-measuring.

## Format

One row per measured data point ("long" format), so new studies and series just
add rows (and, if needed, leave unrelated columns blank) rather than reshaping
the file. Columns:

| group | columns | meaning |
|---|---|---|
| provenance | `timestamp`, `version`, `commit`, `dirty`, `cpu`, `gpu` | when/what/where — identical across all rows of one run |
| identity | `benchmark`, `study`, `label` | which plot/series the point belongs to |
| parameters | `device`, `nranks`, `dim`, `n`, `npts`, `precond`, `maxiter`, `tol` | enough to reproduce the run |
| results | `iters`, `secs`, `gbps`, `E` | the measurement |

- `version` is `git describe --tags --always --dirty`; `commit` is the short
  hash; `dirty=1` flags an uncommitted working tree (avoid for runs you intend
  to keep).
- A **run** is one invocation of a benchmark script: all its rows share one
  `timestamp`. Rendering selects the most recent run for a benchmark by default
  (`--timestamp <date-or-prefix>` picks an older one).

Current `benchmark`/`study` values:

| benchmark | study | series (`label`) |
|---|---|---|
| `homogenization` | `time_vs_size` | config key: `cpu1`, `cpuN`, `gpu1`, `gpuN` |
| `homogenization` | `mpi_scaling` | rank count |
| `homogenization_preconditioner` | `iterations` | `none`, `reference` |
| `homogenization_preconditioner` | `reference_timing` | config key |

## Workflow

Measuring (which appends a new dated run to the DB) and rendering the pages are
**separate steps**, and the measurements are split into three independent cases,
one SLURM job each:

| script | case | study appended |
|---|---|---|
| `job_compare_preconditioner.sh` | unpreconditioned vs reference, CG iteration count (1 CPU core) | `homogenization_preconditioner` / `iterations` |
| `job_scaling_unpreconditioned.sh` | unpreconditioned solve time, device/MPI scaling | `homogenization` / `time_vs_size` |
| `job_scaling_preconditioned.sh` | reference-preconditioned solve time, device/MPI scaling | `homogenization_preconditioner` / `reference_timing` |

```bash
# 1. Measure — submit any subset of the three cases (each appends to the DB):
sbatch benchmarks/job_compare_preconditioner.sh
sbatch benchmarks/job_scaling_unpreconditioned.sh
sbatch benchmarks/job_scaling_preconditioned.sh

# 2. Render ALL pages and plots from the DB (no measuring; runs on the head
#    node). The preconditioner page combines the latest run of each of its two
#    studies, so cases 1 and 3 can come from separate jobs:
benchmarks/make_docs.sh

# 3. Commit the updated database and pages:
git add benchmarks/results.csv docs/benchmark_*
```

The job scripts only collect data (they pass `--collect-only` to the drivers,
which can also restrict to a single `--studies` value). To re-render a single
page directly without `make_docs.sh`:

```bash
python examples/benchmark_homogenization.py --render-only \
    --doc-out docs/benchmark_homogenization.md
```

Shared SLURM/toolchain setup for all four scripts lives in `benchmarks/_env.sh`
(sourced, not submitted). The shared Python helpers (schema, provenance capture,
CSV I/O, device/MPI config vocabulary, per-study selection) live in
`examples/benchmark_db.py`; new benchmarks should import it and append rows with
the same schema.
