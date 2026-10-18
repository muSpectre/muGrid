#!/usr/bin/env python3
"""Scaling benchmark for the µGrid Poisson solver example.

Solves the 3D Poisson equation (`poisson.py`) with the unpreconditioned
conjugate-gradient solver across log-spaced grid sizes on a single CPU core and
(if available) a single GPU, and records the solve wall time. It regenerates the
[Benchmark](../docs/benchmark.md) page: a Markdown table plus a log-log plot of
solve time vs. grid size.

This driver shares the benchmark infrastructure with the homogenization
benchmarks (`examples/benchmark_db.py`): the same append-only CSV database
(`benchmarks/results.csv`), the same provenance capture, the same per-point
result **cache** (so a killed run resumes cheaply — see `benchmark_db.cached_point`),
and the same device/MPI config vocabulary and axis helpers. Data collection and
page rendering are separate steps, exactly as for the homogenization pages.

Each (device, size) point runs `poisson.py` as its own subprocess and reads the
machine-readable result back from a private `--json-out` file, so the runtimes are
unaffected by this driver's own imports and stray stdout cannot corrupt the parse.

Examples
--------
    # measure, append to the DB, and (re)generate the page:
    python examples/benchmark.py --doc-out docs/benchmark.md \
        --plot-out docs/benchmark.png

    # re-render from the latest run already in the DB (no recompute):
    python examples/benchmark.py --render-only --doc-out docs/benchmark.md
"""

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark_db as db  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
POISSON = os.path.join(HERE, "poisson.py")
BENCHMARK = "poisson"
DIM = 3
CONFIG_META = db.CONFIG_META


# --------------------------------------------------------------------------- #
# Running poisson.py
# --------------------------------------------------------------------------- #
_OOM_PATTERNS = re.compile(
    r"out of memory|outofmemory|bad_alloc|MemoryError|cannot allocate memory|"
    r"cudaErrorMemoryAllocation|CUDA_ERROR_OUT_OF_MEMORY|hipErrorOutOfMemory|"
    r"hipErrorMemoryAllocation|failed to allocate", re.IGNORECASE)

OOM = db.OOM


def _looks_like_oom(out):
    if _OOM_PATTERNS.search(out.stderr) or _OOM_PATTERNS.search(out.stdout):
        return True
    return out.returncode == -9  # killed by the OS OOM killer (SIGKILL)


def run(device, n, maxiter):
    """Run one Poisson solve; metrics dict, ``db.OOM``, or ``None`` on failure."""
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "result.json")
        cmd = [sys.executable, POISSON, "-n", f"{n},{n},{n}", "-d", device,
               "-i", str(maxiter), "--json-out", out_path]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=3600)
        except subprocess.SubprocessError:
            return None
        try:
            with open(out_path) as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            if _looks_like_oom(out):
                return db.OOM
            sys.stderr.write(f"  [{device} n={n}] no JSON\n{out.stderr[-400:]}\n")
            return None
    r, c = d["results"], d["config"]
    return dict(npts=c["nb_grid_pts_total"], iters=r["iterations"],
                secs=r["total_time_seconds"],
                gbps=r.get("memory_throughput_GBps"),
                converged=r.get("converged", True))


def _configs(want_gpu):
    """Device configs for the Poisson page: a single CPU core, and (if present)
    a single GPU — the historical CPU-vs-GPU comparison, in the shared vocab."""
    cfgs = [("cpu1", "cpu", 1)]
    if want_gpu:
        cfgs.append(("gpu1", "gpu", 1))
    return cfgs


def _point_fields(key, device, n, args):
    return {"benchmark": BENCHMARK, "study": "time_vs_size", "label": key,
            "device": device, "nranks": 1, "dim": DIM, "n": n,
            "precond": "none", "maxiter": args.maxiter}


def _resolve(key, device, n, prov, args):
    fields = _point_fields(key, device, n, args)
    result, cached = db.cached_point(
        fields, prov, lambda: run(device, n, args.maxiter),
        cache_dir=args.cache_dir, force=args.force)
    return n, result, cached


def collect(args, prov):
    """Run all data points and return DB rows (each point cached as it lands)."""
    _, nb_gpus = db.detect_gpu()
    want_gpu = not args.no_gpu and nb_gpus >= 1
    rows = []
    for key, device, nranks in _configs(want_gpu):
        label = CONFIG_META[key]["label"](nranks)
        cap = args.cpu1_max_size if key == "cpu1" else args.max_size
        sizes = [s for s in args.sizes if s <= cap]
        # The single CPU core is safe to parallelise (one core per point); the
        # single GPU sweeps serially (one device).
        if key == "cpu1" and args.jobs > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
                points = sorted(
                    ex.map(lambda n: _resolve(key, device, n, prov, args), sizes),
                    key=lambda t: t[0])
        else:
            points = []
            for n in sizes:
                point = _resolve(key, device, n, prov, args)
                points.append(point)
                if point[1] == OOM:
                    break
        for n, r, cached in points:
            base = {**prov, "benchmark": BENCHMARK, "study": "time_vs_size",
                    "label": key, "device": device, "nranks": 1, "dim": DIM,
                    "n": n, "npts": n ** DIM, "precond": "none",
                    "maxiter": args.maxiter}
            tag = " [cached]" if cached else ""
            if r is None:
                sys.stderr.write(f"  {label} {n}^3: skipped (run failed)\n")
                continue
            if r == OOM:
                rows.append({**base, "status": "oom"})
                sys.stderr.write(f"  {label} {n}^3: OUT OF MEMORY{tag}\n")
                continue
            conv = "" if r.get("converged", True) else "  [did NOT converge]"
            rows.append({**base, "npts": r["npts"], "status": "ok",
                         "iters": r["iters"], "secs": r["secs"],
                         "gbps": r["gbps"]})
            sys.stderr.write(f"  {label} {n}^3 ({r['npts']} pts): "
                             f"{r['secs']:.3f} s, {r['iters']} it{conv}{tag}\n")
    return rows


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def fmt_points(npts):
    if npts >= 1e6:
        return f"{npts / 1e6:.1f}M"
    if npts >= 1e3:
        return f"{npts / 1e3:.0f}k"
    return str(int(npts))


def merged_from_rows(rows):
    """{config_key: {n: row}} for the time_vs_size study."""
    d = {}
    for r in rows:
        if r["study"] == "time_vs_size":
            d.setdefault(r["label"], {})[r["n"]] = r
    return d


def sizes_in(rows):
    return sorted({r["n"] for r in rows if r["study"] == "time_vs_size"})


def _cell(res, n):
    if n not in res:
        return "—"
    secs = res[n].get("secs")
    if isinstance(secs, (int, float)):
        return f"{secs:.3g}"
    return "OOM" if res[n].get("status") == "oom" else "—"


def table_markdown(sizes, configs, merged):
    cols = [n for n in sizes if any(n in merged.get(k, {}) for k, *_ in configs)]
    header = ("| Configuration | "
              + " | ".join(f"{n}³ ({fmt_points(n ** 3)})" for n in cols) + " |")
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [header, sep]
    for key, label, _style in configs:
        res = merged.get(key)
        if not res:
            continue
        cells = " | ".join(_cell(res, n) for n in cols)
        lines.append(f"| {label} | {cells} |")
    return "\n".join(lines)


def make_plot(configs, merged, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    all_n = set()
    for key, label, style in configs:
        pts = sorted((n, merged[key][n]["secs"]) for n in merged.get(key, {})
                     if isinstance(merged[key][n].get("secs"), (int, float)))
        if not pts:
            continue
        xs, ys = zip(*pts)
        all_n.update(xs)
        ax.loglog(xs, ys, label=label, **style)
    db.set_grid_size_xaxis(ax, all_n, DIM)
    ax.set_ylabel("Solve time (s)")
    ax.set_title("Poisson CG solve: time vs. grid size")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


DOC_TEMPLATE = """# Benchmark: Poisson

## What is measured

Wall time of the unpreconditioned conjugate-gradient solve of the
[Poisson example](examples.md#poisson-equation) (`examples/poisson.py`) on 3D
grids of $N^3$ points, as a function of $N$, on one CPU core and on one GPU.
Problem: $-\\nabla^2 u = f$ on the periodic unit cube with the smooth
zero-mean right-hand side of the example, the 7-point `LaplaceOperator`, no
preconditioner, relative residual $10^{{-6}}$. The reported time is the
example's `total_time_seconds` (the CG solve, excluding setup). One
measurement per point.

The per-iteration work is one 7-point stencil apply and a few BLAS-1
operations, $O(N^3)$ with a small, constant number of floating-point
operations per value loaded, so the time per iteration is bounded by memory
bandwidth. Without preconditioner $\\kappa = O(h^{{-2}})$ and CG needs $O(N)$
iterations, so the solve time grows as $O(N^4)$.

Configurations:

- **CPU (1 core)** — one process without MPI. µGrid's kernels are not
  multithreaded, so this uses one core.
- **GPU (1 device)** — one process on one GPU.

## Provenance

{provenance}

## Commands

Each data point is one invocation of the example, launched by
`examples/benchmark.py` as

```bash
{commands}
```

with `N` the grid size of the column and `--json-out FILE` added to collect the
result.{commands_note}

## Results

{table}

(values are **solve time in seconds**; `OOM` = the run ran out of memory)

![Poisson CG solve time vs. grid size]({plot_name})

## Reproducing
{db_note}
Data points are stored in `benchmarks/results.csv`. This page is generated by
`examples/benchmark.py`; `--render-only` re-renders it from the database, and
a run without it measures and appends a new dated run (see
`benchmarks/README.md`):

```bash
python examples/benchmark.py --doc-out docs/benchmark.md \\
    --plot-out docs/benchmark.png
```
"""


def _commands(rows, configs):
    """One command line per device configuration present in `rows`."""
    lines = []
    note = ""
    for key, label, _style in configs:
        r = next(r for r in rows
                 if r["study"] == "time_vs_size" and r["label"] == key)
        maxiter = r.get("maxiter")
        if not maxiter:
            note = ("\n\n`MAXITER` was not recorded for this run. "
                    + db.todo_metadata("maximum CG iterations (-i)"))
        lines.append(f"# {label}")
        lines.append(db.config_command(
            "poisson.py", f"-n N,N,N -d {{device}} -i {maxiter or 'MAXITER'}",
            r["device"], int(r.get("nranks") or 1)))
    return "\n".join(lines), note


def write_doc_page(path, plot_path, table, meta, rows=(), configs=()):
    commands, note = _commands(rows, configs)
    with open(path, "w") as fh:
        fh.write(DOC_TEMPLATE.format(
            table=table,
            provenance=db.provenance_markdown(
                rows, {"time_vs_size": "time vs. grid size"}),
            commands=commands, commands_note=note,
            db_note=("" if all(r.get("timestamp") for r in rows) else
                     "\nThe results shown are not stored in "
                     "`benchmarks/results.csv`. "
                     + db.todo_metadata("database rows for these results")
                     + "\n"),
            plot_name=os.path.basename(plot_path)))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[32, 48, 64, 96, 128, 192, 256, 384, 512],
                    help="Per-axis grid sizes n (the grid is n x n x n)")
    ap.add_argument("--max-size", type=int, default=512,
                    help="Largest grid size attempted (per config)")
    ap.add_argument("--cpu1-max-size", type=int, default=512,
                    help="Largest grid size attempted for the single CPU core")
    ap.add_argument("--no-gpu", action="store_true", help="Skip the GPU curve")
    ap.add_argument("--maxiter", type=int, default=5000)
    ap.add_argument("--jobs", type=int, default=1,
                    help="Parallel worker processes for the single-CPU-core "
                         "sweep (each point uses one core); the GPU sweeps "
                         "serially (default: 1)")
    ap.add_argument("--cache-dir", default=db.CACHE_DIR,
                    help="Per-point result cache directory. Finished points are "
                         "replayed from here, so a killed run resumes cheaply")
    ap.add_argument("--force", action="store_true",
                    help="Recompute even on a cache hit (and refresh the cache)")
    ap.add_argument("--collect-only", action="store_true",
                    help="Measure and append to the database, then stop "
                         "(render later with --render-only)")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="Do not run: append a fresh run to the database from the "
                         "points already cached for the current commit")
    ap.add_argument("--render-only", action="store_true",
                    help="Skip running; render from the database")
    ap.add_argument("--timestamp", default=None,
                    help="Render this run (timestamp prefix / date) instead of "
                         "the latest")
    ap.add_argument("--db", default=db.DB_PATH, help="Benchmark CSV path")
    ap.add_argument("--doc-out", default=None,
                    help="Write the Markdown benchmark page here")
    ap.add_argument("--plot-out",
                    default=os.path.join(HERE, "..", "docs", "benchmark.png"),
                    help="Write the log-log plot image here")
    args = ap.parse_args()

    if args.aggregate_only:
        prov = db.run_provenance()
        rows = db.cache_rows(BENCHMARK, ["time_vs_size"], prov, args.cache_dir)
        if not rows:
            sys.exit("Nothing cached for the current commit — nothing to record.")
        db.append_rows(rows, args.db)
        sys.stderr.write(f"aggregated {len(rows)} cached points into {args.db}\n")
        return
    elif not args.render_only:
        prov = db.run_provenance()
        rows = collect(args, prov)
        if not rows:
            sys.exit("No successful runs — nothing to record.")
        db.append_rows(rows, args.db)
        sys.stderr.write(f"appended {len(rows)} rows to {args.db}\n")
        if args.collect_only:
            return
        select_ts = prov["timestamp"]
    else:
        select_ts = args.timestamp

    rows = db.select(db.load(args.db), BENCHMARK, select_ts)
    if not rows:
        sys.exit("No matching rows in the database.")

    meta_row = max(rows, key=lambda r: r["timestamp"])
    meta = {k: meta_row[k] for k in ("cpu", "gpu", "version", "timestamp")}
    configs = db.render_configs(rows, "time_vs_size")
    merged = merged_from_rows(rows)

    table = table_markdown(sizes_in(rows), configs, merged)
    print("\n" + table)

    plot_out = os.path.abspath(args.plot_out)
    make_plot(configs, merged, plot_out)
    sys.stderr.write(f"wrote {plot_out}\n")

    if args.doc_out:
        write_doc_page(args.doc_out, plot_out, table, meta, rows, configs)
        sys.stderr.write(f"wrote {os.path.abspath(args.doc_out)}\n")


if __name__ == "__main__":
    main()
