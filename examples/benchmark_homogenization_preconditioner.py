#!/usr/bin/env python3
"""Preconditioner benchmark for the homogenization example (3D by default).

Two studies on the FEM elasticity [homogenization example](examples.md), both
with the **same fused matvec kernel**, run **to convergence** (relative
tolerance), `-P reference` (Ladecký et al., 2023 Green's-function preconditioner)
vs. `-P none`:

1. **CG iterations vs. grid size** — unpreconditioned vs. reference. Device- and
   decomposition-independent, so it runs on one CPU core. The central result:
   the preconditioner makes the count nearly grid-independent.
2. **Reference-solve time vs. grid size, by device / MPI config** — the *same*
   CPU-1-core / full-machine-MPI-CPU / GPU variation as the (unpreconditioned)
   [homogenization benchmark](benchmark_homogenization.md). The reference
   preconditioner applies a forward/inverse FFT every iteration, so this is where
   the FFT-engine paths show up (native cuFFT/rocFFT N-D transform on the GPU,
   slab MPI decomposition on multi-rank runs). A *GPU (N devices, MPI)* curve is
   added automatically on a multi-GPU host (one rank per GPU, round-robin).

Data collection and page generation are separate: a run executes
`homogenization.py` per data point (under `mpiexec` for the MPI configs) and
**appends** results — with date, code version, and machine — to the shared
benchmark database (`benchmarks/results.csv`, see `examples/benchmark_db.py`).
Tables and plots are rendered *from the database*.

Examples
--------
    python examples/benchmark_homogenization_preconditioner.py \
        --doc-out docs/benchmark_homogenization_preconditioner.md

    # re-render from the latest run in the DB, no recompute:
    python examples/benchmark_homogenization_preconditioner.py --render-only \
        --doc-out docs/benchmark_homogenization_preconditioner.md
"""

import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark_db as db  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HOMOG = os.path.join(HERE, "homogenization.py")
BENCHMARK = "homogenization_preconditioner"


# --------------------------------------------------------------------------- #
# Running homogenization.py
# --------------------------------------------------------------------------- #
# Signatures of an out-of-memory failure (host or device, any vendor / MPI).
_OOM_PATTERNS = re.compile(
    r"out of memory|outofmemory|bad_alloc|MemoryError|cannot allocate memory|"
    r"cudaErrorMemoryAllocation|CUDA_ERROR_OUT_OF_MEMORY|hipErrorOutOfMemory|"
    r"hipErrorMemoryAllocation|failed to allocate", re.IGNORECASE)

# Sentinel returned by run() when a data point ran out of memory.
OOM = "oom"


def _looks_like_oom(out):
    if _OOM_PATTERNS.search(out.stderr) or _OOM_PATTERNS.search(out.stdout):
        return True
    return out.returncode == -9  # killed by the OS OOM killer (SIGKILL)


def run(device, precond, n, dim, maxiter, tol, nranks=1):
    """Run one solve; dict of metrics, ``OOM`` if out of memory, else ``None``."""
    grid = ",".join([str(n)] * dim)
    base = [HOMOG, "-n", grid, "-d", device, "-k", "fused", "-P", precond,
            "-i", str(maxiter), "--tol", str(tol), "--inclusion-type", "single",
            "--json"]
    if nranks == 1:
        cmd = [sys.executable] + base
        env = os.environ
    else:
        cmd = ["mpiexec", "-n", str(nranks), sys.executable] + base
        env = dict(os.environ, OMPI_MCA_rmaps_base_oversubscribe="1")
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=7200,
                             env=env)
    except subprocess.SubprocessError:
        return None
    m = re.search(r"\{.*\}", out.stdout, re.DOTALL)
    if not m:
        if _looks_like_oom(out):
            return OOM
        sys.stderr.write(f"  [{device} {precond} n={n} ranks={nranks}] no JSON\n"
                         f"{out.stderr[-400:]}\n")
        return None
    import json
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    r, c = d["results"], d["config"]
    return dict(npts=c["nb_grid_pts_total"], iters=r["total_cg_iterations"],
                secs=r["total_time_seconds"])


def collect_iterations(args, prov):
    """Study `iterations`: CG count, none vs reference, on a single CPU core."""
    common = dict(maxiter=args.maxiter, tol=args.tol, dim=args.dim)
    rows = []
    for n in args.iter_sizes:
        for P in ("none", "reference"):
            r = run("cpu", P, n, args.dim, args.maxiter, args.tol)
            if r is None:
                sys.stderr.write(f"  iter {P} {n}^{args.dim}: skipped\n")
                continue
            rows.append({**prov, **common, "benchmark": BENCHMARK,
                         "study": "iterations", "label": P, "device": "cpu",
                         "nranks": 1, "n": n, "npts": r["npts"], "precond": P,
                         "iters": r["iters"], "secs": r["secs"]})
            sys.stderr.write(f"  iter {P:9s} {n}^{args.dim} ({r['npts']} pts): "
                             f"{r['iters']:5d} it, {r['secs']:.3f} s\n")
    return rows


def collect_timing(args, prov):
    """Study `reference_timing`: reference-solve time across device/MPI configs.

    Each config sweeps grid sizes up to its own cap (a single CPU core tops out
    far sooner than the full node / the GPUs) and stops at the first
    out-of-memory size, which is recorded as an "oom" point (flagged in the
    table, dropped from the plot); larger sizes for that config are not
    attempted.
    """
    _, nb_gpus = db.detect_gpu()
    want_gpu = not args.no_gpu and nb_gpus >= 1
    configs = db.plan_configs(args.mpi_cpu_ranks, nb_gpus, want_gpu)
    common = dict(maxiter=args.maxiter, tol=args.tol, dim=args.dim)
    rows = []
    for key, device, nranks in configs:
        label = db.CONFIG_META[key]["label"](nranks)
        cap = args.cpu1_max_size if key == "cpu1" else args.max_size
        for n in [s for s in args.sizes if s <= cap]:
            r = run(device, "reference", n, args.dim, args.maxiter, args.tol,
                    nranks)
            base = {**prov, **common, "benchmark": BENCHMARK,
                    "study": "reference_timing", "label": key, "device": device,
                    "nranks": nranks, "n": n, "npts": n ** args.dim,
                    "precond": "reference"}
            if r is None:
                sys.stderr.write(f"  time {label} {n}^{args.dim}: skipped\n")
                continue
            if r is OOM:
                rows.append({**base, "status": "oom"})
                sys.stderr.write(f"  time {label} {n}^{args.dim}: OUT OF "
                                 "MEMORY — stopping this config\n")
                break
            rows.append({**base, "npts": r["npts"], "status": "ok",
                         "iters": r["iters"], "secs": r["secs"]})
            sys.stderr.write(f"  time {label} {n}^{args.dim} ({r['npts']} pts): "
                             f"{r['iters']} it, {r['secs']:.3f} s\n")
    return rows


def collect(args, prov):
    """Run the selected studies (`args.studies`); return DB rows (no write)."""
    rows = []
    if "iterations" in args.studies:
        rows += collect_iterations(args, prov)
    if "reference_timing" in args.studies:
        rows += collect_timing(args, prov)
    return rows


# --------------------------------------------------------------------------- #
# Re-shaping DB rows for rendering
# --------------------------------------------------------------------------- #
def fmt_points(npts):
    if npts >= 1e6:
        return f"{npts / 1e6:.1f}M"
    if npts >= 1e3:
        return f"{npts / 1e3:.0f}k"
    return str(int(npts))


def sup(dim):
    return "²" if dim == 2 else "³"


def iters_from_rows(rows):
    """{precond: {n: row}} from the iterations study."""
    d = {}
    for r in rows:
        if r["study"] == "iterations":
            d.setdefault(r["label"], {})[r["n"]] = r
    return d


def timing_from_rows(rows):
    """{config_key: {n: row}} from the reference_timing study."""
    d = {}
    for r in rows:
        if r["study"] == "reference_timing":
            d.setdefault(r["label"], {})[r["n"]] = r
    return d


def sizes_in(rows, study):
    return sorted({r["n"] for r in rows if r["study"] == study})


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def iteration_table(sizes, iters, dim):
    cols = [n for n in sizes if n in iters.get("none", {})]
    head = "| Preconditioner | " + " | ".join(
        f"{n}{sup(dim)} ({fmt_points(n ** dim)})" for n in cols) + " |"
    lines = [head, "|" + "---|" * (len(cols) + 1)]
    for P in ("none", "reference"):
        cells = " | ".join(
            (str(iters[P][n]["iters"]) if n in iters.get(P, {}) else "—")
            for n in cols)
        lines.append(f"| {P} | {cells} |")
    cells = []
    for n in cols:
        a = iters.get("none", {}).get(n)
        b = iters.get("reference", {}).get(n)
        cells.append(f"{a['secs'] / b['secs']:.0f}×" if a and b else "—")
    lines.append("| **CPU-core wall-time speedup** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _cell(res, n):
    """One table cell: solve time, 'OOM' for an out-of-memory point, else '—'."""
    if n not in res:
        return "—"
    secs = res[n].get("secs")
    if isinstance(secs, (int, float)):
        return f"{secs:.3g}"
    return "OOM" if res[n].get("status") == "oom" else "—"


def timing_table(sizes, configs, timing, dim):
    cols = [n for n in sizes if any(n in timing.get(k, {}) for k, *_ in configs)]
    head = ("| Configuration | "
            + " | ".join(f"{n}{sup(dim)} ({fmt_points(n ** dim)})" for n in cols)
            + " |")
    lines = [head, "|" + "---|" * (len(cols) + 1)]
    for key, label, _style in configs:
        res = timing.get(key)
        if not res:
            continue
        cells = " | ".join(_cell(res, n) for n in cols)
        lines.append(f"| {label} | {cells} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def make_iter_plot(iters, dim, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    all_n = set()
    for P, sty in (
            ("none", dict(marker="o", color="#c62828",
                          label="unpreconditioned")),
            ("reference", dict(marker="s", color="#00897b",
                               label="reference preconditioner"))):
        pts = sorted((n, r["iters"]) for n, r in iters.get(P, {}).items())
        if pts:
            xs, ys = zip(*pts)
            all_n.update(xs)
            ax.loglog(xs, ys, **sty)
    db.set_grid_size_xaxis(ax, all_n, dim)
    ax.set_ylabel("CG iterations to converge")
    ax.set_title(f"Homogenization ({dim}D, fused): iterations vs. grid size")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def make_timing_plot(configs, timing, dim, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    all_n = set()
    for key, label, style in configs:
        # Only points with a real measured time — OOM points carry no time.
        pts = sorted((n, r["secs"]) for n, r in timing.get(key, {}).items()
                     if isinstance(r.get("secs"), (int, float)))
        if not pts:
            continue
        xs, ys = zip(*pts)
        all_n.update(xs)
        ax.loglog(xs, ys, label=label, **style)
    db.set_grid_size_xaxis(ax, all_n, dim)
    ax.set_ylabel("Reference-preconditioned solve time (s)")
    ax.set_title(f"Homogenization ({dim}D, fused, reference prec.): "
                 f"time vs. grid size")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Doc page
# --------------------------------------------------------------------------- #
DOC_TEMPLATE = """# Benchmark: preconditioner

Effect of the reference-material (Green's-function) preconditioner of Ladecký et
al., Appl. Math. Comput. 446 (2023) 127835, on the [homogenization
example](examples.md) — `-P reference` vs. `-P none`. Both use the **same fused
matvec kernel** and are run **to convergence** (relative tolerance `{tol}`).

!!! info "Test machine & code version"
    - **CPU:** {cpu}
    - **GPU:** {gpu}
    - **muGrid:** `{version}` — run {timestamp}

Run configuration: {dim}D, single spherical inclusion, fused stiffness kernel,
{ncases} load cases (iterations summed over all cases).

## CG iterations vs. grid size

The central result: unpreconditioned CG iterations grow with the grid (the
condition number is `O(h⁻²)`), while the reference preconditioner makes the count
**nearly independent of grid size**. The count depends only on the operator and
preconditioner, not the device or the MPI decomposition, so this is measured on a
single CPU core.

{iter_table}

(last row: unpreconditioned ÷ reference **wall time** on one CPU core)

![Iterations vs. grid size]({iter_plot})

## Reference solve: device & MPI scaling

This mirrors the (unpreconditioned) [homogenization
benchmark](benchmark_homogenization.md), but for the **reference-preconditioned**
solve: the same single-CPU-core / full-machine-MPI-CPU / single-GPU / multi-GPU
comparison, across {dim}D grid sizes. Because the iteration count is
grid-independent, this isolates the **per-iteration** cost — and each
preconditioned iteration applies a forward/inverse **FFT pair**, so it is where
the FFT-engine paths matter: the native N-D transform on the GPU, and the slab
MPI decomposition on multi-rank runs.

Each configuration is swept to the largest grid that still fits in memory: the
first size that runs **out of memory** is recorded as `OOM` in the table and
dropped from the plot, and larger sizes for that configuration are not attempted.

{timing_table}

(values are **solve time in seconds**, run to convergence; `OOM` = ran out of
memory)

![Reference solve time vs. grid size]({timing_plot})

The preconditioner parallelises cleanly: it is applied in Fourier space by the
FFT engine, which owns its MPI decomposition, and the per-mode block solve is
rank-local. `-P reference` gives identical iteration counts and homogenised
stiffness in serial and under MPI, so the single CPU core is quickly left behind
and the largest grids are reached by MPI domain decomposition across all CPU
cores or across several GPUs (one rank per device, round-robin).

All data points live in the shared benchmark database `benchmarks/results.csv`
(date, code version, machine, parameters, results). This page is generated by
`examples/benchmark_homogenization_preconditioner.py`; re-render from the
database with `--render-only`, or run a fresh measurement that appends a new
dated row set:

```bash
python examples/benchmark_homogenization_preconditioner.py \\
    --doc-out docs/benchmark_homogenization_preconditioner.md
```
"""


def write_doc_page(path, meta, dim, tol, ncases, iter_table, timing_table,
                   iter_plot, timing_plot):
    with open(path, "w") as fh:
        fh.write(DOC_TEMPLATE.format(
            cpu=meta["cpu"], gpu=meta["gpu"], version=meta["version"],
            timestamp=meta["timestamp"], dim=dim, tol=tol, ncases=ncases,
            iter_table=iter_table, timing_table=timing_table,
            iter_plot=os.path.basename(iter_plot),
            timing_plot=os.path.basename(timing_plot)))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dim", type=int, default=3, choices=[2, 3])
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512,
                             768, 1024, 1536, 2048],
                    help="Grid sizes for the reference-solve timing study (each "
                         "config stops at its cap or the first out-of-memory size)")
    ap.add_argument("--max-size", type=int, default=2048,
                    help="Largest grid size attempted for the full-CPU and GPU "
                         "configs")
    ap.add_argument("--cpu1-max-size", type=int, default=128,
                    help="Largest grid size attempted for the single-CPU-core "
                         "config")
    ap.add_argument("--iter-sizes", type=int, nargs="+",
                    default=[16, 24, 32, 48],
                    help="Grid sizes for the none-vs-reference iteration study")
    ap.add_argument("--mpi-cpu-ranks", type=int, default=os.cpu_count())
    ap.add_argument("--no-gpu", action="store_true", help="Skip the GPU curves")
    ap.add_argument("--maxiter", type=int, default=20000)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--studies", nargs="+",
                    choices=["iterations", "reference_timing"],
                    default=["iterations", "reference_timing"],
                    help="Which studies to MEASURE (collection only). "
                         "`iterations` = none-vs-reference CG count (1 CPU "
                         "core); `reference_timing` = device/MPI solve-time "
                         "scaling. Rendering always uses whatever is in the DB.")
    ap.add_argument("--collect-only", action="store_true",
                    help="Measure and append to the database, then stop "
                         "(no plots/page — render later with --render-only)")
    ap.add_argument("--render-only", action="store_true",
                    help="Skip running; render from the database")
    ap.add_argument("--timestamp", default=None,
                    help="Render this run (timestamp prefix / date)")
    ap.add_argument("--db", default=db.DB_PATH, help="Benchmark CSV path")
    ap.add_argument("--doc-out", default=None)
    ap.add_argument("--iter-plot", default=os.path.join(
        HERE, "..", "docs", "benchmark_homogenization_preconditioner_iters.png"))
    ap.add_argument("--time-plot", default=os.path.join(
        HERE, "..", "docs", "benchmark_homogenization_preconditioner_time.png"))
    args = ap.parse_args()

    if not args.render_only:
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

    # Take the latest run of EACH study independently, so the page combines the
    # iteration-count study and the timing study even when they were measured by
    # separate jobs (the two are split across separate submission scripts).
    rows = db.select_studies(db.load(args.db), BENCHMARK,
                             ["iterations", "reference_timing"], select_ts)
    if not rows:
        sys.exit("No matching rows in the database.")

    # Machine/version box: prefer the (heavier) timing run when present.
    meta_rows = [r for r in rows if r["study"] == "reference_timing"] or rows
    meta = {k: meta_rows[0][k] for k in ("cpu", "gpu", "version", "timestamp")}
    dim = next((r["dim"] for r in rows), args.dim)
    tol = next((r["tol"] for r in rows if r["study"] == "iterations"), args.tol)
    configs = db.render_configs(rows, "reference_timing")
    iters = iters_from_rows(rows)
    timing = timing_from_rows(rows)

    it_tab = iteration_table(sizes_in(rows, "iterations"), iters, dim)
    t_tab = timing_table(sizes_in(rows, "reference_timing"), configs, timing,
                         dim)
    print("\n" + it_tab + "\n\n" + t_tab)

    iters_path = os.path.abspath(args.iter_plot)
    time_path = os.path.abspath(args.time_plot)
    make_iter_plot(iters, dim, iters_path)
    make_timing_plot(configs, timing, dim, time_path)
    sys.stderr.write(f"wrote {iters_path}\nwrote {time_path}\n")

    if args.doc_out:
        write_doc_page(args.doc_out, meta, dim, tol,
                       3 if dim == 2 else 6, it_tab, t_tab,
                       iters_path, time_path)
        sys.stderr.write(f"wrote {os.path.abspath(args.doc_out)}\n")


if __name__ == "__main__":
    main()
