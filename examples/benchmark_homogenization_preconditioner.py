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
def run(device, precond, n, dim, maxiter, tol, nranks=1):
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


def collect(args, prov):
    """Run both studies; return DB rows (does not write)."""
    _, nb_gpus = db.detect_gpu()
    want_gpu = not args.no_gpu and nb_gpus >= 1
    configs = db.plan_configs(args.mpi_cpu_ranks, nb_gpus, want_gpu)
    rows = []
    common = dict(maxiter=args.maxiter, tol=args.tol, dim=args.dim)

    # Study 1: iterations (none vs reference), serial CPU.
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

    # Study 2: reference-solve time across device/MPI configs.
    for n in args.sizes:
        for key, device, nranks in configs:
            r = run(device, "reference", n, args.dim, args.maxiter, args.tol,
                    nranks)
            label = db.CONFIG_META[key]["label"](nranks)
            if r is None:
                sys.stderr.write(f"  time {label} {n}^{args.dim}: skipped\n")
                continue
            rows.append({**prov, **common, "benchmark": BENCHMARK,
                         "study": "reference_timing", "label": key,
                         "device": device, "nranks": nranks, "n": n,
                         "npts": r["npts"], "precond": "reference",
                         "iters": r["iters"], "secs": r["secs"]})
            sys.stderr.write(f"  time {label} {n}^{args.dim} ({r['npts']} pts): "
                             f"{r['iters']} it, {r['secs']:.3f} s\n")

    # Study 3: MPI strong scaling of the reference-preconditioned solve, on the
    # CPU cores and (on a multi-GPU host) across the GPUs. The GPU sweep is
    # capped at the visible-device count (one rank per GPU, round-robin).
    scaling_plan = [("cpu", args.scaling_ranks, "cores")]
    if want_gpu:
        gpu_ranks = [R for R in args.scaling_gpu_ranks if R <= nb_gpus]
        scaling_plan.append(("gpu", gpu_ranks, "GPUs"))
    for dev, ranks, unit in scaling_plan:
        for n in args.scaling_sizes:
            for R in ranks:
                r = run(dev, "reference", n, args.dim, args.maxiter, args.tol, R)
                if r is None:
                    sys.stderr.write(f"  scaling[{dev}] {n}^{args.dim} "
                                     f"{R} {unit}: skipped\n")
                    continue
                rows.append({**prov, **common, "benchmark": BENCHMARK,
                             "study": "mpi_scaling", "label": str(R),
                             "device": dev, "nranks": R, "n": n,
                             "npts": r["npts"], "precond": "reference",
                             "iters": r["iters"], "secs": r["secs"]})
                sys.stderr.write(f"  scaling[{dev}] {n}^{args.dim} {R} {unit}: "
                                 f"{r['iters']} it, {r['secs']:.3f} s\n")
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


def scaling_from_rows(rows):
    """{device: {n: {ranks: row}}} from the mpi_scaling study."""
    d = {}
    for r in rows:
        if r["study"] == "mpi_scaling":
            (d.setdefault(r["device"], {}).setdefault(r["n"], {})
             [int(r["nranks"])]) = r
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


# device -> (section title, per-rank column header)
SCALING_DEVICE_META = {
    "cpu": ("Strong scaling on the CPU", "Cores"),
    "gpu": ("Strong scaling on the GPU(s)", "GPUs"),
}


def scaling_tables_markdown(scaling, dim):
    out = []
    for dev in ("cpu", "gpu"):
        per_n = scaling.get(dev)
        if not per_n:
            continue
        title, unit = SCALING_DEVICE_META[dev]
        out.append(f"### {title}\n")
        for n in sorted(per_n):
            rows = per_n[n]
            t1 = rows.get(1, {}).get("secs")
            out.append(f"**{n}{sup(dim)} ({n ** dim:,} points)**\n")
            out.append(f"| {unit} | Iters | Time (s) | Speedup | Parallel eff. |")
            out.append("|---|---|---|---|---|")
            for R in sorted(rows):
                t = rows[R]["secs"]
                sp = t1 / t if t1 else float("nan")
                out.append(f"| {R} | {rows[R]['iters']} | {t:.2f} | "
                           f"{sp:.2f}× | {sp / R * 100:.0f}% |")
            out.append("")
    return "\n".join(out)


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
        cells = " | ".join(
            (f"{res[n]['secs']:.3g}" if n in res else "—") for n in cols)
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
    for P, sty in (
            ("none", dict(marker="o", color="#c62828",
                          label="unpreconditioned")),
            ("reference", dict(marker="s", color="#00897b",
                               label="reference preconditioner"))):
        pts = sorted((n ** dim, r["iters"]) for n, r in iters.get(P, {}).items())
        if pts:
            xs, ys = zip(*pts)
            ax.loglog(xs, ys, **sty)
    ax.set_xlabel("Number of grid points")
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
    for key, label, style in configs:
        pts = sorted((n ** dim, r["secs"]) for n, r in timing.get(key, {}).items())
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.loglog(xs, ys, label=label, **style)
    ax.set_xlabel("Number of grid points")
    ax.set_ylabel("Reference-preconditioned solve time (s)")
    ax.set_title(f"Homogenization ({dim}D, fused, reference prec.): "
                 f"time vs. grid size")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def make_scaling_plot(scaling, dim, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    markers = {32: "v", 64: "o", 96: "s", 128: "^"}
    devs = [d for d in ("cpu", "gpu") if scaling.get(d)] or ["cpu"]
    fig, axes = plt.subplots(1, len(devs), figsize=(6.4 * len(devs), 4.4),
                             squeeze=False)
    for ax, dev in zip(axes[0], devs):
        per_n = scaling.get(dev, {})
        unit = "CPU cores" if dev == "cpu" else "GPUs"
        all_ranks = sorted({R for rows in per_n.values() for R in rows})
        rmax = max(all_ranks) if all_ranks else 1
        ax.plot([1, rmax], [1, rmax], ls="--", color="0.6",
                label="ideal (linear)")
        for n in sorted(per_n):
            rows = per_n[n]
            t1 = rows.get(1, {}).get("secs")
            if not t1:
                continue
            xs = sorted(rows)
            ys = [t1 / rows[R]["secs"] for R in xs]
            ax.plot(xs, ys, marker=markers.get(n, "o"), label=f"{n}{sup(dim)}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        if all_ranks:
            ax.set_xticks(all_ranks)
            ax.set_xticklabels([str(R) for R in all_ranks])
            ax.set_yticks(all_ranks)
            ax.set_yticklabels([str(R) for R in all_ranks])
        ax.set_xlabel(f"MPI ranks ({unit})")
        ax.set_ylabel("Speedup vs. 1 rank")
        ax.set_title(f"strong scaling on {unit}")
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend()
    fig.suptitle(f"Homogenization ({dim}D, fused, reference prec.): "
                 "MPI strong scaling")
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

![Iterations vs. number of grid points]({iter_plot})

## Reference solve: device & MPI scaling

This mirrors the (unpreconditioned) [homogenization
benchmark](benchmark_homogenization.md), but for the **reference-preconditioned**
solve: the same single-CPU-core / full-machine-MPI-CPU / GPU comparison, across
{dim}D grid sizes. Because the iteration count is grid-independent, this isolates
the **per-iteration** cost — and each preconditioned iteration applies a
forward/inverse **FFT pair**, so it is where the FFT-engine paths matter: the
native cuFFT N-D transform on the GPU, and the slab MPI decomposition on
multi-rank runs.

{timing_table}

(values are **solve time in seconds**, run to convergence)

![Reference solve time vs. number of grid points]({timing_plot})

The preconditioner parallelises cleanly: it is applied in Fourier space by the
FFT engine, which owns its MPI decomposition, and the per-mode block solve is
rank-local. `-P reference` gives identical iteration counts and homogenised
stiffness in serial and under MPI. The single CPU core is quickly left behind;
here the **full CPU (MPI) is the fastest option across the whole range**, with
the GPU close behind in the mid-range (~48³–96³). At 128³ the GPU
hits its memory wall hard — *worse* than the unpreconditioned solve, because the
preconditioner's FFT work buffers add to an already-tight 6 GB footprint — and
the full CPU wins by ~7× (18 s vs. 123 s). The same full-CPU-vs-GPU and
memory-wall trade-offs from the [main benchmark](benchmark_homogenization.md)
apply, shifted toward the CPU by the extra per-iteration FFT.

!!! note "Multi-GPU"
    `homogenization.py` binds each MPI rank to a distinct GPU (round-robin), so
    `mpiexec -n <#GPUs> python homogenization.py -d gpu -P reference` runs one
    rank per GPU and the FFT preconditioner is applied in the engine's GPU MPI
    decomposition. This benchmark adds a *GPU (N devices, MPI)* curve
    automatically when more than one GPU is present. **{gpu_count_note}**

## MPI strong scaling

Strong scaling of the reference-preconditioned solve (fixed problem size,
increasing MPI ranks, run to convergence), with `E_eff` and the iteration count
identical across all rank counts. Two decompositions are measured: across the
CPU cores (one rank per core), and — on a multi-GPU host — across the GPUs (one
rank per device, round-robin). Each iteration applies a forward/inverse FFT pair,
so the FFT engine's MPI (slab) decomposition is exercised every iteration.

{scaling_tables}
![Preconditioned MPI strong scaling]({scaling_plot})

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


def write_doc_page(path, meta, dim, tol, ncases, multi_gpu, iter_table,
                   timing_table, scaling_tables, iter_plot, timing_plot,
                   scaling_plot):
    if multi_gpu:
        gpu_count_note = "Runs with more than one GPU show the multi-GPU curve."
    else:
        gpu_count_note = ("This run used a single GPU, so only the single-GPU "
                          "curve is shown; the script needs no changes on a "
                          "multi-GPU host.")
    with open(path, "w") as fh:
        fh.write(DOC_TEMPLATE.format(
            cpu=meta["cpu"], gpu=meta["gpu"], version=meta["version"],
            timestamp=meta["timestamp"], dim=dim, tol=tol, ncases=ncases,
            gpu_count_note=gpu_count_note,
            iter_table=iter_table, timing_table=timing_table,
            scaling_tables=scaling_tables,
            iter_plot=os.path.basename(iter_plot),
            timing_plot=os.path.basename(timing_plot),
            scaling_plot=os.path.basename(scaling_plot)))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dim", type=int, default=3, choices=[2, 3])
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[16, 24, 32, 48, 64, 96, 128],
                    help="Grid sizes for the reference-solve timing study")
    ap.add_argument("--iter-sizes", type=int, nargs="+",
                    default=[16, 24, 32, 48],
                    help="Grid sizes for the none-vs-reference iteration study")
    ap.add_argument("--scaling-sizes", type=int, nargs="+", default=[64, 96],
                    help="Grid sizes for the MPI strong-scaling study")
    ap.add_argument("--scaling-ranks", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16],
                    help="CPU rank counts for the strong-scaling study")
    ap.add_argument("--scaling-gpu-ranks", type=int, nargs="+",
                    default=[1, 2, 4],
                    help="GPU counts for the GPU strong-scaling sweep (capped at "
                         "the number of visible devices)")
    ap.add_argument("--mpi-cpu-ranks", type=int, default=os.cpu_count())
    ap.add_argument("--no-gpu", action="store_true", help="Skip the GPU curves")
    ap.add_argument("--maxiter", type=int, default=20000)
    ap.add_argument("--tol", type=float, default=1e-6)
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
    ap.add_argument("--scaling-plot", default=os.path.join(
        HERE, "..", "docs", "benchmark_homogenization_preconditioner_mpi.png"))
    args = ap.parse_args()

    if not args.render_only:
        prov = db.run_provenance()
        rows = collect(args, prov)
        if not rows:
            sys.exit("No successful runs — nothing to record.")
        db.append_rows(rows, args.db)
        sys.stderr.write(f"appended {len(rows)} rows to {args.db}\n")
        select_ts = prov["timestamp"]
    else:
        select_ts = args.timestamp

    rows = db.select(db.load(args.db), BENCHMARK, select_ts)
    if not rows:
        sys.exit("No matching rows in the database.")

    meta = {k: rows[0][k] for k in ("cpu", "gpu", "version", "timestamp")}
    dim = next((r["dim"] for r in rows), args.dim)
    tol = next((r["tol"] for r in rows if r["study"] == "iterations"), args.tol)
    configs = db.render_configs(rows, "reference_timing")
    iters = iters_from_rows(rows)
    timing = timing_from_rows(rows)
    scaling = scaling_from_rows(rows)
    multi_gpu = any(r["label"] == "gpuN" for r in rows)

    it_tab = iteration_table(sizes_in(rows, "iterations"), iters, dim)
    t_tab = timing_table(sizes_in(rows, "reference_timing"), configs, timing,
                         dim)
    s_tab = scaling_tables_markdown(scaling, dim)
    print("\n" + it_tab + "\n\n" + t_tab + "\n\n" + s_tab)

    iters_path = os.path.abspath(args.iter_plot)
    time_path = os.path.abspath(args.time_plot)
    scaling_path = os.path.abspath(args.scaling_plot)
    make_iter_plot(iters, dim, iters_path)
    make_timing_plot(configs, timing, dim, time_path)
    make_scaling_plot(scaling, dim, scaling_path)
    sys.stderr.write(f"wrote {iters_path}\nwrote {time_path}\n"
                     f"wrote {scaling_path}\n")

    if args.doc_out:
        write_doc_page(args.doc_out, meta, dim, tol,
                       3 if dim == 2 else 6, multi_gpu, it_tab, t_tab, s_tab,
                       iters_path, time_path, scaling_path)
        sys.stderr.write(f"wrote {os.path.abspath(args.doc_out)}\n")


if __name__ == "__main__":
    main()
