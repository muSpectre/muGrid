#!/usr/bin/env python3
"""Scaling benchmark for the homogenization example (3D, fused kernel).

Generates the [homogenization benchmark](../docs/benchmark_homogenization.md)
page, which has two parts:

1. **Time vs. grid size** — a single merged plot comparing, across log-spaced 3D
   grid sizes, a *single CPU core*, the *full machine via MPI* (one rank per
   logical core), and a *single GPU*. On a multi-GPU machine a *multi-GPU MPI*
   curve is added automatically (one rank per GPU); this machine has one GPU, so
   that curve is skipped here.
2. **MPI strong scaling** — speedup vs. rank count at fixed problem size.

Data collection and page generation are separate. A run executes
`homogenization.py` as a subprocess for each data point (under `mpiexec` for the
MPI configurations) and **appends** the results — with date, code version, and
machine — to the shared benchmark database (`benchmarks/results.csv`, see
`examples/benchmark_db.py`). Tables and plots are then rendered *from the
database*, so the page can be regenerated at any time, and historical runs stay
reproducible.

Examples
--------
    # run benchmarks, append to the DB, and (re)generate the page:
    python examples/benchmark_homogenization.py \
        --doc-out docs/benchmark_homogenization.md \
        --plot-out docs/benchmark_homogenization.png

    # just re-render the page from the latest run already in the DB:
    python examples/benchmark_homogenization.py --render-only \
        --doc-out docs/benchmark_homogenization.md

Needs an MPI-enabled muGrid build and `mpi4py` for the MPI configurations; point
`PYTHONPATH` at an MPI (and, for the GPU curves, GPU-enabled) build tree.
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
BENCHMARK = "homogenization"
CONFIG_META = db.CONFIG_META


# --------------------------------------------------------------------------- #
# Running homogenization.py
# --------------------------------------------------------------------------- #
def run(device, n, maxiter, nranks=1):
    """Run one homogenization solve; return a dict of metrics or None."""
    base = [HOMOG, "-n", f"{n},{n},{n}", "-d", device, "-k", "fused",
            "-i", str(maxiter), "--inclusion-type", "single", "--json"]
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
        sys.stderr.write(f"  [{device} n={n} ranks={nranks}] no JSON\n"
                         f"{out.stderr[-400:]}\n")
        return None
    import json
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    r, c = d["results"], d["config"]
    return dict(npts=c["nb_grid_pts_total"], iters=r["total_cg_iterations"],
                secs=r["total_time_seconds"],
                gbps=r.get("memory_throughput_GBps"))


def collect(args, prov):
    """Run all data points and return DB rows (does not write)."""
    _, nb_gpus = db.detect_gpu()
    want_gpu = not args.no_gpu and nb_gpus >= 1
    configs = db.plan_configs(args.mpi_cpu_ranks, nb_gpus, want_gpu)
    rows = []

    # Time vs. size, one curve per device/MPI config.
    for n in args.sizes:
        for key, device, nranks in configs:
            r = run(device, n, args.maxiter, nranks)
            label = CONFIG_META[key]["label"](nranks)
            if r is None:
                sys.stderr.write(f"  {label} {n}^3: skipped (failed / OOM)\n")
                continue
            rows.append({**prov, "benchmark": BENCHMARK, "study": "time_vs_size",
                         "label": key, "device": device, "nranks": nranks,
                         "dim": 3, "n": n, "npts": r["npts"],
                         "maxiter": args.maxiter, "iters": r["iters"],
                         "secs": r["secs"], "gbps": r["gbps"]})
            sys.stderr.write(f"  {label} {n}^3 ({r['npts']} pts): "
                             f"{r['secs']:.3f} s, {r['iters']} it, "
                             f"{r['gbps']:.1f} GB/s\n")

    # MPI strong scaling, on CPU cores and (when present) GPUs. The GPU sweep is
    # capped at the number of visible devices — homogenization.py round-robins
    # one rank per GPU, so more ranks than GPUs would oversubscribe.
    scaling_plan = [("cpu", args.scaling_ranks, "cores")]
    if want_gpu:
        gpu_ranks = [R for R in args.scaling_gpu_ranks if R <= nb_gpus]
        scaling_plan.append(("gpu", gpu_ranks, "GPUs"))
    for dev, ranks, unit in scaling_plan:
        for n in args.scaling_sizes:
            for R in ranks:
                r = run(dev, n, args.maxiter, R)
                if r is None:
                    continue
                rows.append({**prov, "benchmark": BENCHMARK,
                             "study": "mpi_scaling", "label": str(R),
                             "device": dev, "nranks": R, "dim": 3, "n": n,
                             "npts": r["npts"], "maxiter": args.maxiter,
                             "iters": r["iters"], "secs": r["secs"],
                             "gbps": r["gbps"]})
                sys.stderr.write(f"  scaling[{dev}] {n}^3 {R} {unit}: "
                                 f"{r['secs']:.3f} s, {r['gbps']:.1f} GB/s\n")
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


def merged_from_rows(rows):
    """{config_key: {n: {secs, iters, gbps, npts}}}."""
    d = {}
    for r in rows:
        if r["study"] != "time_vs_size":
            continue
        d.setdefault(r["label"], {})[r["n"]] = r
    return d


def scaling_from_rows(rows):
    """{device: {n: {ranks: row}}}."""
    d = {}
    for r in rows:
        if r["study"] != "mpi_scaling":
            continue
        d.setdefault(r["device"], {}).setdefault(r["n"], {})[int(r["nranks"])] = r
    return d


def sizes_in(rows, study):
    return sorted({r["n"] for r in rows if r["study"] == study})


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
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
        cells = " | ".join(
            (f"{res[n]['secs']:.3g}" if n in res else "—") for n in cols)
        lines.append(f"| {label} | {cells} |")
    return "\n".join(lines)


# device -> (section title, per-rank column header)
SCALING_DEVICE_META = {
    "cpu": ("Strong scaling on the CPU", "Cores"),
    "gpu": ("Strong scaling on the GPU(s)", "GPUs"),
}


def scaling_tables_markdown(scaling):
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
            out.append(f"**{n}³ ({n ** 3:,} points)**\n")
            out.append(f"| {unit} | Time (s) | Speedup | Parallel eff. | "
                       "Agg. GB/s |")
            out.append("|---|---|---|---|---|")
            for R in sorted(rows):
                t = rows[R]["secs"]
                sp = t1 / t if t1 else float("nan")
                gbps = rows[R].get("gbps")
                gbps_s = f"{gbps:.1f}" if isinstance(gbps, (int, float)) else "—"
                out.append(f"| {R} | {t:.2f} | {sp:.2f}× | {sp / R * 100:.0f}% | "
                           f"{gbps_s} |")
            out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def make_merged_plot(configs, merged, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for key, label, style in configs:
        pts = sorted((n ** 3, merged[key][n]["secs"]) for n in merged.get(key, {}))
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.loglog(xs, ys, label=label, **style)
    ax.set_xlabel("Number of grid points")
    ax.set_ylabel("Solve time (s)")
    ax.set_title("Homogenization (3D, fused): time vs. grid size")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def make_scaling_plot(scaling, path):
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
            ax.plot(xs, ys, marker=markers.get(n, "o"), label=f"{n}³")
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
    fig.suptitle("Homogenization (3D, fused): MPI strong scaling")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Doc page
# --------------------------------------------------------------------------- #
DOC_TEMPLATE = """# Benchmark: homogenization

Wall time of the FEM elasticity [homogenization example](examples.md)
(`examples/homogenization.py`, fused stiffness kernel), across log-spaced **3D**
grid sizes. Lower is better.

!!! info "Test machine & code version"
    - **CPU:** {cpu}
    - **GPU:** {gpu}
    - **muGrid:** `{version}` — run {timestamp}

Run configuration: 3D single spherical inclusion, fused stiffness kernel,
6 load cases, fixed `{maxiter}` CG iterations per load case — i.e. a **fixed work
budget** so every configuration performs identical arithmetic. Times are the
solver wall time (`total_time_seconds`, excluding setup).

## Time vs. grid size

The plot below merges three ways of running the *same* solve on this machine:

- **CPU (1 core)** — a single core, MPI disabled. muGrid's compute kernels carry
  no OpenMP, so a non-MPI CPU run uses exactly one core.
- **CPU ({ncores} cores, MPI)** — the whole CPU via MPI domain decomposition
  (`mpiexec -n {ncores}`), the grid split into per-rank subdomains that exchange
  ghost layers each iteration.
- **GPU (1 device)** — the whole GPU.
{gpu_mpi_bullet}
{table}

(values are **solve time in seconds**)

![Homogenization solve time vs. number of grid points]({plot_name})

Three regimes are visible. At tiny grids everything is overhead-bound and the
three are within a factor of two. In the **mid-range** (here ~32³–64³) the
GPU dominates: the heavy per-point FEM stiffness kernel keeps it busy, the
working set fits in device memory, and it beats even the full CPU. At the **high
end** the picture flips — a single CPU core is hopeless, but the **whole CPU**
(all {ncores} cores via MPI) overtakes the GPU once the problem outgrows GPU
memory (see below). The fair comparison is full-CPU-vs-GPU, not one-core-vs-GPU:
against all {ncores} cores the GPU's advantage is modest where it leads and
reverses where it does not.

!!! warning "GPU memory wall at large grids"
    A 128³ run needs about 5.85 GB of field storage, which nearly fills this
    GPU's 6 GB. As the working set approaches that limit the allocator
    oversubscribes to host memory and effective throughput collapses — already
    visible at 96³ (throughput drops below the 48³–64³ peak, and the GPU loses to
    the full CPU) and severe at 128³. On this 6 GB card the GPU's fast path tops
    out around 64³; larger 3D grids need a bigger-memory GPU or MPI domain
    decomposition across several GPUs (see the multi-GPU note below).

!!! note "Multi-GPU"
    `homogenization.py` binds each MPI rank to a distinct GPU (round-robin over
    the visible devices), so `mpiexec -n <#GPUs> python homogenization.py -d gpu`
    runs one rank per GPU. This benchmark adds a *GPU (N devices, MPI)* curve
    automatically when more than one GPU is present. **{gpu_count_note}**

## MPI strong scaling

Strong scaling of the same 3D fused solve (fixed problem size, increasing MPI
ranks), with `E_eff` identical across all rank counts. The grid is split into
per-rank subdomains that exchange ghost layers each iteration. Two decompositions
are measured: across the {ncores}-core CPU (one rank per core), and — on a
multi-GPU host — across the GPUs (one rank per device, round-robin).

{scaling_tables}
![Homogenization MPI strong scaling]({scaling_plot_name})

The solve is memory-bandwidth-bound, so aggregate throughput keeps climbing as
ranks are added but parallel efficiency falls once per-rank subdomains get small
and ghost exchange plus CG dot-product reductions start to dominate. On the GPU
side, the per-device working set must stay large enough to hide the
inter-device communication, so the larger grids scale best.

All data points live in the shared benchmark database `benchmarks/results.csv`
(date, code version, machine, parameters, results). This page is generated by
`examples/benchmark_homogenization.py`; re-render it from the database (no
recompute) with `--render-only`, or run a fresh measurement that appends a new
dated row set:

```bash
python examples/benchmark_homogenization.py \\
    --doc-out docs/benchmark_homogenization.md \\
    --plot-out docs/benchmark_homogenization.png
```
"""


def write_doc_page(path, plot_path, scaling_plot_path, table, scaling_tables,
                   meta, ncores, multi_gpu):
    if multi_gpu:
        gpu_mpi_bullet = "- **GPU (N devices, MPI)** — all GPUs, one rank per " \
                         "device.\n"
        gpu_count_note = "Runs with more than one GPU show the multi-GPU curve."
    else:
        gpu_mpi_bullet = ""
        gpu_count_note = ("This run used a single GPU, so only the single-GPU "
                          "curve is shown; the script is ready to produce the "
                          "multi-GPU curve on a multi-GPU host with no changes.")
    with open(path, "w") as fh:
        fh.write(DOC_TEMPLATE.format(
            cpu=meta["cpu"], gpu=meta["gpu"], version=meta["version"],
            timestamp=meta["timestamp"], table=table, maxiter=meta["maxiter"],
            ncores=ncores, gpu_mpi_bullet=gpu_mpi_bullet,
            gpu_count_note=gpu_count_note,
            plot_name=os.path.basename(plot_path),
            scaling_tables=scaling_tables,
            scaling_plot_name=os.path.basename(scaling_plot_path)))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[16, 24, 32, 48, 64, 96, 128])
    ap.add_argument("--mpi-cpu-ranks", type=int, default=os.cpu_count(),
                    help="Ranks for the full-machine MPI CPU curve")
    ap.add_argument("--no-gpu", action="store_true", help="Skip the GPU curves")
    ap.add_argument("--scaling-sizes", type=int, nargs="+", default=[64, 96])
    ap.add_argument("--scaling-ranks", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16])
    ap.add_argument("--scaling-gpu-ranks", type=int, nargs="+",
                    default=[1, 2, 4],
                    help="GPU counts for the GPU strong-scaling sweep (capped at "
                         "the number of visible devices)")
    ap.add_argument("--maxiter", type=int, default=100)
    ap.add_argument("--render-only", action="store_true",
                    help="Skip running; render from the database")
    ap.add_argument("--timestamp", default=None,
                    help="Render this run (timestamp prefix / date) instead of "
                         "the latest")
    ap.add_argument("--db", default=db.DB_PATH, help="Benchmark CSV path")
    ap.add_argument("--doc-out", default=None)
    ap.add_argument("--plot-out",
                    default=os.path.join(HERE, "..", "docs",
                                         "benchmark_homogenization.png"))
    ap.add_argument("--scaling-plot-out",
                    default=os.path.join(HERE, "..", "docs",
                                         "benchmark_homogenization_mpi.png"))
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
    meta["maxiter"] = next((r["maxiter"] for r in rows
                            if r["study"] == "time_vs_size"), args.maxiter)
    configs = db.render_configs(rows, "time_vs_size")
    merged = merged_from_rows(rows)
    scaling = scaling_from_rows(rows)
    ncores = next((r["nranks"] for r in rows if r["label"] == "cpuN"),
                  args.mpi_cpu_ranks)
    multi_gpu = any(r["label"] == "gpuN" for r in rows)

    sizes = sizes_in(rows, "time_vs_size")
    table = table_markdown(sizes, configs, merged)
    scaling_tables = scaling_tables_markdown(scaling)
    print("\n" + table + "\n\n" + scaling_tables)

    plot_out = os.path.abspath(args.plot_out)
    scaling_plot_out = os.path.abspath(args.scaling_plot_out)
    make_merged_plot(configs, merged, plot_out)
    make_scaling_plot(scaling, scaling_plot_out)
    sys.stderr.write(f"wrote {plot_out}\nwrote {scaling_plot_out}\n")

    if args.doc_out:
        write_doc_page(args.doc_out, plot_out, scaling_plot_out, table,
                       scaling_tables, meta, ncores, multi_gpu)
        sys.stderr.write(f"wrote {os.path.abspath(args.doc_out)}\n")


if __name__ == "__main__":
    main()
