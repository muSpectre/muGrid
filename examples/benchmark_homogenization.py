#!/usr/bin/env python3
"""Scaling benchmark for the homogenization example (3D, fused kernel).

Generates the [homogenization benchmark](../docs/benchmark_homogenization.md)
page: **time vs. grid size** — a single merged plot comparing, across log-spaced
3D grid sizes, a *single CPU core*, the *full machine via MPI* (one rank per
logical core), and a *single GPU*. On a multi-GPU machine a *multi-GPU MPI* curve
is added automatically (one rank per GPU). Each configuration is swept to the
largest grid that fits in memory; the first size that runs out of memory is
flagged `OOM` in the table and omitted from the plot, and larger sizes for that
configuration are skipped.

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
HOMOG = os.path.join(HERE, "homogenization.py")
BENCHMARK = "homogenization"
CONFIG_META = db.CONFIG_META


# --------------------------------------------------------------------------- #
# Running homogenization.py
# --------------------------------------------------------------------------- #
# Signatures of an out-of-memory failure (host or device, any vendor / MPI).
_OOM_PATTERNS = re.compile(
    r"out of memory|outofmemory|bad_alloc|MemoryError|cannot allocate memory|"
    r"cudaErrorMemoryAllocation|CUDA_ERROR_OUT_OF_MEMORY|hipErrorOutOfMemory|"
    r"hipErrorMemoryAllocation|failed to allocate", re.IGNORECASE)

# Sentinel returned by run() when a data point ran out of memory (shared with the
# cache, which stores the OOM outcome so a resubmit does not re-attempt it).
OOM = db.OOM


def _looks_like_oom(out):
    """True if a finished subprocess looks like it ran out of memory."""
    if _OOM_PATTERNS.search(out.stderr) or _OOM_PATTERNS.search(out.stdout):
        return True
    # Killed by the OS OOM killer (SIGKILL) with no JSON produced.
    return out.returncode == -9


def run(device, n, maxiter, nranks=1, precision="double"):
    """Run one homogenization solve.

    Returns a dict of metrics on success, the sentinel ``db.OOM`` if the run ran
    out of memory, or ``None`` for any other failure. The child writes its JSON to
    a private file (``--json-out``) that we read back, so stray stdout (MPI/UCX
    banners, warnings) can never corrupt the parse.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "result.json")
        base = [HOMOG, "-n", f"{n},{n},{n}", "-d", device, "-k", "fused",
                "-i", str(maxiter), "--precision", precision,
                "--inclusion-type", "single", "--json-out", out_path]
        if nranks == 1:
            cmd = [sys.executable] + base
            env = os.environ
        else:
            cmd = ["mpiexec", "-n", str(nranks), sys.executable] + base
            env = dict(os.environ, OMPI_MCA_rmaps_base_oversubscribe="1")
        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=7200, env=env)
        except subprocess.SubprocessError:
            return None
        try:
            with open(out_path) as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            if _looks_like_oom(out):
                return db.OOM
            sys.stderr.write(f"  [{device} n={n} ranks={nranks}] no JSON\n"
                             f"{out.stderr[-400:]}\n")
            return None
    r, c = d["results"], d["config"]
    return dict(npts=c["nb_grid_pts_total"], iters=r["total_cg_iterations"],
                secs=r["total_time_seconds"],
                gbps=r.get("memory_throughput_GBps"))


def _point_fields(key, device, nranks, n, args):
    """Identity + parameters of one data point (the cache key, minus results)."""
    return {"benchmark": BENCHMARK, "study": "time_vs_size", "label": key,
            "device": device, "nranks": nranks, "dim": 3, "n": n,
            "maxiter": args.maxiter, "precision": args.precision}


def _resolve(key, device, nranks, n, prov, args):
    """Resolve one point through the cache; return (n, result, from_cache)."""
    fields = _point_fields(key, device, nranks, n, args)
    result, cached = db.cached_point(
        fields, prov,
        lambda: run(device, n, args.maxiter, nranks, args.precision),
        cache_dir=args.cache_dir, force=args.force)
    return n, result, cached


def _sweep_serial(key, device, nranks, sizes, prov, args):
    """Sweep sizes in order, stopping at the first out-of-memory size (larger
    grids for the same config would also OOM)."""
    out = []
    for n in sizes:
        point = _resolve(key, device, nranks, n, prov, args)
        out.append(point)
        if point[1] == OOM:
            break
    return out


def _sweep_parallel(key, device, nranks, sizes, prov, args):
    """Resolve all sizes concurrently (used only for the single-CPU-core config,
    where every point is one core and capped below any OOM threshold)."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(_resolve, key, device, nranks, n, prov, args)
                for n in sizes]
        out = [f.result() for f in futs]
    return sorted(out, key=lambda t: t[0])


def collect(args, prov):
    """Run all data points and return DB rows (does not write the CSV; each point
    is cached as it completes, so a killed run resumes from where it stopped)."""
    _, nb_gpus = db.detect_gpu()
    want_gpu = not args.no_gpu and nb_gpus >= 1
    configs = db.plan_configs(args.mpi_cpu_ranks, nb_gpus, want_gpu)
    rows = []

    # Time vs. size, one curve per device/MPI config. Each config sweeps grid
    # sizes up to its own cap (a single CPU core tops out far sooner than the
    # full node / the GPUs), and stops at the first size that runs out of
    # memory — recorded as an "oom" point (shown in the table, omitted from the
    # plot); larger sizes for that config are not attempted.
    for key, device, nranks in configs:
        label = CONFIG_META[key]["label"](nranks)
        cap = args.cpu1_max_size if key == "cpu1" else args.max_size
        sizes = [s for s in args.sizes if s <= cap]
        # The single CPU core is the one config safe to parallelise: each point
        # uses exactly one core and is capped (cpu1_max_size) below any OOM
        # threshold. Every other config already uses the whole node (MPI) or a
        # whole GPU per point, so those sweep serially and keep the OOM early stop.
        if key == "cpu1" and args.jobs > 1:
            points = _sweep_parallel(key, device, nranks, sizes, prov, args)
        else:
            points = _sweep_serial(key, device, nranks, sizes, prov, args)
        for n, r, cached in points:
            base = {**prov, "benchmark": BENCHMARK, "study": "time_vs_size",
                    "label": key, "device": device, "nranks": nranks,
                    "dim": 3, "n": n, "npts": n ** 3, "maxiter": args.maxiter,
                    "precision": args.precision}
            tag = " [cached]" if cached else ""
            if r is None:
                sys.stderr.write(f"  {label} {n}^3: skipped (run failed)\n")
                continue
            if r == OOM:
                rows.append({**base, "status": "oom"})
                sys.stderr.write(f"  {label} {n}^3: OUT OF MEMORY{tag}\n")
                continue
            rows.append({**base, "npts": r["npts"], "status": "ok",
                         "iters": r["iters"], "secs": r["secs"],
                         "gbps": r["gbps"]})
            sys.stderr.write(f"  {label} {n}^3 ({r['npts']} pts): "
                             f"{r['secs']:.3f} s, {r['iters']} it, "
                             f"{r['gbps']:.1f} GB/s{tag}\n")
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
    """{precision: {config_key: {n: row}}} for the time_vs_size study."""
    d = {}
    for r in rows:
        if r["study"] != "time_vs_size":
            continue
        prec = db.norm_precision(r)
        d.setdefault(prec, {}).setdefault(r["label"], {})[r["n"]] = r
    return d


def precisions_in(merged):
    """Precisions present in `merged`, in canonical (double, single) order."""
    return [p for p in db.PRECISIONS if p in merged]


def sizes_in(rows, study):
    return sorted({r["n"] for r in rows if r["study"] == study})


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def _cell(res, n):
    """One table cell: solve time, 'OOM' for an out-of-memory point, else '—'."""
    if n not in res:
        return "—"
    secs = res[n].get("secs")
    if isinstance(secs, (int, float)):
        return f"{secs:.3g}"
    return "OOM" if res[n].get("status") == "oom" else "—"


def table_markdown(sizes, configs, merged):
    precisions = precisions_in(merged)
    show_prec = len(precisions) > 1
    # Columns: sizes measured by any (config, precision) present.
    cols = [n for n in sizes
            if any(n in merged.get(p, {}).get(k, {})
                   for p in precisions for k, *_ in configs)]
    header = ("| Configuration | "
              + " | ".join(f"{n}³ ({fmt_points(n ** 3)})" for n in cols) + " |")
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [header, sep]
    for key, label, _style in configs:
        for prec in precisions:
            res = merged.get(prec, {}).get(key)
            if not res:
                continue
            row_label = (f"{label}, {db.PRECISION_LABEL[prec]}" if show_prec
                         else label)
            cells = " | ".join(_cell(res, n) for n in cols)
            lines.append(f"| {row_label} | {cells} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def make_merged_plot(configs, merged, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    all_n = set()
    precisions = precisions_in(merged)
    show_prec = len(precisions) > 1
    for key, label, style in configs:
        for prec in precisions:
            res = merged.get(prec, {}).get(key, {})
            # Only points with a real measured time — OOM points carry no time.
            pts = sorted((n, res[n]["secs"]) for n in res
                         if isinstance(res[n].get("secs"), (int, float)))
            if not pts:
                continue
            xs, ys = zip(*pts)
            all_n.update(xs)
            leg = (f"{label}, {db.PRECISION_LABEL[prec]}" if show_prec
                   else label)
            ax.loglog(xs, ys, label=leg,
                      **{**style, **db.PRECISION_STYLE[prec]})
    db.set_grid_size_xaxis(ax, all_n, 3)
    ax.set_ylabel("Solve time (s)")
    ax.set_title("Homogenization (3D, fused): time vs. grid size")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
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
{precision_note}

## Time vs. grid size

The plot below merges several ways of running the *same* solve on this machine:

- **CPU (1 core)** — a single core, MPI disabled. muGrid's compute kernels carry
  no OpenMP, so a non-MPI CPU run uses exactly one core. Only swept up to
  {cpu1_max}³; a single core is hopeless beyond that.
- **CPU ({ncores} cores, MPI)** — the whole CPU via MPI domain decomposition
  (`mpiexec -n {ncores}`), the grid split into per-rank subdomains that exchange
  ghost layers each iteration.
- **GPU (1 device)** — a single GPU.
{gpu_mpi_bullet}
Each configuration is swept to the largest grid that still fits in memory: the
first size that runs **out of memory** is recorded as `OOM` in the table and
dropped from the plot, and larger sizes for that configuration are not attempted.

{table}

(values are **solve time in seconds**; `OOM` = the run ran out of memory)

![Homogenization solve time vs. grid size]({plot_name})

A single CPU core is quickly left behind, so the fair comparison is the full CPU
(all {ncores} cores via MPI) against the GPU(s). The GPU leads in the mid-range,
where the heavy per-point FEM stiffness kernel keeps it busy and the working set
fits in device memory. The largest grids are reached only by MPI domain
decomposition — across all CPU cores, or across several GPUs (one rank per
device, round-robin) — which is also what pushes each curve's memory ceiling out
before the `OOM` cutoff.

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


def _precision_note(precisions):
    """One-line note describing the precision(s) shown, or '' for plain fp64."""
    if len(precisions) > 1:
        return ("\nBoth **double** (`fp64`, solid) and **single** (`fp32`, "
                "dashed) precision are shown; single precision halves the "
                "resident field memory and the bytes moved per iteration.")
    if precisions == ["single"]:
        return "\nAll runs are in **single** precision (`fp32`)."
    return ""


def write_doc_page(path, plot_path, table, meta, ncores, cpu1_max, multi_gpu,
                   precisions):
    if multi_gpu:
        gpu_mpi_bullet = ("- **GPU (N devices, MPI)** — several GPUs via MPI "
                          "domain decomposition, one rank per device "
                          "(round-robin).\n")
    else:
        gpu_mpi_bullet = ""
    with open(path, "w") as fh:
        fh.write(DOC_TEMPLATE.format(
            cpu=meta["cpu"], gpu=meta["gpu"], version=meta["version"],
            timestamp=meta["timestamp"], table=table, maxiter=meta["maxiter"],
            ncores=ncores, cpu1_max=cpu1_max, gpu_mpi_bullet=gpu_mpi_bullet,
            precision_note=_precision_note(precisions),
            plot_name=os.path.basename(plot_path)))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512,
                             768, 1024, 1536, 2048],
                    help="Grid sizes to sweep (each config stops at its cap or "
                         "the first out-of-memory size)")
    ap.add_argument("--max-size", type=int, default=2048,
                    help="Largest grid size attempted for the full-CPU and GPU "
                         "configs")
    ap.add_argument("--cpu1-max-size", type=int, default=128,
                    help="Largest grid size attempted for the single-CPU-core "
                         "config (it is hopeless beyond this)")
    ap.add_argument("--mpi-cpu-ranks", type=int, default=os.cpu_count(),
                    help="Ranks for the full-machine MPI CPU curve")
    ap.add_argument("--no-gpu", action="store_true", help="Skip the GPU curves")
    ap.add_argument("--precision", choices=db.PRECISIONS, default="double",
                    help="Floating-point precision to MEASURE this run in "
                         "(recorded per row). Rendering overlays whatever "
                         "precisions are in the DB — run this once per precision "
                         "to get both curves (default: double)")
    ap.add_argument("--maxiter", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=1,
                    help="Parallel worker processes for the single-CPU-core "
                         "sweep (each point uses one core). Other configs always "
                         "run serially, one per node/GPU (default: 1)")
    ap.add_argument("--cache-dir", default=db.CACHE_DIR,
                    help="Per-point result cache directory. Finished points are "
                         "replayed from here, so a killed run resumes cheaply")
    ap.add_argument("--force", action="store_true",
                    help="Recompute even on a cache hit (and refresh the cache)")
    ap.add_argument("--collect-only", action="store_true",
                    help="Measure and append to the database, then stop "
                         "(no plot/page — render later with --render-only)")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="Do not run: append a fresh run to the database from the "
                         "points already in the cache for the current commit. "
                         "Recovers a sweep that could not finish in one job")
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

    # Latest run PER PRECISION, so the page overlays the fp64 and fp32 curves
    # even though they are measured by separate jobs (separate timestamps).
    rows = db.select_precisions(db.load(args.db), BENCHMARK,
                                ["time_vs_size"], select_ts)
    if not rows:
        sys.exit("No matching rows in the database.")

    # Machine/version box from the most recent run among the precisions shown.
    meta_row = max(rows, key=lambda r: r["timestamp"])
    meta = {k: meta_row[k] for k in ("cpu", "gpu", "version", "timestamp")}
    meta["maxiter"] = next((r["maxiter"] for r in rows
                            if r["study"] == "time_vs_size"), args.maxiter)
    configs = db.render_configs(rows, "time_vs_size")
    merged = merged_from_rows(rows)
    ncores = next((r["nranks"] for r in rows if r["label"] == "cpuN"),
                  args.mpi_cpu_ranks)
    multi_gpu = any(r["label"] == "gpuN" for r in rows)
    cpu1_max = max((r["n"] for r in rows if r["label"] == "cpu1"),
                   default=args.cpu1_max_size)

    sizes = sizes_in(rows, "time_vs_size")
    precisions = precisions_in(merged)
    table = table_markdown(sizes, configs, merged)
    print("\n" + table)

    plot_out = os.path.abspath(args.plot_out)
    make_merged_plot(configs, merged, plot_out)
    sys.stderr.write(f"wrote {plot_out}\n")

    if args.doc_out:
        write_doc_page(args.doc_out, plot_out, table, meta, ncores, cpu1_max,
                       multi_gpu, precisions)
        sys.stderr.write(f"wrote {os.path.abspath(args.doc_out)}\n")


if __name__ == "__main__":
    main()
