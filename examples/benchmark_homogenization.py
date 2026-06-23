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

Every data point runs `homogenization.py` as its own subprocess (under `mpiexec`
for the MPI configurations) with a fixed CG-iteration budget, so all
configurations perform identical arithmetic, and the machine-readable `--json`
output is parsed for the solve wall time.

Example
-------
    python examples/benchmark_homogenization.py \
        --doc-out docs/benchmark_homogenization.md \
        --plot-out docs/benchmark_homogenization.png

Needs an MPI-enabled muGrid build and `mpi4py` for the MPI configurations. Point
`PYTHONPATH` at an MPI (and, for the GPU curves, GPU-enabled) build tree, e.g.

    export PYTHONPATH=$PWD/build-mpi/language_bindings/python:$PWD/language_bindings/python

Use `--from-json results.json` to re-render the page from a previous run.
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HOMOG = os.path.join(HERE, "homogenization.py")


# --------------------------------------------------------------------------- #
# Machine detection
# --------------------------------------------------------------------------- #
def detect_cpu():
    """Human-readable CPU description (model + logical core count)."""
    model = platform.processor() or "unknown CPU"
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return f"{model} ({os.cpu_count()} logical cores)"


def gpu_names():
    """List of GPU name strings from nvidia-smi (empty if none)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        return [r.strip() for r in out.stdout.splitlines() if r.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


def detect_gpu():
    """Human-readable GPU description, or a fallback note."""
    rows = gpu_names()
    if not rows:
        return "no NVIDIA GPU detected", 0
    names = [r.split(",")[0].strip() for r in rows]
    uniq = sorted(set(names))
    label = (f"{len(names)}x {uniq[0]}" if len(names) > 1 and len(uniq) == 1
             else ", ".join(names))
    mem = rows[0].split(",")[1].strip() if "," in rows[0] else ""
    return (f"{label} ({mem})" if mem else label), len(names)


# --------------------------------------------------------------------------- #
# Running homogenization.py
# --------------------------------------------------------------------------- #
def run(device, n, maxiter, nranks=1):
    """Run one homogenization solve; return a dict of metrics or None.

    nranks == 1 runs the process directly; nranks > 1 launches it under
    ``mpiexec`` (one rank per core for CPU, one rank per GPU for GPU — the
    example binds ranks to GPUs round-robin).
    """
    base = [HOMOG, "-n", f"{n},{n},{n}", "-d", device, "-k", "fused",
            "-i", str(maxiter), "--inclusion-type", "single", "--json"]
    if nranks == 1:
        cmd = [sys.executable] + base
        env = os.environ
    else:
        cmd = ["mpiexec", "-n", str(nranks), sys.executable] + base
        # Allow more ranks than physical slots on a laptop / single node.
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
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    r, c = d["results"], d["config"]
    return dict(npts=c["nb_grid_pts_total"], iters=r["total_cg_iterations"],
                secs=r["total_time_seconds"],
                gbps=r.get("memory_throughput_GBps"),
                E=r.get("E_effective_approx"))


def fmt_points(npts):
    if npts >= 1e6:
        return f"{npts / 1e6:.1f}M"
    if npts >= 1e3:
        return f"{npts / 1e3:.0f}k"
    return str(npts)


# --------------------------------------------------------------------------- #
# Configurations for the merged "time vs. size" plot
# --------------------------------------------------------------------------- #
def build_configs(ncores, nb_gpus, want_gpu):
    """Ordered list of configs: (key, label, device, nranks, style)."""
    cfgs = [
        ("cpu1", "CPU (1 core)", "cpu", 1,
         dict(marker="o", color="#5e35b1")),
        ("cpuN", f"CPU ({ncores} cores, MPI)", "cpu", ncores,
         dict(marker="D", color="#3949ab")),
    ]
    if want_gpu and nb_gpus >= 1:
        cfgs.append(("gpu1", "GPU (1 device)", "gpu", 1,
                     dict(marker="s", color="#00897b")))
    # Multi-GPU curve only when the machine actually has >1 GPU.
    if want_gpu and nb_gpus > 1:
        cfgs.append(("gpuN", f"GPU ({nb_gpus} devices, MPI)", "gpu", nb_gpus,
                     dict(marker="^", color="#f4511e")))
    return cfgs


def table_markdown(sizes, configs, results):
    """Rows = configuration, columns = grid size; values = solve time (s)."""
    cols = [n for n in sizes if any(n in results.get(k, {}) for k, *_ in configs)]
    header = ("| Configuration | "
              + " | ".join(f"{n}³ ({fmt_points(n ** 3)})" for n in cols) + " |")
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [header, sep]
    for key, label, *_ in configs:
        res = results.get(key)
        if not res:
            continue
        cells = " | ".join(
            (f"{res[n]['secs']:.3g}" if n in res else "—") for n in cols)
        lines.append(f"| {label} | {cells} |")
    return "\n".join(lines)


def make_merged_plot(sizes, configs, results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for key, label, _dev, _nr, style in configs:
        pts = sorted((n ** 3, results[key][n]["secs"])
                     for n in results.get(key, {}))
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


# --------------------------------------------------------------------------- #
# MPI strong scaling
# --------------------------------------------------------------------------- #
def scaling_tables_markdown(sizes, ranks, res):
    out = []
    for n in sizes:
        rows = res.get(str(n))
        if not rows:
            continue
        t1 = rows.get("1", {}).get("secs")
        out.append(f"**{n}³ ({n ** 3:,} points)**\n")
        out.append("| Ranks | Time (s) | Speedup | Parallel eff. | Agg. GB/s |")
        out.append("|---|---|---|---|---|")
        for R in ranks:
            if str(R) not in rows:
                continue
            t = rows[str(R)]["secs"]
            sp = t1 / t if t1 else float("nan")
            out.append(f"| {R} | {t:.2f} | {sp:.2f}× | {sp / R * 100:.0f}% | "
                       f"{rows[str(R)]['gbps']:.1f} |")
        out.append("")
    return "\n".join(out)


def make_scaling_plot(sizes, ranks, res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    rmax = max(ranks)
    ax.plot([1, rmax], [1, rmax], ls="--", color="0.6", label="ideal (linear)")
    markers = {32: "v", 64: "o", 96: "s", 128: "^"}
    for n in sizes:
        rows = res.get(str(n), {})
        t1 = rows.get("1", {}).get("secs")
        if not t1:
            continue
        xs = [R for R in ranks if str(R) in rows]
        ys = [t1 / rows[str(R)]["secs"] for R in xs]
        ax.plot(xs, ys, marker=markers.get(n, "o"), label=f"{n}³")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(ranks)
    ax.set_xticklabels([str(R) for R in ranks])
    ax.set_yticks(ranks)
    ax.set_yticklabels([str(R) for R in ranks])
    ax.set_xlabel("MPI ranks (CPU cores)")
    ax.set_ylabel("Speedup vs. 1 rank")
    ax.set_title("Homogenization (3D, fused): MPI strong scaling")
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

!!! info "Test machine"
    - **CPU:** {cpu}
    - **GPU:** {gpu}

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
    automatically when more than one GPU is present. **This machine has a single
    GPU**, so only the single-GPU curve is shown; the script is ready to produce
    the multi-GPU curve on a multi-GPU host with no changes.

## MPI strong scaling (CPU)

Strong scaling of the same 3D fused solve (fixed problem size, increasing MPI
ranks) on the {ncores}-core CPU, with `E_eff` identical across all rank counts.

{scaling_tables}
![Homogenization MPI strong scaling]({scaling_plot_name})

Scaling is near-ideal to 4 ranks, then tapers: the solve is
memory-bandwidth-bound, so aggregate throughput keeps climbing as cores are added
(toward ~20 GB/s) but parallel efficiency falls. Once per-rank subdomains get
small, ghost exchange and CG dot-product reductions dominate — at 64³, 16 ranks
*regresses* (only ~16k points/rank; the sweet spot is 8 ranks), whereas the
larger 96³ problem keeps scaling out to 16 cores.

This page is generated by `examples/benchmark_homogenization.py`. Regenerate it
on your own machine (MPI-enabled build + `mpi4py`, GPU build for the GPU curves)
with:

```bash
python examples/benchmark_homogenization.py \\
    --doc-out docs/benchmark_homogenization.md \\
    --plot-out docs/benchmark_homogenization.png
```
"""


def write_doc_page(path, plot_path, scaling_plot_path, table, scaling_tables,
                   maxiter, ncores, nb_gpus):
    gpu, _ = detect_gpu()
    if nb_gpus > 1:
        gpu_mpi_bullet = (
            f"- **GPU ({nb_gpus} devices, MPI)** — all GPUs, one rank per "
            f"device.\n")
    else:
        gpu_mpi_bullet = ""
    with open(path, "w") as fh:
        fh.write(DOC_TEMPLATE.format(
            cpu=detect_cpu(), gpu=gpu, table=table, maxiter=maxiter,
            ncores=ncores, gpu_mpi_bullet=gpu_mpi_bullet,
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
                    default=[16, 24, 32, 48, 64, 96, 128],
                    help="Per-axis grid sizes n for the time-vs-size plot")
    ap.add_argument("--mpi-cpu-ranks", type=int, default=os.cpu_count(),
                    help="Ranks for the full-machine MPI CPU curve "
                         "(default: all logical cores)")
    ap.add_argument("--no-gpu", action="store_true",
                    help="Skip the GPU curves")
    ap.add_argument("--scaling-sizes", type=int, nargs="+", default=[64, 96],
                    help="Grid sizes for the MPI strong-scaling study")
    ap.add_argument("--scaling-ranks", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16],
                    help="Rank counts for the MPI strong-scaling study")
    ap.add_argument("--maxiter", type=int, default=100,
                    help="CG iterations per load case (fixed work budget)")
    ap.add_argument("--from-json", default=None,
                    help="Render the page from a saved results JSON")
    ap.add_argument("--json-out", default=None,
                    help="Save raw results to this JSON")
    ap.add_argument("--doc-out", default=None,
                    help="Write the Markdown benchmark page here")
    ap.add_argument("--plot-out",
                    default=os.path.join(HERE, "..", "docs",
                                         "benchmark_homogenization.png"))
    ap.add_argument("--scaling-plot-out",
                    default=os.path.join(HERE, "..", "docs",
                                         "benchmark_homogenization_mpi.png"))
    args = ap.parse_args()

    _, nb_gpus = detect_gpu()
    want_gpu = not args.no_gpu and nb_gpus >= 1
    ncores = args.mpi_cpu_ranks
    configs = build_configs(ncores, nb_gpus, want_gpu)

    if args.from_json:
        with open(args.from_json) as fh:
            blob = json.load(fh)
        merged = blob["merged"]
        scaling = blob["scaling"]
    else:
        # --- time vs. size ---
        merged = {key: {} for key, *_ in configs}
        for n in args.sizes:
            for key, label, device, nranks, _style in configs:
                res = run(device, n, args.maxiter, nranks)
                if res is None:
                    sys.stderr.write(f"  {label} {n}^3: skipped "
                                     f"(failed / OOM)\n")
                    continue
                merged[key][n] = res
                sys.stderr.write(
                    f"  {label} {n}^3 ({res['npts']} pts): "
                    f"{res['secs']:.3f} s, {res['iters']} it, "
                    f"{res['gbps']:.1f} GB/s\n")

        # --- MPI strong scaling (CPU) ---
        scaling = {str(n): {} for n in args.scaling_sizes}
        for n in args.scaling_sizes:
            for R in args.scaling_ranks:
                res = run("cpu", n, args.maxiter, R)
                if res is None:
                    continue
                scaling[str(n)][str(R)] = res
                sys.stderr.write(f"  scaling {n}^3 ranks={R}: "
                                 f"{res['secs']:.3f} s, "
                                 f"{res['gbps']:.1f} GB/s\n")

        # JSON keys must be strings; remap merged size keys.
        if args.json_out:
            blob = {
                "merged": {k: {str(n): v for n, v in d.items()}
                           for k, d in merged.items()},
                "scaling": scaling,
            }
            with open(args.json_out, "w") as fh:
                json.dump(blob, fh, indent=2)

    # from-json stores size keys as strings; normalise back to int.
    merged = {k: {int(n): v for n, v in d.items()} for k, d in merged.items()}

    table = table_markdown(args.sizes, configs, merged)
    scaling_tables = scaling_tables_markdown(
        args.scaling_sizes, args.scaling_ranks, scaling)
    print("\n" + table + "\n\n(values are solve time in seconds)\n")
    print(scaling_tables)

    plot_out = os.path.abspath(args.plot_out)
    scaling_plot_out = os.path.abspath(args.scaling_plot_out)
    make_merged_plot(args.sizes, configs, merged, plot_out)
    make_scaling_plot(args.scaling_sizes, args.scaling_ranks, scaling,
                      scaling_plot_out)
    sys.stderr.write(f"wrote {plot_out}\nwrote {scaling_plot_out}\n")

    if args.doc_out:
        write_doc_page(args.doc_out, plot_out, scaling_plot_out, table,
                       scaling_tables, args.maxiter, ncores, nb_gpus)
        sys.stderr.write(f"wrote {os.path.abspath(args.doc_out)}\n")


if __name__ == "__main__":
    main()
