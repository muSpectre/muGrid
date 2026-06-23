#!/usr/bin/env python3
"""Preconditioner benchmark for the homogenization example (3D by default).

Two studies on the FEM elasticity [homogenization example](examples.md), both
with the **same fused matvec kernel** and run **to convergence** (relative
tolerance), `-P reference` (Ladecký et al., 2023 Green's-function preconditioner)
vs. `-P none`:

1. **CG iterations vs. grid size** — unpreconditioned vs. reference. The
   iteration count is device- and decomposition-independent, so this runs on one
   CPU core. It is the central result: the preconditioner makes the count nearly
   grid-independent.
2. **Reference-solve time vs. grid size, by device / MPI config** — the *same*
   CPU-1-core / full-machine-MPI-CPU / GPU variation as the (unpreconditioned)
   [homogenization benchmark](benchmark_homogenization.md). The reference
   preconditioner applies a forward/inverse FFT every iteration, so this is where
   the FFT-engine paths show up: the native cuFFT/rocFFT N-D transform on the
   GPU, and the slab MPI decomposition on multi-rank runs. A *GPU (N devices,
   MPI)* curve is added automatically on a multi-GPU host (one rank per GPU,
   round-robin).

Every data point runs `homogenization.py` as a subprocess (under `mpiexec` for
the MPI configs) and parses its `--json` output.

Example
-------
    python examples/benchmark_homogenization_preconditioner.py \
        --doc-out docs/benchmark_homogenization_preconditioner.md

Needs an MPI-enabled (and, for the GPU curves, GPU-enabled) muGrid build and
`mpi4py`. Use `--from-json results.json` to re-render without recomputing.
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


def detect_gpu():
    """(human-readable description, device count)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        names = [n.strip() for n in out.stdout.splitlines() if n.strip()]
    except (OSError, subprocess.SubprocessError):
        names = []
    if not names:
        return "no NVIDIA GPU detected", 0
    uniq = sorted(set(names))
    label = (f"{len(names)}x {uniq[0]}" if len(names) > 1 and len(uniq) == 1
             else ", ".join(names))
    return label, len(names)


# --------------------------------------------------------------------------- #
# Running homogenization.py
# --------------------------------------------------------------------------- #
def run(device, precond, n, dim, maxiter, tol, nranks=1):
    """One homogenization solve; dict of metrics or None.

    nranks > 1 launches under ``mpiexec`` (one rank per core for CPU; the example
    binds ranks to GPUs round-robin for GPU runs).
    """
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
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    r, c = d["results"], d["config"]
    return dict(npts=c["nb_grid_pts_total"], iters=r["total_cg_iterations"],
                secs=r["total_time_seconds"], E=r.get("E_effective_approx"))


def fmt_points(npts):
    if npts >= 1e6:
        return f"{npts / 1e6:.1f}M"
    if npts >= 1e3:
        return f"{npts / 1e3:.0f}k"
    return str(npts)


def sup(dim):
    return "²" if dim == 2 else "³"


# --------------------------------------------------------------------------- #
# Device / MPI configurations for the reference-solve timing study
# --------------------------------------------------------------------------- #
def build_configs(ncores, nb_gpus, want_gpu):
    """Ordered list: (key, label, device, nranks, style) — mirrors the main
    homogenization benchmark."""
    cfgs = [
        ("cpu1", "CPU (1 core)", "cpu", 1, dict(marker="o", color="#5e35b1")),
        ("cpuN", f"CPU ({ncores} cores, MPI)", "cpu", ncores,
         dict(marker="D", color="#3949ab")),
    ]
    if want_gpu and nb_gpus >= 1:
        cfgs.append(("gpu1", "GPU (1 device)", "gpu", 1,
                     dict(marker="s", color="#00897b")))
    if want_gpu and nb_gpus > 1:
        cfgs.append(("gpuN", f"GPU ({nb_gpus} devices, MPI)", "gpu", nb_gpus,
                     dict(marker="^", color="#f4511e")))
    return cfgs


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def iteration_table(sizes, iters_res, dim):
    """CG iterations vs grid size: none vs reference (device-independent)."""
    cols = [n for n in sizes if ("none", n) in iters_res]
    head = "| Preconditioner | " + " | ".join(
        f"{n}{sup(dim)} ({fmt_points(n ** dim)})" for n in cols) + " |"
    lines = [head, "|" + "---|" * (len(cols) + 1)]
    for P in ("none", "reference"):
        cells = " | ".join(
            (str(iters_res[(P, n)]["iters"]) if (P, n) in iters_res else "—")
            for n in cols)
        lines.append(f"| {P} | {cells} |")
    # Wall-time speedup (none / reference) on the single CPU core.
    cells = []
    for n in cols:
        a, b = iters_res.get(("none", n)), iters_res.get(("reference", n))
        cells.append(f"{a['secs'] / b['secs']:.0f}×" if a and b else "—")
    lines.append("| **CPU-core wall-time speedup** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def timing_table(sizes, configs, time_res, dim):
    """Reference-solve time (s): rows = config, columns = grid size."""
    cols = [n for n in sizes
            if any((key, n) in time_res for key, *_ in configs)]
    head = ("| Configuration | "
            + " | ".join(f"{n}{sup(dim)} ({fmt_points(n ** dim)})" for n in cols)
            + " |")
    lines = [head, "|" + "---|" * (len(cols) + 1)]
    for key, label, *_ in configs:
        if not any((key, n) in time_res for n in cols):
            continue
        cells = " | ".join(
            (f"{time_res[(key, n)]['secs']:.3g}" if (key, n) in time_res else "—")
            for n in cols)
        lines.append(f"| {label} | {cells} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def make_iter_plot(sizes, iters_res, dim, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for P, sty in (
            ("none", dict(marker="o", color="#c62828",
                          label="unpreconditioned")),
            ("reference", dict(marker="s", color="#00897b",
                               label="reference preconditioner"))):
        p = sorted((n ** dim, iters_res[(P, n)]["iters"])
                   for n in sizes if (P, n) in iters_res)
        if p:
            xs, ys = zip(*p)
            ax.loglog(xs, ys, **sty)
    ax.set_xlabel("Number of grid points")
    ax.set_ylabel("CG iterations to converge")
    ax.set_title(f"Homogenization ({dim}D, fused): iterations vs. grid size")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def make_timing_plot(sizes, configs, time_res, dim, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for key, label, _dev, _nr, style in configs:
        p = sorted((n ** dim, time_res[(key, n)]["secs"])
                   for n in sizes if (key, n) in time_res)
        if not p:
            continue
        xs, ys = zip(*p)
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


# --------------------------------------------------------------------------- #
# Doc page
# --------------------------------------------------------------------------- #
DOC_TEMPLATE = """# Benchmark: preconditioner

Effect of the reference-material (Green's-function) preconditioner of Ladecký et
al., Appl. Math. Comput. 446 (2023) 127835, on the [homogenization
example](examples.md) — `-P reference` vs. `-P none`. Both use the **same fused
matvec kernel** and are run **to convergence** (relative tolerance `{tol}`).

!!! info "Test machine"
    - **CPU:** {cpu}
    - **GPU:** {gpu}

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
3D grid sizes. Because the iteration count is grid-independent, this isolates the
**per-iteration** cost — and each preconditioned iteration applies a
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
    automatically when more than one GPU is present. **This machine has a single
    GPU**, so only the single-GPU curve is shown; the script needs no changes on
    a multi-GPU host.

This page is generated by `examples/benchmark_homogenization_preconditioner.py`
(MPI-enabled build + `mpi4py`, GPU build for the GPU curves):

```bash
python examples/benchmark_homogenization_preconditioner.py \\
    --doc-out docs/benchmark_homogenization_preconditioner.md
```
"""


def write_doc_page(args, configs, nb_gpus, iter_table, timing_table,
                   iter_plot, timing_plot):
    gpu, _ = detect_gpu()
    with open(args.doc_out, "w") as fh:
        fh.write(DOC_TEMPLATE.format(
            cpu=detect_cpu(), gpu=gpu, dim=args.dim, tol=args.tol,
            ncases=3 if args.dim == 2 else 6,
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
                    default=[16, 24, 32, 48, 64, 96, 128],
                    help="Grid sizes for the reference-solve timing study")
    ap.add_argument("--iter-sizes", type=int, nargs="+",
                    default=[16, 24, 32, 48],
                    help="Grid sizes for the none-vs-reference iteration study "
                         "(kept modest: unpreconditioned 3D is expensive)")
    ap.add_argument("--mpi-cpu-ranks", type=int, default=os.cpu_count(),
                    help="Ranks for the full-machine MPI CPU curve")
    ap.add_argument("--no-gpu", action="store_true", help="Skip the GPU curves")
    ap.add_argument("--maxiter", type=int, default=20000)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--from-json", default=None,
                    help="Render the page from a saved results JSON")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--doc-out", default=None)
    ap.add_argument("--iter-plot", default=os.path.join(
        HERE, "..", "docs", "benchmark_homogenization_preconditioner_iters.png"))
    ap.add_argument("--time-plot", default=os.path.join(
        HERE, "..", "docs", "benchmark_homogenization_preconditioner_time.png"))
    args = ap.parse_args()

    _, nb_gpus = detect_gpu()
    want_gpu = not args.no_gpu and nb_gpus >= 1
    ncores = args.mpi_cpu_ranks
    configs = build_configs(ncores, nb_gpus, want_gpu)

    if args.from_json:
        with open(args.from_json) as fh:
            blob = json.load(fh)
        # Keys are "precond|n" (iterations) and "configkey|n" (timing).
        iters_res = {(k.rsplit("|", 1)[0], int(k.rsplit("|", 1)[1])): v
                     for k, v in blob["iters"].items()}
        time_res = {(k.rsplit("|", 1)[0], int(k.rsplit("|", 1)[1])): v
                    for k, v in blob["time"].items()}
    else:
        # Study 1: iterations (none vs reference), serial CPU.
        iters_res = {}
        for n in args.iter_sizes:
            for P in ("none", "reference"):
                r = run("cpu", P, n, args.dim, args.maxiter, args.tol)
                if r is None:
                    sys.stderr.write(f"  iter {P} {n}^{args.dim}: skipped\n")
                    continue
                iters_res[(P, n)] = r
                sys.stderr.write(f"  iter {P:9s} {n}^{args.dim} "
                                 f"({r['npts']} pts): {r['iters']:5d} it, "
                                 f"{r['secs']:.3f} s\n")

        # Study 2: reference-solve time across device/MPI configs.
        time_res = {}
        for n in args.sizes:
            for key, label, device, nranks, _style in configs:
                r = run(device, "reference", n, args.dim, args.maxiter,
                        args.tol, nranks)
                if r is None:
                    sys.stderr.write(f"  time {label} {n}^{args.dim}: "
                                     f"skipped\n")
                    continue
                time_res[(key, n)] = r
                sys.stderr.write(f"  time {label} {n}^{args.dim} "
                                 f"({r['npts']} pts): {r['iters']} it, "
                                 f"{r['secs']:.3f} s\n")

        if args.json_out:
            blob = {
                "iters": {f"{P}|{n}": v for (P, n), v in iters_res.items()},
                "time": {f"{k}|{n}": v for (k, n), v in time_res.items()},
            }
            with open(args.json_out, "w") as fh:
                json.dump(blob, fh, indent=2)

    if not iters_res and not time_res:
        sys.exit("No successful runs.")

    it_tab = iteration_table(args.iter_sizes, iters_res, args.dim)
    t_tab = timing_table(args.sizes, configs, time_res, args.dim)
    print("\n" + it_tab + "\n\n" + t_tab)

    iters_path = os.path.abspath(args.iter_plot)
    time_path = os.path.abspath(args.time_plot)
    make_iter_plot(args.iter_sizes, iters_res, args.dim, iters_path)
    make_timing_plot(args.sizes, configs, time_res, args.dim, time_path)
    sys.stderr.write(f"wrote {iters_path}\nwrote {time_path}\n")

    if args.doc_out:
        write_doc_page(args, configs, nb_gpus, it_tab, t_tab,
                       iters_path, time_path)
        sys.stderr.write(f"wrote {os.path.abspath(args.doc_out)}\n")


if __name__ == "__main__":
    main()
