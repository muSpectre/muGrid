#!/usr/bin/env python3
"""CPU-vs-GPU scaling benchmark for the homogenization example.

Runs `homogenization.py` (FEM elasticity homogenization, fused stiffness kernel)
over a range of log-spaced 3D grid sizes on the CPU and (if available) the GPU,
with a fixed iteration budget so both devices do identical work, and records the
solve wall time. It regenerates the documentation
[homogenization benchmark](../docs/benchmark_homogenization.md) page: a Markdown
table plus a log-log plot of solve time vs. number of grid points.

Each (device, size) point runs `homogenization.py` as its own subprocess and the
machine-readable ``--json`` output is parsed.

Example
-------
    python examples/benchmark_homogenization.py \
        --doc-out docs/benchmark_homogenization.md \
        --plot-out docs/benchmark_homogenization.png
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


def detect_gpu():
    """Human-readable GPU description from nvidia-smi, or a fallback note."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        rows = [r.strip() for r in out.stdout.splitlines() if r.strip()]
    except (OSError, subprocess.SubprocessError):
        rows = []
    if not rows:
        return "no NVIDIA GPU detected"
    names = [r.split(",")[0].strip() for r in rows]
    uniq = sorted(set(names))
    label = (f"{len(names)}x {uniq[0]}" if len(names) > 1 and len(uniq) == 1
             else ", ".join(names))
    mem = rows[0].split(",")[1].strip() if "," in rows[0] else ""
    return f"{label} ({mem})" if mem else label


def run(device, n, maxiter):
    """Run one homogenization solve; return a dict of metrics or None."""
    cmd = [sys.executable, HOMOG, "-n", f"{n},{n},{n}", "-d", device,
           "-k", "fused", "-i", str(maxiter), "--inclusion-type", "single",
           "--json"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    except subprocess.SubprocessError:
        return None
    m = re.search(r"\{.*\}", out.stdout, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    r, c = d["results"], d["config"]
    return dict(npts=c["nb_grid_pts_total"], iters=r["total_cg_iterations"],
                secs=r["total_time_seconds"],
                gbps=r.get("memory_throughput_GBps"))


def fmt_points(npts):
    if npts >= 1e6:
        return f"{npts / 1e6:.1f}M"
    if npts >= 1e3:
        return f"{npts / 1e3:.0f}k"
    return str(npts)


def table_markdown(sizes, results):
    """Rows = device, columns = grid size (n³); values = solve time (s)."""
    cols = [n for n in sizes if any(n in results[d] for d in results)]
    header = ("| Implementation | "
              + " | ".join(f"{n}³ ({fmt_points(n ** 3)})" for n in cols) + " |")
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [header, sep]
    labels = {"cpu": "CPU (1 core)", "gpu": "GPU"}
    for dev in ("cpu", "gpu"):
        if not results.get(dev):
            continue
        cells = " | ".join(
            (f"{results[dev][n]['secs']:.3g}" if n in results[dev] else "—")
            for n in cols)
        lines.append(f"| {labels[dev]} | {cells} |")
    # Speedup row if both present
    if results.get("cpu") and results.get("gpu"):
        cells = []
        for n in cols:
            c, g = results["cpu"].get(n), results["gpu"].get(n)
            cells.append(f"{c['secs'] / g['secs']:.2f}×" if c and g else "—")
        lines.append("| **Speedup (CPU/GPU)** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def make_plot(sizes, results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    style = {"cpu": dict(label="CPU (1 core)", marker="o", color="#5e35b1"),
             "gpu": dict(label="GPU", marker="s", color="#00897b")}
    for dev in ("cpu", "gpu"):
        pts = sorted((n ** 3, results[dev][n]["secs"])
                     for n in results.get(dev, {}))
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.loglog(xs, ys, **style[dev])
    ax.set_xlabel("Number of grid points")
    ax.set_ylabel("Solve time (s)")
    ax.set_title("Homogenization (3D, fused): time vs. grid size")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


DOC_TEMPLATE = """# Benchmark: homogenization

Wall time of the FEM elasticity [homogenization example](examples.md)
(`examples/homogenization.py`, fused stiffness kernel), across log-spaced **3D**
grid sizes, on the CPU and the GPU. Lower is better.

!!! info "Test machine"
    - **CPU:** {cpu}
    - **GPU:** {gpu}

!!! warning "The CPU baseline is single-threaded"
    muGrid's compute kernels carry no OpenMP, and this build has MPI disabled, so
    the example runs on **one CPU core**. The comparison below is therefore *one
    CPU core* vs. the whole GPU. MPI domain decomposition is available for
    multi-core/multi-node CPU runs (`mpiexec -n N`, an MPI-enabled build), but is
    not exercised here.

Run configuration: 3D single spherical inclusion, fused stiffness kernel,
6 load cases, fixed `{maxiter}` CG iterations per load case — i.e. a **fixed work
budget** so both devices perform identical arithmetic. Times are the solver wall
time (`total_time_seconds`, excluding setup).

{table}

(values are **solve time in seconds**)

![Homogenization solve time vs. number of grid points]({plot_name})

The 3D operator runs 6 load cases with a heavy per-point FEM stiffness kernel, so
the GPU amortizes its kernel-launch overhead early — it overtakes a single CPU
core at roughly 24³ and reaches ~7× by 96³.

!!! warning "GPU memory wall at 128³"
    A 128³ run needs about 5.85 GB of field storage, which nearly fills this
    GPU's 6 GB. The working set no longer fits comfortably, the allocator
    oversubscribes to host memory, and GPU throughput collapses to roughly the
    single-core CPU level. On a 6 GB card, ~96³ is the largest 3D problem that
    stays on the fast path; larger grids need a bigger-memory GPU or MPI domain
    decomposition across several GPUs.

This page is generated by `examples/benchmark_homogenization.py`. Regenerate it
on your own machine with:

```bash
python examples/benchmark_homogenization.py \\
    --doc-out docs/benchmark_homogenization.md \\
    --plot-out docs/benchmark_homogenization.png
```
"""


def write_doc_page(path, plot_path, table, maxiter):
    with open(path, "w") as fh:
        fh.write(DOC_TEMPLATE.format(
            cpu=detect_cpu(), gpu=detect_gpu(), table=table,
            plot_name=os.path.basename(plot_path), maxiter=maxiter))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[16, 24, 32, 48, 64, 96, 128],
                    help="Per-axis grid sizes n (the grid is n x n x n)")
    ap.add_argument("--devices", nargs="+", default=["cpu", "gpu"],
                    choices=["cpu", "gpu"])
    ap.add_argument("--maxiter", type=int, default=100,
                    help="CG iterations per load case (fixed work budget)")
    ap.add_argument("--doc-out", default=None,
                    help="Write the Markdown benchmark page here")
    ap.add_argument("--plot-out",
                    default=os.path.join(HERE, "..", "docs",
                                         "benchmark_homogenization.png"))
    args = ap.parse_args()

    results = {d: {} for d in args.devices}
    for n in args.sizes:
        for dev in args.devices:
            res = run(dev, n, args.maxiter)
            if res is None:
                print(f"  {dev} {n}^3: skipped (failed / OOM)", file=sys.stderr)
                continue
            results[dev][n] = res
            print(f"  {dev} {n}^3 ({res['npts']} pts): {res['secs']:.3f} s, "
                  f"{res['iters']} it, {res['gbps']:.1f} GB/s", file=sys.stderr)

    results = {d: r for d, r in results.items() if r}
    if not results:
        sys.exit("No successful runs — nothing to report.")

    table = table_markdown(args.sizes, results)
    print("\n" + table + "\n\n(values are solve time in seconds)")

    plot_out = os.path.abspath(args.plot_out)
    make_plot(args.sizes, results, plot_out)
    print(f"\nwrote {plot_out}", file=sys.stderr)

    if args.doc_out:
        write_doc_page(args.doc_out, plot_out, table, args.maxiter)
        print(f"wrote {os.path.abspath(args.doc_out)}", file=sys.stderr)


if __name__ == "__main__":
    main()
