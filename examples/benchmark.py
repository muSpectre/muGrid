#!/usr/bin/env python3
"""Scaling benchmark for the µGrid Poisson solver example.

Solves the 3D Poisson equation (`poisson.py`) with the unpreconditioned
conjugate-gradient solver over a range of log-spaced grid sizes on the CPU and
(if available) the GPU, and records the solve wall time. It regenerates the
documentation [Benchmark](../docs/benchmark.md) page: a Markdown table plus a
log-log plot of solve time vs. number of grid points.

Each (device, size) point runs `poisson.py` as its own subprocess and the
machine-readable ``--json`` output is parsed, so the runtimes are unaffected by
this driver's own imports.

Example
-------
    python examples/benchmark.py --doc-out docs/benchmark.md \
        --plot-out docs/benchmark.png
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
POISSON = os.path.join(HERE, "poisson.py")


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
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        names = [n.strip() for n in out.stdout.splitlines() if n.strip()]
    except (OSError, subprocess.SubprocessError):
        names = []
    if not names:
        return "no NVIDIA GPU detected"
    uniq = sorted(set(names))
    if len(names) > 1 and len(uniq) == 1:
        return f"{len(names)}x {uniq[0]}"
    return ", ".join(names)


def run(device, n, maxiter):
    """Run one Poisson solve; return (npts, iters, seconds, converged) or None."""
    cmd = [sys.executable, POISSON, "-n", f"{n},{n},{n}",
           "-d", device, "-i", str(maxiter), "--json"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
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
    return (c["nb_grid_pts_total"], r["iterations"],
            r["total_time_seconds"], r["converged"])


def fmt_points(npts):
    """Compact grid-point count, e.g. 16.8M, 884k."""
    if npts >= 1e6:
        return f"{npts / 1e6:.1f}M"
    if npts >= 1e3:
        return f"{npts / 1e3:.0f}k"
    return str(npts)


def table_markdown(sizes, results):
    """Markdown table: rows = device, columns = grid size (n³)."""
    cols = [n for n in sizes if any(n in results[d] for d in results)]
    header = ("| Implementation | "
              + " | ".join(f"{n}³ ({fmt_points(n ** 3)})" for n in cols) + " |")
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [header, sep]
    labels = {"cpu": "CPU", "gpu": "GPU"}
    for dev in ("cpu", "gpu"):
        if dev not in results or not results[dev]:
            continue
        cells = " | ".join(
            (f"{results[dev][n][2]:.3g}" if n in results[dev] else "—")
            for n in cols)
        lines.append(f"| {labels[dev]} | {cells} |")
    return "\n".join(lines)


def make_plot(sizes, results, path):
    """Log-log plot of solve time vs. number of grid points; one line/device."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    style = {"cpu": dict(label="CPU", marker="o", color="#5e35b1"),
             "gpu": dict(label="GPU", marker="s", color="#00897b")}
    for dev in ("cpu", "gpu"):
        pts = sorted((n ** 3, results[dev][n][2])
                     for n in results.get(dev, {}))
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.loglog(xs, ys, **style[dev])

    ax.set_xlabel("Number of grid points")
    ax.set_ylabel("Solve time (s)")
    ax.set_title("Poisson CG solve: time vs. grid size")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


DOC_TEMPLATE = """# Benchmark

Wall time of the unpreconditioned conjugate-gradient solve in the
[Poisson example](examples.md) (`examples/poisson.py`), across log-spaced 3D
grid sizes, on the CPU and the GPU. Lower is better.

!!! info "Test machine"
    - **CPU:** {cpu}
    - **GPU:** {gpu}

Run configuration: 3D grid, hard-coded Laplace operator, no preconditioner,
relative tolerance `1e-6`. Times are the solver wall time (`total_time_seconds`,
excluding setup) for a single run per size.

{table}

(values are **solve time in seconds**)

![Poisson CG solve time vs. number of grid points]({plot_name})

The problem is memory-bandwidth-bound (arithmetic intensity ≈ 0.16 FLOP/byte),
so the time tracks memory throughput rather than peak FLOPs, and the
unpreconditioned CG iteration count grows with grid size — hence the
slightly-steeper-than-linear slope on the log-log plot.

This page is generated by `examples/benchmark.py`. Regenerate it on your own
machine with:

```bash
python examples/benchmark.py --doc-out docs/benchmark.md \\
    --plot-out docs/benchmark.png
```
"""


def write_doc_page(path, plot_path, table):
    plot_name = os.path.basename(plot_path)
    with open(path, "w") as fh:
        fh.write(DOC_TEMPLATE.format(cpu=detect_cpu(), gpu=detect_gpu(),
                                     table=table, plot_name=plot_name))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[32, 48, 64, 96, 128, 192, 256],
                    help="Per-axis grid sizes n (the grid is n x n x n)")
    ap.add_argument("--devices", nargs="+", default=["cpu", "gpu"],
                    choices=["cpu", "gpu"])
    ap.add_argument("--maxiter", type=int, default=5000)
    ap.add_argument("--doc-out", default=None,
                    help="Write the Markdown benchmark page here")
    ap.add_argument("--plot-out", default=os.path.join(HERE, "..", "docs",
                                                        "benchmark.png"),
                    help="Write the log-log plot image here")
    args = ap.parse_args()

    results = {d: {} for d in args.devices}
    for dev in args.devices:
        for n in args.sizes:
            res = run(dev, n, args.maxiter)
            if res is None:
                print(f"  {dev} n={n}: skipped (failed / unavailable)",
                      file=sys.stderr)
                continue
            npts, iters, secs, conv = res
            results[dev][n] = res
            flag = "" if conv else "  [did NOT converge]"
            print(f"  {dev} {n}^3 ({npts} pts): {secs:.3f} s, {iters} it{flag}",
                  file=sys.stderr)

    # Drop devices that produced nothing (e.g. no GPU present)
    results = {d: r for d, r in results.items() if r}
    if not results:
        sys.exit("No successful runs — nothing to report.")

    table = table_markdown(args.sizes, results)
    print("\n" + table + "\n\n(values are solve time in seconds)")

    plot_out = os.path.abspath(args.plot_out)
    make_plot(args.sizes, results, plot_out)
    print(f"\nwrote {plot_out}", file=sys.stderr)

    if args.doc_out:
        write_doc_page(args.doc_out, plot_out, table)
        print(f"wrote {os.path.abspath(args.doc_out)}", file=sys.stderr)


if __name__ == "__main__":
    main()
