#!/usr/bin/env python3
"""Preconditioner benchmark for the homogenization example.

Compares the unpreconditioned CG solve against the reference-material Fourier
preconditioner (Ladecký et al., 2023) across grid sizes, on CPU and GPU, using
the same (fused) matvec kernel for both so the comparison is apples-to-apples.
Both are run *to convergence* (relative tolerance), since the point of the
preconditioner is the number of iterations needed — which becomes nearly
independent of grid size — and the resulting wall-time win.

Regenerates the documentation page and two log-log plots:
  - CG iterations vs. number of grid points (the grid-independence result);
  - solve wall time vs. number of grid points.

Example
-------
    python examples/benchmark_homogenization_preconditioner.py \
        --doc-out docs/benchmark_homogenization_preconditioner.md
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
    return f"{len(names)}x {uniq[0]}" if len(names) > 1 and len(uniq) == 1 \
        else ", ".join(names)


def run(device, precond, n, dim, maxiter, tol):
    grid = ",".join([str(n)] * dim)
    cmd = [sys.executable, HOMOG, "-n", grid, "-d", device, "-k", "fused",
           "-P", precond, "-i", str(maxiter), "--tol", str(tol),
           "--inclusion-type", "single", "--json"]
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
                secs=r["total_time_seconds"], E=r.get("E_effective_approx"))


def fmt_points(npts):
    if npts >= 1e6:
        return f"{npts / 1e6:.1f}M"
    if npts >= 1e3:
        return f"{npts / 1e3:.0f}k"
    return str(npts)


def iteration_table(sizes, res, dim):
    """CG iterations (device-independent) vs grid size: none vs reference."""
    cols = [n for n in sizes if ("cpu", "none", n) in res]
    head = "| Preconditioner | " + " | ".join(
        f"{n}{'²' if dim == 2 else '³'} ({fmt_points(n ** dim)})" for n in cols
    ) + " |"
    lines = [head, "|" + "---|" * (len(cols) + 1)]
    for P, label in (("none", "none"), ("reference", "reference")):
        cells = " | ".join(str(res[("cpu", P, n)]["iters"]) for n in cols)
        lines.append(f"| {label} | {cells} |")
    return "\n".join(lines)


def time_table(sizes, res, dim):
    """Solve wall time (s) for each device × preconditioner."""
    cols = [n for n in sizes if ("cpu", "none", n) in res]
    head = "| Run | " + " | ".join(
        f"{n}{'²' if dim == 2 else '³'}" for n in cols) + " |"
    lines = [head, "|" + "---|" * (len(cols) + 1)]
    for dev in ("cpu", "gpu"):
        for P in ("none", "reference"):
            if not any((dev, P, n) in res for n in cols):
                continue
            cells = " | ".join(
                (f"{res[(dev, P, n)]['secs']:.3g}" if (dev, P, n) in res else "—")
                for n in cols)
            lines.append(f"| {dev.upper()}, {P} | {cells} |")
    # Speedup (none/reference) per device
    for dev in ("cpu", "gpu"):
        if not any((dev, "reference", n) in res for n in cols):
            continue
        cells = []
        for n in cols:
            a, b = res.get((dev, "none", n)), res.get((dev, "reference", n))
            cells.append(f"{a['secs'] / b['secs']:.1f}×" if a and b else "—")
        lines.append(f"| **{dev.upper()} speedup** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def make_plots(sizes, res, dim, iters_path, time_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def pts(dev, P):
        return sorted((n ** dim, res[(dev, P, n)])
                      for n in sizes if (dev, P, n) in res)

    # Iterations vs grid size (device-independent -> use CPU)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for P, sty in (("none", dict(marker="o", color="#c62828", label="unpreconditioned")),
                   ("reference", dict(marker="s", color="#00897b",
                                      label="reference preconditioner"))):
        p = pts("cpu", P)
        if p:
            xs, rs = zip(*p)
            ax.loglog(xs, [r["iters"] for r in rs], **sty)
    ax.set_xlabel("Number of grid points")
    ax.set_ylabel("CG iterations to converge")
    ax.set_title(f"Homogenization ({dim}D, fused): iterations vs. grid size")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(iters_path, dpi=120)
    plt.close(fig)

    # Wall time vs grid size (CPU/GPU × none/reference)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    styles = {
        ("cpu", "none"): dict(marker="o", ls="--", color="#c62828", label="CPU, unprec."),
        ("cpu", "reference"): dict(marker="s", ls="-", color="#c62828", label="CPU, ref."),
        ("gpu", "none"): dict(marker="o", ls="--", color="#1565c0", label="GPU, unprec."),
        ("gpu", "reference"): dict(marker="s", ls="-", color="#1565c0", label="GPU, ref."),
    }
    for key, sty in styles.items():
        p = pts(*key)
        if p:
            xs, rs = zip(*p)
            ax.loglog(xs, [r["secs"] for r in rs], **sty)
    ax.set_xlabel("Number of grid points")
    ax.set_ylabel("Solve time (s)")
    ax.set_title(f"Homogenization ({dim}D, fused): time vs. grid size")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(time_path, dpi=120)
    plt.close(fig)


DOC_TEMPLATE = """# Benchmark: preconditioner

Effect of the reference-material (Green's-function) preconditioner of Ladecký et
al., Appl. Math. Comput. 446 (2023) 127835, on the [homogenization
example](examples.md) — `-P reference` vs. `-P none`. Both use the **same fused
matvec kernel** and are run **to convergence** (relative tolerance `{tol}`), so
the comparison isolates the preconditioner.

!!! info "Test machine"
    - **CPU:** {cpu}
    - **GPU:** {gpu}

Run configuration: {dim}D, single spherical inclusion, fused stiffness kernel,
{ncases} load cases (iterations summed over all cases).

## CG iterations vs. grid size

This is the central result: unpreconditioned CG iterations grow with the grid
(the condition number is `O(h⁻²)`), while the reference preconditioner makes the
count **nearly independent of grid size**. Iterations depend only on the
operator and preconditioner, not the device.

{iter_table}

![Iterations vs. number of grid points]({iter_plot})

## Solve wall time

Each preconditioned iteration costs more (a forward/inverse FFT pair plus the
per-mode block solve), so the wall-time win is smaller than the iteration-count
ratio — but still large, and it grows with problem size.

{time_table}

(values are **solve time in seconds**; speedup is unpreconditioned ÷
preconditioned)

![Solve time vs. number of grid points]({time_plot})

The preconditioner also parallelizes: it is applied in Fourier space by the FFT
engine, which handles its own MPI decomposition, and the per-mode block solve is
rank-local. `-P reference` gives identical iteration counts and homogenized
stiffness in serial and under MPI (verified 1 vs. 4 ranks). On a single GPU the
per-iteration cost is dominated by the preconditioner FFTs (already C++/cuFFT),
not Python.

This page is generated by `examples/benchmark_homogenization_preconditioner.py`.
Regenerate it with:

```bash
python examples/benchmark_homogenization_preconditioner.py \\
    --doc-out docs/benchmark_homogenization_preconditioner.md
```
"""


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--dim", type=int, default=2, choices=[2, 3])
    ap.add_argument("--devices", nargs="+", default=["cpu", "gpu"],
                    choices=["cpu", "gpu"])
    ap.add_argument("--maxiter", type=int, default=20000)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--doc-out", default=None)
    ap.add_argument("--iter-plot", default=os.path.join(
        HERE, "..", "docs", "benchmark_homogenization_preconditioner_iters.png"))
    ap.add_argument("--time-plot", default=os.path.join(
        HERE, "..", "docs", "benchmark_homogenization_preconditioner_time.png"))
    args = ap.parse_args()

    res = {}
    for n in args.sizes:
        for dev in args.devices:
            for P in ("none", "reference"):
                r = run(dev, P, n, args.dim, args.maxiter, args.tol)
                if r is None:
                    print(f"  {dev} {P} {n}^{args.dim}: skipped", file=sys.stderr)
                    continue
                res[(dev, P, n)] = r
                print(f"  {dev} {P:9s} {n}^{args.dim} ({r['npts']} pts): "
                      f"{r['iters']:5d} it, {r['secs']:.3f} s", file=sys.stderr)

    if not res:
        sys.exit("No successful runs.")

    it_tab = iteration_table(args.sizes, res, args.dim)
    t_tab = time_table(args.sizes, res, args.dim)
    print("\n" + it_tab + "\n\n" + t_tab)

    iters_path = os.path.abspath(args.iter_plot)
    time_path = os.path.abspath(args.time_plot)
    make_plots(args.sizes, res, args.dim, iters_path, time_path)
    print(f"\nwrote {iters_path}\nwrote {time_path}", file=sys.stderr)

    if args.doc_out:
        with open(args.doc_out, "w") as fh:
            fh.write(DOC_TEMPLATE.format(
                cpu=detect_cpu(), gpu=detect_gpu(), dim=args.dim, tol=args.tol,
                ncases=3 if args.dim == 2 else 6,
                iter_table=it_tab, time_table=t_tab,
                iter_plot=os.path.basename(iters_path),
                time_plot=os.path.basename(time_path)))
        print(f"wrote {os.path.abspath(args.doc_out)}", file=sys.stderr)


if __name__ == "__main__":
    main()
