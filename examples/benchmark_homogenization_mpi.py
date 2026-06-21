#!/usr/bin/env python3
"""MPI strong-scaling benchmark for the homogenization example (3D, fused).

For each grid size, runs `homogenization.py` with an increasing number of MPI
ranks and reports wall time, speedup, parallel efficiency, and the aggregate
memory throughput. Produces a strong-scaling plot (speedup vs. ranks) and the
Markdown tables used on the [homogenization benchmark](../docs/benchmark_homogenization.md)
page.

This needs an **MPI-enabled** muGrid build and `mpi4py`. Either install an
MPI-enabled muGrid, or point `PYTHONPATH` at an MPI build tree, e.g.

    cmake -S . -B build-mpi -DCMAKE_BUILD_TYPE=Release -DMUGRID_ENABLE_MPI=ON
    cmake --build build-mpi --target _muGrid
    export PYTHONPATH=$PWD/build-mpi/language_bindings/python:$PWD/language_bindings/python

then run:

    python examples/benchmark_homogenization_mpi.py \
        --plot-out docs/benchmark_homogenization_mpi.png

Use `--from-json results.json` to re-render the plot/tables from a previous run
without recomputing.
"""

import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HOMOG = os.path.join(HERE, "homogenization.py")


def run(nranks, n, maxiter):
    cmd = ["mpiexec", "-n", str(nranks), sys.executable, HOMOG,
           "-n", f"{n},{n},{n}", "-d", "cpu", "-k", "fused", "-i", str(maxiter),
           "--inclusion-type", "single", "--json"]
    env = dict(os.environ, OMPI_MCA_rmaps_base_oversubscribe="1")
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=7200,
                             env=env)
    except subprocess.SubprocessError:
        return None
    m = re.search(r"\{.*\}", out.stdout, re.DOTALL)
    if not m:
        sys.stderr.write(f"  [n={n} ranks={nranks}] no JSON\n{out.stderr[-400:]}\n")
        return None
    r = json.loads(m.group(0))["results"]
    return dict(secs=r["total_time_seconds"], iters=r["total_cg_iterations"],
                gbps=r.get("memory_throughput_GBps"),
                E=r.get("E_effective_approx"))


def tables_markdown(sizes, ranks, res):
    """One strong-scaling table per grid size."""
    out = []
    for n in sizes:
        if not res.get(str(n)):
            continue
        rows = res[str(n)]
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


def make_plot(sizes, ranks, res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    rmax = max(ranks)
    ax.plot([1, rmax], [1, rmax], ls="--", color="0.6", label="ideal (linear)")
    markers = {64: "o", 96: "s", 128: "^"}
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


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[64, 96])
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--maxiter", type=int, default=100)
    ap.add_argument("--from-json", default=None,
                    help="Render plot/tables from a saved results JSON instead "
                         "of running")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--plot-out",
                    default=os.path.join(HERE, "..", "docs",
                                         "benchmark_homogenization_mpi.png"))
    args = ap.parse_args()

    if args.from_json:
        with open(args.from_json) as fh:
            res = json.load(fh)
    else:
        res = {str(n): {} for n in args.sizes}
        for n in args.sizes:
            for R in args.ranks:
                r = run(R, n, args.maxiter)
                if r:
                    res[str(n)][str(R)] = r
                    sys.stderr.write(f"  {n}^3 ranks={R}: {r['secs']:.3f}s "
                                     f"{r['gbps']:.1f} GB/s E={r['E']:.6f}\n")
        if args.json_out:
            with open(args.json_out, "w") as fh:
                json.dump(res, fh, indent=2)

    print("\n" + tables_markdown(args.sizes, args.ranks, res))
    make_plot(args.sizes, args.ranks, res, os.path.abspath(args.plot_out))
    sys.stderr.write(f"wrote {os.path.abspath(args.plot_out)}\n")


if __name__ == "__main__":
    main()
