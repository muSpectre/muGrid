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
BENCHMARK = "homogenization_preconditioner"


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
    if _OOM_PATTERNS.search(out.stderr) or _OOM_PATTERNS.search(out.stdout):
        return True
    return out.returncode == -9  # killed by the OS OOM killer (SIGKILL)


def _setup_seconds(d):
    """Preconditioner setup time from the example's timer tree (or None)."""
    for t in d.get("timing", {}).get("timers", []):
        if t.get("name") == "preconditioner_setup":
            return float(t["total_seconds"])
    return None


def run(device, precond, n, dim, maxiter, tol, nranks=1, precision="double",
        prec_precision=None, scale_file=None):
    """Run one solve; dict of metrics, ``db.OOM`` if out of memory, else ``None``.

    The child writes its JSON to a private file (``--json-out``) that we read
    back, so stray stdout (MPI/UCX banners, warnings) cannot corrupt the parse.
    `prec_precision` runs the preconditioner in its own precision; `scale_file`
    replaces the single inclusion by a per-voxel material scale (phase field).
    """
    grid = ",".join([str(n)] * dim)
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "result.json")
        base = [HOMOG, "-n", grid, "-d", device, "-k", "fused", "-P", precond,
                "-i", str(maxiter), "--tol", str(tol), "--precision", precision,
                "--json-out", out_path]
        if scale_file:
            base += ["--inclusion-type", "file", "--scale-file", scale_file]
        else:
            base += ["--inclusion-type", "single"]
        if prec_precision:
            base += ["--preconditioner-precision", prec_precision]
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
            sys.stderr.write(f"  [{device} {precond} n={n} ranks={nranks}] "
                             f"no JSON\n{out.stderr[-400:]}\n")
            return None
    r, c = d["results"], d["config"]
    return dict(npts=c["nb_grid_pts_total"], iters=r["total_cg_iterations"],
                secs=r["total_time_seconds"], setup=_setup_seconds(d))


def _resolve(fields, runner, prov, args):
    """Resolve one point through the cache; return (fields, result, from_cache)."""
    result, cached = db.cached_point(fields, prov, runner,
                                     cache_dir=args.cache_dir, force=args.force)
    return fields, result, cached


def collect_iterations(args, prov):
    """Study `iterations`: CG count of each preconditioner of
    `--iter-preconditioners`, on a single CPU core.

    Every point is a single core, so the whole (size x preconditioner) grid is
    resolved through a worker pool of `--jobs` (each point cached as it lands)."""
    tasks = []
    for n in args.iter_sizes:
        for P in args.iter_preconditioners:
            fields = {"benchmark": BENCHMARK, "study": "iterations", "label": P,
                      "device": "cpu", "nranks": 1, "dim": args.dim, "n": n,
                      "precond": P, "precision": args.precision,
                      "maxiter": args.maxiter, "tol": args.tol}
            tasks.append(fields)

    def work(fields):
        return _resolve(
            fields,
            lambda: run("cpu", fields["precond"], fields["n"], args.dim,
                        args.maxiter, args.tol, precision=args.precision),
            prov, args)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        results = list(ex.map(work, tasks))

    rows = []
    for fields, r, cached in results:
        P, n = fields["precond"], fields["n"]
        tag = " [cached]" if cached else ""
        if r is None or r == OOM:
            sys.stderr.write(f"  iter {P} {n}^{args.dim}: skipped{tag}\n")
            continue
        rows.append({**prov, **fields, "npts": r["npts"],
                     "iters": r["iters"], "secs": r["secs"],
                     "setup_secs": r.get("setup")})
        sys.stderr.write(f"  iter {P:9s} {n}^{args.dim} ({r['npts']} pts): "
                         f"{r['iters']:5d} it, {r['secs']:.3f} s{tag}\n")
    return rows


def collect_timing(args, prov):
    """Study `reference_timing`: preconditioned solve time across device/MPI
    configs, for each preconditioner of `--timing-preconditioners`.

    Each (config, preconditioner) sweeps grid sizes up to the config's cap (a
    single CPU core tops out far sooner than the full node / the GPUs) and
    stops at the first out-of-memory size, which is recorded as an "oom" point
    (flagged in the table, dropped from the plot); larger sizes are not
    attempted. `--configs` restricts the configs (one job per node type).
    """
    _, nb_gpus = db.detect_gpu()
    want_gpu = not args.no_gpu and nb_gpus >= 1
    configs = db.plan_configs(args.mpi_cpu_ranks, nb_gpus, want_gpu)
    if args.configs:
        configs = [c for c in configs if c[0] in args.configs]
    rows = []

    def fields_for(key, device, nranks, n, P):
        return {"benchmark": BENCHMARK, "study": "reference_timing",
                "label": key, "device": device, "nranks": nranks, "dim": args.dim,
                "n": n, "precond": P, "precision": args.precision,
                "maxiter": args.maxiter, "tol": args.tol}

    def resolve(key, device, nranks, n, P):
        f = fields_for(key, device, nranks, n, P)
        return _resolve(
            f, lambda: run(device, P, n, args.dim, args.maxiter,
                           args.tol, nranks, args.precision), prov, args)

    for key, device, nranks in configs:
        label = db.CONFIG_META[key]["label"](nranks)
        cap = args.cpu1_max_size if key == "cpu1" else args.max_size
        sizes = [s for s in args.sizes if s <= cap]
        for P in args.timing_preconditioners:
            # cpu1 is the only node-safe config to parallelise (one core per
            # point, capped below OOM); the rest use the whole node/GPU per
            # point and keep the first-OOM early stop.
            points = []
            if key == "cpu1" and args.jobs > 1:
                with concurrent.futures.ThreadPoolExecutor(
                        max_workers=args.jobs) as ex:
                    points = sorted(
                        ex.map(lambda n: resolve(key, device, nranks, n, P),
                               sizes),
                        key=lambda t: t[0]["n"])
            else:
                for n in sizes:
                    point = resolve(key, device, nranks, n, P)
                    points.append(point)
                    if point[1] == OOM:
                        break
            for f, r, cached in points:
                n = f["n"]
                base = {**prov, **f, "npts": n ** args.dim}
                tag = " [cached]" if cached else ""
                if r is None:
                    sys.stderr.write(f"  time {label} {P} {n}^{args.dim}: "
                                     "skipped\n")
                    continue
                if r == OOM:
                    rows.append({**base, "status": "oom"})
                    sys.stderr.write(f"  time {label} {P} {n}^{args.dim}: OUT "
                                     f"OF MEMORY{tag}\n")
                    continue
                rows.append({**base, "npts": r["npts"], "status": "ok",
                             "iters": r["iters"], "secs": r["secs"],
                             "setup_secs": r.get("setup")})
                sys.stderr.write(f"  time {label} {P} {n}^{args.dim} "
                                 f"({r['npts']} pts): {r['iters']} it, "
                                 f"{r['secs']:.3f} s{tag}\n")
    return rows


def phase_field_scale(args, n):
    """Material scale s per voxel of the high-contrast phase field, written to
    an .npy file (cached) for `homogenization.py --scale-file`.

    The Cahn-Hilliard geometry `--phase-field-geometry` (1 = stiff) is
    block-averaged (or repeated) to n^3 and thresholded; phi = (1 + tanh(d /
    2 eta)) / 2 from the periodic signed distance d in pixels, on the nodes;
    per voxel the mean of its 8 corners phi_v and s = phi_v^2 (1 - k) + k with
    k = 1 / contrast (as muFFTTO's scaling_case_3d.py)."""
    k = 1.0 / args.phase_field_contrast
    eta = args.phase_field_eta
    tag = (f"{os.path.splitext(os.path.basename(args.phase_field_geometry))[0]}"
           f"_N{n}_eta{eta:g}_c{args.phase_field_contrast:g}")
    path = os.path.join(args.cache_dir, "phase_field", f"s_{tag}.npy")
    if os.path.exists(path):
        return path
    import numpy as np
    from scipy import ndimage
    phase = np.load(args.phase_field_geometry).astype(float)
    m = phase.shape[0]
    if n < m:
        f = m // n
        solid = phase.reshape(n, f, n, f, n, f).mean(axis=(1, 3, 5)) > 0.5
    elif n == m:
        solid = phase > 0.5
    else:
        solid = np.kron(phase > 0.5, np.ones((n // m,) * 3, dtype=bool))
    big = np.tile(solid, (3, 3, 3))
    window = (slice(n, 2 * n),) * 3
    d_in = ndimage.distance_transform_edt(big)[window]
    d_out = ndimage.distance_transform_edt(~big)[window]
    del big
    distance = np.where(solid, d_in - 0.5, -(d_out - 0.5))
    phi = 0.5 * (1 + np.tanh(distance / (2 * eta)))
    corners = sum(np.roll(phi, shift=(-a, -b, -c), axis=(0, 1, 2))
                  for a in (0, 1) for b in (0, 1) for c in (0, 1)) / 8
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, corners ** 2 * (1 - k) + k)
    return path


def collect_phase_field(args, prov):
    """Study `phase_field`: the high-contrast phase field on one device (GPU
    when available), each preconditioner of `--phase-field-preconditioners`
    in `--phase-field-precision` between double-precision CG vectors."""
    if args.dim != 3:
        sys.stderr.write("  phase field: 3D only, skipped\n")
        return []
    if not args.phase_field_geometry:
        sys.exit("--studies phase_field needs --phase-field-geometry")
    _, nb_gpus = db.detect_gpu()
    device = "gpu" if (not args.no_gpu and nb_gpus >= 1) else "cpu"
    material = (f"{os.path.basename(args.phase_field_geometry)}"
                f"|eta={args.phase_field_eta:g}"
                f"|contrast={args.phase_field_contrast:g}")
    rows = []
    for n in args.phase_field_sizes:
        scale_file = phase_field_scale(args, n)
        for P in args.phase_field_preconditioners:
            fields = {"benchmark": BENCHMARK, "study": "phase_field",
                      "label": P, "device": device, "nranks": 1, "dim": 3,
                      "n": n, "precond": P, "precision": "double",
                      "prec_precision": args.phase_field_precision,
                      "maxiter": args.maxiter, "tol": args.tol,
                      "material": material}
            f, r, cached = _resolve(
                fields, lambda: run(device, P, n, 3, args.maxiter, args.tol, 1,
                                    "double", args.phase_field_precision,
                                    scale_file), prov, args)
            tag = " [cached]" if cached else ""
            base = {**prov, **f, "npts": n ** 3}
            if r is None:
                sys.stderr.write(f"  phase {P} {n}^3: skipped{tag}\n")
                continue
            if r == OOM:
                rows.append({**base, "status": "oom"})
                sys.stderr.write(f"  phase {P} {n}^3: OUT OF MEMORY{tag}\n")
                continue
            rows.append({**base, "npts": r["npts"], "status": "ok",
                         "iters": r["iters"], "secs": r["secs"],
                         "setup_secs": r.get("setup")})
            sys.stderr.write(f"  phase {P} {n}^3: {r['iters']} it, "
                             f"{r['secs']:.3f} s (setup {r.get('setup')}){tag}\n")
    return rows


def collect(args, prov):
    """Run the selected studies (`args.studies`); return DB rows (no write)."""
    rows = []
    if "iterations" in args.studies:
        rows += collect_iterations(args, prov)
    if "reference_timing" in args.studies:
        rows += collect_timing(args, prov)
    if "phase_field" in args.studies:
        rows += collect_phase_field(args, prov)
    return rows


# --------------------------------------------------------------------------- #
# Re-shaping DB rows for rendering
# --------------------------------------------------------------------------- #
#: Preconditioners in display order, their names and plot styles.
PRECOND_ORDER = ["none", "reference", "green-jacobi", "multigrid", "hybrid",
                 "heterogeneous-multigrid"]
PRECOND_NAME = {"none": "none", "reference": "reference (Green)",
                "green-jacobi": "Green-Jacobi",
                "multigrid": "reference multigrid", "hybrid": "hybrid",
                "heterogeneous-multigrid": "heterogeneous multigrid"}
PRECOND_STYLE = {"none": dict(marker="o", color="#c62828", linestyle="-"),
                 "reference": dict(marker="s", color="#00897b", linestyle="-"),
                 "green-jacobi": dict(marker="^", color="#f9a825",
                                      linestyle="-."),
                 "multigrid": dict(marker="v", color="#6d4c41", linestyle=":"),
                 "hybrid": dict(marker="P", color="#8e24aa", linestyle=":"),
                 "heterogeneous-multigrid": dict(marker="D", color="#1565c0",
                                                 linestyle="--")}


def preconds_in(*dicts):
    """Preconditioners present as keys of any of the given dicts, in order."""
    present = set()
    for d in dicts:
        present.update(d)
    return [P for P in PRECOND_ORDER if P in present]


def fmt_points(npts):
    if npts >= 1e6:
        return f"{npts / 1e6:.1f}M"
    if npts >= 1e3:
        return f"{npts / 1e3:.0f}k"
    return str(int(npts))


def sup(dim):
    return "²" if dim == 2 else "³"


def iters_from_rows(rows):
    """{precision: {precond: {n: row}}} from the iterations study."""
    d = {}
    for r in rows:
        if r["study"] == "iterations":
            d.setdefault(db.norm_precision(r), {}) \
             .setdefault(r["label"], {})[r["n"]] = r
    return d


def timing_from_rows(rows, study="reference_timing"):
    """{precision: {(config_key, precond): {n: row}}} from the timing study
    (rows of older runs carry no precond: they are the reference)."""
    d = {}
    for r in rows:
        if r["study"] == study:
            d.setdefault(db.norm_precision(r), {}) \
             .setdefault((r["label"], r.get("precond") or "reference"),
                         {})[r["n"]] = r
    return d


def precisions_in(merged):
    """Precisions present in a {precision: ...} dict, in canonical order."""
    return [p for p in db.PRECISIONS if p in merged]


def sizes_in(rows, study):
    return sorted({r["n"] for r in rows if r["study"] == study})


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def iteration_table(sizes, iters, dim):
    precisions = precisions_in(iters)
    show_prec = len(precisions) > 1
    Ps = preconds_in(*(iters.get(p, {}) for p in precisions))
    cols = [n for n in sizes
            if any(n in iters.get(p, {}).get(P, {})
                   for p in precisions for P in Ps)]
    head = "| Preconditioner | " + " | ".join(
        f"{n}{sup(dim)} ({fmt_points(n ** dim)})" for n in cols) + " |"
    lines = [head, "|" + "---|" * (len(cols) + 1)]
    for prec in precisions:
        pi = iters.get(prec, {})
        suffix = f", {db.PRECISION_LABEL[prec]}" if show_prec else ""
        for P in Ps:
            cells = " | ".join(
                (str(pi[P][n]["iters"]) if n in pi.get(P, {}) else "—")
                for n in cols)
            lines.append(f"| {PRECOND_NAME[P]}{suffix} | {cells} |")
        for P in Ps:
            if P == "none":
                continue
            cells = []
            for n in cols:
                a = pi.get("none", {}).get(n)
                b = pi.get(P, {}).get(n)
                cells.append(f"{a['secs'] / b['secs']:.0f}×" if a and b
                             else "—")
            if any(c != "—" for c in cells):
                lines.append(f"| **CPU-core speedup, {PRECOND_NAME[P]}"
                             f"{suffix}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _cell(res, n, field="secs"):
    """One table cell: a time, 'OOM' for an out-of-memory point, else '—'."""
    if n not in res:
        return "—"
    secs = res[n].get(field)
    if isinstance(secs, (int, float)):
        return f"{secs:.3g}"
    return "OOM" if res[n].get("status") == "oom" else "—"


def timing_table(sizes, configs, timing, dim, field="secs"):
    """Rows (config, preconditioner[, precision]) x grid sizes of `field`
    (`secs`: solve time, `setup_secs`: preconditioner setup)."""
    precisions = precisions_in(timing)
    show_prec = len(precisions) > 1
    keys = set()
    for p in precisions:
        keys.update(timing.get(p, {}))
    Ps = [P for P in PRECOND_ORDER if any(k[1] == P for k in keys)]
    cols = [n for n in sizes
            if any(n in timing.get(p, {}).get((k, P), {})
                   for p in precisions for k, *_ in configs for P in Ps)]
    head = ("| Configuration | Preconditioner | "
            + " | ".join(f"{n}{sup(dim)} ({fmt_points(n ** dim)})" for n in cols)
            + " |")
    lines = [head, "|" + "---|" * (len(cols) + 2)]
    for key, label, _style in configs:
        for P in Ps:
            for prec in precisions:
                res = timing.get(prec, {}).get((key, P))
                if not res or (field != "secs" and not any(
                        isinstance(r.get(field), (int, float))
                        for r in res.values())):
                    continue
                row_label = (f"{label}, {db.PRECISION_LABEL[prec]}"
                             if show_prec else label)
                cells = " | ".join(_cell(res, n, field) for n in cols)
                lines.append(f"| {row_label} | {PRECOND_NAME[P]} | {cells} |")
    return "\n".join(lines)


def phase_field_table(rows):
    """Preconditioners x grid sizes of the phase-field study: CG iterations,
    solve time and setup time per cell."""
    pf = [r for r in rows if r["study"] == "phase_field"]
    if not pf:
        return ""
    sizes = sorted({r["n"] for r in pf})
    by = {}
    for r in pf:
        by.setdefault(r["precond"], {})[r["n"]] = r
    Ps = preconds_in(by)
    head = "| Preconditioner | " + " | ".join(
        f"{n}³ ({fmt_points(n ** 3)})" for n in sizes) + " |"
    lines = [head, "|" + "---|" * (len(sizes) + 1)]
    for P in Ps:
        cells = []
        for n in sizes:
            r = by[P].get(n)
            if r is None:
                cells.append("—")
            elif r.get("status") == "oom":
                cells.append("OOM")
            else:
                setup = r.get("setup_secs")
                setup = (f" + {setup:.3g} s setup"
                         if isinstance(setup, (int, float)) else "")
                cells.append(f"{r['iters']} it, {r['secs']:.3g} s{setup}")
        lines.append(f"| {PRECOND_NAME[P]} | " + " | ".join(cells) + " |")
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
    precisions = precisions_in(iters)
    show_prec = len(precisions) > 1
    for prec in precisions:
        for P in preconds_in(iters.get(prec, {})):
            pts = sorted((n, r["iters"])
                         for n, r in iters.get(prec, {}).get(P, {}).items())
            if not pts:
                continue
            xs, ys = zip(*pts)
            all_n.update(xs)
            leg = (f"{PRECOND_NAME[P]}, {db.PRECISION_LABEL[prec]}"
                   if show_prec else PRECOND_NAME[P])
            ax.loglog(xs, ys, label=leg,
                      **{**PRECOND_STYLE[P], **db.PRECISION_STYLE[prec]})
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
    precisions = precisions_in(timing)
    show_prec = len(precisions) > 1
    keys = set()
    for p in precisions:
        keys.update(timing.get(p, {}))
    Ps = [P for P in PRECOND_ORDER if any(k[1] == P for k in keys)]
    for key, label, style in configs:
        for P in Ps:
            for prec in precisions:
                res = timing.get(prec, {}).get((key, P), {})
                # Only points with a real measured time; OOM carries none.
                pts = sorted((n, r["secs"]) for n, r in res.items()
                             if isinstance(r.get("secs"), (int, float)))
                if not pts:
                    continue
                xs, ys = zip(*pts)
                all_n.update(xs)
                leg = f"{label}, {PRECOND_NAME[P]}"
                if show_prec:
                    leg += f", {db.PRECISION_LABEL[prec]}"
                line = {**style, "marker": PRECOND_STYLE[P]["marker"],
                        "linestyle": PRECOND_STYLE[P]["linestyle"]}
                ax.loglog(xs, ys, label=leg,
                          **{**line, **(db.PRECISION_STYLE[prec]
                                        if show_prec else {})})
    db.set_grid_size_xaxis(ax, all_n, dim)
    ax.set_ylabel("Preconditioned solve time (s)")
    ax.set_title(f"Homogenization ({dim}D, fused): solve time vs. grid size")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Doc page
# --------------------------------------------------------------------------- #
DOC_TEMPLATE = """# Benchmark: preconditioners

## What is measured

The preconditioners of [Preconditioners](linalg.md#preconditioners) on the
conjugate-gradient solves of the [homogenization example](examples.md#homogenization)
(`examples/homogenization.py`):
{precond_list}

Problem of studies 1 and 2: $N^{dim}$ grid points, single spherical
inclusion (radius 0.25, $E_\\text{{inclusion}}/E_\\text{{matrix}} = 10$,
$\\nu = 0.3$), fused `IsotropicStiffnessOperator{dim}D` matvec, the {ncases}
unit load cases, each solved to a relative residual of `{tol}`. Iteration
counts are summed over the load cases; times are the example's
`total_time_seconds` (all solves including right-hand-side assembly and stress
averaging, excluding the preconditioner setup, which is tabled separately).
One measurement per point.
{precision_note}

Studies:

1. **CG iterations vs. grid size** on one CPU core. The iteration count is a
   property of the operator and preconditioner, not of the device or
   decomposition. Without preconditioner $\\kappa(\\mathbf{{K}}) = O(h^{{-2}})$,
   so CG needs $O(N)$ iterations; with the reference preconditioner
   $\\kappa \\le c_\\text{{max}}/c_\\text{{min}}$ for material data bounded by
   multiples of the reference, independent of $N$.
2. **Preconditioned solve time vs. grid size** for the device and MPI
   configurations of the [homogenization benchmark](benchmark_homogenization.md).
   Each (configuration, preconditioner) is swept in $N$ until the first size
   that runs out of memory (`OOM`); larger sizes are not attempted.
{phase_field_intro}
## Provenance

{provenance}

## Commands

Each data point is one invocation of the example, launched by
`examples/benchmark_homogenization_preconditioner.py` as

```bash
{commands}
```

with `N` the grid size of the column and `--json-out FILE` added to collect the
result.

## CG iterations vs. grid size

{iter_table}

(preconditioner rows: CG iterations summed over the {ncases} load cases;
speedup rows: wall time without preconditioner ÷ wall time with it, one CPU
core)

![Iterations vs. grid size]({iter_plot})

## Solve time: device and MPI configurations

{timing_table}

(values are **solve time in seconds**, run to convergence; `OOM` = ran out of
memory)

![Solve time vs. grid size]({timing_plot})
{setup_section}{phase_field_section}
## Reproducing

All data points are stored in `{db_name}`. This page is generated
by `examples/benchmark_homogenization_preconditioner.py`; `--render-only`
re-renders it from the database, and a run without it measures and appends a
new dated run (see `benchmarks/README.md`):

```bash
python examples/benchmark_homogenization_preconditioner.py \\
    --doc-out {doc_name}
```
"""


PRECOND_DOC = {
    "none": "`none`: no preconditioner",
    "reference": ("`reference`: the Green's function of the volume-mean "
                  "reference material, applied by FFT (Ladecký et al., Appl. "
                  "Math. Comput. 446 (2023) 127835)"),
    "green-jacobi": ("`green-jacobi`: the reference preconditioner scaled "
                     "symmetrically by the Jacobi diagonal of the actual "
                     "operator (J-FFT)"),
    "multigrid": ("`multigrid`: a V-cycle approximating the reference "
                  "preconditioner (uniform reference operator)"),
    "hybrid": ("`hybrid`: the reference preconditioner by FFT in the "
               "rank-local axes and a block-tridiagonal solve in the "
               "distributed one"),
    "heterogeneous-multigrid": (
        "`heterogeneous-multigrid`: a V(2,2)-cycle on the heterogeneous "
        "operator itself (`muGrid.Multigrid`): Galerkin coarse operators "
        "probed from the operator, Chebyshev-Jacobi smoothing with a "
        "guaranteed eigenvalue bound, no FFT"),
}


def _precision_note(precisions):
    """One-line note describing the precision(s) shown."""
    if len(precisions) > 1:
        return ("\nBoth **double** (`fp64`, solid) and **single** (`fp32`, "
                "dashed) precision are shown; single precision halves the "
                "resident field memory and the bytes moved per iteration.")
    if precisions == ["single"]:
        return "\nAll runs are in **single** precision (`fp32`)."
    return "\nAll runs are in **double** precision (`fp64`)."


def _commands(rows, dim, precisions):
    """Command lines of the studies, one per configuration present."""
    prec = (precisions[0] if len(precisions) == 1
            else "{" + "|".join(precisions) + "}")
    grid = ",".join(["N"] * dim)
    lines = []

    def args(precond, r, extra="--inclusion-type single", precision=prec):
        return (f"-n {grid} -d {{device}} -k fused -P {precond} "
                f"-i {r['maxiter']} --tol {r['tol']} --precision {precision} "
                f"{extra}")

    it = [r for r in rows if r["study"] == "iterations"]
    if it:
        Ps = preconds_in({r["precond"] for r in it})
        lines.append("# CG iterations vs. grid size (one CPU core)")
        lines.append(db.config_command("homogenization.py",
                                       args("{" + "|".join(Ps) + "}", it[0]),
                                       "cpu", 1))
    for key, label, _style in db.render_configs(rows, "reference_timing"):
        cfg = [r for r in rows
               if r["study"] == "reference_timing" and r["label"] == key]
        Ps = preconds_in({r.get("precond") or "reference" for r in cfg})
        lines.append(f"# Solve time, {label}")
        lines.append(db.config_command(
            "homogenization.py",
            args(Ps[0] if len(Ps) == 1 else "{" + "|".join(Ps) + "}", cfg[0]),
            cfg[0]["device"], int(cfg[0]["nranks"])))
    pf = [r for r in rows if r["study"] == "phase_field"]
    if pf:
        Ps = preconds_in({r["precond"] for r in pf})
        r = pf[0]
        lines.append("# High-contrast phase field (scale file written by the "
                     "driver)")
        lines.append(db.config_command(
            "homogenization.py",
            args("{" + "|".join(Ps) + "}", r,
                 extra="--inclusion-type file --scale-file s_N.npy "
                       f"--preconditioner-precision {r.get('prec_precision')}",
                 precision="double"),
            r["device"], 1))
    return "\n".join(lines)


def write_doc_page(path, meta, dim, tol, ncases, iter_table, timing_table,
                   iter_plot, timing_plot, precisions, rows=(), setup_table="",
                   phase_table="", db_path=None):
    present = preconds_in({r.get("precond") or "reference" for r in rows})
    precond_list = "\n".join(f"- {PRECOND_DOC[P]}" for P in present)
    pf = [r for r in rows if r["study"] == "phase_field"]
    phase_intro = phase_section = ""
    if pf:
        r = pf[0]
        material = str(r.get("material") or "")
        phase_intro = (
            "3. **High-contrast phase field**: a Cahn-Hilliard geometry with "
            "a smooth tanh interface, material $s(x) C(E, \\nu)$ with "
            "$s = \\phi^2 (1 - k) + k$, contrast $1/k$ "
            f"(`{material}`), on one {r['device'].upper()}; the "
            f"preconditioners run in `{r.get('prec_precision')}` precision "
            "between double-precision CG vectors.\n")
        phase_section = (
            "\n## High-contrast phase field\n\n" + phase_table + "\n\n"
            f"(CG iterations summed over the {ncases} load cases, solve time "
            "and preconditioner setup time; relative residual "
            f"`{r['tol']}`)\n")
    setup_section = ""
    if setup_table:
        setup_section = ("\n## Preconditioner setup time\n\n" + setup_table
                         + "\n\n(seconds, once per material; excluded from "
                         "the solve times above)\n")
    with open(path, "w") as fh:
        fh.write(DOC_TEMPLATE.format(
            dim=dim, tol=tol, ncases=ncases,
            precision_note=_precision_note(precisions),
            precond_list=precond_list,
            phase_field_intro=phase_intro,
            provenance=db.provenance_markdown(
                rows, {"iterations": "CG iterations vs. grid size",
                       "reference_timing": "solve timing",
                       "phase_field": "high-contrast phase field"}),
            commands=_commands(rows, dim, precisions),
            iter_table=iter_table, timing_table=timing_table,
            iter_plot=os.path.basename(iter_plot),
            timing_plot=os.path.basename(timing_plot),
            setup_section=setup_section, phase_field_section=phase_section,
            db_name=os.path.relpath(db_path or db.DB_PATH,
                                    os.path.join(HERE, "..")),
            doc_name=os.path.relpath(os.path.abspath(path),
                                     os.path.join(HERE, ".."))))


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
    ap.add_argument("--iter-preconditioners", nargs="+",
                    choices=PRECOND_ORDER, default=["none", "reference"],
                    help="Preconditioners of the iteration study")
    ap.add_argument("--timing-preconditioners", nargs="+",
                    choices=PRECOND_ORDER[1:], default=["reference"],
                    help="Preconditioners of the solve-time study")
    ap.add_argument("--configs", nargs="+", choices=db.CONFIG_ORDER,
                    default=None,
                    help="Restrict the solve-time study to these configs "
                         "(e.g. one job per node type; default: all that the "
                         "machine has)")
    ap.add_argument("--phase-field-geometry", default=None,
                    help="Cahn-Hilliard geometry (.npy, 1 = stiff) of the "
                         "phase-field study")
    ap.add_argument("--phase-field-sizes", type=int, nargs="+",
                    default=[64, 128])
    ap.add_argument("--phase-field-preconditioners", nargs="+",
                    choices=PRECOND_ORDER[1:],
                    default=["reference", "green-jacobi",
                             "heterogeneous-multigrid"])
    ap.add_argument("--phase-field-eta", type=float, default=2.0,
                    help="Interface width in pixels")
    ap.add_argument("--phase-field-contrast", type=float, default=1e4)
    ap.add_argument("--phase-field-precision", choices=["single", "double"],
                    default="single",
                    help="Precision of the preconditioners of the phase-field "
                         "study (CG vectors stay double)")
    ap.add_argument("--mpi-cpu-ranks", type=int, default=os.cpu_count())
    ap.add_argument("--no-gpu", action="store_true", help="Skip the GPU curves")
    ap.add_argument("--precision", choices=db.PRECISIONS, default="double",
                    help="Floating-point precision to MEASURE this run in "
                         "(recorded per row). Rendering overlays whatever "
                         "precisions are in the DB — run once per precision to "
                         "get both curves (default: double)")
    ap.add_argument("--maxiter", type=int, default=20000)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--jobs", type=int, default=1,
                    help="Parallel worker processes for the single-CPU-core "
                         "points (the iteration study and the cpu1 timing sweep; "
                         "each point uses one core). MPI/GPU configs always run "
                         "serially, one per node/GPU (default: 1)")
    ap.add_argument("--cache-dir", default=db.CACHE_DIR,
                    help="Per-point result cache directory. Finished points are "
                         "replayed from here, so a killed run resumes cheaply")
    ap.add_argument("--force", action="store_true",
                    help="Recompute even on a cache hit (and refresh the cache)")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="Do not run: append a fresh run to the database from the "
                         "points already in the cache for the current commit. "
                         "Recovers a sweep that could not finish in one job")
    ap.add_argument("--studies", nargs="+",
                    choices=["iterations", "reference_timing", "phase_field"],
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

    if args.aggregate_only:
        prov = db.run_provenance()
        rows = db.cache_rows(BENCHMARK, args.studies, prov, args.cache_dir)
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

    # Take the latest run of EACH study independently and, within a study, of
    # EACH precision independently — so the page combines the iteration-count
    # and timing studies AND overlays the fp64 and fp32 curves, even though all
    # four are measured by separate jobs (separate submission scripts, separate
    # timestamps).
    rows = db.select_precisions(db.load(args.db), BENCHMARK,
                                ["iterations", "reference_timing",
                                 "phase_field"], select_ts)
    if not rows:
        sys.exit("No matching rows in the database.")

    # Machine/version box: prefer the (heavier) timing run, most recent one.
    meta_rows = [r for r in rows if r["study"] == "reference_timing"] or rows
    meta = max(meta_rows, key=lambda r: r["timestamp"])
    meta = {k: meta[k] for k in ("cpu", "gpu", "version", "timestamp")}
    dim = next((r["dim"] for r in rows), args.dim)
    tol = next((r["tol"] for r in rows if r["study"] == "iterations"), args.tol)
    configs = db.render_configs(rows, "reference_timing")
    iters = iters_from_rows(rows)
    timing = timing_from_rows(rows)

    it_tab = iteration_table(sizes_in(rows, "iterations"), iters, dim)
    t_tab = timing_table(sizes_in(rows, "reference_timing"), configs, timing,
                         dim)
    s_tab = timing_table(sizes_in(rows, "reference_timing"), configs, timing,
                         dim, field="setup_secs")
    p_tab = phase_field_table(rows)
    print("\n" + it_tab + "\n\n" + t_tab + "\n\n" + s_tab + "\n\n" + p_tab)

    # Precisions present anywhere on the page (union of the two studies).
    precisions = [p for p in db.PRECISIONS
                  if p in iters or p in timing]

    iters_path = os.path.abspath(args.iter_plot)
    time_path = os.path.abspath(args.time_plot)
    make_iter_plot(iters, dim, iters_path)
    make_timing_plot(configs, timing, dim, time_path)
    sys.stderr.write(f"wrote {iters_path}\nwrote {time_path}\n")

    if args.doc_out:
        write_doc_page(args.doc_out, meta, dim, tol,
                       3 if dim == 2 else 6, it_tab, t_tab,
                       iters_path, time_path, precisions, rows,
                       setup_table=s_tab, phase_table=p_tab, db_path=args.db)
        sys.stderr.write(f"wrote {os.path.abspath(args.doc_out)}\n")


if __name__ == "__main__":
    main()
