#!/usr/bin/env python3
"""Tiny git-friendly benchmark database shared by the benchmark scripts.

The database is a single append-only CSV at ``benchmarks/results.csv`` (one row
per measured data point, "long" format). Every benchmark run appends a fresh
batch of rows stamped with the date, code version, and machine, so the file is a
growing history that can be committed to git and diffed line-by-line.

Plot/page generation is fully separated from data collection: the benchmark
scripts read rows back from this CSV (by default the most recent run for the
benchmark in question) and render tables and plots from them — so a page can be
regenerated at any time without re-running anything, and old runs stay
reproducible.

Schema (columns)
----------------
Provenance (same for every row of one run):
  timestamp  ISO-8601 local time the run started
  version    `git describe --tags --always --dirty`
  commit     short git hash
  dirty      1 if the working tree had uncommitted changes, else 0
  cpu        CPU model + logical core count
  gpu        GPU description (or "none")
Identity of the data point:
  benchmark  e.g. "homogenization", "homogenization_preconditioner", "poisson"
  study      sub-plot within a benchmark, e.g. "time_vs_size", "mpi_scaling",
             "iterations", "reference_timing"
  label      series within a study, e.g. a config key ("cpu1"/"cpuN"/"gpu1"/
             "gpuN"), a rank count, or a preconditioner name
Run parameters:
  device nranks dim n npts precond maxiter tol
Results:
  iters secs gbps E
Unused columns for a given row are left blank.
"""

import csv
import datetime
import os
import platform
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DB_PATH = os.path.join(REPO_ROOT, "benchmarks", "results.csv")

FIELDS = [
    # provenance
    "timestamp", "version", "commit", "dirty", "cpu", "gpu",
    # identity
    "benchmark", "study", "label",
    # run parameters
    "device", "nranks", "dim", "n", "npts", "precond", "maxiter", "tol",
    # results
    "iters", "secs", "gbps", "E",
    # status: blank/"ok" for a normal point, "oom" for a run that ran out of
    # memory (recorded so the tables can flag it; left out of the plots)
    "status",
]


# --------------------------------------------------------------------------- #
# Provenance helpers
# --------------------------------------------------------------------------- #
def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT,
                              capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def git_version():
    return _git("describe", "--tags", "--always", "--dirty") or "unknown"


def git_commit():
    return _git("rev-parse", "--short", "HEAD") or "unknown"


def git_dirty():
    return 1 if _git("status", "--porcelain") else 0


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


def _label_from_names(names):
    uniq = sorted(set(names))
    return (f"{len(names)}x {uniq[0]}" if len(names) > 1 and len(uniq) == 1
            else ", ".join(names))


def _nvidia_gpu_names():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        if out.returncode != 0:  # stub nvidia-smi on a non-NVIDIA host prints
            return []             # its error to stdout, so check the exit code
        return [n.strip() for n in out.stdout.splitlines() if n.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


def _rocm_gpu_names():
    """AMD/ROCm device names (one per GPU), best-effort across tool versions."""
    # rocm-smi: lines look like "GPU[0]\t\t: Card series: Instinct MI300A".
    try:
        out = subprocess.run(["rocm-smi", "--showproductname"],
                             capture_output=True, text=True, timeout=30)
        by_idx = {}
        for line in out.stdout.splitlines():
            m = re.match(r"\s*GPU\[(\d+)\]", line)
            if not m:
                continue
            idx = int(m.group(1))
            if ":" in line and any(k in line for k in
                                   ("series", "model", "name", "Name")):
                by_idx[idx] = line.rsplit(":", 1)[1].strip()
            else:
                by_idx.setdefault(idx, "AMD GPU")
        if by_idx:
            return [by_idx[i] for i in sorted(by_idx)]
    except (OSError, subprocess.SubprocessError):
        pass
    # amd-smi fallback: count "GPU: <n>" stanzas.
    try:
        out = subprocess.run(["amd-smi", "list"],
                             capture_output=True, text=True, timeout=30)
        n = len(re.findall(r"^\s*GPU:\s*\d+", out.stdout, re.MULTILINE))
        if n:
            return ["AMD GPU"] * n
    except (OSError, subprocess.SubprocessError):
        pass
    return []


def detect_gpu():
    """(human-readable description, device count).

    Detection order: an explicit ``MUGRID_BENCH_GPU_COUNT`` override (set by batch
    scripts on hosts where the vendor CLI is unreliable — e.g. an AMD node that
    still has a stub ``nvidia-smi``), then NVIDIA (``nvidia-smi``), then AMD/ROCm
    (``rocm-smi`` / ``amd-smi``).
    """
    env_n = os.environ.get("MUGRID_BENCH_GPU_COUNT")
    if env_n is not None:
        try:
            n = int(env_n)
        except ValueError:
            n = 0
        if n <= 0:
            return "none", 0
        name = os.environ.get("MUGRID_BENCH_GPU_NAME", "GPU")
        return (f"{n}x {name}" if n > 1 else name), n

    names = _nvidia_gpu_names() or _rocm_gpu_names()
    if not names:
        return "none", 0
    return _label_from_names(names), len(names)


def now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def run_provenance(timestamp=None):
    """Provenance fields shared by every row of one run (captured once)."""
    gpu, _ = detect_gpu()
    return {
        "timestamp": timestamp or now_iso(),
        "version": git_version(),
        "commit": git_commit(),
        "dirty": git_dirty(),
        "cpu": detect_cpu(),
        "gpu": gpu,
    }


# --------------------------------------------------------------------------- #
# Device / MPI configuration vocabulary (shared by the device-comparison plots)
# --------------------------------------------------------------------------- #
# key -> plot style + legend-label template. The label is derived from the rank
# count stored with each data point, so historical runs render with the right
# core/device count regardless of the machine doing the rendering.
CONFIG_META = {
    "cpu1": dict(style=dict(marker="o", color="#5e35b1"),
                 label=lambda nr: "CPU (1 core)"),
    "cpuN": dict(style=dict(marker="D", color="#3949ab"),
                 label=lambda nr: f"CPU ({nr} cores, MPI)"),
    "gpu1": dict(style=dict(marker="s", color="#00897b"),
                 label=lambda nr: "GPU (1 device)"),
    "gpuN": dict(style=dict(marker="^", color="#f4511e"),
                 label=lambda nr: f"GPU ({nr} devices, MPI)"),
}
CONFIG_ORDER = ["cpu1", "cpuN", "gpu1", "gpuN"]


def plan_configs(ncores, nb_gpus, want_gpu):
    """Configs to RUN on this machine, FASTEST FIRST: list of (key, device,
    nranks).

    Ordered multi-GPU, single GPU, full-node MPI, single CPU core — so the
    quickest configurations are measured first and the slow single core (the one
    most likely to be cut off by a wall-clock limit) comes last. Rendering
    re-sorts by `CONFIG_ORDER`, so this only affects execution order.
    """
    cfgs = []
    if want_gpu and nb_gpus > 1:
        cfgs.append(("gpuN", "gpu", nb_gpus))
    if want_gpu and nb_gpus >= 1:
        cfgs.append(("gpu1", "gpu", 1))
    cfgs.append(("cpuN", "cpu", ncores))
    cfgs.append(("cpu1", "cpu", 1))
    return cfgs


def grid_label(n, dim):
    """Per-axis grid size formatted as geometry, e.g. 128 -> "128³" (3D)."""
    return f"{n}{'²' if dim == 2 else '³'}"


def set_grid_size_xaxis(ax, ns, dim):
    """Label the (log) x-axis with grid *geometry* ticks (e.g. 128³) at the
    measured per-axis grid sizes `ns`, instead of total grid-point counts."""
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

    ns = sorted(ns)
    ax.xaxis.set_major_locator(FixedLocator(ns))
    ax.xaxis.set_major_formatter(
        FixedFormatter([grid_label(n, dim) for n in ns]))
    ax.xaxis.set_minor_locator(NullLocator())
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45)
        lbl.set_horizontalalignment("right")
    ax.set_xlabel("Grid size")


def render_configs(rows, study):
    """Ordered (key, label, style) for the configs present in `study` rows."""
    nr = {}
    for r in rows:
        if r["study"] == study:
            nr.setdefault(r["label"], r["nranks"])
    out = []
    for key in CONFIG_ORDER:
        if key in nr:
            meta = CONFIG_META[key]
            out.append((key, meta["label"](nr[key]), meta["style"]))
    return out


# --------------------------------------------------------------------------- #
# CSV I/O
# --------------------------------------------------------------------------- #
def _migrate_header(path):
    """Rewrite an existing CSV in place if its header predates a FIELDS change
    (e.g. a column was appended), so old and new rows stay column-aligned."""
    if not os.path.exists(path):
        return
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return
        if header == FIELDS:
            return
        old_rows = list(csv.DictReader(fh, fieldnames=header))
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in old_rows:
            writer.writerow({k: r.get(k, "") for k in FIELDS})


def append_rows(rows, path=DB_PATH):
    """Append rows (list of dicts; subset of FIELDS) to the CSV, writing the
    header if the file does not exist yet."""
    if not rows:
        return
    is_new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not is_new:
        _migrate_header(path)
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in FIELDS})


def _num(v):
    """Parse a CSV string into int/float where possible, else keep as str."""
    if v == "" or v is None:
        return None
    try:
        i = int(v)
        return i
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def load(path=DB_PATH):
    """All rows as dicts with numeric fields coerced to int/float."""
    if not os.path.exists(path):
        return []
    numeric = {"dirty", "nranks", "dim", "n", "npts", "maxiter", "tol",
               "iters", "secs", "gbps", "E"}
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append({k: (_num(v) if k in numeric else v)
                         for k, v in r.items()})
    return rows


def runs_for(rows, benchmark):
    """Distinct run timestamps for a benchmark, newest first."""
    ts = {r["timestamp"] for r in rows if r["benchmark"] == benchmark}
    return sorted(ts, reverse=True)


def select(rows, benchmark, timestamp=None):
    """Rows for a benchmark from one run.

    timestamp=None selects the most recent run; otherwise the run whose
    timestamp starts with the given string (so a date like "2026-06-23" or a
    full timestamp both work).
    """
    sub = [r for r in rows if r["benchmark"] == benchmark]
    if not sub:
        return []
    if timestamp is None:
        target = max(r["timestamp"] for r in sub)
        return [r for r in sub if r["timestamp"] == target]
    return [r for r in sub if r["timestamp"].startswith(timestamp)]


def select_studies(rows, benchmark, studies, timestamp=None):
    """Rows for the given studies of a benchmark, each from its OWN latest run.

    Unlike `select`, which pins every row to a single run, this picks the most
    recent run *per study* independently (or, with `timestamp`, the run whose
    timestamp starts with the given string). A page that combines several
    studies (e.g. the preconditioner's iteration-count and timing studies) can
    therefore show the freshest data for each, even when the studies were
    measured by separate jobs that ran at different times.
    """
    out = []
    for study in studies:
        sub = [r for r in rows
               if r["benchmark"] == benchmark and r["study"] == study]
        if not sub:
            continue
        if timestamp is None:
            target = max(r["timestamp"] for r in sub)
            out.extend(r for r in sub if r["timestamp"] == target)
        else:
            out.extend(r for r in sub if r["timestamp"].startswith(timestamp))
    return out
