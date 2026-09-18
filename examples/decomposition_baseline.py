"""
What the FFT engine's decomposition costs, independently of the FFT.

Replacing the reference preconditioner's fine-grid FFT by a V-cycle removes two
things at once, and they are easy to confuse:

1. the **all-to-all** transposes, four per apply, each a full barrier; and
2. the **decomposition** the FFT engine imposes on the whole solve.

The second is the one that gets overlooked. ``muGrid.FFTEngine`` is not just a
``CartesianDecomposition`` with transforms bolted on — it also decides how the
domain is split, and its split is a *slab*: ``[1, 1, P]``, with only the last
axis ever distributed. Every halo exchange in the matvec pays for that, whether
or not a transform is ever applied. Two consequences:

- **Surface-to-volume degrades faster.** A 3D split shrinks all three extents
  together; a slab shrinks one. At 64³ on 8 ranks a Cartesian subdomain is
  32x32x32 and a slab is 64x64x8, and the slab exchanges about 1.7x the halo.
- **There is a hard rank ceiling.** A slab cannot use more ranks than the grid
  has planes: 64 for a 64³ grid, and long before that each slab is a few planes
  thick and the halo is comparable to the interior. A 3D split at the same rank
  count is nowhere near its limit.

So a benchmark that swaps the preconditioner alone cannot say which of the two
effects it moved. This script separates them by holding the preconditioner
fixed at ``-P none`` and varying only the decomposition, via
``homogenization.py --decomposition``. Whatever difference appears is
attributable to the split alone, because no transform runs in either arm.

That makes this the baseline every later multigrid-versus-FFT number is read
against: the V-cycle's win over the FFT preconditioner is the *product* of the
two effects, and this measures one of them on its own.

It needs no GPU. The effect is a property of how the domain is divided among
ranks, so CPU ranks measure it directly, and a single device cannot measure it
at all.

Examples
--------
The baseline sweep::

    python examples/decomposition_baseline.py -n 128 -p 1 2 4 8

With a different grid and rank ladder::

    python examples/decomposition_baseline.py -n 96 -p 1 2 3 6 --maxiter 10
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
HOMOGENIZATION = os.path.join(HERE, "homogenization.py")


def find_timer(node, path):
    """Look up a slash-separated path in muTimer's ``to_dict()`` tree."""
    timers = node["timers"] if "timers" in node else node.get("children", [])
    head, _, tail = path.partition("/")
    for timer in timers:
        if timer["name"] == head:
            return find_timer(timer, tail) if tail else timer
    return None


def per_call(node, path):
    """Seconds per call at ``path``, or None if the timer was never hit."""
    timer = find_timer(node, path)
    if timer is None or not timer["calls"]:
        return None
    return timer["total_seconds"] / timer["calls"]


def halo_points(nb_subdomain_grid_pts):
    """Points in a one-deep halo around a subdomain.

    The shell between the subdomain grown by one in every direction and the
    subdomain itself. This is geometry, not a measurement, and is printed
    beside the timings so an anomaly can be checked against what the split
    makes unavoidable.
    """
    interior = 1
    grown = 1
    for extent in nb_subdomain_grid_pts:
        interior *= extent
        grown *= extent + 2
    return grown - interior


def run(nb_grid_pts, nb_ranks, decomposition, args):
    """One ``-P none`` solve at a given rank count and split."""
    grid = ",".join(str(n) for n in nb_grid_pts)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        out = handle.name
    cmd = [*args.mpirun.split(), "-n", str(nb_ranks)]
    if args.oversubscribe:
        cmd.append("--oversubscribe")
    cmd += [args.python, HOMOGENIZATION,
            "-n", grid,
            "-d", args.device,
            "-k", args.kernel,
            "-P", "none",
            "--decomposition", decomposition,
            "--maxiter", str(args.maxiter),
            "--precision", args.precision,
            "--json-out", out]
    if args.device != "cpu":
        cmd.append("--sync-timers")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-12:]
            raise RuntimeError(f"{' '.join(cmd)}\nexited {proc.returncode}:\n"
                               + "\n".join(tail))
        with open(out) as stream:
            return json.load(stream)
    finally:
        if os.path.exists(out):
            os.unlink(out)


def probe(nb_grid_pts, nb_ranks, decomposition, args):
    result = run(nb_grid_pts, nb_ranks, decomposition, args)
    config, timing = result["config"], result["timing"]
    it = "total_solve/iteration"
    row = {
        "nb_ranks": nb_ranks,
        "decomposition": decomposition,
        "nb_subdivisions": config["nb_subdivisions"],
        "nb_subdomain_grid_pts": config["nb_subdomain_grid_pts"],
        "halo_points": halo_points(config["nb_subdomain_grid_pts"]),
        "t_matvec": per_call(timing, f"{it}/hessp/apply_stiffness"),
        "t_ghost": per_call(
            timing, f"{it}/hessp/apply_stiffness/communicate_ghosts"),
        "E_eff": result["results"]["E_effective_approx"],
    }
    if row["t_matvec"] and row["t_ghost"]:
        row["halo_share"] = row["t_ghost"] / row["t_matvec"]
    return row


def report(rows, nb_grid_pts, args):
    grid = "x".join(str(n) for n in nb_grid_pts)
    by_ranks = {}
    for row in rows:
        by_ranks.setdefault(row["nb_ranks"], {})[row["decomposition"]] = row

    serial = {row["decomposition"]: row for row in rows if row["nb_ranks"] == 1}

    print()
    print(f"-P none on {grid}, matvec cost by decomposition "
          f"(milliseconds per apply)")
    print(f"{'ranks':>6} {'split':>10} {'subdivisions':>14} {'subdomain':>16} "
          f"{'halo pts':>10} {'matvec':>9} {'halo':>9} {'halo %':>7} "
          f"{'speedup':>8}")
    for nb_ranks in sorted(by_ranks):
        for decomposition in ("cartesian", "fft"):
            row = by_ranks[nb_ranks].get(decomposition)
            if row is None:
                continue
            base = serial.get(decomposition, {}).get("t_matvec")
            speedup = (base / row["t_matvec"]
                       if base and row["t_matvec"] else None)
            print(f"{nb_ranks:>6} {decomposition:>10} "
                  f"{'x'.join(str(x) for x in row['nb_subdivisions']):>14} "
                  f"{'x'.join(str(x) for x in row['nb_subdomain_grid_pts']):>16} "
                  f"{row['halo_points']:>10,} "
                  f"{row['t_matvec'] * 1e3:>9.2f} "
                  f"{row['t_ghost'] * 1e3:>9.2f} "
                  f"{row.get('halo_share', 0) * 100:>6.1f}% "
                  + (f"{speedup:>8.2f}" if speedup else f"{'-':>8}"))

    print()
    print("Slab penalty: FFT-engine split relative to a genuine 3D split")
    print(f"{'ranks':>6} {'halo pts':>10} {'matvec':>9} {'halo':>9}")
    for nb_ranks in sorted(by_ranks):
        pair = by_ranks[nb_ranks]
        if len(pair) < 2:
            continue
        cart, fft = pair["cartesian"], pair["fft"]
        print(f"{nb_ranks:>6} "
              f"{fft['halo_points'] / cart['halo_points']:>9.2f}x "
              f"{fft['t_matvec'] / cart['t_matvec']:>8.2f}x "
              + (f"{fft['t_ghost'] / cart['t_ghost']:>8.2f}x"
                 if cart["t_ghost"] else f"{'-':>9}"))

    ceiling = min(nb_grid_pts)
    print()
    print(f"The slab's rank ceiling on this grid is {ceiling} "
          f"(one plane per rank); a 3D split's is {ceiling ** len(nb_grid_pts):,}.")
    print("Both arms ran -P none, so no transform executed in either: the")
    print("difference above is the decomposition alone, and a V-cycle that")
    print("also removes the all-to-all wins this on top of that.")

    values = {round(row["E_eff"], 9) for row in rows}
    print(f"Physics agrees across every run: E_eff = {values.pop():.6f}"
          if len(values) == 1 else
          f"WARNING: runs disagree on E_eff: {sorted(values)}")


def main():
    parser = argparse.ArgumentParser(
        prog="decomposition_baseline",
        description="Cost of the FFT engine's slab decomposition, measured "
                    "with the preconditioner held fixed at none.")
    parser.add_argument(
        "-n", "--nb-grid-pts", default="128",
        help="Grid: a bare integer N means N,N,N (default: 128)")
    parser.add_argument(
        "-p", "--nb-ranks", nargs="+", type=int, default=[1, 2, 4, 8],
        help="Rank counts to sweep (default: 1 2 4 8)")
    parser.add_argument(
        "-d", "--device", choices=["cpu", "gpu"], default="cpu",
        help="Device (default: cpu). The effect is a property of the domain "
             "split, so CPU ranks measure it directly")
    parser.add_argument(
        "-k", "--kernel", choices=["fused", "generic"], default="fused",
        help="Matvec kernel (default: fused)")
    parser.add_argument(
        "-i", "--maxiter", type=int, default=20,
        help="CG iterations per strain case. This measures per-apply cost, "
             "not convergence, so a small number is enough (default: 20)")
    parser.add_argument(
        "--precision", choices=["double", "single"], default="double",
        help="Solver precision (default: double)")
    parser.add_argument(
        "--mpirun", default="mpirun",
        help="Launcher, split on spaces (default: mpirun)")
    parser.add_argument(
        "--oversubscribe", action="store_true",
        help="Pass --oversubscribe, for rank counts above the core count")
    parser.add_argument(
        "--python", default=sys.executable,
        help="Interpreter for the child runs (default: this one)")
    parser.add_argument(
        "--json-out", default=None, metavar="FILE",
        help="Also write the collected rows to FILE as JSON")
    args = parser.parse_args()

    spec = args.nb_grid_pts
    nb_grid_pts = ([int(x) for x in spec.split(",")] if "," in spec
                   else [int(spec)] * 3)

    rows = []
    for nb_ranks in args.nb_ranks:
        for decomposition in ("cartesian", "fft"):
            print(f"running {nb_ranks} rank(s), {decomposition} ...",
                  flush=True)
            rows.append(probe(nb_grid_pts, nb_ranks, decomposition, args))

    report(rows, nb_grid_pts, args)

    if args.json_out:
        with open(args.json_out, "w") as stream:
            json.dump({"config": vars(args), "nb_grid_pts": nb_grid_pts,
                       "rows": rows}, stream, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
