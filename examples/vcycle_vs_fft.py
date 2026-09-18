"""
V-cycle vs. FFT cost of the reference-stiffness preconditioner, on one device.

Both preconditioners in ``examples/homogenization.py`` apply the *same* operator
``M⁻¹ ≈ (Kʳᵉᶠ)⁻¹``; they differ only in how. ``-P reference`` does it exactly with
one forward FFT, a per-mode block solve and one inverse FFT. ``-P multigrid``
does it approximately with a V-cycle: damped-Jacobi smoothing and halo exchange
on every level, and an exact FFT solve only on the coarsest grid.

The quantity that decides whether the V-cycle is worth having is

    R = cost of one V-cycle apply / cost of one FFT apply      (on one device)

measured on a *single* device, where the FFT communicates nothing at all and is
therefore at its strongest. R > 1 is expected and is not a failure: it is the
arithmetic deficit the FFT's all-to-all has to make up as ranks are added. The
V-cycle's cost is halo-only and scales with rank count; the FFT's two all-to-all
transposes per transform (four per apply) do not. Knowing R at P = 1 predicts the
crossover instead of waiting to observe it.

Three ways to get R, depending on what the device can run:

``model``
    R is inferred from the matvec, without running a V-cycle at all. A V-cycle's
    arithmetic is dominated by ``apply_uniform`` matvecs — ``ν`` pre-smoothing,
    ``ν`` post-smoothing and one residual per level — and each level is
    ``2^-dim`` the size of the one above, so in fine-grid matvec units a cycle
    costs ``W = (2ν+1) · Σ_ℓ 2^(-dim·ℓ)``, which for ν = 2 in 3D is ≈ 5.71. Then
    ``R_model = W · t(apply_stiffness) / t(prec)``. This needs only a ``-P
    reference`` run, so it works wherever the fused matvec works — in particular
    on the GPU, where the V-cycle itself cannot yet run.

``parts`` (``--measure-parts``, GPU)
    The cycle is priced from its pieces, each timed on the device: the
    smoothing steps, the residual, the transfers and the coarse solve. This
    captures what the model omits — above all the smoother's vector
    operations, which on a bandwidth-bound device are not a rounding error.
    It exists because the V-cycle cannot be run end to end on the GPU:
    ``GridTransfer`` declares host-space overloads only (see
    ``src/libmugrid/operators/transfer.hh``), so ``restrict`` and ``prolong``
    have no device kernels. Those two terms are therefore charged by the
    traffic they move rather than measured; they come to a few percent of the
    cycle, and the table prints R with and without them.

``measured`` (``--measure``, CPU)
    ``R_meas = t(vcycle) / t(prec_fft)``, from an additional ``-P multigrid``
    run. The real number, end to end, available wherever the V-cycle runs —
    today that means the CPU. Use it to check the other two: the script prints
    ``inflation``, the factor by which the matvec-only model understates the
    true cycle.

``--inflation`` applies such a factor to a modelled R by hand, for carrying a
correction from one device to another. Prefer ``--measure-parts`` on the GPU:
it measures the same effect there instead of assuming it transfers.

R is a per-apply ratio, and the V-cycle buys its cheaper apply with more CG
iterations. ``--measure`` gets that penalty directly; ``--iteration-ratio``
supplies one measured elsewhere, which is legitimate because iteration counts
are a property of the operator and not of the device. Either way the script
prints ``R_eff``, the ratio a whole solve actually pays.

All timings are taken from the ``iteration`` phase only. The ``startup`` phase
carries kernel JIT and FFT plan creation and would otherwise dominate.

On the GPU the child runs get ``--sync-timers``, without which the whole
exercise is void: kernel launches are asynchronous, so an unsynchronised
host-side timer around a region that only *launches* work measures the launch.
The totals stay right, because the next dot product pulls a scalar back to the
host and synchronises, but the breakdown does not -- the work is charged to
whichever region is open at that point. It shows up as an FFT apply that stops
growing with the grid, or shrinks. ``--no-sync`` reproduces that if you want to
see it.

Examples
--------
The CPU sweep, which measures the real cycle and the iteration penalty::

    python examples/vcycle_vs_fft.py -n 64 128 -d cpu --measure

The GPU sweep, pricing the cycle from its parts and carrying the iteration
penalty over from that CPU run::

    python examples/vcycle_vs_fft.py -n 128 192 256 -d gpu \
        --measure-parts --iteration-ratio 1.50
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
HOMOGENIZATION = os.path.join(HERE, "homogenization.py")

# Defaults of MultigridReferencePreconditioner; the work estimate has to match
# the cycle that would actually run, so keep these in step with that class.
DEFAULT_NU = 2
DEFAULT_MIN_COARSE = 32


def resolve_nb_levels(nb_grid_pts, min_coarse=DEFAULT_MIN_COARSE):
    """Levels the V-cycle would use: coarsen while every direction stays even
    and at least ``min_coarse`` wide. Mirrors
    ``MultigridReferencePreconditioner._resolve_nb_levels``.
    """
    nb_levels = 1
    while (all(n % (2 ** nb_levels) == 0 for n in nb_grid_pts) and
           all(n // (2 ** nb_levels) >= min_coarse for n in nb_grid_pts)):
        nb_levels += 1
    return nb_levels


def vcycle_work(nb_levels, dim, nu=DEFAULT_NU):
    """V-cycle cost in fine-grid matvec units.

    Every level but the coarsest costs ``2ν + 1`` matvecs (ν pre-smoothing, ν
    post-smoothing, one residual) at ``2^(-dim·ℓ)`` of the fine-grid size. The
    coarsest level is an FFT solve, not a matvec, and is excluded.
    """
    return sum((2 * nu + 1) * 2.0 ** (-dim * lvl)
               for lvl in range(nb_levels - 1))


def measure_parts(nb_grid_pts, lam_ref, mu_ref, nu, min_coarse, repeats):
    """Time a GPU V-cycle from its parts, since it cannot be run end to end.

    Everything a cycle does is measurable on the device today except the two
    grid transfers, whose kernels are host-only (``GridTransfer`` in
    ``src/libmugrid/operators/transfer.hh`` declares host-space overloads only).
    So the cycle is priced piece by piece, with the transfers charged by the
    traffic they move: ``restrict`` reads a fine field and writes a coarse one,
    ``prolong`` the reverse, and both are single-pass stencils, so a
    ``linalg.copy`` of the fine field at that level is a fair stand-in for
    either. It is a proxy, not a measurement -- the row is reported separately
    so it can be read with that in mind.

    Returns a dict of per-cycle seconds, broken down by term.
    """
    import time

    import cupy as cp

    import muGrid
    from muGrid import linalg
    from muGrid.Preconditioners import MultigridReferencePreconditioner

    muGrid.route_cupy_through_mugrid()
    dim = len(nb_grid_pts)
    comm = muGrid.Communicator()
    decomposition = muGrid.CartesianDecomposition(
        comm, list(nb_grid_pts), nb_subdivisions=[1] * dim,
        nb_ghosts_left=(1,) * dim, nb_ghosts_right=(1,) * dim, device="gpu")
    prec = MultigridReferencePreconditioner(
        decomposition, [1.0 / n for n in nb_grid_pts], lam_ref, mu_ref,
        communicator=comm, nu=nu, min_coarse=min_coarse)

    sync = cp.cuda.runtime.deviceSynchronize

    def bench(call):
        call()
        sync()
        start = time.perf_counter()
        for _ in range(repeats):
            call()
        sync()
        return (time.perf_counter() - start) / repeats

    smoothing = residual = transfer = 0.0
    for lvl, level in enumerate(prec.levels[:-1]):
        # 2 nu smoothing steps: each is one apply plus the Jacobi update.
        smoothing += 2 * nu * bench(lambda l=level: l.smooth(1, prec.omega))
        # The residual: one apply, one axpby, one halo exchange before the
        # restriction.
        residual += bench(lambda l=level: l.apply(l.z, l.t))
        residual += bench(lambda l=level: linalg.axpby(1.0, l.r, -1.0, l.t))
        residual += bench(lambda l=level: l.decomp.communicate_ghosts(l.t))
        # restrict down, prolong back up, and the correction axpy.
        transfer += 2 * bench(lambda l=level: linalg.copy(l.r, l.t))
        transfer += bench(lambda l=level: linalg.axpy(1.0, l.t, l.z))

    bottom = prec.levels[-1]
    coarse = bench(lambda: prec._coarse_prec.apply(bottom.r, bottom.z))

    return {
        "smoothing": smoothing,
        "residual": residual,
        "transfer_proxy": transfer,
        "coarse": coarse,
        "cycle": smoothing + residual + transfer + coarse,
        "cycle_without_transfer": smoothing + residual + coarse,
        "nb_levels": prec.nb_levels,
    }


def find_timer(node, path):
    """Look up a slash-separated path in muTimer's ``to_dict()`` tree.

    Returns the node dict, or None if any path component is missing.
    """
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


def run(nb_grid_pts, device, preconditioner, kernel, precision, tol, maxiter,
        python=sys.executable, extra=()):
    """Run one homogenization solve and return its parsed JSON results."""
    grid = ",".join(str(n) for n in nb_grid_pts)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        out = handle.name
    cmd = [python, HOMOGENIZATION,
           "-n", grid,
           "-d", device,
           "-k", kernel,
           "-P", preconditioner,
           "--precision", precision,
           "--tol", str(tol),
           "--maxiter", str(maxiter),
           "--json-out", out,
           *extra]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-12:]
            raise RuntimeError(
                f"{' '.join(cmd)}\nexited {proc.returncode}:\n"
                + "\n".join(tail))
        with open(out) as stream:
            return json.load(stream)
    finally:
        if os.path.exists(out):
            os.unlink(out)


def probe(nb_grid_pts, device, args):
    """Time both preconditioner routes at one grid size."""
    dim = len(nb_grid_pts)
    nb_levels = resolve_nb_levels(nb_grid_pts, args.min_coarse)
    work = vcycle_work(nb_levels, dim, args.nu)

    extra = ("--sync-timers",) if args.sync else ()
    reference = run(nb_grid_pts, device, "reference", args.kernel,
                    args.precision, args.tol, args.maxiter, args.python, extra)
    timing = reference["timing"]

    it = "total_solve/iteration"
    row = {
        "nb_grid_pts": nb_grid_pts,
        "nb_levels": nb_levels,
        "work": work,
        "iterations_reference": reference["results"]["total_cg_iterations"],
        # One fused matvec, as the V-cycle's smoother would pay for it: halo
        # exchange plus apply. This is the unit `work` counts.
        "t_matvec": per_call(timing, f"{it}/hessp/apply_stiffness"),
        "t_kernel": per_call(timing, f"{it}/hessp/apply_stiffness/fused_kernel"),
        "t_ghost": per_call(
            timing, f"{it}/hessp/apply_stiffness/communicate_ghosts"),
        # The FFT apply as the solver actually pays for it, including whatever
        # the fft/scale/ifft children do not cover.
        "t_prec": per_call(timing, f"{it}/prec"),
        "t_fft": per_call(timing, f"{it}/prec/fft"),
        "t_scale": per_call(timing, f"{it}/prec/scale"),
        "t_ifft": per_call(timing, f"{it}/prec/ifft"),
    }
    row["R_model"] = (row["work"] * row["t_matvec"] / row["t_prec"]
                      if row["t_matvec"] and row["t_prec"] else None)

    if args.measure:
        multigrid = run(nb_grid_pts, device, "multigrid", args.kernel,
                        args.precision, args.tol, args.maxiter, args.python,
                        extra)
        mg_timing = multigrid["timing"]
        row["iterations_multigrid"] = multigrid["results"][
            "total_cg_iterations"]
        row["t_vcycle"] = per_call(mg_timing, f"{it}/prec/vcycle")
        row["t_coarse"] = per_call(mg_timing, f"{it}/prec/vcycle/coarse")
        row["R_meas"] = (row["t_vcycle"] / row["t_prec"]
                         if row["t_vcycle"] and row["t_prec"] else None)
        row["inflation"] = (row["R_meas"] / row["R_model"]
                            if row["R_meas"] and row["R_model"] else None)

    if args.inflation is not None and row["R_model"] is not None:
        row["R_corrected"] = row["R_model"] * args.inflation

    # R is a per-apply ratio, but the V-cycle is an *approximate* inverse and
    # buys its cheaper apply with more CG iterations. What the solve actually
    # pays is R times that penalty, so carry the ratio through wherever it is
    # known -- measured here by --measure, or supplied by --iteration-ratio
    # from a run on another device (iteration counts do not depend on one).
    penalty = args.iteration_ratio
    if penalty is None and row.get("iterations_multigrid"):
        penalty = row["iterations_multigrid"] / row["iterations_reference"]
    row["iteration_penalty"] = penalty

    # The uniform reference material, recovered from the run's own config. The
    # example takes the volume mean of the per-pixel Lame fields, and those are
    # linear in the phase indicator, so the mean is the phase-weighted mean of
    # the two constituents -- no need to rebuild the microstructure.
    config = reference["config"]
    poisson, v_f = config["nu"], config["volume_fraction"]

    def lame(young):
        return (young * poisson / ((1 + poisson) * (1 - 2 * poisson)),
                young / (2 * (1 + poisson)))

    lam_m, mu_m = lame(config["E_matrix"])
    lam_i, mu_i = lame(config["E_inclusion"])
    row["lambda_ref"] = (1 - v_f) * lam_m + v_f * lam_i
    row["mu_ref"] = (1 - v_f) * mu_m + v_f * mu_i

    return row


def micro(seconds):
    return "      -" if seconds is None else f"{seconds * 1e6:7.1f}"


def ratio(value):
    return "    -" if value is None else f"{value:5.2f}"


def report(rows, args):
    print()
    print("Per-apply timings, iteration phase only (microseconds)")
    print(f"{'grid':>16} {'lvl':>4} {'W':>6} {'matvec':>8} {'kernel':>8} "
          f"{'ghost':>8} {'fft':>8} {'scale':>8} {'ifft':>8} {'prec':>8} "
          f"{'vcycle':>8}")
    for row in rows:
        grid = "x".join(str(n) for n in row["nb_grid_pts"])
        print(f"{grid:>16} {row['nb_levels']:>4} {row['work']:>6.2f} "
              f"{micro(row['t_matvec'])} {micro(row['t_kernel'])} "
              f"{micro(row['t_ghost'])} {micro(row['t_fft'])} "
              f"{micro(row['t_scale'])} {micro(row['t_ifft'])} "
              f"{micro(row['t_prec'])} {micro(row.get('t_vcycle'))}")

    print()
    print("R = V-cycle apply / FFT apply")
    header = f"{'grid':>16} {'R_model':>9}"
    if args.measure:
        header += f" {'R_meas':>9} {'inflation':>10}"
    if args.inflation is not None:
        header += f" {'R_corr':>9}"
    if args.measure_parts:
        header += f" {'R_parts':>9}"
    header += f" {'it(fft)':>9}"
    if args.measure:
        header += f" {'it(mg)':>9}"
    print(header)
    for row in rows:
        grid = "x".join(str(n) for n in row["nb_grid_pts"])
        line = f"{grid:>16} {ratio(row['R_model']):>9}"
        if args.measure:
            line += f" {ratio(row.get('R_meas')):>9}"
            line += f" {ratio(row.get('inflation')):>10}"
        if args.inflation is not None:
            line += f" {ratio(row.get('R_corrected')):>9}"
        if args.measure_parts:
            line += f" {ratio(row.get('R_parts')):>9}"
        line += f" {row['iterations_reference']:>9}"
        if args.measure:
            line += f" {row.get('iterations_multigrid', '-'):>9}"
        print(line)

    penalised = [row for row in rows if row.get("iteration_penalty")]
    if penalised:
        print()
        print("R_eff = R x (V-cycle iterations / FFT iterations): what a whole "
              "solve pays")
        print(f"{'grid':>16} {'penalty':>9} {'R_eff(model)':>13} "
              f"{'R_eff(parts)':>13}")
        for row in penalised:
            grid = "x".join(str(n) for n in row["nb_grid_pts"])
            penalty = row["iteration_penalty"]
            best = {key: (row[key] * penalty if row.get(key) else None)
                    for key in ("R_model", "R_parts")}
            print(f"{grid:>16} {penalty:>9.2f} "
                  f"{ratio(best['R_model']):>13} {ratio(best['R_parts']):>13}")

    if any("parts" in row for row in rows):
        print()
        print("V-cycle priced from its parts on the device (microseconds per "
              "cycle)")
        print(f"{'grid':>16} {'smoothing':>10} {'residual':>10} "
              f"{'transfer*':>10} {'coarse':>10} {'cycle':>10} {'prec':>10} "
              f"{'R_parts':>9}")
        for row in rows:
            parts = row.get("parts")
            if parts is None:
                continue
            grid = "x".join(str(n) for n in row["nb_grid_pts"])
            print(f"{grid:>16} {micro(parts['smoothing']):>10} "
                  f"{micro(parts['residual']):>10} "
                  f"{micro(parts['transfer_proxy']):>10} "
                  f"{micro(parts['coarse']):>10} {micro(parts['cycle']):>10} "
                  f"{micro(row['t_prec']):>10} {ratio(row.get('R_parts')):>9}")
        print("* transfer is a traffic proxy (linalg.copy), not a measurement: "
              "restrict/prolong")
        print("  have no device kernels yet. R_parts excluding it: "
              + ", ".join(
                  f"{'x'.join(str(n) for n in row['nb_grid_pts'])} "
                  f"{row['R_parts_no_transfer']:.2f}"
                  for row in rows if row.get("R_parts_no_transfer")))

    print()
    print("Reading R: 1-2 wins at modest rank counts; 3-5 needs the FFT to")
    print("degrade 3-5x, which it does by roughly 8-32 ranks; above 8 the flop")
    print("cost is too steep to be recovered by removing the all-to-all.")
    print("Iteration counts are the other half of the trade: R is a per-apply")
    print("ratio, so a V-cycle that needs more iterations pays R times that.")


def main():
    parser = argparse.ArgumentParser(
        prog="vcycle_vs_fft",
        description="Cost of a V-cycle apply relative to an FFT apply, on one "
                    "device.")
    parser.add_argument(
        "-n", "--nb-grid-pts", nargs="+", default=["128", "192", "256"],
        help="Grid sizes. A bare integer N means a cubic N,N,N grid; a "
             "comma-separated list is used verbatim (default: 128 192 256)")
    parser.add_argument(
        "-d", "--device", choices=["cpu", "gpu"], default="gpu",
        help="Device for both runs (default: gpu)")
    parser.add_argument(
        "-k", "--kernel", choices=["fused", "generic"], default="fused",
        help="Matvec kernel (default: fused)")
    parser.add_argument(
        "--precision", choices=["double", "single"], default="double",
        help="Solver precision (default: double)")
    parser.add_argument(
        "-t", "--tol", type=float, default=1e-6,
        help="Relative CG tolerance (default: 1e-6)")
    parser.add_argument(
        "-i", "--maxiter", type=int, default=500,
        help="Maximum CG iterations (default: 500)")
    parser.add_argument(
        "--measure", action="store_true",
        help="Also run '-P multigrid' and time the real V-cycle. Requires the "
             "V-cycle to be available on the chosen device (CPU only for now)")
    parser.add_argument(
        "--inflation", type=float, default=None,
        help="Multiply the modelled R by this factor, to carry a "
             "CPU-measured model correction over to a device where the "
             "V-cycle cannot yet run")
    parser.add_argument(
        "--nu", type=int, default=DEFAULT_NU,
        help=f"Smoothing steps per level, for the work estimate "
             f"(default: {DEFAULT_NU})")
    parser.add_argument(
        "--min-coarse", type=int, default=DEFAULT_MIN_COARSE,
        help=f"Coarsest grid the cycle targets, for the level count "
             f"(default: {DEFAULT_MIN_COARSE})")
    parser.add_argument(
        "--iteration-ratio", type=float, default=None,
        help="V-cycle CG iterations divided by FFT CG iterations, when "
             "measured elsewhere. Iteration counts are a property of the "
             "operator, not of the device, so a CPU measurement transfers to "
             "the GPU unchanged. Implied by --measure")
    parser.add_argument(
        "--measure-parts", action="store_true",
        help="Price the V-cycle from its parts on the device: smoothing, "
             "residual, transfers and the coarse solve, each timed directly. "
             "Gives a GPU R without needing an end-to-end V-cycle, which the "
             "missing device grid-transfer kernels currently rule out. GPU "
             "only")
    parser.add_argument(
        "--repeats", type=int, default=20,
        help="Timed repetitions per part in --measure-parts (default: 20)")
    parser.add_argument(
        "--no-sync", dest="sync", action="store_false",
        help="Do not pass --sync-timers to the child runs. The GPU breakdown "
             "is then launch-time only and R is meaningless; this exists to "
             "show that, not to be used")
    parser.add_argument(
        "--python", default=sys.executable,
        help="Interpreter used for the child runs (default: this one)")
    parser.add_argument(
        "--json-out", default=None, metavar="FILE",
        help="Also write the collected rows to FILE as JSON")
    args = parser.parse_args()

    grids = []
    for spec in args.nb_grid_pts:
        if "," in spec:
            grids.append([int(x) for x in spec.split(",")])
        else:
            grids.append([int(spec)] * 3)

    rows = []
    for nb_grid_pts in grids:
        grid = "x".join(str(n) for n in nb_grid_pts)
        print(f"running {grid} on {args.device} ...", flush=True)
        rows.append(probe(nb_grid_pts, args.device, args))

    # Deliberately after every child run: this one builds the hierarchy in
    # *this* process, and holding a fine-grid hierarchy on the device would
    # take memory away from the runs still to come.
    if args.measure_parts:
        if args.device != "gpu":
            parser.error("--measure-parts is for the GPU; on the CPU run "
                         "--measure, which times the real cycle")
        for row in rows:
            grid = "x".join(str(n) for n in row["nb_grid_pts"])
            print(f"pricing the {grid} cycle from its parts ...", flush=True)
            parts = measure_parts(
                row["nb_grid_pts"], row["lambda_ref"], row["mu_ref"],
                args.nu, args.min_coarse, args.repeats)
            row["parts"] = parts
            if row["t_prec"]:
                row["R_parts"] = parts["cycle"] / row["t_prec"]
                row["R_parts_no_transfer"] = (
                    parts["cycle_without_transfer"] / row["t_prec"])

    report(rows, args)

    if args.json_out:
        with open(args.json_out, "w") as stream:
            json.dump({"config": vars(args), "rows": rows}, stream, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
