#!/usr/bin/env python3
"""
Where a V-cycle and a setup of the multigrid spend their time, operation by
operation, for muGrid's MultigridPreconditioner and for the muFFTTO
reference, on the problem of ``compare_muffto.py`` (whose problem set-up this
script reuses, with the same command-line options).

    python profile_vcycle.py --method mugrid --grid 128 --precision single \\
        --device gpu --muffto <muFFTTO checkout> --data <CH3D folder>

Every operation of the cycle is wrapped: the device is synchronised before
and after it, so each is charged with exactly its own kernels (the total of
the synchronised cycle is therefore a little above the plain one, which is
printed too). Categories: the operator applies per level, the Jacobi
scaling, the vector updates (copy, axpy, axpby, scal), the grid transfers,
precision casts and the coarsest solve; the rest (ghost exchanges called by
the cycle itself, Python) is "other". Then the setup is run ``--setups``
times (the material does not change; the first run includes one-off costs
such as kernel compilation), synchronised.

Prints one JSON line (``--out`` appends it to a file).
"""
import collections
import json
import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# the problem and the options of compare_muffto.py (its main() is not run);
# our own options are taken off the command line first
own = {"--setups": 3, "--cycles": 20}
argv = []
iterator = iter(sys.argv[1:])
for token in iterator:
    if token in own:
        own[token] = int(next(iterator))
    else:
        argv.append(token)
sys.argv = [str(HERE / "compare_muffto.py")] + argv
cmp = runpy.run_path(str(HERE / "compare_muffto.py"), run_name="compare_muffto")
args = cmp["args"]

xp = None
if args.device == "gpu":
    import cupy as xp


def sync():
    if xp is not None:
        xp.cuda.Device().synchronize()


times = collections.defaultdict(float)
calls = collections.Counter()
active = {"on": False}


def timed(category, function):
    """``function`` charged to ``category`` while ``active['on']``."""
    def wrapper(*a, **k):
        if not active["on"]:
            return function(*a, **k)
        sync()
        start = time.perf_counter()
        result = function(*a, **k)
        sync()
        name = category(*a) if callable(category) else category
        times[name] += time.perf_counter() - start
        calls[name] += 1
        return result
    return wrapper


s = cmp["material_scale"]()
if args.method == "mugrid":
    import muGrid.Multigrid as module

    setup = cmp["setup_mugrid"](s)
    mg = setup["prec"]
    module._Level.apply = timed(
        lambda level, *a: f"apply L{level.index}"
        + (" (operator)" if level.operator is not None else " (stencil)"),
        module._Level.apply)
    module.MultigridPreconditioner._jacobi = staticmethod(timed(
        "jacobi", module.MultigridPreconditioner._jacobi))
    module.MultigridPreconditioner._solve_coarsest = timed(
        "coarsest", module.MultigridPreconditioner._solve_coarsest)
    module._copy_cast = timed("precision casts", module._copy_cast)
    linalg = module.linalg
else:
    setup = cmp["setup_muffto"](s)
    from muFFTTO import multigrid as module

    mg = setup["setup"].__self__
    module.Level.apply_K = timed(
        lambda level, *a: f"apply L{level.index}"
        + (" (fused)" if getattr(level, "fused", False) else ""),
        module.Level.apply_K)
    module.Multigrid._jacobi = staticmethod(timed(
        "jacobi", module.Multigrid._jacobi))
    module.Multigrid._solve_coarsest = timed(
        "coarsest", module.Multigrid._solve_coarsest)
    module.Multigrid._convert = staticmethod(timed(
        "precision casts", module.Multigrid._convert))
    linalg = module.linalg
for name in ("copy", "axpy", "axpby", "scal"):
    setattr(linalg, name, timed("vector updates", getattr(linalg, name)))
mg.transfer.restrict = timed("transfer", mg.transfer.restrict)
mg.transfer.prolong = timed("transfer", mg.transfer.prolong)

# setups, synchronised (the material is unchanged)
setup_seconds = []
for _ in range(own["--setups"]):
    sync()
    start = time.perf_counter()
    setup["setup"]()
    sync()
    setup_seconds.append(time.perf_counter() - start)

# V-cycles: plain, then operation by operation
r, z = setup["rhs"], setup["vector"]("profile-z")
apply = setup["prec"]
for _ in range(3):
    apply(r, z)
sync()
start = time.perf_counter()
for _ in range(own["--cycles"]):
    apply(r, z)
sync()
plain = (time.perf_counter() - start) / own["--cycles"]

active["on"] = True
sync()
start = time.perf_counter()
for _ in range(own["--cycles"]):
    apply(r, z)
sync()
total = (time.perf_counter() - start) / own["--cycles"]
active["on"] = False

per_cycle = {name: 1e3 * seconds / own["--cycles"]
             for name, seconds in sorted(times.items())}
per_cycle["other"] = 1e3 * total - sum(per_cycle.values())
record = {
    "method": args.method, "N": args.grid, "precision": args.precision,
    "device": args.device, "vcycle_ms": 1e3 * plain,
    "vcycle_synchronised_ms": 1e3 * total, "per_cycle_ms": per_cycle,
    "calls_per_cycle": {name: n // own["--cycles"]
                        for name, n in sorted(calls.items())},
    "setup_s": setup_seconds}
line = json.dumps(record)
print(line)
if args.out:
    with open(args.out, "a") as f:
        f.write(line + "\n")
