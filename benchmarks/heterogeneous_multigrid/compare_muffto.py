#!/usr/bin/env python3
"""
Cross-check of muGrid's heterogeneous multigrid against the muFFTTO reference.

One process solves one case:

    python compare_muffto.py --method mugrid --grid 64 --precision single --device gpu
    python compare_muffto.py --method muffto --grid 64 --precision single --device gpu

Problem (as muFFTTO's experiments/Multigrid_preconditioning/scaling_case_3d.py):
the Cahn-Hilliard geometry ``data/CH3D/ch3d_<geometry>.npy`` (128³, 1 = stiff),
block-averaged to N; a tanh interface of width eta (in pixels) from the periodic
signed distance, phi = (1 + tanh(d / 2 eta)) / 2 on the nodes; per voxel the
mean of its 8 corners phi_v and s = phi_v² (1 - k) + k with k = 1e-4 (contrast
1e4); material s C_0 (K = 1, mu = 1/2), trilinear hexahedra (the fused
IsotropicStiffnessOperator3D, one material value per voxel), load E_xx.

Both methods use the same plain PCG (muGrid.Solvers.conjugate_gradients, relative
residual ``--rtol``) on the same right-hand side; muFFTTO with its multigrid
(operator='fused', re-discretised coarse levels), muGrid with
MultigridPreconditioner (Galerkin coarse levels). Recorded: CG iterations, the
homogenised stress, setup time, time per V-cycle, CG time, and the GPU memory of
this process from ``nvidia-smi --query-compute-apps`` (peak and after the solve,
minus the CUDA context measured before anything is allocated).

The muFFTTO reference is imported from ``--muffto`` (read only); the geometry
is read from ``--data`` (muFFTTO's
``experiments/Multigrid_preconditioning/data/CH3D``).
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# An editable install's import redirector would win over PYTHONPATH; drop it so
# a build tree on PYTHONPATH (as in runtests.py) is what gets imported.
sys.meta_path = [f for f in sys.meta_path
                 if type(f).__name__ != "ScikitBuildRedirectingFinder"]

import numpy as np  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--method", required=True,
                    choices=["mugrid", "muffto", "green", "green_jacobi"],
                    help="mugrid: MultigridPreconditioner; muffto: the muFFTTO "
                         "multigrid; green / green_jacobi: muGrid's FFT reference-"
                         "material preconditioners (reference C_0, the stiff phase)")
parser.add_argument("--maxiter", type=int, default=2000)
parser.add_argument("--grid", type=int, required=True)
parser.add_argument("--precision", choices=["double", "single"], default="single")
parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
parser.add_argument("--eta", type=float, default=2.0, help="interface width / h")
parser.add_argument("--geometry", default="ch_s12")
parser.add_argument("--muffto", default=os.environ.get("MUFFTTO_PATH"),
                    help="muFFTTO checkout with muFFTTO/multigrid.py "
                         "(--method muffto only; default: $MUFFTTO_PATH)")
parser.add_argument("--data", default=os.environ.get("MUFFTTO_CH3D_DATA"),
                    help="directory with the Cahn-Hilliard geometries "
                         "ch3d_<geometry>.npy (default: $MUFFTTO_CH3D_DATA; "
                         "not needed once the material is cached)")
parser.add_argument("--rtol", type=float, default=1e-6)
parser.add_argument("--lmax", choices=["bound", "power"], default="bound",
                    help="muFFTTO reference: smoother bound or power estimate")
parser.add_argument("--repeat", type=int, default=20, help="V-cycles timed")
parser.add_argument("--cache", default=str(Path(__file__).resolve().parent / "cache"))
parser.add_argument("--out", default=None, help="append the result (JSON line) here")
args = parser.parse_args()
if args.method == "muffto" and not args.muffto:
    parser.error("--method muffto needs --muffto (or $MUFFTTO_PATH)")

N, K_SOFT = args.grid, 1e-4
LAMBDA_0, MU_0 = 1.0 - 2.0 / 3.0 * 0.5, 0.5        # K = 1, mu = 1/2
E_XX = [1.0, 0, 0, 0, 0, 0, 0, 0, 0]


# ------------------------------------------------- GPU memory of this process
class GpuMemory:
    """Peak and current memory of this process on the GPU (nvidia-smi)."""

    def __init__(self):
        self.pid = os.getpid()
        self.peak = 0.0
        self._stop = False
        self._thread = None

    def now(self):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True,
                timeout=10).stdout
        except (OSError, subprocess.TimeoutExpired):
            return float("nan")
        for line in out.splitlines():
            pid, used = (part.strip() for part in line.split(","))
            if int(pid) == self.pid:
                return float(used)
        return 0.0

    def start(self):
        def poll():
            while not self._stop:
                self.peak = max(self.peak, self.now())
                time.sleep(0.1)
        self._thread = threading.Thread(target=poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True
        self._thread.join()
        self.peak = max(self.peak, self.now())


# ------------------------------------------------------ geometry (host, cached)
def material_scale():
    """s per voxel, (N, N, N)."""
    cache = Path(args.cache) / f"s_{args.geometry}_N{N}_eta{args.eta:g}.npy"
    if cache.exists():
        return np.load(cache)
    if not args.data:
        sys.exit("the material is not cached: pass --data (or "
                 "$MUFFTTO_CH3D_DATA), the directory of ch3d_<geometry>.npy")
    from scipy import ndimage
    phase = np.load(Path(args.data) / f"ch3d_{args.geometry}.npy").astype(float)
    M = phase.shape[0]
    if N < M:
        f = M // N
        solid = phase.reshape(N, f, N, f, N, f).mean(axis=(1, 3, 5)) > 0.5
    elif N == M:
        solid = phase > 0.5
    else:
        solid = np.kron(phase > 0.5, np.ones((N // M,) * 3, dtype=bool))
    big = np.tile(solid, (3, 3, 3))
    window = (slice(N, 2 * N),) * 3
    d_in = ndimage.distance_transform_edt(big)[window]
    d_out = ndimage.distance_transform_edt(~big)[window]
    del big
    distance = np.where(solid, d_in - 0.5, -(d_out - 0.5))
    phi = 0.5 * (1 + np.tanh(distance / (2 * args.eta)))
    corners = sum(np.roll(phi, shift=(-a, -b, -c), axis=(0, 1, 2))
                  for a in (0, 1) for b in (0, 1) for c in (0, 1)) / 8
    s = corners ** 2 * (1 - K_SOFT) + K_SOFT
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, s)
    return s


# ------------------------------------------------------------ the two methods
def sync():
    if args.device == "gpu":
        import cupy
        cupy.cuda.Device().synchronize()


def to_device(values):
    if args.device == "gpu":
        import cupy
        return cupy.asarray(values)
    return values


def setup_mugrid(s):
    import muGrid
    from muGrid.Multigrid import MultigridPreconditioner, ScaledLocalMatrices
    device = muGrid.Device.gpu() if args.device == "gpu" else None
    kwargs = {"device": device} if device is not None else {}
    comm = muGrid.Communicator()
    decomposition = muGrid.CartesianDecomposition(
        comm, [N] * 3, nb_subdivisions=[1] * 3, nb_ghosts_left=[1] * 3,
        nb_ghosts_right=[1] * 3, **kwargs)
    op = muGrid.IsotropicStiffnessOperator3D([1.0 / N] * 3, muGrid.FEMElement.q1)
    scale = decomposition.real_field("scale")
    lam, mu = decomposition.real_field("lambda"), decomposition.real_field("mu")
    scale.s[0] = to_device(s)
    lam.s[0] = to_device(LAMBDA_0 * s)
    mu.s[0] = to_device(MU_0 * s)
    single = args.precision == "single"
    if single:
        lam32 = decomposition.real_field("lambda32", dtype=np.float32)
        mu32 = decomposition.real_field("mu32", dtype=np.float32)
        lam32.s[0] = to_device((LAMBDA_0 * s).astype(np.float32))
        mu32.s[0] = to_device((MU_0 * s).astype(np.float32))
    for field in [scale, lam, mu] + ([lam32, mu32] if single else []):
        decomposition.communicate_ghosts(field)

    def apply(u, f):
        decomposition.communicate_ghosts(u)
        op.apply(u, lam, mu, f)

    def apply_single(u, f):
        decomposition.communicate_ghosts(u)
        op.apply(u, lam32, mu32, f)

    t0 = time.perf_counter()
    mg = MultigridPreconditioner(
        apply, decomposition, 3,
        local_matrices=ScaledLocalMatrices.isotropic_stiffness(
            op, LAMBDA_0, MU_0, scale),
        precision=args.precision, operator_single=apply_single if single else None)
    t_build = time.perf_counter() - t0
    rhs = decomposition.real_field("rhs", (3,))
    op.apply_macro_rhs(lam, mu, E_XX, rhs)
    rhs.s[...] *= -1.0

    return dict(comm=comm, collection=decomposition.collection, hessp=apply, prec=mg,
                setup=mg.setup, rhs=rhs, x=decomposition.real_field("x", (3,)),
                vector=lambda name: decomposition.real_field(name, (3,)),
                stress=lambda x: op.average_stress(x, lam, mu, E_XX), t_build=t_build,
                levels=lambda: [(lvl.nb_grid_pts[0], lvl.bound)
                                for lvl in mg.levels[:-1]])


def setup_green(s, jacobi):
    """muGrid's Green (reference stiffness C_0, FFT) or Green-Jacobi
    preconditioner on an FFT engine, in the precision of ``--precision``
    between float64 CG vectors (as the multigrid's float32 cycle)."""
    import muGrid
    from muGrid.Preconditioners import (make_green_jacobi_preconditioner,
                                        make_reference_stiffness_preconditioner)
    device = muGrid.Device.gpu() if args.device == "gpu" else None
    kwargs = {"device": device} if device is not None else {}
    comm = muGrid.Communicator()
    engine = muGrid.FFTEngine([N] * 3, comm, nb_ghosts_left=[1] * 3,
                              nb_ghosts_right=[1] * 3, **kwargs)
    fc = engine.real_space_collection
    op = muGrid.IsotropicStiffnessOperator3D([1.0 / N] * 3, muGrid.FEMElement.q1)
    dtype = np.float32 if args.precision == "single" else np.float64
    lam, mu = fc.real_field("lambda"), fc.real_field("mu")
    lam.s[0] = to_device(LAMBDA_0 * s)
    mu.s[0] = to_device(MU_0 * s)
    lam_p = fc.real_field("lambda-prec", dtype=dtype)
    mu_p = fc.real_field("mu-prec", dtype=dtype)
    lam_p.s[0] = to_device((LAMBDA_0 * s).astype(dtype))
    mu_p.s[0] = to_device((MU_0 * s).astype(dtype))
    for field in (lam, mu, lam_p, mu_p):
        engine.communicate_ghosts(field)

    def apply(u, f):
        engine.communicate_ghosts(u)
        op.apply(u, lam, mu, f)

    t0 = time.perf_counter()
    if jacobi:
        prec = make_green_jacobi_preconditioner(
            engine, op, lam_p, mu_p, 3, reference_lambda=LAMBDA_0,
            reference_mu=MU_0, element=muGrid.FEMElement.q1,
            grid_spacing=[1.0 / N] * 3, dtype=dtype)
    else:
        prec = make_reference_stiffness_preconditioner(
            engine, nb_components=3, dtype=dtype, element=muGrid.FEMElement.q1,
            grid_spacing=[1.0 / N] * 3, lambda_ref=LAMBDA_0, mu_ref=MU_0)
    t_build = time.perf_counter() - t0
    if dtype == np.float32:
        r32 = engine.real_space_field("prec-r32", (3,), dtype=np.float32)
        z32 = engine.real_space_field("prec-z32", (3,), dtype=np.float32)

        def apply_prec(r, z):
            r32.sg[...] = r.sg
            prec(r32, z32)
            z.sg[...] = z32.sg
    else:
        apply_prec = prec
    rhs = engine.real_space_field("rhs", (3,))
    op.apply_macro_rhs(lam, mu, E_XX, rhs)
    rhs.s[...] *= -1.0
    return dict(comm=comm, collection=fc, hessp=apply, prec=apply_prec,
                setup=lambda: None, rhs=rhs, x=engine.real_space_field("x", (3,)),
                vector=lambda name: engine.real_space_field(name, (3,)),
                stress=lambda x: op.average_stress(x, lam, mu, E_XX),
                t_build=t_build, levels=lambda: [])


def setup_muffto(s):
    sys.path.insert(0, args.muffto)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from muFFTTO import domain, material_models, multigrid
    C_0 = material_models.get_elastic_material_tensor(dim=3, K=1.0, mu=0.5,
                                                      kind="linear")
    cell = domain.PeriodicUnitCell(domain_size=[1, 1, 1], problem_type="elasticity")
    disc = domain.Discretization(cell=cell, nb_of_pixels_global=(N,) * 3,
                                 discretization_type="finite_element",
                                 element_type="trilinear_hexahedron_1Q",
                                 device=args.device)
    t0 = time.perf_counter()
    # muFFTTO versions before the guaranteed bound have no ``lmax`` option
    # (they estimate it); pass it only where it exists
    import inspect
    options = ({"lmax": args.lmax} if "lmax" in inspect.signature(
        multigrid.Multigrid).parameters else {})
    mg = multigrid.Multigrid(disc, C_0, operator="fused", precision=args.precision,
                             **options)
    t_build = time.perf_counter() - t0
    mg.scale.s[0, 0] = to_device(s[None])
    fine = mg.fine_operator
    fine.update_material()
    rhs = disc.get_unknown_size_field(name="rhs")
    fine.operator.apply_macro_rhs(fine.lam, fine.mu, E_XX, rhs)
    rhs.s[...] *= -1.0
    return dict(comm=disc.communicator, collection=disc.field_collection,
                hessp=mg.apply_K, prec=mg.apply, setup=mg.setup, rhs=rhs,
                x=disc.get_unknown_size_field(name="x"),
                vector=lambda name: disc.get_unknown_size_field(name=name),
                stress=lambda x: fine.operator.average_stress(x, fine.lam, fine.mu,
                                                              E_XX),
                t_build=t_build,
                levels=lambda: [(lvl.disc.nb_of_pixels_global[0], lvl.lmax)
                                for lvl in mg.levels if lvl.lmax is not None])


# --------------------------------------------------------------------------- run
def main():
    memory = GpuMemory()
    if args.device == "gpu":
        import cupy
        cupy.zeros(1)                                  # CUDA context, nothing else
        sync()
        context = memory.now()
        memory.start()
    s = material_scale()
    case = {"mugrid": setup_mugrid, "muffto": setup_muffto,
            "green": lambda s: setup_green(s, False),
            "green_jacobi": lambda s: setup_green(s, True)}[args.method](s)
    sync()
    t0 = time.perf_counter()
    case["setup"]()
    sync()
    t_setup = time.perf_counter() - t0

    from muGrid.Solvers import conjugate_gradients
    rhs, x = case["rhs"], case["x"]
    z = case["vector"]("bench-z")
    case["prec"](rhs, z)                               # warm-up
    sync()
    t0 = time.perf_counter()
    for _ in range(args.repeat):
        case["prec"](rhs, z)
    sync()
    t_vcycle = (time.perf_counter() - t0) / args.repeat

    x.set_zero()
    iterations = []
    sync()
    t0 = time.perf_counter()
    conjugate_gradients(case["comm"], case["collection"], rhs, x, hessp=case["hessp"],
                        prec=case["prec"], rtol=args.rtol, maxiter=args.maxiter,
                        callback=lambda it, state: iterations.append(it))
    sync()
    t_cg = time.perf_counter() - t0
    # true residual; the operator also refreshes x's ghosts, which
    # average_stress reads (CG leaves them stale)
    residual = case["vector"]("bench-residual")
    case["hessp"](x, residual)
    xp = __import__("cupy") if args.device == "gpu" else np
    true_residual = float(xp.linalg.norm(residual.s - rhs.s) / xp.linalg.norm(rhs.s))
    stress = [float(v) for v in case["stress"](x)]

    result = dict(method=args.method, N=N, precision=args.precision, device=args.device,
                  geometry=args.geometry, eta=args.eta, rtol=args.rtol,
                  lmax=args.lmax if args.method == "muffto" else "-",
                  iterations=max(iterations), true_residual=true_residual,
                  C11=stress[0], t_build=case["t_build"], t_setup=t_setup,
                  t_vcycle=t_vcycle, t_cg=t_cg,
                  levels=[(int(n), float(b)) for n, b in case["levels"]()])
    if args.device == "gpu":
        after = memory.now()
        memory.stop()
        voxels = N ** 3
        result.update(gpu_context_MB=context, gpu_peak_MB=memory.peak - context,
                      gpu_after_MB=after - context,
                      gpu_peak_B_per_voxel=(memory.peak - context) * 2 ** 20 / voxels,
                      gpu_after_B_per_voxel=(after - context) * 2 ** 20 / voxels)
    line = json.dumps(result)
    print(line, flush=True)
    if args.out:
        with open(args.out, "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
