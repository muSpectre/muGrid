"""
Single- vs double-precision homogenization (Phase E end-to-end example).

This example computes the effective elastic stiffness of a 2D periodic unit
cell with a stiff circular inclusion, using the *fused* isotropic stiffness
operator (``K = Bᵀ C B``) and a matrix-free conjugate-gradient solve. It runs
the *same* computation twice — once in double precision (``float64``) and once
in single precision (``float32``) — and reports:

* the homogenized stiffness tensor ``C_eff`` in each precision,
* the relative difference between them, and
* the per-field memory footprint, which single precision halves.

Single precision is selected purely through the ``dtype=`` argument of the
field-creation API (``fc.real_field(name, comps, dtype=np.float32)``); the
fused operator, the linalg primitives and the CG solver all pick up the
precision from the fields they are handed — no separate code path.

Run on the GPU with ``-d gpu`` (requires a CUDA/HIP build).
"""

import argparse

import numpy as np

import muGrid
from muGrid import parprint
from muGrid.Solvers import conjugate_gradients

try:
    from mpi4py import MPI

    comm = muGrid.Communicator(MPI.COMM_WORLD)
except ImportError:
    comm = muGrid.Communicator()


def isotropic_lame(E, nu):
    """Lamé parameters (plane strain) from Young's modulus and Poisson ratio."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lam, mu


def run(dtype, args, device, arr):
    """Solve the homogenization problem in the requested precision and return
    the homogenized 3x3 (Voigt) stiffness plus the resident field bytes."""
    dim = 2
    nb_quad = 4  # Q1 (bilinear quad) in 2D
    domain_size = np.ones(dim)
    grid_spacing = domain_size / np.array(args.nb_grid_pts)

    decomposition = muGrid.CartesianDecomposition(
        comm,
        args.nb_grid_pts,
        nb_subdivisions=(comm.size, 1),
        nb_ghosts_left=(1,) * dim,
        nb_ghosts_right=(1,) * dim,
        nb_sub_pts={"quad": nb_quad},
        device=device,
    )
    fc = decomposition

    # Microstructure: stiff circular inclusion in a soft matrix.
    coords = decomposition.coords
    center = [0.5] * dim
    r2 = sum((coords[d] - center[d]) ** 2 for d in range(dim))
    phase = np.where(r2 < args.inclusion_radius**2, 1.0, 0.0)

    lam_m, mu_m = isotropic_lame(args.E_matrix, args.nu)
    lam_i, mu_i = isotropic_lame(args.E_inclusion, args.nu)

    # Material (Lamé) fields and solver work fields, all in `dtype`.
    lambda_field = fc.real_field("lambda", dtype=dtype)
    mu_field = fc.real_field("mu", dtype=dtype)
    lambda_field.p[...] = arr.asarray(
        (lam_m * (1 - phase) + lam_i * phase).astype(dtype)
    )
    mu_field.p[...] = arr.asarray(
        (mu_m * (1 - phase) + mu_i * phase).astype(dtype)
    )
    decomposition.communicate_ghosts(lambda_field)
    decomposition.communicate_ghosts(mu_field)

    u_field = fc.real_field("u", (dim,), dtype=dtype)
    rhs_field = fc.real_field("rhs", (dim,), dtype=dtype)

    op = muGrid.IsotropicStiffnessOperator2D(
        tuple(grid_spacing), muGrid.FEMElement.q1
    )

    def apply_stiffness(u_in, f_out):
        decomposition.communicate_ghosts(u_in)
        op.apply(u_in, lambda_field, mu_field, f_out)

    # Three independent macroscopic strain load cases: xx, yy, xy.
    voigt_labels = ["xx", "yy", "xy"]
    strain_cases = [(0, 0), (1, 1), (0, 1)]
    voigt_index = {(0, 0): 0, (1, 1): 1, (0, 1): 2, (1, 0): 2}

    C_eff = np.zeros((3, 3))
    total_iters = 0
    for (i, j) in strain_cases:
        E_macro = np.zeros((dim, dim))
        E_macro[i, j] = 1.0
        if i != j:
            E_macro[j, i] = 1.0
        E_flat = [float(E_macro[a, b]) for a in range(dim) for b in range(dim)]

        # RHS: f = -Bᵀ C E_macro (assembled streaming from the Lamé fields).
        op.apply_macro_rhs(lambda_field, mu_field, E_flat, rhs_field)
        rhs_field.s[...] *= -1.0

        u_field.s[...] = 0.0
        it = [0]

        def cb(iteration, state):
            it[0] = iteration

        conjugate_gradients(
            comm, fc, rhs_field, u_field, hessp=apply_stiffness,
            rtol=args.tol, maxiter=args.maxiter, callback=cb,
        )
        total_iters += it[0]

        # Homogenized stress Σ = (1/|Ω|) ∫ C:(E_macro + sym∇u) dΩ.
        decomposition.communicate_ghosts(u_field)
        local_int = op.average_stress(u_field, lambda_field, mu_field, E_flat)
        sig = np.zeros((dim, dim))
        for k in range(dim):
            for L in range(dim):
                sig[k, L] = comm.sum(local_int[k * dim + L])
        sig /= np.prod(domain_size)

        col = voigt_index[(i, j)]
        for k in range(dim):
            for L in range(dim):
                if k <= L:
                    C_eff[voigt_index[(k, L)], col] = sig[k, L]

    # Resident bytes of the principal fields (those that scale with the grid).
    field_bytes = sum(
        f.s.nbytes for f in (lambda_field, mu_field, u_field, rhs_field)
    )
    return C_eff, total_iters, field_bytes, voigt_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-n", "--nb-grid-pts", default=[64, 64],
        type=lambda s: [int(x) for x in s.split(",")],
        help="Grid points as nx,ny (default: 64,64)",
    )
    parser.add_argument("-d", "--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("-i", "--maxiter", type=int, default=2000)
    parser.add_argument("-t", "--tol", type=float, default=1e-5,
                        help="Relative CG tolerance (default: 1e-5)")
    parser.add_argument("--E-matrix", type=float, default=1.0)
    parser.add_argument("--E-inclusion", type=float, default=10.0)
    parser.add_argument("--nu", type=float, default=0.3)
    parser.add_argument("--inclusion-radius", type=float, default=0.25)
    args = parser.parse_args()

    if args.device == "cpu":
        import numpy as arr
        device = muGrid.Device.cpu()
    else:
        import cupy as arr
        device = muGrid.Device.gpu()
        muGrid.route_cupy_through_mugrid()

    parprint("=" * 64, comm=comm)
    parprint("Single- vs double-precision homogenization", comm=comm)
    parprint("=" * 64, comm=comm)
    parprint(f"Grid: {args.nb_grid_pts}   device: {device.device_string}   "
             f"CG rtol: {args.tol}", comm=comm)
    parprint(f"E_matrix={args.E_matrix}  E_inclusion={args.E_inclusion}  "
             f"nu={args.nu}  inclusion_radius={args.inclusion_radius}", comm=comm)
    parprint("", comm=comm)

    results = {}
    for label, dtype in (("float64", np.float64), ("float32", np.float32)):
        C, iters, nbytes, voigt_labels = run(dtype, args, device, arr)
        results[label] = (C, iters, nbytes)
        parprint(f"--- {label} ---  ({iters} total CG iterations)", comm=comm)
        header = "      " + "".join(f"{lbl:>12}" for lbl in voigt_labels)
        parprint(header, comm=comm)
        for r, rl in enumerate(voigt_labels):
            row = f"{rl:>4}  " + "".join(f"{C[r, c]:12.6f}" for c in range(3))
            parprint(row, comm=comm)
        parprint("", comm=comm)

    C64, _, bytes64 = results["float64"]
    C32, _, bytes32 = results["float32"]
    denom = np.linalg.norm(C64)
    rel = np.linalg.norm(C32 - C64) / denom if denom > 0 else 0.0

    parprint("=" * 64, comm=comm)
    parprint("Comparison", comm=comm)
    parprint("=" * 64, comm=comm)
    parprint(f"Relative difference ||C32 - C64|| / ||C64||  = {rel:.3e}",
             comm=comm)
    parprint(f"Max abs component difference                 = "
             f"{np.max(np.abs(C32 - C64)):.3e}", comm=comm)
    parprint(f"Field memory: float64 = {bytes64 / 1e6:.3f} MB   "
             f"float32 = {bytes32 / 1e6:.3f} MB   "
             f"(ratio {bytes32 / bytes64:.2f})", comm=comm)

    tol = 1e-4
    ok = rel < tol
    parprint("", comm=comm)
    parprint(f"RESULT: {'PASS' if ok else 'FAIL'} "
             f"(fp32 matches fp64 within {tol:g} relative)", comm=comm)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
