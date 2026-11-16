#!/usr/bin/env python3
"""
muGrid's heterogeneous multigrid on muFFTTO's own operators, against the muFFTTO
multigrid on the same operator.

    python cross_check_operators.py --element linear_triangles --grid 128
    python cross_check_operators.py --element trilinear_hexahedron_1Q_hg --grid 32

The operator is muFFTTO's stencil pipeline (gradient, material s C_0 per
quadrature point, divergence, hourglass control) of ``--element``, handed to
muGrid's MultigridPreconditioner as a black-box callable on muFFTTO's own FFT
engine and fields. Its local matrices (one per quadrature point; for the
one-point hexahedron the point together with its hourglass term) are probed from
the same operator on a 4^d grid, so the smoother bound is the guaranteed one.

The reference coarsens by re-discretisation (linear triangles: mean of the four
fine triangles; hexahedra: octant means; one-point hexahedra: fully integrated
coarse levels), muGrid by the Galerkin product with multilinear P. Both are
plain PCG (relative residual ``--rtol``) on the same right-hand side (E_xx);
compared are the iterations and the solutions.

Material: a seeded smooth random field, s = 10^(-c (1 + tanh(g / w)) / 2) with
g a smooth Gaussian random field, contrast 10^c. The muFFTTO reference is
imported from ``--muffto`` (read only).
"""
import argparse
import json
import os
import sys

sys.meta_path = [f for f in sys.meta_path
                 if type(f).__name__ != "ScikitBuildRedirectingFinder"]

import numpy as np  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--element", required=True,
                    choices=["linear_triangles", "bilinear_rectangle",
                             "trilinear_hexahedron", "trilinear_hexahedron_1Q_hg"])
parser.add_argument("--grid", type=int, required=True)
parser.add_argument("--contrast", type=float, default=4.0, help="log10 of the contrast")
parser.add_argument("--rtol", type=float, default=1e-8)
parser.add_argument("--hg-eps", type=float, default=0.3)
parser.add_argument("--lmax", choices=["bound", "power"], default="bound")
parser.add_argument("--muffto", default=os.environ.get("MUFFTTO_PATH"),
                    help="muFFTTO checkout with muFFTTO/multigrid.py "
                         "(default: $MUFFTTO_PATH)")
parser.add_argument("--out", default=None)
args = parser.parse_args()
if not args.muffto:
    parser.error("needs --muffto (or $MUFFTTO_PATH)")

sys.path.insert(0, args.muffto)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from muGrid.Multigrid import (MultigridPreconditioner,  # noqa: E402
                              ScaledLocalMatrices, probe_pixel_matrix)
from muGrid.Solvers import conjugate_gradients  # noqa: E402
from muFFTTO import domain, material_models, multigrid  # noqa: E402

DIM = 2 if args.element in ("linear_triangles", "bilinear_rectangle") else 3
N = args.grid
C_0 = material_models.get_elastic_material_tensor(dim=DIM, K=1.0, mu=0.5, kind="linear")
HG = args.hg_eps if args.element.endswith("_hg") else None


def discretization(n):
    """n^d pixels of the size of the problem's (1/N): a 4^d probe grid must
    keep the pixel size, since a 3D element matrix scales with h."""
    cell = domain.PeriodicUnitCell(domain_size=[n / N] * DIM, problem_type="elasticity")
    return domain.Discretization(cell=cell, nb_of_pixels_global=(n,) * DIM,
                                 discretization_type="finite_element",
                                 element_type=args.element, hourglass_eps=HG)


def smooth_scale(nq):
    """s per quadrature point, (nq, N, ..., N): a smooth random two-phase field."""
    rng = np.random.default_rng(42)
    k = np.meshgrid(*[np.fft.fftfreq(N) * N for _ in range(DIM)], indexing="ij")
    k2 = sum(kk ** 2 for kk in k)
    noise = np.fft.ifftn(np.fft.fftn(rng.standard_normal((N,) * DIM))
                         * np.exp(-k2 / 8.0)).real
    noise /= noise.std()
    s = 10.0 ** (-args.contrast * 0.5 * (1 + np.tanh(noise / 0.3)))
    return np.repeat(s[None], nq, axis=0)


def local_matrices(scale_field):
    """K_q of muFFTTO's operator, probed on one pixel of a 4^d grid."""
    small = discretization(4)
    level = multigrid.Level(small, C_0, 0)
    pixel = (1,) * DIM
    matrices = []
    for q in range(level.scale.s.shape[2]):
        level.scale.s[...] = 0.0
        level.scale.s[(0, 0, q) + pixel] = 1.0
        level.update_material()
        matrices.append(probe_pixel_matrix(level.apply_K, small.fft, DIM, pixel))
    return ScaledLocalMatrices(np.array(matrices), scale_field)


def main():
    disc = discretization(N)
    reference = multigrid.Multigrid(disc, C_0, lmax=args.lmax)
    nq = reference.scale.s.shape[2]
    reference.scale.s[0, 0] = smooth_scale(nq)
    reference.setup()
    ours = MultigridPreconditioner(
        reference.apply_K, disc.fft, DIM,
        local_matrices=local_matrices(reference.scale),
        communicator=disc.communicator)
    ours.setup()

    E = np.zeros((DIM, DIM))
    E[0, 0] = 1.0
    macro = disc.get_gradient_size_field(name="macro_gradient_field")
    disc.get_macro_gradient_field_mugrid(macro_gradient_ij=E,
                                         macro_gradient_field_ijqxyz=macro)
    rhs = disc.get_unknown_size_field(name="rhs")
    disc.get_rhs_mugrid(material_data_field_ijklqxyz=reference.material,
                        macro_gradient_field_ijqxyz=macro, rhs_inxyz=rhs)

    results = {}
    for label, prec in (("muffto", reference.apply), ("mugrid", ours)):
        x = disc.get_unknown_size_field(name=f"x-{label}")
        x.set_zero()
        iterations = []
        conjugate_gradients(disc.communicator, disc.field_collection, rhs, x,
                            hessp=reference.apply_K, prec=prec, rtol=args.rtol,
                            maxiter=3000,
                            callback=lambda it, state: iterations.append(it))
        energy = float(np.sum(np.asarray(x.s) * rhs.s))
        results[label] = (max(iterations), np.array(x.s), energy)
    difference = (np.abs(results["mugrid"][1] - results["muffto"][1]).max()
                  / np.abs(results["muffto"][1]).max())
    line = json.dumps(dict(
        element=args.element, N=N, contrast=10 ** args.contrast, rtol=args.rtol,
        reference_lmax=args.lmax,
        iterations_muffto=results["muffto"][0], iterations_mugrid=results["mugrid"][0],
        energy_muffto=results["muffto"][2], energy_mugrid=results["mugrid"][2],
        solution_difference=difference,
        bounds_muffto=[float(lv.lmax) for lv in reference.levels
                       if lv.lmax is not None],
        bounds_mugrid=[float(b) for _, b, _ in ours.level_summary()]))
    print(line, flush=True)
    if args.out:
        with open(args.out, "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
