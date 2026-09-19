#!/usr/bin/env python3
"""Stage 1 prototype for the multigrid-accelerated reference preconditioner.

See `docs/multigrid_preconditioner_plan.md`. This is a throwaway serial,
host-only, NumPy prototype whose only job is to answer the Stage 1 questions
before any C++ is written:

1. What is the nodal ``dim x dim`` block ``K_nn`` of the *uniform* operator,
   per element kind and dimension?  (``--probe``)
2. Are the grid-transfer operators exact adjoints, and does prolongation
   reproduce linear displacement fields exactly?  (``--check``)
3. What is the V-cycle convergence factor ``rho`` on uniform material, and is
   it independent of the grid size?  (``--rho``)
4. Does the V-cycle, used as a CG preconditioner in place of the exact
   fine-grid FFT solve, keep the iteration count grid-independent?  (``--cg``)

Everything operates on plain NumPy arrays of shape ``(dim, *nb_grid_pts)``;
muGrid fields are touched only inside `Level.apply`, which calls the same fused
``apply_uniform`` kernel the production code would use. Ghost exchange is the
serial periodic self-copy, so the operator is genuinely periodic.

Run with the workspace venv and homebrew MPI on the path:

    source benchenv.sh
    python muGrid/mg_prototype.py --probe
    python muGrid/mg_prototype.py --check
    python muGrid/mg_prototype.py --rho --dim 3 -n 64
"""

import argparse

import numpy as np

import muGrid
from muGrid.Preconditioners import make_reference_stiffness_preconditioner

# --------------------------------------------------------------------------- #
# Grid transfer: multilinear prolongation P, and its exact adjoint R = P^T.
#
# Nodal 2:1 coarsening on a periodic grid; coarse node c sits on fine node 2c.
# Both are tensor products of a 1D rule applied along each spatial axis, so the
# d-dimensional operator is just d successive 1D passes. Arrays are
# (dim, *spatial) -- axis 0 is the displacement component and is never touched,
# which is the whole point: the transfers do not couple components.
# --------------------------------------------------------------------------- #


def _axis_slice(ax, s):
    """Index tuple selecting `s` along spatial axis `ax` of a (dim, *spatial)."""
    return (slice(None),) * (ax + 1) + (s,)


def _prolong_axis(a, ax):
    """1D linear interpolation along spatial axis `ax`: n -> 2n.

    fine[2c]   = coarse[c]
    fine[2c+1] = (coarse[c] + coarse[c+1]) / 2
    """
    axis = ax + 1
    n = a.shape[axis]
    out = np.empty(a.shape[:axis] + (2 * n,) + a.shape[axis + 1:], dtype=a.dtype)
    out[_axis_slice(ax, slice(0, None, 2))] = a
    out[_axis_slice(ax, slice(1, None, 2))] = 0.5 * (
        a + np.roll(a, -1, axis=axis))
    return out


def _restrict_axis(a, ax):
    """Exact adjoint of `_prolong_axis` along spatial axis `ax`: 2n -> n.

    coarse[c] = fine[2c] + (fine[2c-1] + fine[2c+1]) / 2

    Note this is P^T, *not* the 1/2^d-normalised full weighting. P^T is the
    variational restriction that pairs with a rediscretised FE coarse operator:
    the residual is a force (a functional), and restricting a functional carries
    no measure factor. Using P^T is also what makes the two-grid operator --
    and hence the preconditioner -- symmetric.
    """
    axis = ax + 1
    even = a[_axis_slice(ax, slice(0, None, 2))]   # fine[2c]
    odd = a[_axis_slice(ax, slice(1, None, 2))]    # fine[2c+1]
    return even + 0.5 * (odd + np.roll(odd, 1, axis=axis))


def prolong(coarse, dim):
    out = coarse
    for ax in range(dim):
        out = _prolong_axis(out, ax)
    return out


def restrict(fine, dim):
    out = fine
    for ax in range(dim):
        out = _restrict_axis(out, ax)
    return out


def project_mean_out(a):
    """Remove the nullspace of the periodic operator: the `dim` constant
    translations. (Rigid rotations are not periodic, so they are not in it.)"""
    spatial = tuple(range(1, a.ndim))
    return a - a.mean(axis=spatial, keepdims=True)


# --------------------------------------------------------------------------- #
# One multigrid level
# --------------------------------------------------------------------------- #


class Level:
    """A single grid level carrying the *uniform* reference operator K_ref."""

    def __init__(self, dim, n, spacing, element, lam, mu, with_fft=False):
        self.dim = dim
        self.n = n
        self.shape = (dim,) + (n,) * dim
        self.spacing = spacing
        self.lam = lam
        self.mu = mu

        cls = (muGrid.IsotropicStiffnessOperator2D if dim == 2
               else muGrid.IsotropicStiffnessOperator3D)
        self.op = cls(tuple(spacing), element)

        ghosts = dict(nb_ghosts_left=(1,) * dim, nb_ghosts_right=(1,) * dim)
        if with_fft:
            # The coarsest level needs FFT-transformable fields, so its
            # decomposition *is* an FFTEngine -- exactly as homogenization.py
            # does for `-P reference`.
            self.decomp = muGrid.FFTEngine(
                (n,) * dim, muGrid.Communicator(), **ghosts)
            self.fc = self.decomp.real_space_collection
        else:
            self.decomp = muGrid.CartesianDecomposition(
                muGrid.Communicator(), (n,) * dim,
                nb_subdivisions=(1,) * dim, **ghosts)
            self.fc = self.decomp.collection

        self._u = self.fc.real_field("mg-u", (dim,))
        self._f = self.fc.real_field("mg-f", (dim,))

        self.node_block = self._probe_node_block()
        self.node_block_inv = np.linalg.inv(self.node_block)

    # -- operator ---------------------------------------------------------- #

    def apply(self, u_arr):
        """Return K_ref @ u for a (dim, *spatial) array."""
        self._u.s[:, 0] = u_arr
        self.decomp.communicate_ghosts(self._u)
        self.op.apply_uniform(self._u, self.lam, self.mu, self._f)
        return np.array(self._f.s[:, 0])

    def _probe_node_block(self):
        """The nodal dim x dim block K_nn, by impulse response.

        On a uniform periodic grid every node has an identical element
        neighbourhood, so this one matrix describes every node on the level.
        """
        node = (self.n // 2,) * self.dim
        block = np.zeros((self.dim, self.dim))
        for beta in range(self.dim):
            e = np.zeros(self.shape)
            e[(beta,) + node] = 1.0
            block[:, beta] = self.apply(e)[(slice(None),) + node]
        return block

    # -- smoother ---------------------------------------------------------- #

    def lambda_max(self, nb_it=100):
        """Largest eigenvalue of D^-1 K by power iteration.

        D^-1 K is invariant under uniform refinement -- D and K carry the same
        power of h -- so one level's estimate serves the whole hierarchy, and
        the coarsest is the cheapest place to measure it.
        """
        rng = np.random.default_rng(0)
        v = project_mean_out(rng.standard_normal(self.shape))
        v /= np.linalg.norm(v)
        lam = 0.0
        for _ in range(nb_it):
            w = np.einsum("ij,j...->i...", self.node_block_inv, self.apply(v))
            lam = np.linalg.norm(w)
            v = w / lam
        return lam

    def smooth(self, r, z, nb_steps, omega):
        """`nb_steps` damped-Jacobi sweeps of z <- z + omega * D^-1 (r - K z).

        D is the constant nodal block. When it is c*I -- Q1 in any dimension,
        P1 in 3D, isotropic spacing -- this degenerates to a scalar axpy and
        needs no kernel of its own; see `--probe`.
        """
        for _ in range(nb_steps):
            t = r - self.apply(z)
            z = z + omega * np.einsum("ij,j...->i...", self.node_block_inv, t)
        return z


# --------------------------------------------------------------------------- #
# The hierarchy
# --------------------------------------------------------------------------- #


class MultigridReference:
    """V-cycle approximation of K_ref^-1, with an exact FFT solve at the bottom.

    Used as a drop-in for the fine-grid FFT solve inside the reference
    preconditioner. Heterogeneity is *not* handled here -- it stays in the
    J^{1/2} . G . J^{1/2} scaling of GreenJacobiPreconditioner.
    """

    #: omega = SAFETY / lambda_max(D^-1 K). Measured optimum is 1.7 across
    #: {2D, 3D} x {Q1, P1}; the stability limit is 2.0 and divergence sets in
    #: sharply at 1.9, so this keeps ~15% margin. See the Stage 1 findings in
    #: docs/multigrid_preconditioner_plan.md.
    SAFETY = 1.7

    def __init__(self, dim, n, spacing, element, lam, mu,
                 nb_levels=None, coarsest=8, nu1=2, nu2=2, omega=None):
        self.dim = dim
        self.nu1, self.nu2 = nu1, nu2

        if nb_levels is None:
            nb_levels = 1
            while n // (2 ** nb_levels) >= coarsest and n % (2 ** nb_levels) == 0:
                nb_levels += 1
        self.nb_levels = nb_levels

        self.levels = []
        for lvl in range(nb_levels):
            nl = n // (2 ** lvl)
            hl = np.asarray(spacing) * (2 ** lvl)
            self.levels.append(
                Level(dim, nl, hl, element, lam, mu,
                      with_fft=(lvl == nb_levels - 1)))

        # A hand-picked omega is fragile: the divergence threshold moves with
        # dimension and element kind (3D Q1 diverges at 0.7, where 2D Q1 is at
        # its optimum). Deriving it from a measured lambda_max removes the
        # tuning knob entirely.
        self.omega = (omega if omega is not None
                      else self.SAFETY / self.levels[-1].lambda_max())

        # Coarsest level: the exact block-Fourier inverse of K_ref, reusing the
        # production assembly verbatim. It already replaces the singular q = 0
        # block by its pseudo-inverse, which is the nullspace handling we need.
        bottom = self.levels[-1]
        self._coarse_prec = make_reference_stiffness_preconditioner(
            bottom.decomp,
            lambda u_in, f_out: self._bottom_apply(bottom, u_in, f_out),
            dim, name="mg-coarse")
        self._coarse_r = bottom.fc.real_field("mg-coarse-r", (dim,))
        self._coarse_z = bottom.fc.real_field("mg-coarse-z", (dim,))

    @staticmethod
    def _bottom_apply(level, u_in, f_out):
        level.decomp.communicate_ghosts(u_in)
        level.op.apply_uniform(u_in, level.lam, level.mu, f_out)

    def _solve_coarsest(self, r):
        self._coarse_r.s[:, 0] = r
        self._coarse_prec.apply(self._coarse_r, self._coarse_z)
        return np.array(self._coarse_z.s[:, 0])

    def vcycle(self, r, lvl=0):
        level = self.levels[lvl]
        if lvl == self.nb_levels - 1:
            return self._solve_coarsest(r)

        z = np.zeros_like(r)
        z = level.smooth(r, z, self.nu1, self.omega)
        residual = r - level.apply(z)
        z = z + prolong(self.vcycle(restrict(residual, self.dim), lvl + 1),
                        self.dim)
        z = level.smooth(r, z, self.nu2, self.omega)
        return z

    def apply(self, r, nb_cycles=1):
        """M^-1 r. A *fixed* number of cycles and smoothing steps keeps this a
        linear, symmetric operator, which plain CG requires."""
        r = project_mean_out(r)
        z = np.zeros_like(r)
        for _ in range(nb_cycles):
            z = z + self.vcycle(r - self.levels[0].apply(z))
        return project_mean_out(z)


# --------------------------------------------------------------------------- #
# Stage 1 experiments
# --------------------------------------------------------------------------- #


def cmd_probe(args):
    """Question 1: what is the nodal block K_nn?"""
    print("Nodal block K_nn of the uniform operator "
          f"(lambda={args.lam}, mu={args.mu})\n")
    for dim in (2, 3):
        for el_name in ("q1", "p1"):
            for spacing, tag in (((1.0,) * dim, "isotropic h"),
                                 ((1.0,) + (2.0,) * (dim - 1), "anisotropic h")):
                lvl = Level(dim, 8, spacing, getattr(muGrid.FEMElement, el_name),
                            args.lam, args.mu)
                K = lvl.node_block
                off = np.abs(K - np.diag(np.diag(K))).max()
                spread = np.ptp(np.diag(K))
                if off > 1e-12:
                    verdict = "FULL BLOCK  (needs the dim x dim path)"
                elif spread > 1e-12:
                    verdict = "diagonal, unequal entries"
                else:
                    verdict = "c * I  -> smoother is a scalar axpy"
                print(f"  {dim}D {el_name}, {tag:<14} "
                      f"|offdiag|={off:8.2e}  spread={spread:8.2e}   {verdict}")
        print()


def cmd_check(args):
    """Question 2: are the transfers correct?"""
    rng = np.random.default_rng(0)
    dim, n = args.dim, args.n
    print(f"Transfer checks, {dim}D, fine n={n}\n")

    # Exact adjointness <P c, f> == <c, P^T f>.
    c = rng.standard_normal((dim,) + (n // 2,) * dim)
    f = rng.standard_normal((dim,) + (n,) * dim)
    lhs = float((prolong(c, dim) * f).sum())
    rhs = float((c * restrict(f, dim)).sum())
    rel = abs(lhs - rhs) / abs(lhs)
    print(f"  adjointness   <Pc,f> = {lhs: .12e}")
    print(f"                <c,Rf> = {rhs: .12e}   rel.err = {rel:.2e}"
          f"   {'OK' if rel < 1e-13 else 'FAIL'}")

    # Prolongation must reproduce linear displacement fields exactly: that is
    # what puts every rigid-body mode and every constant strain in range(P).
    coords_c = np.meshgrid(*[np.arange(n // 2) * 2.0] * dim, indexing="ij")
    coords_f = np.meshgrid(*[np.arange(n) * 1.0] * dim, indexing="ij")
    A = rng.standard_normal((dim, dim))
    b = rng.standard_normal(dim)
    lin_c = np.stack([sum(A[i, j] * coords_c[j] for j in range(dim)) + b[i]
                      for i in range(dim)])
    lin_f = np.stack([sum(A[i, j] * coords_f[j] for j in range(dim)) + b[i]
                      for i in range(dim)])
    # Periodic wrap makes the last interval non-linear; compare the interior.
    interior = (slice(None),) + (slice(0, n - 2),) * dim
    err = np.abs(prolong(lin_c, dim)[interior] - lin_f[interior]).max()
    print(f"  linear repro  max|P(lin_H) - lin_h| = {err:.2e}"
          f"   {'OK' if err < 1e-10 else 'FAIL'}")

    # The preconditioner must be symmetric, or CG is not valid.
    mg = _build(args)
    a = project_mean_out(rng.standard_normal(mg.levels[0].shape))
    b2 = project_mean_out(rng.standard_normal(mg.levels[0].shape))
    lhs = float((mg.apply(a) * b2).sum())
    rhs = float((a * mg.apply(b2)).sum())
    rel = abs(lhs - rhs) / abs(lhs)
    print(f"  symmetry      <M^-1 a, b> = {lhs: .12e}")
    print(f"                <a, M^-1 b> = {rhs: .12e}   rel.err = {rel:.2e}"
          f"   {'OK' if rel < 1e-10 else 'FAIL'}")


def _build(args, n=None):
    n = n or args.n
    spacing = (1.0 / n,) * args.dim
    return MultigridReference(
        args.dim, n, spacing, getattr(muGrid.FEMElement, args.element),
        args.lam, args.mu, nb_levels=args.levels, coarsest=args.coarsest,
        nu1=args.nu1, nu2=args.nu2, omega=args.omega)


def cmd_rho(args):
    """Question 3: V-cycle convergence factor, and its grid dependence."""
    rng = np.random.default_rng(1)
    print(f"V-cycle convergence factor, {args.dim}D {args.element}, "
          f"nu=({args.nu1},{args.nu2})\n")
    print(f"  {'n':>5} {'levels':>7} {'coarsest':>9} {'omega':>7} "
          f"{'rho':>9} {'iters to 1e-8':>14}")
    for n in args.sizes:
        mg = _build(args, n)
        f = project_mean_out(rng.standard_normal(mg.levels[0].shape))
        u = np.zeros_like(f)
        r0 = np.linalg.norm(f)
        factors, nrm = [], r0
        for it in range(args.maxiter):
            u = u + mg.vcycle(project_mean_out(f - mg.levels[0].apply(u)))
            new = np.linalg.norm(project_mean_out(f - mg.levels[0].apply(u)))
            factors.append(new / nrm)
            nrm = new
            if nrm / r0 < 1e-8:
                break
        rho = float(np.mean(factors[-3:])) if factors else float("nan")
        print(f"  {n:>5} {mg.nb_levels:>7} {mg.levels[-1].n:>9} "
              f"{mg.omega:>7.4f} {rho:>9.4f} {it + 1:>14}")


# --------------------------------------------------------------------------- #
# Hybrid preconditioner: FFT in the *local* axes, block-tridiagonal solve along
# the *distributed* one.
#
# Under the slab decomposition [1, 1, P] that muGrid's FFT engine imposes, x and
# y are rank-local and only z is divided. That split is deliberate -- leaving two
# axes local is what allows a batched 2D rocFFT/cuFFT call instead of 1D
# transforms plus a transpose -- but it is also what forces the all-to-all, since
# the third transform needs data the rank does not hold.
#
# The hybrid keeps the cheap half and replaces the expensive half. Two facts make
# it exact rather than approximate:
#
#   1. K_ref is translation-invariant in x and y (it is *uniform* by
#      construction), so transforming those axes block-diagonalises it: each
#      (qx, qy) mode decouples completely.
#   2. The stencil reaches exactly one node in z -- measured, for Q1 (27 nodes)
#      and P1 (19 nodes) alike -- so what is left along z after that transform is
#      block-tridiagonal, with dim x dim blocks and dim*dim entries per mode.
#
# Neither fact needs the operator to separate as a Kronecker sum. Q1 elasticity
# does not, which is why the "the Laplacian factorises" argument is the wrong one
# to lean on even though it points at the right answer.
#
# Solve that tridiagonal system exactly and the result *is* K_ref^-1 -- the same
# operator `-P reference` applies, reached without ever transforming z. So the
# CG count should be the FFT column's, not the V-cycle's, which is the whole
# point: the V-cycle's 1.5x iteration penalty disappears.
#
# The catch is that z is both the tridiagonal direction and the distributed one.
# `z_solve="mg"` models the cheapest answer to that -- semi-coarsening multigrid
# in z alone, halo-only and free of the log P that cyclic reduction would cost.
# --------------------------------------------------------------------------- #


def _stencil(dim, spacing, element, lam, mu, n=8):
    """The uniform operator's stencil, by impulse response.

    Returns S with ``f(y) = sum_d S[d] u(y - d)``, indexed ``S[dx+1, dy+1, ...]``.
    `n` only has to be large enough that a one-node-wide stencil cannot wrap.
    """
    decomp = muGrid.CartesianDecomposition(
        muGrid.Communicator(), (n,) * dim, nb_subdivisions=(1,) * dim,
        nb_ghosts_left=(1,) * dim, nb_ghosts_right=(1,) * dim)
    cls = (muGrid.IsotropicStiffnessOperator2D if dim == 2
           else muGrid.IsotropicStiffnessOperator3D)
    op = cls(tuple(spacing), element)
    u = decomp.collection.real_field("stencil-u", (dim,))
    f = decomp.collection.real_field("stencil-f", (dim,))

    S = np.zeros((3,) * dim + (dim, dim))
    c = n // 2
    for beta in range(dim):
        u.set_zero()
        u.s[(beta, 0) + (c,) * dim] = 1.0
        decomp.communicate_ghosts(u)
        op.apply_uniform(u, lam, mu, f)
        arr = np.asarray(f.s)[:, 0]
        for off in np.ndindex((3,) * dim):
            S[off + (slice(None), beta)] = arr[
                (slice(None),) + tuple(c + o - 1 for o in off)]
    return S


def _z_blocks(S, dim, local_shape):
    """Transform the local axes; return the three z-coupling blocks per mode.

    ``A[m]`` for ``m = dz + 1`` has shape ``(*local_shape, dim, dim)``, and the
    z-operator for each mode is
    ``(T v)(k) = A[0] v(k+1) + A[1] v(k) + A[2] v(k-1)``.
    """
    qs = [2 * np.pi * np.fft.fftfreq(n) for n in local_shape]
    grids = np.meshgrid(*qs, indexing="ij") if local_shape else []
    A = np.zeros((3,) + tuple(local_shape) + (dim, dim), dtype=complex)
    for off in np.ndindex((3,) * dim):
        phase = np.ones(local_shape, dtype=complex)
        for ax in range(dim - 1):
            phase = phase * np.exp(-1j * grids[ax] * (off[ax] - 1))
        A[off[-1]] += phase[..., None, None] * S[off]
    return A


def _apply_T(A, v):
    """``(T v)(k) = A[0] v(k+1) + A[1] v(k) + A[2] v(k-1)``, batched over modes.

    `v` has shape ``(*modes, nz, dim)``; the blocks broadcast along z.
    """
    def mul(block, w):
        return np.einsum("...ij,...j->...i", block[..., None, :, :], w)
    return (mul(A[0], np.roll(v, -1, axis=-2))
            + mul(A[1], v)
            + mul(A[2], np.roll(v, 1, axis=-2)))


def _dense_T(A, nz):
    """The z-operator as an explicit ``(*modes, nz*dim, nz*dim)`` matrix."""
    dim = A.shape[-1]
    modes = A.shape[1:-2]
    M = np.zeros(modes + (nz * dim, nz * dim), dtype=complex)
    for k in range(nz):
        sl = slice(k * dim, (k + 1) * dim)
        M[..., sl, slice(((k + 1) % nz) * dim, ((k + 1) % nz) * dim + dim)] += A[0]
        M[..., sl, sl] += A[1]
        M[..., sl, slice(((k - 1) % nz) * dim, ((k - 1) % nz) * dim + dim)] += A[2]
    return M


def _invert_modes(M):
    """Batched inverse, with a pseudo-inverse wherever the mode is singular.

    Only the all-zero local mode is singular: there the z-operator still has the
    constant-in-z nullspace, which is exactly the rigid translation the reference
    preconditioner also pseudo-inverts at q = 0.
    """
    flat = M.reshape((-1,) + M.shape[-2:])
    out = np.empty_like(flat)
    eye = np.eye(flat.shape[-1], dtype=flat.dtype)
    for i in range(flat.shape[0]):
        try:
            out[i] = np.linalg.inv(flat[i])
        except np.linalg.LinAlgError:
            out[i] = np.linalg.pinv(flat[i])
            continue
        # inv() does not always raise on a singular mode; it returns garbage.
        # Checking the residual is cheaper than a condition number and catches
        # both cases.
        residual = np.abs(flat[i] @ out[i] - eye).max()
        if not np.isfinite(residual) or residual > 1e-6:
            out[i] = np.linalg.pinv(flat[i])
    return out.reshape(M.shape)


def _prolong_z(coarse):
    """Linear interpolation along z, ``(*modes, nz, dim) -> (*modes, 2nz, dim)``."""
    nz = coarse.shape[-2]
    fine = np.empty(coarse.shape[:-2] + (2 * nz, coarse.shape[-1]),
                    dtype=coarse.dtype)
    fine[..., 0::2, :] = coarse
    fine[..., 1::2, :] = 0.5 * (coarse + np.roll(coarse, -1, axis=-2))
    return fine


def _restrict_z(fine):
    """The exact adjoint of :func:`_prolong_z`, which keeps the cycle symmetric."""
    even = fine[..., 0::2, :]
    odd = fine[..., 1::2, :]
    return even + 0.5 * (odd + np.roll(odd, 1, axis=-2))


class HybridFourierTridiagonal:
    """``K_ref^-1`` by FFT in the local axes and a tridiagonal solve in z.

    Parameters
    ----------
    z_solve : {"exact", "mg"}
        ``"exact"`` factorises each mode's z-operator once and is therefore the
        exact inverse -- the ceiling this design can reach, and what a parallel
        partitioned-Thomas or cyclic-reduction solver would compute. ``"mg"``
        replaces it with semi-coarsening multigrid in z, which needs only halo
        exchange and is the variant that would actually run under MPI.
    """

    SAFETY = 1.7

    def __init__(self, dim, n, spacing, element, lam, mu,
                 z_solve="exact", nu=2, cycles=1, coarsest=4):
        self.dim, self.n, self.nz = dim, n, n
        self.z_solve, self.nu, self.cycles = z_solve, nu, cycles
        self.local_shape = (n,) * (dim - 1)
        self.local_axes = tuple(range(1, dim))

        spacing = np.asarray(spacing, dtype=float)
        self.A = _z_blocks(_stencil(dim, spacing, element, lam, mu),
                           dim, self.local_shape)

        if z_solve == "exact":
            self.Tinv = _invert_modes(_dense_T(self.A, n))
            self.nb_z_levels = 1
            return

        # Semi-coarsening: z halves, the local axes do not, so each level is a
        # rediscretisation of the same operator at a doubled z spacing.
        self.levels = []
        nz, nb = n, 1
        while nz // 2 >= coarsest and nz % 2 == 0:
            nb, nz = nb + 1, nz // 2
        self.nb_z_levels = nb
        for lvl in range(nb):
            h = spacing.copy()
            h[-1] *= 2 ** lvl
            A = _z_blocks(_stencil(dim, h, element, lam, mu),
                          dim, self.local_shape)
            self.levels.append({"A": A, "nz": n // 2 ** lvl,
                                "Dinv": _invert_modes(A[1])})
        self.levels[-1]["Tinv"] = _invert_modes(
            _dense_T(self.levels[-1]["A"], self.levels[-1]["nz"]))
        for level in self.levels[:-1]:
            level["omega"] = self.SAFETY / self._lambda_max(level)

    # -- smoother ---------------------------------------------------------- #

    def _lambda_max(self, level, nb_it=40):
        """Largest eigenvalue of ``D^-1 T``, per mode, by power iteration.

        Per-mode rather than global: the modes are decoupled, and a single
        damping that suited the stiffest of them would badly under-relax the
        rest. A real per-mode omega keeps the smoother linear and symmetric.
        """
        rng = np.random.default_rng(0)
        shape = self.local_shape + (level["nz"], self.dim)
        v = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
        lam = np.ones(self.local_shape)
        for _ in range(nb_it):
            w = self._apply_Dinv(level, _apply_T(level["A"], v))
            lam = np.sqrt((np.abs(w) ** 2).sum(axis=(-2, -1)))
            v = w / np.maximum(lam, 1e-300)[..., None, None]
        return np.maximum(lam, 1e-12)

    @staticmethod
    def _apply_Dinv(level, v):
        return np.einsum("...ij,...j->...i", level["Dinv"][..., None, :, :], v)

    def _smooth(self, level, r, z, nb_steps):
        omega = level["omega"][..., None, None]
        for _ in range(nb_steps):
            z = z + omega * self._apply_Dinv(
                level, r - _apply_T(level["A"], z))
        return z

    # -- the z-cycle ------------------------------------------------------- #

    def _z_vcycle(self, r, lvl=0):
        level = self.levels[lvl]
        if lvl == self.nb_z_levels - 1:
            flat = r.reshape(self.local_shape + (-1,))
            out = np.einsum("...ij,...j->...i", level["Tinv"], flat)
            return out.reshape(r.shape)
        z = self._smooth(level, r, np.zeros_like(r), self.nu)
        residual = r - _apply_T(level["A"], z)
        z = z + _prolong_z(self._z_vcycle(_restrict_z(residual), lvl + 1))
        return self._smooth(level, r, z, self.nu)

    def _solve_z(self, v):
        if self.z_solve == "exact":
            flat = v.reshape(self.local_shape + (-1,))
            out = np.einsum("...ij,...j->...i", self.Tinv, flat)
            return out.reshape(v.shape)
        z = np.zeros_like(v)
        for _ in range(self.cycles):
            z = z + self._z_vcycle(v - _apply_T(self.levels[0]["A"], z))
        return z

    # -- the preconditioner ------------------------------------------------ #

    def apply(self, r):
        """``M^-1 r`` for a real ``(dim, *spatial)`` array."""
        r = project_mean_out(r)
        rh = np.fft.fftn(r, axes=self.local_axes)
        v = np.moveaxis(rh, 0, -1)

        # The all-zero local mode keeps the constant-in-z nullspace -- the rigid
        # translation. Hold the solve orthogonal to it rather than letting the
        # cycle wander along it.
        zero = (0,) * (self.dim - 1)
        v[zero] -= v[zero].mean(axis=0, keepdims=True)
        z = self._solve_z(v)
        z[zero] -= z[zero].mean(axis=0, keepdims=True)

        out = np.fft.ifftn(np.moveaxis(z, -1, 0), axes=self.local_axes)
        return project_mean_out(np.real(out))


class Heterogeneous:
    """The actual homogenization system: K(lambda(x), mu(x)) with a spherical
    inclusion, plus the diagonal needed for the J^{1/2} . G . J^{1/2} scaling.

    The reference material is the volume mean of the Lame fields, exactly as
    `examples/homogenization.py` picks it.
    """

    def __init__(self, level, contrast, kind="inclusion"):
        self.level = level
        dim, n = level.dim, level.n
        fc = level.fc

        coords = np.meshgrid(*[(np.arange(n) + 0.5) / n] * dim, indexing="ij")
        if kind == "inclusion":
            # Sharp binary interface -- the classic homogenization test case.
            radius = np.sqrt(sum((c - 0.5) ** 2 for c in coords))
            weight = (radius < 0.25).astype(float)
        elif kind == "smooth":
            # Smoothly varying high-contrast data: the regime the J-FFT scheme
            # of Ladecky et al. is actually designed for, and what SIMP
            # topology optimization with density filtering produces.
            weight = np.prod([0.5 * (1 + np.cos(2 * np.pi * c))
                              for c in coords], axis=0)
        else:
            raise ValueError(f"unknown material kind {kind!r}")

        scale = 1.0 + (contrast - 1.0) * weight
        lam = level.lam * scale
        mu = level.mu * scale
        self.lam_ref, self.mu_ref = float(lam.mean()), float(mu.mean())

        self._lam = fc.real_field("het-lambda")
        self._mu = fc.real_field("het-mu")
        self._lam.s[0] = lam
        self._mu.s[0] = mu
        level.decomp.communicate_ghosts(self._lam)
        level.decomp.communicate_ghosts(self._mu)

        self._u = fc.real_field("het-u", (dim,))
        self._f = fc.real_field("het-f", (dim,))

        diag = fc.real_field("het-diag", (dim,))
        level.op.assemble_diagonal(self._lam, self._mu, diag)
        d = np.array(diag.s[:, 0])
        self.jhalf = np.where(d > 0, 1.0 / np.sqrt(np.where(d > 0, d, 1.0)), 1.0)

    def apply(self, u_arr):
        self._u.s[:, 0] = u_arr
        self.level.decomp.communicate_ghosts(self._u)
        self.level.op.apply(self._u, self._lam, self._mu, self._f)
        return np.array(self._f.s[:, 0])


def _cg(apply_op, b, prec, tol, maxiter):
    """Plain PCG; returns the iteration count, or -1 if it did not converge."""
    x = np.zeros_like(b)
    r = b.copy()
    z = prec(r)
    p = z.copy()
    rz = float((r * z).sum())
    nb = np.linalg.norm(b)
    for it in range(maxiter):
        Ap = apply_op(p)
        alpha = rz / float((p * Ap).sum())
        x += alpha * p
        r -= alpha * Ap
        if np.linalg.norm(r) / nb < tol:
            return it + 1
        z = prec(r)
        rz_new = float((r * z).sum())
        p = z + (rz_new / rz) * p
        rz = rz_new
    return -1


def cmd_cg(args):
    """Question 4: does the V-cycle keep the CG count grid-independent when it
    replaces the exact fine-grid FFT solve?

    Measured on the *heterogeneous* problem -- on the uniform one the exact FFT
    is the exact inverse and converges in a single step, which says nothing.
    """
    rng = np.random.default_rng(2)
    print(f"PCG on the heterogeneous problem, {args.dim}D {args.element}, "
          f"contrast={args.contrast}, material={args.material}, "
          f"tol={args.tol}")
    print(f"V-cycle: nu=({args.nu1},{args.nu2}) cycles={args.cycles}\n")
    print(f"  {'n':>5} {'lvls':>5} {'none':>7} {'FFT':>7} {'MG':>7} "
          f"{'J.FFT.J':>9} {'J.MG.J':>9}")

    for n in args.sizes:
        mg = _build(args, n)
        het = Heterogeneous(mg.levels[0], args.contrast, args.material)

        # Both preconditioners approximate the SAME K_ref, built from the
        # volume-mean Lame parameters of the heterogeneous medium.
        kw = dict(nb_levels=args.levels, coarsest=args.coarsest,
                  nu1=args.nu1, nu2=args.nu2, omega=args.omega)
        el = getattr(muGrid.FEMElement, args.element)
        mg = MultigridReference(args.dim, n, (1.0 / n,) * args.dim, el,
                                het.lam_ref, het.mu_ref, **kw)
        exact = MultigridReference(args.dim, n, (1.0 / n,) * args.dim, el,
                                   het.lam_ref, het.mu_ref,
                                   **{**kw, "nb_levels": 1})

        b = project_mean_out(rng.standard_normal(mg.levels[0].shape))
        op = lambda v: project_mean_out(het.apply(v))  # noqa: E731

        def green_fft(r):
            return exact._solve_coarsest(project_mean_out(r))

        def green_mg(r):
            return mg.apply(r, nb_cycles=args.cycles)

        def jacobi_wrap(green):
            def inner(r):
                return het.jhalf * green(het.jhalf * r)
            return inner

        res = [_cg(op, b, p, args.tol, args.maxiter) for p in (
            project_mean_out, green_fft, green_mg,
            jacobi_wrap(green_fft), jacobi_wrap(green_mg))]
        print(f"  {n:>5} {mg.nb_levels:>5} " + "".join(f"{v:>7}" for v in res[:3])
              + "".join(f"{v:>9}" for v in res[3:]))


def cmd_hybrid(args):
    """Does the hybrid keep the FFT preconditioner's CG count?

    The V-cycle's weakness is not its cost per apply but that it is an
    *approximate* inverse: it buys a cheaper apply with ~1.5x the iterations.
    The hybrid is exact whenever its z-solve is, so the question is what an
    affordable, halo-only z-solve costs in iterations.

    Columns: no preconditioner; the exact fine-grid FFT; the 3D V-cycle; the
    hybrid with an exact z-solve; the hybrid with a z-only V-cycle. Then the
    same five wrapped in the J^{1/2} . G . J^{1/2} scaling.
    """
    rng = np.random.default_rng(2)
    print(f"PCG on the heterogeneous problem, {args.dim}D {args.element}, "
          f"contrast={args.contrast}, material={args.material}, "
          f"tol={args.tol}")
    print(f"3D V-cycle: nu=({args.nu1},{args.nu2}) cycles={args.cycles}   "
          f"hybrid z-cycle: nu={args.nu1} cycles={args.z_cycles}\n")
    print(f"  {'n':>5} {'zlvl':>5} {'none':>7} {'FFT':>7} {'MG':>7} "
          f"{'HybEx':>7} {'HybMG':>7}   {'J.FFT.J':>9} {'J.MG.J':>9} "
          f"{'J.HybEx.J':>10} {'J.HybMG.J':>10}")

    el = getattr(muGrid.FEMElement, args.element)
    for n in args.sizes:
        h = (1.0 / n,) * args.dim
        base = _build(args, n)
        het = Heterogeneous(base.levels[0], args.contrast, args.material)

        kw = dict(nb_levels=args.levels, coarsest=args.coarsest,
                  nu1=args.nu1, nu2=args.nu2, omega=args.omega)
        mg = MultigridReference(args.dim, n, h, el,
                                het.lam_ref, het.mu_ref, **kw)
        exact = MultigridReference(args.dim, n, h, el, het.lam_ref,
                                   het.mu_ref, **{**kw, "nb_levels": 1})
        hyb_exact = HybridFourierTridiagonal(
            args.dim, n, h, el, het.lam_ref, het.mu_ref, z_solve="exact")
        hyb_mg = HybridFourierTridiagonal(
            args.dim, n, h, el, het.lam_ref, het.mu_ref, z_solve="mg",
            nu=args.nu1, cycles=args.z_cycles, coarsest=args.z_coarsest)

        b = project_mean_out(rng.standard_normal(base.levels[0].shape))
        op = lambda v: project_mean_out(het.apply(v))  # noqa: E731

        greens = [
            lambda r: exact._solve_coarsest(project_mean_out(r)),
            lambda r: mg.apply(r, nb_cycles=args.cycles),
            hyb_exact.apply,
            hyb_mg.apply,
        ]

        def jacobi_wrap(green):
            def inner(r):
                return het.jhalf * green(het.jhalf * r)
            return inner

        plain = [_cg(op, b, project_mean_out, args.tol, args.maxiter)]
        plain += [_cg(op, b, g, args.tol, args.maxiter) for g in greens]
        scaled = [_cg(op, b, jacobi_wrap(g), args.tol, args.maxiter)
                  for g in greens]
        print(f"  {n:>5} {hyb_mg.nb_z_levels:>5} "
              + "".join(f"{v:>7}" for v in plain)
              + "  " + "".join(f"{v:>9}" for v in scaled[:2])
              + "".join(f"{v:>10}" for v in scaled[2:]))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe", action="store_true", help="nodal block K_nn")
    p.add_argument("--check", action="store_true", help="transfer + symmetry")
    p.add_argument("--rho", action="store_true", help="V-cycle factor vs n")
    p.add_argument("--cg", action="store_true", help="CG iteration count vs n")
    p.add_argument("--hybrid", action="store_true",
                   help="CG counts for the FFT-in-xy / tridiagonal-in-z hybrid")
    p.add_argument("--z-cycles", type=int, default=1,
                   help="z-only V-cycles per hybrid apply (default: 1)")
    p.add_argument("--z-coarsest", type=int, default=4,
                   help="coarsest z extent for the hybrid's cycle (default: 4)")
    p.add_argument("--dim", type=int, default=2, choices=(2, 3))
    p.add_argument("--element", default="q1", choices=("q1", "p1"))
    p.add_argument("-n", type=int, default=64, help="fine grid points/direction")
    p.add_argument("--sizes", type=int, nargs="+", default=None)
    p.add_argument("--levels", type=int, default=None)
    p.add_argument("--coarsest", type=int, default=8)
    p.add_argument("--nu1", type=int, default=2)
    p.add_argument("--nu2", type=int, default=2)
    p.add_argument("--omega", type=float, default=None,
                   help="damping; default: 1.7/lambda_max (auto)")
    p.add_argument("--cycles", type=int, default=1)
    p.add_argument("--lam", type=float, default=1.3)
    p.add_argument("--mu", type=float, default=0.7)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--contrast", type=float, default=10.0)
    p.add_argument("--material", default="inclusion",
                   choices=("inclusion", "smooth"))
    p.add_argument("--maxiter", type=int, default=200)
    args = p.parse_args()

    if args.sizes is None:
        args.sizes = [args.n] if (args.probe or args.check) else (
            [32, 64, 128, 256] if args.dim == 2 else [16, 32, 64])

    if not any((args.probe, args.check, args.rho, args.cg, args.hybrid)):
        p.error("pick at least one of --probe / --check / --rho / --cg / "
                "--hybrid")
    if args.probe:
        cmd_probe(args)
    if args.check:
        cmd_check(args)
    if args.rho:
        cmd_rho(args)
    if args.cg:
        cmd_cg(args)
    if args.hybrid:
        cmd_hybrid(args)


if __name__ == "__main__":
    main()
