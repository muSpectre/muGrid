"""
Tests of the heterogeneous multigrid (muGrid.Multigrid): the variable-stencil
operator, Galerkin coarse operators by probing and by algebraic RAP.

Everything is compared with dense matrices assembled column by column on small
serial grids.
"""


import warnings

import numpy as np
import pytest

import muGrid
from muGrid.Multigrid import (
    StencilLayout,
    axis_colouring,
    galerkin_by_probing,
    galerkin_rap,
    make_stencil_field,
    stencil_operator,
)

# Every test here is serial (it builds its own one-rank decompositions), so under
# an MPI launcher each rank would only repeat it; the distributed tests are in
# python_heterogeneous_multigrid_mpi_tests.py.
pytestmark = pytest.mark.skipif(
    muGrid.has_mpi and muGrid.Communicator.world().size > 1,
    reason="serial tests; the MPI ones are in "
           "python_heterogeneous_multigrid_mpi_tests.py")


def serial_communicator():
    """A one-rank communicator, also under an MPI launcher (where every rank
    then runs the serial tests on its own)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
                                message=".*serial muGrid Communicator.*")
        return muGrid.Communicator()


def decomposition(nb_grid_pts, nb_sub_pts=None, device=None, comm=None):
    """Serial (default) or distributed decomposition with one ghost layer."""
    dim = len(nb_grid_pts)
    kwargs = {} if nb_sub_pts is None else {"nb_sub_pts": nb_sub_pts}
    if device is not None:
        kwargs["device"] = device
    if comm is None:
        comm = serial_communicator()
        kwargs["nb_subdivisions"] = [1] * dim
    return muGrid.CartesianDecomposition(
        comm, list(nb_grid_pts), nb_ghosts_left=[1] * dim,
        nb_ghosts_right=[1] * dim, **kwargs)


def put(target, values):
    """``target[...] = values`` for a host or device view."""
    if type(target).__module__.startswith("cupy"):
        import cupy as cp
        values = cp.asarray(values)
    target[...] = values


def host(values):
    return values.get() if hasattr(values, "get") else np.asarray(values)


def dense(apply, decomp, nc, name="dense"):
    """Dense matrix of ``apply(x, y)`` on nodal fields, column by column."""
    x = decomp.real_field(f"{name}-x", (nc,))
    y = decomp.real_field(f"{name}-y", (nc,))
    shape = x.s.shape
    n = int(np.prod(shape))
    A = np.empty((n, n))
    for column, entry in enumerate(np.ndindex(*shape)):
        x.set_zero()
        x.s[entry] = 1.0
        apply(x, y)
        A[:, column] = np.asarray(y.s).ravel()
    return A


def dense_prolongation(fine, coarse, nc, dim, name="dense-P"):
    """Dense P (multilinear, component-wise) from GridTransfer.prolong."""
    transfer = muGrid.GridTransfer(dim)

    def apply(xc, yf):
        coarse.communicate_ghosts(xc)
        transfer.prolong(xc, yf)

    xc = coarse.real_field(f"{name}-xc", (nc,))
    yf = fine.real_field(f"{name}-yf", (nc,))
    shape = xc.s.shape
    P = np.empty((int(np.prod(yf.s.shape)), int(np.prod(shape))))
    for column, entry in enumerate(np.ndindex(*shape)):
        xc.set_zero()
        xc.s[entry] = 1.0
        apply(xc, yf)
        P[:, column] = np.asarray(yf.s).ravel()
    return P


def stencil_apply(decomp, layout, stencil):
    """``apply(x, y)`` of the variable-stencil operator with ghost exchange."""
    op = stencil_operator(layout)

    def apply(x, y):
        decomp.communicate_ghosts(x)
        op.apply(stencil, x, y)
    return apply


def reference_dense_from_stencil(values, layout, nb_grid_pts):
    """Independent dense assembly of a periodic stencil field (numpy)."""
    nc = layout.nb_components
    nodes = list(np.ndindex(*nb_grid_pts))
    shape = (nc,) + (1,) + tuple(nb_grid_pts)
    n = int(np.prod(shape))
    A = np.zeros((n, n))

    def dof(i, node):
        return np.ravel_multi_index((i, 0) + tuple(node), shape)

    def wrap(node):
        return tuple(int(c) % m for c, m in zip(node, nb_grid_pts))

    for node in nodes:
        for k, s in enumerate(layout.half_offsets):
            other = wrap(np.add(node, s))
            for i in range(nc):
                for j in range(nc):
                    a = values[(layout.block_entry(k, i, j), 0) + node]
                    A[dof(i, node), dof(j, other)] += a
                    A[dof(j, other), dof(i, node)] += a
        for (i, j), entry in layout.centre_entry.items():
            A[dof(i, node), dof(j, node)] += values[(entry, 0) + node]
            if i != j:
                A[dof(j, node), dof(i, node)] += values[(entry, 0) + node]
    return A


# --------------------------------------------------------------------------- #
# Operators to coarsen
# --------------------------------------------------------------------------- #


def material(decomp, seed, contrast=1e3):
    """Heterogeneous Lame fields (with ghosts), contrast 1e3."""
    rng = np.random.default_rng(seed)
    lam = decomp.real_field("lambda")
    mu = decomp.real_field("mu")
    scale = 10.0 ** rng.uniform(-np.log10(contrast), 0, lam.s.shape)
    lam.s[...] = 2.0 * scale
    mu.s[...] = scale
    decomp.communicate_ghosts(lam)
    decomp.communicate_ghosts(mu)
    return lam, mu


def fused_elasticity(decomp, dim, element, seed=0):
    """Fused isotropic stiffness with a heterogeneous material."""
    nb_grid_pts = tuple(decomp.nb_domain_grid_pts)
    op = muGrid.IsotropicStiffnessOperator(
        dim, [1.0 / n for n in nb_grid_pts], element)
    lam, mu = material(decomp, seed)

    def apply(u, f):
        decomp.communicate_ghosts(u)
        op.apply(u, lam, mu, f)
    return apply


def fem_conductivity(nb_grid_pts, element, seed=0):
    """Scalar Bᵀ W k_q B with a heterogeneous coefficient per quadrature
    point (nb_components = 1); returns the decomposition and the apply."""
    dim = len(nb_grid_pts)
    grad = muGrid.FEMGradientOperator(
        dim, grid_spacing=[1.0 / n for n in nb_grid_pts], element=element)
    decomp = decomposition(nb_grid_pts, {"quad": grad.nb_quad_pts})
    gradient = decomp.real_field("grad", (dim,), "quad")
    rng = np.random.default_rng(seed)
    coefficient = 10.0 ** rng.uniform(-3, 0, gradient.s.shape[1:])

    def apply(u, f):
        decomp.communicate_ghosts(u)
        grad.apply(u, gradient)
        gradient.s[...] *= coefficient
        decomp.communicate_ghosts(gradient)
        grad.transpose(gradient, f)
    return decomp, apply


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", list(range(3, 41)))
def test_axis_colouring_separates_neighbours(n):
    """Points at periodic distance 1 or 2 get different colours, with 3 colours
    when n is divisible by 3 and at most 4 otherwise (5 for n = 5)."""
    colours = axis_colouring(n)
    for d in (1, 2):
        assert np.all(colours != np.roll(colours, d))
    expected = 3 if n % 3 == 0 else (5 if n == 5 else 4)
    assert colours.max() + 1 == expected


@pytest.mark.parametrize("nb_grid_pts, nc, dtype", [
    ((4, 5), 1, np.float64), ((5, 4), 2, np.float64), ((4, 3, 5), 3, np.float64),
    ((3, 4, 4), 1, np.float64), ((4, 5), 2, np.float32), ((4, 4, 3), 3, np.float32)])
def test_variable_stencil_operator_matches_dense_assembly(nb_grid_pts, nc,
                                                          dtype):
    """The C++ kernel applies the symmetric stencil it stores: equal to an
    independent dense assembly, periodic through the ghosts, for every
    component count and both precisions."""
    dim = len(nb_grid_pts)
    layout = StencilLayout(dim, nc)
    decomp = decomposition(nb_grid_pts)
    stencil = make_stencil_field(decomp, layout, "stencil", dtype)
    rng = np.random.default_rng(1)
    stencil.s[...] = rng.standard_normal(stencil.s.shape)
    decomp.communicate_ghosts(stencil)
    expected = reference_dense_from_stencil(
        np.asarray(stencil.s, dtype=np.float64), layout, nb_grid_pts)
    op = stencil_operator(layout)
    x = decomp.real_field("x", (nc,), dtype=dtype)
    y = decomp.real_field("y", (nc,), dtype=dtype)
    x.s[...] = rng.standard_normal(x.s.shape)
    decomp.communicate_ghosts(x)
    op.apply(stencil, x, y)
    tol = 1e-12 if dtype == np.float64 else 1e-5
    np.testing.assert_allclose(
        np.asarray(y.s, dtype=np.float64).ravel(),
        expected @ np.asarray(x.s, dtype=np.float64).ravel(),
        rtol=tol, atol=tol * np.abs(expected).max() * 10)
    y_before = np.array(y.s)
    op.apply_increment(stencil, x, -2.0, y)
    np.testing.assert_allclose(np.asarray(y.s), -y_before, rtol=tol,
                               atol=tol * np.abs(y_before).max())


def test_variable_stencil_operator_rejects_bad_fields():
    """Wrong component counts, a stencil field of the wrong size, aliasing x
    and y, and fields without ghosts raise."""
    layout = StencilLayout(2, 2)
    decomp = decomposition((4, 4))
    op = stencil_operator(layout)
    stencil = make_stencil_field(decomp, layout, "stencil")
    x = decomp.real_field("x", (2,))
    y = decomp.real_field("y", (2,))
    with pytest.raises(RuntimeError, match="components"):
        op.apply(stencil, decomp.real_field("x1", (1,)), y)
    with pytest.raises(RuntimeError, match="nb_entries"):
        op.apply(decomp.real_field("bad", (5,)), x, y)
    with pytest.raises(RuntimeError, match="different"):
        op.apply(stencil, x, x)
    no_ghosts = muGrid.CartesianDecomposition(
        muGrid.Communicator(), [4, 4], nb_subdivisions=[1, 1])
    with pytest.raises(RuntimeError, match="ghost"):
        op.apply(make_stencil_field(no_ghosts, layout, "s"),
                 no_ghosts.real_field("x", (2,)),
                 no_ghosts.real_field("y", (2,)))
    with pytest.raises(RuntimeError, match="nb_components"):
        muGrid.VariableStencilOperator(2, 4)


def coarse_galerkin(apply_fine, fine, nb_grid_pts, nc, name="coarse"):
    """Probe R A P on the coarse grid of half the size."""
    dim = len(nb_grid_pts)
    coarse = decomposition(tuple(n // 2 for n in nb_grid_pts))
    layout = StencilLayout(dim, nc)
    stencil = make_stencil_field(coarse, layout, f"{name}-stencil")
    galerkin_by_probing(apply_fine, muGrid.GridTransfer(dim), fine, coarse,
                        layout, stencil)
    return coarse, layout, stencil


# coarse grids (4, 6) ... are coloured periodically (strided extraction);
# (7, 5) by blocks of 3 and 4 and by 5 colours (index arrays)
ELASTICITY = [((8, 12), muGrid.FEMElement.q1), ((12, 8), muGrid.FEMElement.p1),
              ((14, 10), muGrid.FEMElement.q1),
              ((8, 8, 8), muGrid.FEMElement.q1), ((8, 6, 8), muGrid.FEMElement.p1)]


@pytest.mark.parametrize("nb_grid_pts, element", ELASTICITY)
def test_probed_coarse_operator_is_the_galerkin_product(nb_grid_pts, element):
    """Heterogeneous fused elasticity (contrast 1e3, 2D/3D, Q1/P1): the probed
    coarse stencil equals Pᵀ A P assembled densely, so applying it reproduces
    R A P on any coarse vector."""
    dim = len(nb_grid_pts)
    fine = decomposition(nb_grid_pts)
    apply_fine = fused_elasticity(fine, dim, element, seed=2)
    coarse, layout, stencil = coarse_galerkin(apply_fine, fine, nb_grid_pts,
                                              dim)
    A = dense(apply_fine, fine, dim, "A")
    P = dense_prolongation(fine, coarse, dim, dim)
    expected = P.T @ A @ P
    probed = dense(stencil_apply(coarse, layout, stencil), coarse, dim, "AH")
    np.testing.assert_allclose(probed, expected, rtol=1e-12,
                               atol=1e-12 * np.abs(expected).max())


@pytest.mark.parametrize("nb_grid_pts, element", [
    ((8, 12), muGrid.FEMElement.q1), ((12, 8), muGrid.FEMElement.p1),
    ((8, 8, 6), muGrid.FEMElement.q1)])
def test_probed_scalar_operator_is_the_galerkin_product(nb_grid_pts, element):
    """The same for a scalar FE chain Bᵀ W k_q B with one coefficient per
    quadrature point (nb_components = 1): the probing is not tied to the
    fused elasticity operator."""
    fine, apply_fine = fem_conductivity(nb_grid_pts, element, seed=4)
    coarse, layout, stencil = coarse_galerkin(apply_fine, fine, nb_grid_pts,
                                              1)
    A = dense(apply_fine, fine, 1, "A")
    P = dense_prolongation(fine, coarse, 1, len(nb_grid_pts))
    expected = P.T @ A @ P
    probed = dense(stencil_apply(coarse, layout, stencil), coarse, 1, "AH")
    np.testing.assert_allclose(probed, expected, rtol=1e-12,
                               atol=1e-12 * np.abs(expected).max())


@pytest.mark.parametrize("fine_pts, nc", [((12, 8), 2), ((8, 12), 1),
                                          ((8, 8, 6), 3), ((6, 8, 8), 1)])
def test_rap_is_the_galerkin_product(fine_pts, nc):
    """Algebraic RAP on an arbitrary symmetric stencil equals Pᵀ A P."""
    dim = len(fine_pts)
    layout = StencilLayout(dim, nc)
    fine = decomposition(fine_pts)
    coarse = decomposition(tuple(n // 2 for n in fine_pts))
    fine_stencil = make_stencil_field(fine, layout, "fine-stencil")
    rng = np.random.default_rng(5)
    fine_stencil.s[...] = rng.standard_normal(fine_stencil.s.shape)
    fine.communicate_ghosts(fine_stencil)
    coarse_stencil = make_stencil_field(coarse, layout, "coarse-stencil")
    galerkin_rap(fine_stencil, fine, coarse_stencil, coarse, layout)
    A = dense(stencil_apply(fine, layout, fine_stencil), fine, nc, "A")
    P = dense_prolongation(fine, coarse, nc, dim)
    expected = P.T @ A @ P
    rap = dense(stencil_apply(coarse, layout, coarse_stencil), coarse, nc,
                "AH")
    np.testing.assert_allclose(rap, expected, rtol=1e-12,
                               atol=1e-12 * np.abs(expected).max())


def test_two_levels_rap_after_probing():
    """Level 2 by RAP of the probed level 1 equals (P₁P₂)ᵀ A (P₁P₂): the two
    constructions compose to the Galerkin hierarchy."""
    nb_grid_pts = (24, 16)
    fine = decomposition(nb_grid_pts)
    apply_fine = fused_elasticity(fine, 2, muGrid.FEMElement.q1, seed=6)
    level1, layout, stencil1 = coarse_galerkin(apply_fine, fine, nb_grid_pts,
                                               2, "l1")
    level2 = decomposition((6, 4))
    stencil2 = make_stencil_field(level2, layout, "l2-stencil")
    galerkin_rap(stencil1, level1, stencil2, level2, layout)
    A = dense(apply_fine, fine, 2, "A")
    P1 = dense_prolongation(fine, level1, 2, 2, "P1")
    P2 = dense_prolongation(level1, level2, 2, 2, "P2")
    P = P1 @ P2
    expected = P.T @ A @ P
    got = dense(stencil_apply(level2, layout, stencil2), level2, 2, "A2")
    np.testing.assert_allclose(got, expected, rtol=1e-11,
                               atol=1e-12 * np.abs(expected).max())


# --------------------------------------------------------------------------- #
# Stage 3: the multigrid (serial, host)
# --------------------------------------------------------------------------- #

from muGrid.Multigrid import (  # noqa: E402
    MultigridPreconditioner,
    ScaledLocalMatrices,
)
from muGrid.Solvers import conjugate_gradients  # noqa: E402

LAMBDA_0, MU_0 = 0.6, 0.4


def phase_field(nb_grid_pts, contrast, width=0.04, seed=0):
    """Smooth two-phase geometry: tanh interfaces around random discs/balls;
    s = 1 in the matrix, 1/contrast in the inclusions."""
    rng = np.random.default_rng(seed)
    dim = len(nb_grid_pts)
    coords = np.meshgrid(*[np.arange(n) / n for n in nb_grid_pts],
                         indexing="ij")
    distance = np.full(nb_grid_pts, np.inf)
    for _ in range(6):
        centre = rng.uniform(0, 1, dim)
        radius = rng.uniform(0.08, 0.18)
        d2 = sum((np.minimum(np.abs(c - x0), 1 - np.abs(c - x0))) ** 2
                 for c, x0 in zip(coords, centre))
        distance = np.minimum(distance, np.sqrt(d2) - radius)
    inside = 0.5 * (1 - np.tanh(distance / width))
    return 10.0 ** (-np.log10(contrast) * inside)


def crack(nb_grid_pts, depth=1e-4):
    """Homogeneous material cut by a straight crack through the whole cell
    (one pixel wide, s = depth): a localised mode the previous setup did not
    see."""
    s = np.ones(nb_grid_pts)
    s[nb_grid_pts[0] // 2] = depth
    return s


class FusedProblem:
    """Fused isotropic elasticity A = s(x) K(λ₀, μ₀), one s per pixel.

    ``coarsening``: "galerkin" (default); "rediscretised" (the coarsening
    rule with λ/μ = λ₀/μ₀, one family); "rediscretised-lame" (the rule with
    two families λ, μ)."""

    def __init__(self, nb_grid_pts, element=muGrid.FEMElement.q1,
                 structured=True, device=None, precision="double",
                 single_fine=True, comm=None, coarsening="galerkin"):
        self.dim = dim = len(nb_grid_pts)
        self.nb_grid_pts = nb_grid_pts
        self.comm = comm
        self.decomposition = d = decomposition(nb_grid_pts, device=device,
                                               comm=comm)
        self.window = tuple(slice(int(o), int(o) + int(n)) for o, n in zip(
            d.subdomain_locations, d.nb_subdomain_grid_pts))
        self.op = muGrid.IsotropicStiffnessOperator(
            dim, [1.0 / n for n in nb_grid_pts], element)
        self.scale = d.real_field("scale")
        self.lam = d.real_field("lambda")
        self.mu = d.real_field("mu")
        self.lam32 = d.real_field("lambda32", dtype=np.float32)
        self.mu32 = d.real_field("mu32", dtype=np.float32)
        self.set_scale(np.ones(nb_grid_pts))
        local = (ScaledLocalMatrices.isotropic_stiffness(
            self.op, LAMBDA_0, MU_0, self.scale) if structured else None)
        if coarsening != "galerkin":
            from muGrid.MultigridCoarsening import IsotropicRediscretisation
            coarsening = IsotropicRediscretisation(
                element, [1.0 / n for n in nb_grid_pts], (self.lam, self.mu),
                lambda_over_mu=(LAMBDA_0 / MU_0
                                if coarsening == "rediscretised" else None))
        self.mg = MultigridPreconditioner(
            self.apply, d, dim, local_matrices=local, precision=precision,
            operator_single=self.apply_single if single_fine else None,
            communicator=comm, coarsening=coarsening)

    def set_scale(self, s):
        """``s`` on the global grid; this rank's part is used."""
        s = s[self.window]
        put(self.scale.s[0], s)
        put(self.lam.s[0], LAMBDA_0 * s)
        put(self.mu.s[0], MU_0 * s)
        put(self.lam32.s[0], (LAMBDA_0 * s).astype(np.float32))
        put(self.mu32.s[0], (MU_0 * s).astype(np.float32))
        for f in (self.scale, self.lam, self.mu, self.lam32, self.mu32):
            self.decomposition.communicate_ghosts(f)

    def apply(self, u, f):
        self.decomposition.communicate_ghosts(u)
        self.op.apply(u, self.lam, self.mu, f)

    def apply_single(self, u, f):
        self.decomposition.communicate_ghosts(u)
        self.op.apply(u, self.lam32, self.mu32, f)


def cg_iterations(problem, seed=0, rtol=1e-8, maxiter=300):
    """PCG with the multigrid on a random zero-mean right-hand side."""
    d, nc = problem.decomposition, problem.dim
    b = d.real_field("cg-b", (nc,))
    x = d.real_field("cg-x", (nc,))
    put(b.s, global_rhs(problem, nc, seed))
    x.set_zero()
    iterations = []
    comm = getattr(problem, "comm", None) or serial_communicator()
    conjugate_gradients(comm, d.collection, b, x,
                        hessp=problem.apply, prec=problem.mg, rtol=rtol,
                        maxiter=maxiter,
                        callback=lambda it, state: iterations.append(it))
    return max(iterations)


def global_rhs(problem, nc, seed):
    """A zero-mean random right-hand side on the global grid, this rank's
    part (the same numbers for every decomposition)."""
    rng = np.random.default_rng(seed)
    grid = tuple(int(n) for n in problem.decomposition.nb_domain_grid_pts)
    values = rng.standard_normal((nc, 1) + grid)
    values -= values.mean(axis=tuple(range(2, values.ndim)), keepdims=True)
    window = getattr(problem, "window", tuple(slice(None) for _ in grid))
    return values[(slice(None), slice(None)) + window]


def level_matrix(mg, index, name="lvl"):
    """Dense operator of level ``index`` and its diagonal."""
    level = mg.levels[index]
    A = dense(level.apply, level.decomposition, mg.nb_components,
              f"{name}{index}")
    return A


def true_lambda_max(A):
    """λmax(D⁻¹A) by a dense eigensolver (Lanczos-free, exact)."""
    d = np.diag(A).copy()
    d[d == 0] = 1.0
    s = 1 / np.sqrt(d)
    return float(np.linalg.eigvalsh(s[:, None] * A * s[None, :])[-1])


@pytest.mark.parametrize("nb_grid_pts, element", [
    ((8, 8), muGrid.FEMElement.q1), ((8, 8), muGrid.FEMElement.p1),
    ((8, 8, 8), muGrid.FEMElement.q1), ((8, 8, 8), muGrid.FEMElement.p1)])
def test_local_matrices_reproduce_the_operator(nb_grid_pts, element):
    """The probed pixel matrix of the fused operator reproduces it (the
    construction-time safeguard passes) and wrong data is refused."""
    problem = FusedProblem(nb_grid_pts, element)
    rng = np.random.default_rng(0)
    problem.set_scale(10.0 ** rng.uniform(-4, 0, nb_grid_pts))
    lm = problem.mg.local_matrices
    assert lm.check_operator(problem.apply, problem.decomposition,
                             problem.dim) < 1e-12
    wrong = ScaledLocalMatrices(1.1 * lm.matrices, problem.scale)
    with pytest.raises(ValueError, match="not Σ_pixels"):
        wrong.check_operator(problem.apply, problem.decomposition,
                             problem.dim)


def test_negative_scale_is_refused():
    problem = FusedProblem((16, 16))
    s = np.ones((16, 16))
    s[3, 4] = -1e-3
    problem.set_scale(s)
    with pytest.raises(ValueError, match=">= 0"):
        problem.mg.setup()


MATERIALS = {
    "homogeneous": lambda n: np.ones(n),
    "random 1e4": lambda n: 10.0 ** np.random.default_rng(7).uniform(-4, 0, n),
    "crack": crack,
    "phase field 1e4": lambda n: phase_field(n, 1e4),
}


@pytest.mark.parametrize("material", list(MATERIALS))
@pytest.mark.parametrize("nb_grid_pts, element", [
    ((16, 16), muGrid.FEMElement.q1), ((16, 16), muGrid.FEMElement.p1),
    ((16, 16, 8), muGrid.FEMElement.q1)])
def test_bound_is_above_lambda_max_on_every_level(nb_grid_pts, element,
                                                  material):
    """b_l ≥ λmax(D_l⁻¹A_l) on every smoothed level, for homogeneous,
    extreme random, crack-like and phase-field materials (dense eigensolver on
    the exact level operators). Both the bound used and the
    material-independent c_l of the local matrices are checked; on coarse
    levels the bound used is min(c_l, Gershgorin of the stencil)."""
    problem = FusedProblem(nb_grid_pts, element)
    problem.set_scale(MATERIALS[material](nb_grid_pts))
    problem.mg.setup()
    lm = problem.mg.local_matrices
    for level in problem.mg.levels[:-1]:
        if np.prod(level.nb_grid_pts) * problem.dim > 2600:
            continue                       # dense eigensolver too large
        lmax = true_lambda_max(level_matrix(problem.mg, level.index))
        c_l = lm.bound(level.index)
        assert level.bound_source in ("local matrices", "gershgorin")
        assert level.bound <= c_l
        assert lmax <= level.bound * (1 + 1e-12), (level.index, lmax,
                                                   level.bound)
        assert lmax <= c_l * (1 + 1e-12)


def test_cycle_is_symmetric_positive_definite_3d():
    """3D: the cycle is linear, symmetric and positive on zero-mean random
    vectors (the dense spectrum is checked in 2D below)."""
    n = (8, 8, 8)
    problem = FusedProblem(n)
    problem.set_scale(MATERIALS["random 1e4"](n))
    problem.mg.setup()
    d = problem.decomposition
    rng = np.random.default_rng(13)
    vectors = []
    for _ in range(3):
        v = rng.standard_normal((3, 1) + n)
        vectors.append(v - v.mean(axis=(2, 3, 4), keepdims=True))
    r = d.real_field("spd-r", (3,))
    z = d.real_field("spd-z", (3,))

    def M(v):
        put(r.s, v)
        problem.mg(r, z)
        return host(z.s).copy()

    a, b, c = vectors
    Ma, Mb = M(a), M(b)
    scale = np.abs(Ma).max() * np.abs(b).max() * a.size
    assert abs(np.sum(b * Ma) - np.sum(a * Mb)) < 1e-10 * scale
    for v, Mv in ((a, Ma), (b, Mb), (c, M(c))):
        assert np.sum(v * Mv) > 0
    np.testing.assert_allclose(M(2 * a - 3 * b), 2 * Ma - 3 * Mb,
                               atol=1e-10 * np.abs(Ma).max())


@pytest.mark.parametrize("nb_grid_pts", [(16, 16)])
def test_cycle_is_symmetric_positive_definite(nb_grid_pts):
    """The V-cycle is a fixed linear SPD operator on zero-mean vectors."""
    problem = FusedProblem(nb_grid_pts)
    problem.set_scale(MATERIALS["random 1e4"](nb_grid_pts))
    problem.mg.setup()
    d, nc = problem.decomposition, problem.dim
    M = dense(problem.mg, d, nc, "M")
    zero_mean = np.eye(M.shape[0]) - np.kron(
        np.eye(nc), np.full((np.prod(nb_grid_pts),) * 2,
                            1 / np.prod(nb_grid_pts)))
    M0 = zero_mean @ M @ zero_mean
    np.testing.assert_allclose(M0, M0.T, atol=1e-10 * np.abs(M0).max())
    eigenvalues = np.linalg.eigvalsh(0.5 * (M0 + M0.T))
    assert (eigenvalues > -1e-10 * eigenvalues.max()).all()
    assert np.sum(eigenvalues > 1e-10 * eigenvalues.max()) == M.shape[0] - nc


def test_cg_iterations_are_grid_independent_on_a_phase_field():
    """Phase field, contrast 1e4: PCG iterations stay flat under refinement."""
    counts = []
    for n in (32, 64, 128):
        problem = FusedProblem((n, n))
        problem.set_scale(phase_field((n, n), 1e4))
        problem.mg.setup()
        counts.append(cg_iterations(problem))
    print("phase field 2D, N = 32/64/128:", counts)
    assert max(counts) < 30
    assert max(counts) - min(counts) <= 4


def test_localised_mode_after_setup_keeps_cg_fast():
    """The AT2 failure mode of the power estimate: a crack appears after the
    previous setup. The material-independent bounds c_l are the same before
    and after, the bounds used stay above λmax, and CG stays fast."""
    n = (16, 16)
    problem = FusedProblem(n)
    problem.mg.setup()
    lm = problem.mg.local_matrices
    before = [lm.bound(level.index) for level in problem.mg.levels[:-1]]
    homogeneous = cg_iterations(problem)
    problem.set_scale(crack(n))
    problem.mg.setup()
    after = [lm.bound(level.index) for level in problem.mg.levels[:-1]]
    assert before == after
    for level in problem.mg.levels[:-1]:
        lmax = true_lambda_max(level_matrix(problem.mg, level.index, "c"))
        assert lmax <= level.bound * (1 + 1e-12)
    cracked = cg_iterations(problem)
    print("homogeneous / cracked:", homogeneous, cracked)
    assert cracked < 40


@pytest.mark.parametrize("nb_grid_pts, element", [
    ((64, 64), muGrid.FEMElement.q1), ((64, 64), muGrid.FEMElement.p1),
    ((16, 16, 16), muGrid.FEMElement.q1)])
def test_cg_converges_fast_for_fused_elasticity(nb_grid_pts, element):
    problem = FusedProblem(nb_grid_pts, element)
    problem.set_scale(phase_field(nb_grid_pts, 1e3))
    problem.mg.setup()
    iterations = cg_iterations(problem)
    print(nb_grid_pts, element, iterations, problem.mg.level_summary())
    assert iterations < 30


class FEMProblem:
    """Bᵀ W (s_q C) B with a fixed (possibly anisotropic) tensor C and one
    scale per quadrature point: elasticity (nc = d) or conductivity (nc = 1).
    """

    def __init__(self, nb_grid_pts, element, tensor, structured=True,
                 seed=0):
        self.dim = dim = len(nb_grid_pts)
        nc = np.asarray(tensor).shape[0]
        self.nb_components = nc
        self.grad = muGrid.FEMGradientOperator(
            dim, grid_spacing=[1.0 / n for n in nb_grid_pts], element=element)
        nq = self.grad.nb_quad_pts
        self.decomposition = d = decomposition(nb_grid_pts, {"quad": nq})
        self.tensor = np.asarray(tensor)
        self.gradient = d.real_field("fem-grad", (nc, dim), "quad")
        self.flux = d.real_field("fem-flux", (nc, dim), "quad")
        self.scale = d.real_field("fem-scale", (), "quad")
        rng = np.random.default_rng(seed)
        pixel = phase_field(nb_grid_pts, 1e3)
        self.scale.s[...] = pixel[None] * rng.uniform(1.0, 1.5, (nq,) + tuple(
            nb_grid_pts))
        local = (ScaledLocalMatrices.fem(self.grad, self.tensor, self.scale)
                 if structured else None)
        self.mg = MultigridPreconditioner(self.apply, d, nc,
                                          local_matrices=local)

    def apply(self, u, f):
        self.decomposition.communicate_ghosts(u)
        self.grad.apply(u, self.gradient)
        self.flux.s[...] = np.einsum("ikjl,jlq...->ikq...", self.tensor,
                                     self.gradient.s) * self.scale.s
        self.decomposition.communicate_ghosts(self.flux)
        self.grad.transpose(self.flux, f)


def anisotropic_elasticity_tensor(dim, seed=3):
    """A fixed anisotropic, minor- and major-symmetric positive definite
    stiffness tensor C_ijkl (random Voigt matrix)."""
    rng = np.random.default_rng(seed)
    pairs = [(i, j) for i in range(dim) for j in range(i, dim)]
    m = len(pairs)
    G = rng.standard_normal((m, m))
    voigt = G @ G.T + m * np.eye(m)
    C = np.zeros((dim,) * 4)
    for a, (i, j) in enumerate(pairs):
        for b, (k, l) in enumerate(pairs):
            for (p, q) in {(i, j), (j, i)}:
                for (r, s) in {(k, l), (l, k)}:
                    C[p, q, r, s] = voigt[a, b]
    return C


FEM_CASES = {
    "2D P1 elasticity, anisotropic": ((16, 16), muGrid.FEMElement.p1, 2),
    "2D Q1 elasticity, anisotropic": ((16, 16), muGrid.FEMElement.q1, 2),
    "3D Q1 elasticity, anisotropic": ((8, 8, 8), muGrid.FEMElement.q1, 3),
    "2D Q1 conductivity, anisotropic": ((32, 32), muGrid.FEMElement.q1, 1),
}


def fem_tensor(dim, nc):
    if nc == 1:
        A = np.array([[2.0, 0.7], [0.7, 1.0]]) if dim == 2 else np.diag(
            [2.0, 1.0, 1.5])
        return A[None, :, None, :]
    return anisotropic_elasticity_tensor(dim)


@pytest.mark.parametrize("case", list(FEM_CASES))
def test_fem_chain_with_per_quadrature_point_material(case):
    """Bᵀ W (s_q C) B, anisotropic C, a scale per quadrature point: the
    structure check passes, the bound holds on the levels a dense eigensolver
    can afford, and CG converges fast."""
    nb_grid_pts, element, nc = FEM_CASES[case]
    dim = len(nb_grid_pts)
    problem = FEMProblem(nb_grid_pts, element, fem_tensor(dim, nc))
    problem.mg.setup()
    for level in problem.mg.levels[:-1]:
        if np.prod(level.nb_grid_pts) * nc > 1200:     # dense eigensolve
            continue
        lmax = true_lambda_max(level_matrix(problem.mg, level.index, "f"))
        assert lmax <= level.bound * (1 + 1e-12)
    problem.dim = nc                                    # for cg_iterations
    iterations = cg_iterations(problem)
    print(case, iterations, problem.mg.level_summary())
    assert iterations < 30


def test_fallback_bound_without_local_structure():
    """Without local matrices the bound comes from the entries (Gershgorin,
    probed on the fine level): still above λmax on every level, and CG still
    converges, with a looser smoother."""
    n = (16, 16)
    structured = FusedProblem(n)
    black_box = FusedProblem(n, structured=False)
    for problem in (structured, black_box):
        problem.set_scale(phase_field(n, 1e4))
        problem.mg.setup()
    for level in black_box.mg.levels[:-1]:
        assert level.bound_source.startswith("gershgorin")
        lmax = true_lambda_max(level_matrix(black_box.mg, level.index, "g"))
        assert lmax <= level.bound * (1 + 1e-12)
    counts = cg_iterations(structured), cg_iterations(black_box)
    print("structured / black box:", counts,
          structured.mg.level_summary(), black_box.mg.level_summary())
    assert counts[1] < 60


# --------------------------------------------------------------------------- #
# Stage 4: GPU
# --------------------------------------------------------------------------- #

try:
    import cupy  # noqa: F401
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

gpu = pytest.mark.skipif(
    not (muGrid.has_gpu and HAS_CUPY and muGrid.is_gpu_available()),
    reason="needs a GPU build, a GPU and CuPy")


@gpu
@pytest.mark.parametrize("nb_grid_pts, nc, dtype", [
    ((6, 5), 1, np.float64), ((5, 6), 2, np.float32), ((6, 5, 4), 3, np.float64),
    ((4, 5, 6), 2, np.float32), ((5, 4, 6), 1, np.float64)])
def test_variable_stencil_operator_on_device_matches_host(nb_grid_pts, nc,
                                                          dtype):
    layout = StencilLayout(len(nb_grid_pts), nc)
    rng = np.random.default_rng(8)
    stencil_values = rng.standard_normal((layout.nb_entries, 1) + nb_grid_pts)
    x_values = rng.standard_normal((nc, 1) + nb_grid_pts)
    results = []
    for device in (None, muGrid.Device.gpu()):
        d = decomposition(nb_grid_pts, device=device)
        stencil = make_stencil_field(d, layout, "stencil", dtype)
        x = d.real_field("x", (nc,), dtype=dtype)
        y = d.real_field("y", (nc,), dtype=dtype)
        put(stencil.s, stencil_values.astype(dtype))
        put(x.s, x_values.astype(dtype))
        d.communicate_ghosts(stencil)
        d.communicate_ghosts(x)
        op = stencil_operator(layout)
        op.apply(stencil, x, y)
        op.apply_increment(stencil, x, 0.5, y)
        results.append(host(y.s))
    tol = 1e-12 if dtype == np.float64 else 2e-5
    np.testing.assert_allclose(results[1], results[0], rtol=tol,
                               atol=tol * np.abs(results[0]).max())


@gpu
@pytest.mark.parametrize("fine_pts, nc, dtype", [
    ((8, 12), 2, np.float64), ((12, 8), 1, np.float32),
    ((8, 6, 10), 3, np.float64), ((6, 8, 4), 3, np.float32)])
def test_grid_transfer_on_device_matches_host(fine_pts, nc, dtype):
    dim = len(fine_pts)
    coarse_pts = tuple(n // 2 for n in fine_pts)
    rng = np.random.default_rng(9)
    fine_values = rng.standard_normal((nc, 1) + fine_pts).astype(dtype)
    coarse_values = rng.standard_normal((nc, 1) + coarse_pts).astype(dtype)
    results = []
    for device in (None, muGrid.Device.gpu()):
        fd = decomposition(fine_pts, device=device)
        cd = decomposition(coarse_pts, device=device)
        f_in = fd.real_field("f-in", (nc,), dtype=dtype)
        f_out = fd.real_field("f-out", (nc,), dtype=dtype)
        c_in = cd.real_field("c-in", (nc,), dtype=dtype)
        c_out = cd.real_field("c-out", (nc,), dtype=dtype)
        put(f_in.s, fine_values)
        put(c_in.s, coarse_values)
        fd.communicate_ghosts(f_in)
        cd.communicate_ghosts(c_in)
        transfer = muGrid.GridTransfer(dim)
        transfer.restrict(f_in, c_out)
        transfer.prolong(c_in, f_out)
        results.append((host(c_out.s), host(f_out.s)))
    tol = 1e-13 if dtype == np.float64 else 1e-6
    for got, expected in zip(results[1], results[0]):
        np.testing.assert_allclose(got, expected, rtol=tol, atol=tol)


@gpu
@pytest.mark.parametrize("nb_grid_pts", [(64, 64), (16, 16, 16)])
def test_multigrid_on_device_matches_host(nb_grid_pts):
    """The whole multigrid on the GPU: the same coarse stencils, bounds and
    V-cycle output as on the host, and the same CG iteration count."""
    problems = [FusedProblem(nb_grid_pts),
                FusedProblem(nb_grid_pts, device=muGrid.Device.gpu())]
    scale = phase_field(nb_grid_pts, 1e4)
    for problem in problems:
        problem.set_scale(scale)
        problem.mg.setup()
    cpu, gpu_ = (p.mg for p in problems)
    np.testing.assert_allclose([lvl.bound for lvl in gpu_.levels[:-1]],
                               [lvl.bound for lvl in cpu.levels[:-1]],
                               rtol=1e-12)
    for a, b in zip(cpu.levels[1:], gpu_.levels[1:]):
        expected = host(a.stencil.s)
        np.testing.assert_allclose(host(b.stencil.s), expected, rtol=1e-11,
                                   atol=1e-11 * np.abs(expected).max())
    rng = np.random.default_rng(10)
    dim = len(nb_grid_pts)
    r_values = rng.standard_normal((dim, 1) + nb_grid_pts)
    r_values -= r_values.mean(axis=tuple(range(2, 2 + dim)), keepdims=True)
    outputs = []
    for problem in problems:
        d = problem.decomposition
        r = d.real_field("vc-r", (dim,))
        z = d.real_field("vc-z", (dim,))
        put(r.s, r_values)
        problem.mg(r, z)
        outputs.append(host(z.s))
    np.testing.assert_allclose(outputs[1], outputs[0], rtol=1e-9,
                               atol=1e-10 * np.abs(outputs[0]).max())
    counts = [cg_iterations(p) for p in problems]
    print(nb_grid_pts, "CPU / GPU iterations:", counts)
    assert counts[0] == counts[1]


# --------------------------------------------------------------------------- #
# Stage 5: float32 cycle
# --------------------------------------------------------------------------- #


def devices():
    out = [None]
    if muGrid.has_gpu and HAS_CUPY and muGrid.is_gpu_available():
        out.append(muGrid.Device.gpu())
    return out


@pytest.mark.parametrize("device", devices(),
                         ids=lambda d: "gpu" if d is not None else "cpu")
@pytest.mark.parametrize("nb_grid_pts", [(64, 64), (16, 16, 16)])
def test_single_precision_cycle_iterations(nb_grid_pts, device):
    """A float32 cycle between float64 CG vectors keeps CG's accuracy (the
    true residual reaches 1e-8) and costs at most a few extra iterations;
    the difference to float64 is printed. Both variants: the fine level in
    float32 (operator_single) and in float64."""
    scale = phase_field(nb_grid_pts, 1e4)
    counts = {}
    for label, precision, single_fine in (("double", "double", True),
                                          ("single", "single", True),
                                          ("single, f64 fine", "single",
                                           False)):
        problem = FusedProblem(nb_grid_pts, device=device,
                               precision=precision, single_fine=single_fine)
        problem.set_scale(scale)
        problem.mg.setup()
        counts[label] = cg_iterations(problem)
        if precision == "single":
            stencil = problem.mg.levels[1].stencil
            assert np.dtype(stencil.dtype) == np.float32
            names = _collection_names(problem.mg.levels[1].decomposition)
            assert not any("stencil64" in n or "dinv64" in n for n in names)
    print(nb_grid_pts, "device" if device else "host", counts)
    assert counts["single"] <= counts["double"] + 3
    assert counts["single, f64 fine"] <= counts["double"] + 3


def _collection_names(decomposition):
    collection = decomposition.collection
    return list(collection.field_names)


def test_single_precision_cycle_is_symmetric_to_round_off():
    """The float32 cycle is SPD to single-precision round-off."""
    n = (16, 16)
    problem = FusedProblem(n, precision="single")
    problem.set_scale(MATERIALS["random 1e4"](n))
    problem.mg.setup()
    M = dense(problem.mg, problem.decomposition, 2, "M32")
    P0 = np.eye(M.shape[0]) - np.kron(np.eye(2), np.full((256, 256), 1 / 256))
    M0 = P0 @ M @ P0
    assert np.abs(M0 - M0.T).max() < 1e-5 * np.abs(M0).max()
    eigenvalues = np.linalg.eigvalsh(0.5 * (M0 + M0.T))
    assert eigenvalues.min() > -1e-5 * eigenvalues.max()


@pytest.mark.parametrize("precision, single_fine", [
    ("double", True), ("single", True), ("single", False)])
def test_apply_reads_only_its_input(precision, single_fine):
    """M r depends on r only, and r is not modified, whichever fields earlier
    applies were given (regression: the second apply used to copy its r into
    the first caller's r)."""
    n = (16, 16)
    problem = FusedProblem(n, precision=precision, single_fine=single_fine)
    problem.set_scale(MATERIALS["random 1e4"](n))
    problem.mg.setup()
    d = problem.decomposition
    rng = np.random.default_rng(12)
    fields = {}
    for name in ("r1", "z1", "r2", "z2"):
        fields[name] = d.real_field(name, (2,))
    for name in ("r1", "r2"):
        values = rng.standard_normal(fields[name].s.shape)
        values -= values.mean(axis=(2, 3), keepdims=True)
        put(fields[name].s, values)
    r1, r2 = host(fields["r1"].s).copy(), host(fields["r2"].s).copy()
    problem.mg(fields["r1"], fields["z1"])
    z1 = host(fields["z1"].s).copy()
    problem.mg(fields["r2"], fields["z2"])
    problem.mg(fields["r1"], fields["z1"])
    np.testing.assert_array_equal(host(fields["r1"].s), r1)
    np.testing.assert_array_equal(host(fields["r2"].s), r2)
    np.testing.assert_allclose(host(fields["z1"].s), z1, rtol=1e-12, atol=0)


def test_reexported_from_preconditioners():
    """The multigrid is importable with the other preconditioners."""
    from muGrid.Preconditioners import MultigridPreconditioner as M
    from muGrid.Preconditioners import ScaledLocalMatrices as S
    assert M is MultigridPreconditioner and S is ScaledLocalMatrices
    with pytest.raises(AttributeError):
        from muGrid import Preconditioners
        Preconditioners.NoSuchPreconditioner  # noqa: B018


# --------------------------------------------------------------------------- #
# Lean setup and cycle
# --------------------------------------------------------------------------- #

LEFTOVER_TAGS = ("mg-check", "stencil64", "dinv64", "-probe-", "-setup-",
                 "rap-block")


@pytest.mark.parametrize("device", devices(),
                         ids=lambda d: "gpu" if d is not None else "cpu")
@pytest.mark.parametrize("precision", ["double", "single"])
def test_only_the_cycle_fields_stay(precision, device):
    """After construction and two setups no field of the operator check, the
    probing, the setup's double copies or the RAP is left in any level's
    collection (the operator check's two fine vectors used to stay for the
    whole run)."""
    problem = FusedProblem((16, 16, 16), device=device, precision=precision)
    problem.set_scale(phase_field((16, 16, 16), 1e4))
    problem.mg.setup()
    problem.mg.setup()
    decompositions = [problem.decomposition] + [
        level.decomposition for level in problem.mg.levels[1:]]
    leftover = [name for d in decompositions for name in _collection_names(d)
                if any(tag in name for tag in LEFTOVER_TAGS)]
    assert leftover == []


@pytest.mark.parametrize("device", devices(),
                         ids=lambda d: "gpu" if d is not None else "cpu")
def test_setup_needs_little_memory_beyond_the_cycle(device):
    """A repeated setup of a float32 cycle allocates at most ~50 bytes per
    fine voxel beyond the fields the cycle keeps (the double diagonal of the
    fine level, the RAP block): the coarse stencils are measured in place,
    the probes use the fine level's vectors. It used to be ~250 B/voxel,
    mostly double copies of the stencils."""
    n = (32, 32, 32)
    problem = FusedProblem(n, device=device, precision="single")
    problem.set_scale(phase_field(n, 1e4))
    problem.mg.setup()
    pool = "host" if device is None else "cuda"

    def bytes_of(report, key):
        return sum(p[key] for name, p in report["pools"].items()
                   if name.startswith(pool))

    muGrid.memory_profiler.enable()
    try:
        muGrid.memory_profiler.reset()
        problem.mg.setup()
        report = muGrid.memory_profiler.report()
    finally:
        muGrid.memory_profiler.disable()
    transient = bytes_of(report, "peak_bytes") - bytes_of(report,
                                                          "current_bytes")
    per_voxel = transient / np.prod(n)
    print(f"setup transient {per_voxel:.1f} B/voxel")
    assert per_voxel < 80


@pytest.mark.skipif(len(devices()) < 2, reason="needs a GPU")
@pytest.mark.parametrize("precision", ["double", "single"])
def test_jacobi_on_whole_buffers_is_exact(precision):
    """On a device the Jacobi step multiplies whole contiguous buffers; on the
    interior it equals D⁻¹ r exactly, also with arbitrary finite values in
    the ghosts of r and z."""
    from muGrid.Multigrid import _contiguous_view

    problem = FusedProblem((16, 16, 16), device=devices()[1],
                           precision=precision)
    problem.set_scale(phase_field((16, 16, 16), 1e4))
    mg = problem.mg
    mg.setup()
    rng = np.random.default_rng(3)
    for level in mg.levels[:-1]:
        r, z = level.vector("jacobi-r"), level.vector("jacobi-z")
        for field in (r, z):
            put(field.sg, rng.standard_normal(field.sg.shape))
        assert all(_contiguous_view(f, 3) is not None
                   for f in (level.dinv, r, z))
        expected = host(level.dinv.s) * host(r.s)
        MultigridPreconditioner._jacobi(level, r, z)
        np.testing.assert_array_equal(host(z.s), expected)


@pytest.mark.parametrize("device", devices(),
                         ids=lambda d: "gpu" if d is not None else "cpu")
def test_precision_cast_copies_whole_buffers(device):
    """The float64 <-> float32 copies between CG and cycle copy every entry,
    ghosts included (on a device over the contiguous buffers)."""
    from muGrid.Multigrid import _copy_cast

    d = decomposition((8, 6, 4), device=device)
    rng = np.random.default_rng(4)
    double = d.real_field("cast-double", (3,))
    single = d.real_field("cast-single", (3,), dtype=np.float32)
    back = d.real_field("cast-back", (3,))
    values = rng.standard_normal(double.sg.shape)
    put(double.sg, values)
    _copy_cast(double, single)
    np.testing.assert_array_equal(host(single.sg), values.astype(np.float32))
    _copy_cast(single, back)
    np.testing.assert_array_equal(host(back.sg),
                                  values.astype(np.float32).astype(np.float64))


# --------------------------------------------------------------------------- #
# Re-discretised coarse levels (muGrid.MultigridCoarsening)
# --------------------------------------------------------------------------- #

RULES = ["rediscretised", "rediscretised-lame"]


def block_mean(s, levels):
    """The material ``s`` averaged ``levels`` times over 2^d child voxels."""
    for _ in range(levels):
        dim = s.ndim
        s = np.mean([s[tuple(slice(o, None, 2) for o in offset)]
                     for offset in np.ndindex(*(2,) * dim)], axis=0)
    return s


@pytest.mark.parametrize("coarsening", RULES)
@pytest.mark.parametrize("nb_grid_pts", [(16, 16), (8, 8, 8)])
def test_rediscretised_level_is_the_coarse_fused_operator(nb_grid_pts,
                                                          coarsening):
    """Every coarse level applies the fused operator of the coarse grid
    (spacing 2^l h) with λ, μ the means of the child voxels, built here
    independently on its own decomposition (dense, on the host; the device
    is compared with the host below)."""
    device = None
    s = MATERIALS["random 1e4"](nb_grid_pts)
    problem = FusedProblem(nb_grid_pts, device=device, coarsening=coarsening)
    problem.set_scale(s)
    problem.mg.setup()
    dim = len(nb_grid_pts)
    for level in problem.mg.levels[1:]:
        # the fused operator on its own coarse decomposition
        d = decomposition(level.nb_grid_pts, device=device)
        op = muGrid.IsotropicStiffnessOperator(
            dim, [2 ** level.index / n for n in nb_grid_pts],
            muGrid.FEMElement.q1)
        lam, mu = d.real_field("ref-lambda"), d.real_field("ref-mu")
        coarse = block_mean(s, level.index)
        put(lam.s[0], LAMBDA_0 * coarse)
        put(mu.s[0], MU_0 * coarse)
        d.communicate_ghosts(lam)
        d.communicate_ghosts(mu)

        def reference(u, f, d=d, op=op, lam=lam, mu=mu):
            d.communicate_ghosts(u)
            op.apply(u, lam, mu, f)

        A = dense(level.apply, level.decomposition, dim, f"rule{level.index}")
        expected = dense(reference, d, dim, f"ref{level.index}")
        np.testing.assert_allclose(A, expected, rtol=1e-12,
                                   atol=1e-12 * np.abs(expected).max())


@pytest.mark.parametrize("nb_grid_pts", [(16, 16), (8, 8, 8)])
def test_rediscretised_equals_galerkin_for_a_homogeneous_material(
        nb_grid_pts):
    """For a homogeneous Q1 material the re-discretised and the Galerkin
    coarse operators are the same (multilinear interpolation is exact)."""
    dim = len(nb_grid_pts)
    operators = []
    for coarsening in ("galerkin", "rediscretised"):
        problem = FusedProblem(nb_grid_pts, coarsening=coarsening)
        problem.set_scale(np.ones(nb_grid_pts))
        problem.mg.setup()
        level = problem.mg.levels[1]
        operators.append(dense(level.apply, level.decomposition, dim,
                               f"homogeneous-{coarsening}"))
    np.testing.assert_allclose(operators[1], operators[0], rtol=1e-11,
                               atol=1e-11 * np.abs(operators[0]).max())


@pytest.mark.parametrize("coarsening", RULES)
@pytest.mark.parametrize("material", list(MATERIALS))
@pytest.mark.parametrize("nb_grid_pts", [(16, 16), (16, 16, 8)])
def test_rediscretised_bound_is_above_lambda_max(nb_grid_pts, material,
                                                 coarsening):
    """b_l ≥ λmax(D_l⁻¹A_l) on every smoothed re-discretised level (dense
    eigensolver), for all test materials, with one family (λ/μ fixed) and
    with two (λ, μ)."""
    problem = FusedProblem(nb_grid_pts, coarsening=coarsening)
    problem.set_scale(MATERIALS[material](nb_grid_pts))
    problem.mg.setup()
    for level in problem.mg.levels[:-1]:
        if np.prod(level.nb_grid_pts) * problem.dim > 2600:
            continue                       # dense eigensolver too large
        lmax = true_lambda_max(level_matrix(problem.mg, level.index))
        assert lmax <= level.bound * (1 + 1e-12), (level.index, lmax,
                                                   level.bound)


@pytest.mark.parametrize("coarsening", RULES)
def test_rediscretised_cycle_is_symmetric_positive_definite(coarsening):
    """The cycle with re-discretised levels is a fixed linear SPD operator on
    zero-mean vectors (dense spectrum, 2D)."""
    n = (16, 16)
    problem = FusedProblem(n, coarsening=coarsening)
    problem.set_scale(MATERIALS["random 1e4"](n))
    problem.mg.setup()
    d, nc = problem.decomposition, problem.dim
    M = dense(problem.mg, d, nc, "M-rule")
    zero_mean = np.eye(M.shape[0]) - np.kron(
        np.eye(nc), np.full((np.prod(n),) * 2, 1 / np.prod(n)))
    M0 = zero_mean @ M @ zero_mean
    np.testing.assert_allclose(M0, M0.T, atol=1e-10 * np.abs(M0).max())
    eigenvalues = np.linalg.eigvalsh(0.5 * (M0 + M0.T))
    assert (eigenvalues > -1e-10 * eigenvalues.max()).all()
    assert np.sum(eigenvalues > 1e-10 * eigenvalues.max()) == M.shape[0] - nc


@pytest.mark.parametrize("device", devices(),
                         ids=lambda d: "gpu" if d is not None else "cpu")
@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("nb_grid_pts", [(64, 64), (32, 32, 32)])
def test_rediscretised_cg_iterations_are_close_to_galerkin(nb_grid_pts,
                                                           precision, device):
    """Phase field, contrast 1e4: the re-discretised levels cost at most a
    few CG iterations more than Galerkin's, in both precisions, and leave no
    stencil."""
    scale = phase_field(nb_grid_pts, 1e4)
    counts = {}
    for coarsening in ("galerkin", "rediscretised"):
        problem = FusedProblem(nb_grid_pts, device=device,
                               precision=precision, coarsening=coarsening)
        problem.set_scale(scale)
        problem.mg.setup()
        counts[coarsening] = cg_iterations(problem)
        if coarsening != "galerkin":
            assert all(level.stencil is None
                       for level in problem.mg.levels)
    print(nb_grid_pts, precision, counts)
    assert counts["rediscretised"] <= counts["galerkin"] + 4


def test_rediscretised_cg_iterations_are_grid_independent():
    """Phase field 2D, contrast 1e4: PCG iterations stay flat under
    refinement with re-discretised levels too."""
    counts = []
    for n in (32, 64, 128):
        problem = FusedProblem((n, n), coarsening="rediscretised")
        problem.set_scale(phase_field((n, n), 1e4))
        problem.mg.setup()
        counts.append(cg_iterations(problem))
    print("phase field 2D, re-discretised, N = 32/64/128:", counts)
    assert max(counts) < 30
    assert max(counts) - min(counts) <= 4


@pytest.mark.skipif(len(devices()) < 2, reason="needs a GPU")
def test_rediscretised_on_device_matches_host():
    """The same V-cycle output on the device as on the host (double)."""
    n = (16, 16, 16)
    scale = phase_field(n, 1e4)
    outputs = []
    for device in devices():
        problem = FusedProblem(n, device=device, coarsening="rediscretised")
        problem.set_scale(scale)
        problem.mg.setup()
        d = problem.decomposition
        r = d.real_field("rule-r", (3,))
        z = d.real_field("rule-z", (3,))
        put(r.s, global_rhs(problem, 3, 5))
        problem.mg(r, z)
        outputs.append(host(z.s).copy())
    np.testing.assert_allclose(outputs[1], outputs[0], rtol=1e-9,
                               atol=1e-9 * np.abs(outputs[0]).max())


def test_rediscretisation_refuses_p1_and_a_wrong_grid_spacing():
    """P1 has no nested re-discretisation; a grid spacing that does not match
    the fine operator is caught at construction (in 3D: the 2D elastic
    element matrix does not depend on an isotropic h)."""
    from muGrid.MultigridCoarsening import IsotropicRediscretisation

    with pytest.raises(ValueError, match="q1"):
        FusedProblem((16, 16), element=muGrid.FEMElement.p1,
                     coarsening="rediscretised")
    problem = FusedProblem((8, 8, 8))
    wrong = IsotropicRediscretisation(muGrid.FEMElement.q1, [2 / 8] * 3,
                                      (problem.lam, problem.mu))
    with pytest.raises(ValueError, match="does not reproduce"):
        MultigridPreconditioner(problem.apply, problem.decomposition, 3,
                                coarsening=wrong)
    with pytest.raises(ValueError, match="coarsening"):
        MultigridPreconditioner(problem.apply, problem.decomposition, 3,
                                coarsening="algebraic")


@pytest.mark.parametrize("device", devices(),
                         ids=lambda d: "gpu" if d is not None else "cpu")
def test_rediscretised_setup_needs_little_memory(device):
    """A repeated setup with re-discretised levels allocates almost nothing
    beyond the cycle's fields: the double fine diagonal (~26 B/voxel)."""
    n = (32, 32, 32)
    problem = FusedProblem(n, device=device, precision="single",
                           coarsening="rediscretised")
    problem.set_scale(phase_field(n, 1e4))
    problem.mg.setup()
    pool = "host" if device is None else "cuda"
    muGrid.memory_profiler.enable()
    try:
        muGrid.memory_profiler.reset()
        problem.mg.setup()
        report = muGrid.memory_profiler.report()
    finally:
        muGrid.memory_profiler.disable()
    transient = sum(p["peak_bytes"] - p["current_bytes"]
                    for name, p in report["pools"].items()
                    if name.startswith(pool))
    per_voxel = transient / np.prod(n)
    print(f"re-discretised setup transient {per_voxel:.1f} B/voxel")
    assert per_voxel < 40
