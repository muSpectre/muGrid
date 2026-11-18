"""
Tests for muGrid.Solvers.conjugate_gradients_energy and EnergyErrorEstimate:
conjugate gradients stopped on the adaptive-delay lower estimate of the
energy-norm error (Strakoš & Tichý 2002; Meurant, Papež & Tichý 2021).
"""

import numpy as np
import pytest
from conftest import (
    create_device,
    get_array_module,
    get_test_devices,
    skip_if_gpu_unavailable,
)

import muGrid
from muGrid import Communicator, GlobalFieldCollection
from muGrid.Solvers import (
    ConvergenceError,
    EnergyErrorEstimate,
    conjugate_gradients_energy,
)

serial_only = pytest.mark.skipif(
    Communicator().size > 1, reason="Test only works in serial"
)


# ---------------------------------------------------------------------------
# Reference: the algorithm as it is written in the muFFTTO solver the muGrid
# one is ported from, on plain numpy vectors. It keeps the delay counter of
# that code, which muGrid replaces by an equivalent count of steps.
# ---------------------------------------------------------------------------


def reference_safety_factor(tails, steps, first):
    tails = np.array(tails)
    larger = np.where((tails[first] / tails) <= 1e-4)[0]
    start = larger[-1] if larger.size > 0 else 0
    return np.max(tails[start:-1] / np.asarray(steps)[start:-1])


class ReferenceEstimate:
    """Accepted estimates for a sequence of steps, reference bookkeeping."""

    def __init__(self, tau=0.25, d_min=0):
        self.tau, self.d_min = tau, d_min
        self.first, self.delay = 0, 0
        self.steps, self.tails = [], []
        self.error_sq = None

    def update(self, alpha, rz):
        self.steps.append(alpha * rz)
        self.tails = [t + self.steps[-1] for t in self.tails] + [
            self.steps[-1]
        ]
        if len(self.steps) > 2:
            safety = reference_safety_factor(
                self.tails, self.steps, self.first
            )
            numerator = safety * self.steps[-1]
            denominator = float(np.sum(self.steps[self.first:-1]))
            while (
                self.delay >= 0
                and denominator > 0
                and numerator / denominator <= self.tau
                and len(self.steps) - self.first >= self.d_min
            ):
                self.error_sq = denominator + self.steps[-1]
                self.first, self.delay = self.first + 1, self.delay - 1
                denominator = float(np.sum(self.steps[self.first:-1]))
            self.delay += 1


def reference_cg(matvec, prec, b, x0, rtol, estimator, maxiter=10000):
    """Dense PCG with the energy stopping test. Returns the solution, the
    number of iterations and, per iteration, the estimator's state
    (iterate the estimate belongs to, estimate) and all iterates."""
    x = x0.copy()
    Ax = matvec(x)
    start_energy_sq = float(x @ Ax)
    r = b - Ax
    z = prec(r)
    p = z.copy()
    rz = float(r @ z)
    iterates, history = [x.copy()], []
    for iteration in range(1, maxiter + 1):
        Ap = matvec(p)
        alpha = rz / float(p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        z = prec(r)
        next_rz = float(r @ z)
        p = z + next_rz / rz * p
        estimator.update(alpha, rz)
        rz = next_rz
        iterates.append(x.copy())
        history.append(estimator.error_sq)
        energy_sq = (
            start_energy_sq if start_energy_sq > 0 else estimator.tails[0]
        )
        if (
            estimator.error_sq is not None
            and estimator.error_sq < rtol**2 * energy_sq
        ):
            return x, iteration, history, iterates
    raise RuntimeError("reference CG did not converge")


def spd_matrix(n, condition, seed=0):
    """Random symmetric positive definite matrix with the given condition
    number and eigenvalues spread evenly on a logarithmic scale."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return (q * np.logspace(0, np.log10(condition), n)) @ q.T


# ---------------------------------------------------------------------------
# The estimator alone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d_min", [0, 5, 10])
@pytest.mark.parametrize(
    "kind", ["linear", "stagnation", "superlinear", "noisy"]
)
def test_estimator_accepts_as_the_reference(kind, d_min):
    """Same accepted estimates, bit for bit, as the reference bookkeeping,
    for step sequences of different convergence behaviour."""
    rng = np.random.default_rng(1)
    k = np.arange(120)
    steps = {
        "linear": 0.8**k,
        "stagnation": np.where(k < 60, 0.995**k, 0.995**60 * 0.5 ** (k - 60)),
        "superlinear": np.exp(-0.002 * k**2),
        "noisy": 0.9**k * rng.uniform(0.2, 5.0, k.size),
    }[kind]
    ours = EnergyErrorEstimate(d_min=d_min)
    reference = ReferenceEstimate(d_min=d_min)
    accepted = 0
    for step in steps:
        ours.update(2.0, step / 2.0)
        reference.update(2.0, step / 2.0)
        assert ours.error_sq == reference.error_sq
        if ours.error_sq is not None:
            accepted += 1
            assert ours.iteration == reference.first - 1
        assert ours.initial_error_sq == reference.tails[0]
    assert accepted > 0


@pytest.mark.parametrize("condition", [1e2, 1e5])
def test_estimate_is_a_sharp_lower_bound(condition):
    """On a system with a known solution the estimate never exceeds the true
    squared energy error of its iterate. On the well-conditioned system,
    where CG converges steadily, it is also not far below it (the target
    1 - tau is a heuristic, so only half the error is required)."""
    n = 200
    A = spd_matrix(n, condition)
    rng = np.random.default_rng(2)
    solution = rng.standard_normal(n)
    b = A @ solution
    tau = 0.25
    estimator = EnergyErrorEstimate(tau=tau)
    x, r = np.zeros(n), b.copy()
    p, rz = r.copy(), float(r @ r)
    true_error_sq = [float(solution @ A @ solution)]
    checked = 0
    for _ in range(150):
        Ap = A @ p
        alpha = rz / float(p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        next_rz = float(r @ r)
        p = r + next_rz / rz * p
        estimator.update(alpha, rz)
        rz = next_rz
        true_error_sq.append(float((solution - x) @ A @ (solution - x)))
        if estimator.error_sq is not None:
            truth = true_error_sq[estimator.iteration]
            if truth < 1e-20 * true_error_sq[0]:
                break  # below what round-off lets the scalars describe
            assert estimator.error_sq <= truth * (1 + 1e-10)
            if condition <= 1e2:
                assert estimator.error_sq >= 0.5 * truth
            checked += 1
    assert checked > 20


def test_estimator_rejects_bad_parameters():
    with pytest.raises(ValueError):
        EnergyErrorEstimate(tau=1.0)
    with pytest.raises(ValueError):
        EnergyErrorEstimate(tau=0.0)
    with pytest.raises(ValueError):
        EnergyErrorEstimate(d_min=-1)


# ---------------------------------------------------------------------------
# The solver on a dense system
# ---------------------------------------------------------------------------


def dense_problem(A, b, x0):
    fc = GlobalFieldCollection((len(b),))
    rhs = fc.real_field("rhs")
    rhs.p[...] = b
    solution = fc.real_field("solution")
    solution.p[...] = x0
    counts = {"hessp": 0, "prec": 0}

    def hessp(x, Ax):
        counts["hessp"] += 1
        Ax.p[...] = A @ x.p

    diagonal = np.diag(A)

    def jacobi(r, z):
        counts["prec"] += 1
        z.p[...] = r.p / diagonal

    return fc, rhs, solution, hessp, jacobi, counts


@serial_only
@pytest.mark.parametrize("d_min", [0, 10])
@pytest.mark.parametrize("warm_start", [False, True])
@pytest.mark.parametrize("preconditioned", [False, True])
def test_same_iterates_as_the_reference(preconditioned, warm_start, d_min):
    """Iteration count and solution equal those of the reference algorithm,
    for cold and warm starts, with and without a preconditioner, and the
    solve costs one operator and one preconditioner product per iteration
    plus one each at the start."""
    # A sparse matrix, the screened Laplacian of a periodic 20 x 20 grid
    # with a random coefficient: its products carry little round-off, so the
    # iteration counts of two implementations can be compared exactly. (A
    # dense random matrix shifts the stopping iteration by one now and then.)
    rng = np.random.default_rng(3)
    m = 20
    n = m * m
    shift = np.roll(np.eye(m), 1, axis=0)
    laplace_1d = 2 * np.eye(m) - shift - shift.T
    A = (np.kron(laplace_1d, np.eye(m)) + np.kron(np.eye(m), laplace_1d)
         + np.diag(10.0 ** rng.uniform(-2, 1, n)))
    b = rng.standard_normal(n)
    x0 = rng.standard_normal(n) if warm_start else np.zeros(n)
    rtol = 1e-6

    def reference_prec(r):
        return r / np.diag(A) if preconditioned else r.copy()

    x_ref, iterations_ref, history_ref, _ = reference_cg(
        lambda v: A @ v, reference_prec, b, x0, rtol,
        ReferenceEstimate(d_min=d_min),
    )

    fc, rhs, solution, hessp, jacobi, counts = dense_problem(A, b, x0)
    history = []
    conjugate_gradients_energy(
        Communicator(), fc, rhs, solution, hessp,
        prec=jacobi if preconditioned else None, rtol=rtol, d_min=d_min,
        maxiter=5000,
        callback=lambda it, state: history.append(state["energy_error_sq"]),
    )
    iterations = len(history) - 1  # the callback is also called at the start

    assert iterations == iterations_ref
    # the estimates agree to round-off (numpy's and muGrid's dot products
    # sum in different orders)
    for estimate, estimate_ref in zip(history[1:], history_ref):
        assert (estimate is None) == (estimate_ref is None)
        if estimate is not None:
            assert estimate == pytest.approx(estimate_ref, rel=1e-8)
    np.testing.assert_allclose(solution.p, x_ref, rtol=1e-8, atol=1e-10)
    assert counts["hessp"] == iterations + 1
    assert counts["prec"] == (iterations + 1 if preconditioned else 0)


@serial_only
@pytest.mark.parametrize("warm_start", [False, True])
def test_requested_energy_accuracy_is_reached(warm_start):
    """The true energy error of the returned iterate is below rtol times the
    energy of the solution, up to the underestimation the rule allows."""
    n = 300
    A = spd_matrix(n, 1e5, seed=4)
    rng = np.random.default_rng(5)
    exact = rng.standard_normal(n)
    b = A @ exact
    # the warm start is a perturbed solution, as in a sequence of solves
    x0 = exact * (1 + 1e-2 * rng.standard_normal(n)) if warm_start else 0.0
    rtol, tau = 1e-5, 0.25
    fc, rhs, solution, hessp, _, _ = dense_problem(A, b, x0)
    conjugate_gradients_energy(
        Communicator(), fc, rhs, solution, hessp, rtol=rtol, tau=tau
    )
    error = solution.p - exact
    energy_error = np.sqrt(error @ A @ error)
    energy = np.sqrt(exact @ A @ exact)
    assert energy_error <= 1.5 * rtol * energy
    # and not wastefully far below it
    assert energy_error >= 1e-3 * rtol * energy


@serial_only
def test_start_vector_that_solves_the_system_returns_at_once():
    A = spd_matrix(20, 10.0)
    fc, rhs, solution, hessp, _, counts = dense_problem(
        A, np.zeros(20), np.zeros(20)
    )
    conjugate_gradients_energy(Communicator(), fc, rhs, solution, hessp)
    assert counts["hessp"] == 1
    np.testing.assert_array_equal(solution.p, 0.0)


@serial_only
def test_small_system_ends_with_the_exact_solution():
    """CG on a 3x3 system terminates before any estimate can be accepted."""
    A = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 2.0]])
    b = np.array([1.0, 2.0, 3.0])
    fc, rhs, solution, hessp, _, _ = dense_problem(A, b, np.zeros(3))
    try:
        conjugate_gradients_energy(
            Communicator(), fc, rhs, solution, hessp, rtol=1e-10, maxiter=10
        )
    except ConvergenceError:
        # round-off left a residual of 1e-16 instead of exactly zero; the
        # iterate is then still the solution
        pass
    np.testing.assert_allclose(solution.p, np.linalg.solve(A, b), rtol=1e-12)


@serial_only
def test_maxiter_raises():
    A = spd_matrix(200, 1e6)
    fc, rhs, solution, hessp, _, _ = dense_problem(
        A, np.ones(200), np.zeros(200)
    )
    with pytest.raises(ConvergenceError, match="did not converge"):
        conjugate_gradients_energy(
            Communicator(), fc, rhs, solution, hessp, maxiter=5
        )


@serial_only
def test_indefinite_operator_raises():
    A = np.diag([1.0, -1.0, 2.0, 3.0])
    fc, rhs, solution, hessp, _, _ = dense_problem(
        A, np.ones(4), np.zeros(4)
    )
    with pytest.raises(ConvergenceError, match="not positive"):
        conjugate_gradients_energy(Communicator(), fc, rhs, solution, hessp)


@serial_only
def test_indefinite_preconditioner_raises():
    A = spd_matrix(50, 100.0)
    fc, rhs, solution, hessp, _, _ = dense_problem(
        A, np.ones(50), np.zeros(50)
    )

    def negative(r, z):
        z.p[...] = -r.p

    with pytest.raises(ConvergenceError, match="not positive definite"):
        conjugate_gradients_energy(
            Communicator(), fc, rhs, solution, hessp, prec=negative
        )


# ---------------------------------------------------------------------------
# The solver on a grid: MPI, GPU and single precision
# ---------------------------------------------------------------------------

NB_GRID_PTS = (32, 32)
GRID_SPACING = 1 / NB_GRID_PTS[0]


def global_coefficient_and_rhs():
    """Coefficient c and right-hand side of the screened Poisson problem
    (-Laplace + c) u = f on the whole grid, the same on every rank."""
    x, y = np.meshgrid(
        *(np.arange(n) / n for n in NB_GRID_PTS), indexing="ij"
    )
    c = 1 + 1e4 * (np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)) ** 2
    rng = np.random.default_rng(6)
    return c, rng.standard_normal(NB_GRID_PTS)


def global_reference(rtol, d_min, warm_start):
    """Serial numpy solve of the whole problem with the reference algorithm,
    Jacobi preconditioned."""
    c, f = global_coefficient_and_rhs()

    def matvec(v):
        u = v.reshape(NB_GRID_PTS)
        laplace = sum(
            np.roll(u, 1, axis) + np.roll(u, -1, axis) - 2 * u
            for axis in range(2)
        )
        return (-laplace / GRID_SPACING**2 + c * u).ravel()

    diagonal = (4 / GRID_SPACING**2 + c).ravel()
    x0 = (f / c).ravel() if warm_start else np.zeros(f.size)
    x, iterations, _, _ = reference_cg(
        matvec, lambda r: r / diagonal, f.ravel(), x0, rtol,
        ReferenceEstimate(d_min=d_min),
    )
    return x.reshape(NB_GRID_PTS), iterations


def to_host(array):
    return array.get() if hasattr(array, "get") else np.asarray(array)


@pytest.mark.parametrize("device", get_test_devices())
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("warm_start", [False, True])
@pytest.mark.parametrize("d_min", [0, 10])
def test_grid_solve_matches_serial_reference(
    comm, device, dtype, warm_start, d_min
):
    """A distributed solve, on the host or the device, makes the same number
    of iterations as the serial reference on the whole grid and returns its
    solution: the stopping test uses global scalars only."""
    skip_if_gpu_unavailable(device)
    xp = get_array_module(device)
    # single precision carries the recursion to about 1e-5 of the energy
    rtol = 1e-6 if dtype == np.float64 else 1e-4

    laplace = muGrid.LaplaceOperator(2, -1.0 / GRID_SPACING**2)
    engine = muGrid.FFTEngine(
        NB_GRID_PTS, comm, ghosts=laplace, device=create_device(device)
    )
    ox, oy = engine.subdomain_locations
    lx, ly = engine.nb_subdomain_grid_pts
    c_global, f_global = global_coefficient_and_rhs()
    local = (slice(ox, ox + lx), slice(oy, oy + ly))
    c = xp.asarray(c_global[local], dtype=dtype)
    diagonal = 4 / GRID_SPACING**2 + c

    fc = engine.real_space_collection
    rhs = fc.real_field("rhs", dtype=dtype)
    solution = fc.real_field("solution", dtype=dtype)
    rhs.p[...] = xp.asarray(f_global[local], dtype=dtype)
    if warm_start:
        solution.p[...] = xp.asarray((f_global / c_global)[local], dtype=dtype)

    def hessp(x_field, Ax_field):
        engine.communicate_ghosts(x_field)
        laplace.apply(x_field, Ax_field)
        Ax_field.p[...] += c * x_field.p

    def jacobi(r, z):
        z.p[...] = r.p / diagonal

    iterations = []
    conjugate_gradients_energy(
        comm, fc, rhs, solution, hessp, prec=jacobi, rtol=rtol, d_min=d_min,
        callback=lambda it, state: iterations.append(it),
    )

    x_ref, iterations_ref = global_reference(rtol, d_min, warm_start)
    if dtype == np.float64:
        assert max(iterations) == iterations_ref
        tolerance = 1e-9
    else:
        # round-off in single precision may shift the stopping iteration
        assert abs(max(iterations) - iterations_ref) <= 2
        tolerance = 1e-3
    scale = np.abs(x_ref).max()
    np.testing.assert_allclose(
        to_host(solution.p), x_ref[local], atol=tolerance * scale
    )
